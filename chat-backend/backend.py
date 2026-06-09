"""
chat-backend/backend.py
=======================
MCP-native chat backend using direct httpx MCP protocol implementation.

Speaks true MCP Streamable HTTP protocol (POST with JSON-RPC) but uses
httpx directly instead of the mcp library's streamablehttp_client transport,
which has TaskGroup issues in Cloud Run to Cloud Run communication.

The protocol is identical — tools/list and tools/call JSON-RPC over HTTP.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL    = os.environ.get(
    "MCP_SERVER_URL",
    "https://retail-mcp-server-630538663455.us-central1.run.app"
)
MCP_BEARER_TOKEN  = os.environ.get("MCP_BEARER_TOKEN", "")

# Trailing slash prevents 307 redirect which strips Authorization header
MCP_ENDPOINT = f"{MCP_SERVER_URL}/mcp/"

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── MCP protocol over httpx ────────────────────────────────
# We implement the MCP Streamable HTTP protocol directly using httpx.
# This is the same protocol the mcp library uses internally, but without
# the TaskGroup async complexity that causes Cloud Run to Cloud Run failures.
#
# MCP Streamable HTTP protocol:
#   POST /mcp/
#   Content-Type: application/json
#   Authorization: Bearer <token>
#   Body: {"jsonrpc": "2.0", "id": N, "method": "...", "params": {...}}

def _mcp_headers() -> dict:
    """Build headers for MCP requests."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"
    return headers


async def _mcp_request(method: str, params: dict, request_id: int = 1) -> dict:
    """
    Send a single MCP JSON-RPC request and return the result.
    Uses httpx directly — no TaskGroup, no session state, no SSE streaming.
    Each call is a simple POST → response cycle.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=False,  # Don't follow — trailing slash already correct
    ) as client:
        resp = await client.post(
            MCP_ENDPOINT,
            headers=_mcp_headers(),
            json=payload,
        )

    if resp.status_code == 401:
        raise Exception("MCP server returned 401 — check MCP_BEARER_TOKEN")
    if resp.status_code == 403:
        raise Exception("MCP server returned 403 — bearer token rejected")
    if resp.status_code not in (200, 202):
        raise Exception(f"MCP server returned HTTP {resp.status_code}: {resp.text[:200]}")

    # MCP Streamable HTTP can return SSE or JSON
    # If SSE, parse the data lines; if JSON, parse directly
    content_type = resp.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        # Parse SSE — extract data: lines and combine JSON
        result_data = None
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    result_data = json.loads(line[6:])
                    break
                except json.JSONDecodeError:
                    continue
        if result_data is None:
            raise Exception(f"Could not parse SSE response: {resp.text[:200]}")
        return result_data
    else:
        return resp.json()


async def get_mcp_tools() -> list[dict]:
    """
    Discover tools from MCP server using tools/list.
    Returns tool definitions in Anthropic's format for Claude.
    """
    response = await _mcp_request("tools/list", {})

    if "error" in response:
        raise Exception(f"MCP tools/list error: {response['error']}")

    tools = response.get("result", {}).get("tools", [])
    logger.info(f"Discovered {len(tools)} tools from MCP server")

    # Convert MCP schema → Anthropic format
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


async def run_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """
    Call a tool on the MCP server using tools/call.
    Returns the tool result as a string for Claude to process.
    """
    logger.info(f"MCP tools/call: {tool_name}({json.dumps(tool_input)[:100]})")

    response = await _mcp_request(
        "tools/call",
        {"name": tool_name, "arguments": tool_input},
        request_id=2,
    )

    if "error" in response:
        raise Exception(f"MCP tools/call error: {response['error']}")

    result = response.get("result", {})
    content = result.get("content", [])

    if content and isinstance(content, list):
        parts = [block.get("text", "") for block in content if "text" in block]
        return "\n".join(parts) if parts else json.dumps(result)

    return json.dumps(result)


# ── FastAPI app ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Retail Chat UI starting ===")
    logger.info(f"MCP endpoint : {MCP_ENDPOINT}")
    logger.info(f"Anthropic key: {'set' if ANTHROPIC_API_KEY else 'MISSING'}")
    logger.info(f"Bearer token : {'set' if MCP_BEARER_TOKEN else 'not set'}")
    try:
        tools = await get_mcp_tools()
        logger.info(f"MCP connected — tools: {[t['name'] for t in tools]}")
    except Exception as e:
        logger.warning(f"MCP not reachable at startup (will retry per request): {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Retail Intelligence Chat",
    description="MCP-native chat UI for retail PostgreSQL database",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []


SYSTEM_PROMPT = """You are a retail operations assistant with direct access to a live
PostgreSQL database containing inventory, supplier, promotion, and product data for a
retail confectionery business.

Always use your tools to fetch real data before answering. Be specific — cite actual
product names, quantities, and supplier names from the results. If a question spans
multiple tools, call them all. Highlight urgent situations prominently."""


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Discover tools via MCP protocol
    try:
        tools = await get_mcp_tools()
    except Exception as e:
        logger.error(f"Cannot reach MCP server: {e}")
        return ChatResponse(
            reply=f"Cannot reach the MCP server at {MCP_ENDPOINT}. Error: {str(e)}",
            tool_calls=[],
        )

    messages = req.history[-10:] + [{"role": "user", "content": req.message}]
    tool_calls_log = []

    # Agentic loop — Claude picks tools, we execute via MCP
    for iteration in range(10):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            reply = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return ChatResponse(reply=reply, tool_calls=tool_calls_log)

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_log.append({"tool": block.name, "input": block.input})
                    logger.info(f"Iteration {iteration}: calling {block.name}")

                    try:
                        result_text = await run_mcp_tool(block.name, block.input)
                    except Exception as e:
                        result_text = f"Tool error: {str(e)}"
                        logger.error(f"Tool {block.name} failed: {e}")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_text,
                    })

            messages.append({"role": "user", "content": tool_results})

    return ChatResponse(
        reply="Reached maximum reasoning steps. Please try a more specific question.",
        tool_calls=tool_calls_log,
    )


@app.get("/health")
async def health():
    """Always returns 200 — MCP status in body."""
    try:
        tools = await get_mcp_tools()
        return {
            "status":   "healthy",
            "mcp":      MCP_ENDPOINT,
            "tools":    len(tools),
            "protocol": "MCP JSON-RPC over HTTP",
        }
    except Exception as e:
        logger.error(f"Health MCP error: {e}")
        return {"status": "degraded", "mcp": MCP_ENDPOINT, "error": str(e)}


@app.get("/")
async def root():
    return {"service": "retail-chat-ui", "mcp": MCP_ENDPOINT}


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")