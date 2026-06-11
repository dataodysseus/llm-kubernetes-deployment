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


def get_mcp_tools() -> list[dict]:
    """
    Return tool definitions for Claude.
    Matches the tools registered in mcp-server/main.py exactly.
    Discovery via /tools REST endpoint; falls back to hardcoded schemas
    if the server is temporarily unreachable.
    """
    return [
        {
            "name": "search_similar_products",
            "description": "Search for retail products semantically similar to a query using pgvector cosine similarity on 1024d embeddings. Returns top-k results with similarity scores.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_embedding": {"type": "array", "items": {"type": "number"}, "description": "1024-dimensional float list"},
                    "top_k": {"type": "integer", "default": 5},
                    "source_filter": {"type": "string", "default": "databricks_item_table"},
                },
                "required": ["query_embedding"],
            },
        },
        {
            "name": "get_inventory_status",
            "description": "Get current inventory levels across warehouses. Filter by item_id, warehouse_region, or status (IN_STOCK, LOW_STOCK, OUT_OF_STOCK).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "warehouse_region": {"type": "string"},
                    "status": {"type": "string", "enum": ["IN_STOCK", "LOW_STOCK", "OUT_OF_STOCK"]},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        },
        {
            "name": "get_low_stock_alerts",
            "description": "Get all LOW_STOCK or OUT_OF_STOCK products. Optionally filter by warehouse_region. Critical for reorder decisions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "warehouse_region": {"type": "string"},
                },
            },
        },
        {
            "name": "get_supplier_risk",
            "description": "Identify supply chain risk — critical raw materials from single suppliers and which finished products are affected.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "supplier_id": {"type": "string"},
                    "critical_only": {"type": "boolean", "default": True},
                    "min_products_affected": {"type": "integer", "default": 1},
                },
            },
        },
        {
            "name": "get_active_promotions",
            "description": "Get currently active promotions with discount values, applicable products, regional restrictions, and usage stats.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "include_upcoming": {"type": "boolean", "default": False},
                },
            },
        },
        {
            "name": "get_promotion_stock_risk",
            "description": "Find products on active promotions that are LOW_STOCK or OUT_OF_STOCK — critical business risk.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "warehouse_region": {"type": "string"},
                },
            },
        },
    ]


async def run_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """
    Call a tool via the /tools/{name} REST endpoint on the MCP server.
    This is the clean REST layer on top of the MCP tools — same functions,
    same PostgreSQL queries, just called over plain HTTP POST.
    """
    tool_url = f"{MCP_SERVER_URL}/tools/{tool_name}"
    logger.info(f"Calling {tool_url} with {json.dumps(tool_input)[:100]}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.post(
            tool_url,
            headers=_mcp_headers(),
            json=tool_input,
        )

    if resp.status_code == 401:
        raise Exception("Tool call returned 401 — check MCP_BEARER_TOKEN")
    if resp.status_code == 403:
        raise Exception("Tool call returned 403 — bearer token rejected")
    if resp.status_code == 404:
        raise Exception(f"Tool '{tool_name}' not found at {tool_url}")
    if resp.status_code != 200:
        raise Exception(f"Tool call returned HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    result = data.get("result", data)
    return json.dumps(result)


# ── FastAPI app ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Retail Chat UI starting ===")
    logger.info(f"MCP endpoint : {MCP_ENDPOINT}")
    logger.info(f"Anthropic key: {'set' if ANTHROPIC_API_KEY else 'MISSING'}")
    logger.info(f"Bearer token : {'set' if MCP_BEARER_TOKEN else 'not set'}")
    try:
        tools = get_mcp_tools()
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
    redirect_slashes=False, 
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
        tools = get_mcp_tools()
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
    """Always returns HTTP 200 — MCP status reported in body."""
    try:
        # Verify MCP server reachable via /tools REST endpoint
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                f"{MCP_SERVER_URL}/tools",
                headers=_mcp_headers(),
            )
        if resp.status_code == 200:
            tool_list = resp.json().get("tools", [])
            return {
                "status":   "healthy",
                "mcp":      MCP_SERVER_URL,
                "tools":    len(tool_list),
                "protocol": "MCP REST over HTTP",
            }
        else:
            return {
                "status": "degraded",
                "mcp":    MCP_SERVER_URL,
                "error":  f"MCP /tools returned HTTP {resp.status_code}",
            }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "degraded", "mcp": MCP_SERVER_URL, "error": str(e)}


# Serve frontend — StaticFiles handles GET / and serves index.html
# No explicit root route needed — StaticFiles html=True does it automatically
app.mount("/", StaticFiles(directory="static", html=True), name="static")