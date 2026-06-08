"""
chat-backend/backend.py
=======================
MCP-native chat backend for the Retail Intelligence UI.

Key difference from the REST-based version:
  - Uses the `mcp` Python library's ClientSession to speak the real
    MCP Streamable HTTP protocol (tools/list + tools/call)
  - Tools are discovered dynamically at runtime, not hardcoded
  - Claude drives tool selection; this backend executes them via MCP

Architecture:
  Browser → POST /chat → Claude (picks tools) → MCP ClientSession
         → tools/list (discover) + tools/call (execute)
         → MCP Server on Cloud Run → PostgreSQL

Environment variables (set in .env locally, Cloud Run env vars in prod):
  ANTHROPIC_API_KEY   Your Anthropic API key
  MCP_SERVER_URL      Base URL of your MCP server Cloud Run service
  MCP_BEARER_TOKEN    Bearer token protecting the /mcp endpoint
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── MCP client imports ─────────────────────────────────────
# These are the key pieces that make this a true MCP client:
#   - ClientSession     : manages the MCP session lifecycle
#   - streamablehttp    : the Streamable HTTP transport (matches
#                         what FastMCP's streamable_http_app() serves)
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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

# The full MCP endpoint URL — this is what the protocol connects to
MCP_ENDPOINT = f"{MCP_SERVER_URL}/mcp"

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── MCP session helper ─────────────────────────────────────
# Each call to this creates a fresh MCP session, sends the request,
# and cleanly closes the session. Cloud Run is stateless so we don't
# keep persistent sessions — one session per tool call is correct here.

async def run_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """
    Open an MCP client session, call one tool, return the result as a string.

    Protocol flow inside this function:
      1. streamablehttp_client opens HTTP connection to /mcp endpoint
      2. ClientSession.initialize() sends the MCP handshake
         → server responds with its capabilities and protocol version
      3. session.call_tool() sends:
         POST /mcp  {"jsonrpc":"2.0","method":"tools/call",
                     "params":{"name":"...","arguments":{...}}}
      4. Server executes the tool against PostgreSQL, returns result
      5. Context manager closes the session cleanly
    """
    headers = {}
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"

    logger.info(f"MCP call: {tool_name}({json.dumps(tool_input)[:100]})")

    async with streamablehttp_client(MCP_ENDPOINT, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Handshake — exchanges protocol version and capabilities
            await session.initialize()

            # The actual tool call — pure MCP protocol
            result = await session.call_tool(tool_name, tool_input)

            # Result comes back as a list of content blocks
            # Text blocks contain the serialised tool output
            if result.content:
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts) if parts else "No result returned"
            return "Tool returned empty result"


async def get_mcp_tools() -> list[dict]:
    """
    Discover available tools from the MCP server using tools/list.

    This is what makes MCP different from REST — the client doesn't
    need to know the tool schemas in advance. It asks the server at
    runtime and gets back the full schema for each tool, which is then
    passed directly to Claude so it knows exactly how to call them.

    Protocol flow:
      1. Open MCP session (same handshake as above)
      2. session.list_tools() sends:
         POST /mcp  {"jsonrpc":"2.0","method":"tools/list","params":{}}
      3. Server returns all tool definitions with name, description,
         and inputSchema for each tool
      4. We convert to Anthropic's tool format for Claude
    """
    headers = {}
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"

    async with streamablehttp_client(MCP_ENDPOINT, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()

    # Convert MCP tool schema → Anthropic tool format
    # MCP uses "inputSchema", Anthropic uses "input_schema" — small rename
    anthropic_tools = []
    for tool in tools_result.tools:
        anthropic_tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
        })

    logger.info(f"Discovered {len(anthropic_tools)} tools from MCP server")
    return anthropic_tools


# ── FastAPI app ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: verify MCP server is reachable and discover tools.
    This gives a clear error at boot time rather than on first request.
    """
    logger.info(f"Connecting to MCP server: {MCP_ENDPOINT}")
    try:
        tools = await get_mcp_tools()
        tool_names = [t["name"] for t in tools]
        logger.info(f"MCP tools available: {tool_names}")
    except Exception as e:
        logger.warning(f"MCP server not reachable at startup: {e}")
        logger.warning("Will retry on first request")
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


# ── Request / response models ──────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []


# ── System prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a retail operations assistant with direct access to a live 
PostgreSQL database containing inventory, supplier, promotion, and product data for a 
retail confectionery business.

Always use your tools to fetch real data before answering. Be specific — cite actual 
product names, quantities, and supplier names from the results. If a question spans 
multiple tools, call them all. Highlight urgent situations prominently."""


# ── Chat endpoint ──────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Flow:
      1. Discover tools from MCP server (tools/list)
      2. Send user message to Claude with tool schemas
      3. Claude decides which tools to call
      4. For each tool_use block: call the MCP server (tools/call)
      5. Feed results back to Claude
      6. Repeat until Claude produces a final text response
    """
    # Step 1: discover tools via MCP protocol
    try:
        tools = await get_mcp_tools()
    except Exception as e:
        logger.error(f"Could not reach MCP server: {e}")
        return ChatResponse(
            reply=f"Cannot reach the MCP server at {MCP_ENDPOINT}. "
                  f"Please check it is running. Error: {str(e)}",
            tool_calls=[],
        )

    messages = req.history[-10:] + [{"role": "user", "content": req.message}]
    tool_calls_log = []

    # Step 2-5: agentic loop
    for iteration in range(10):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Claude finished — return final text response
        if response.stop_reason == "end_turn":
            reply = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return ChatResponse(reply=reply, tool_calls=tool_calls_log)

        # Claude wants to use tools
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_log.append({
                        "tool":  block.name,
                        "input": block.input,
                    })
                    logger.info(f"Iteration {iteration}: calling {block.name}")

                    # Step 4: call the tool via true MCP protocol
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

            # Step 5: feed results back to Claude
            messages.append({"role": "user", "content": tool_results})

    return ChatResponse(
        reply="Reached maximum reasoning steps. Please try a more specific question.",
        tool_calls=tool_calls_log,
    )


# ── Health check ───────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Health check for Cloud Run.
    Also verifies MCP server reachability and reports tool count.
    """
    try:
        tools = await get_mcp_tools()
        return {
            "status":     "healthy",
            "mcp_server": MCP_ENDPOINT,
            "tools":      len(tools),
            "protocol":   "MCP Streamable HTTP",
        }
    except Exception as e:
        # Return 200 even if MCP is temporarily unreachable
        # so Cloud Run doesn't kill the container
        return {
            "status":     "degraded",
            "mcp_server": MCP_ENDPOINT,
            "error":      str(e),
        }


# ── Serve frontend static files ────────────────────────────
# The same index.html from chat-ui/static/ is copied into
# chat-backend/static/ — one Cloud Run service serves everything
app.mount("/", StaticFiles(directory="static", html=True), name="static")