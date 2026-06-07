"""
Chat UI Backend
Bridges the web chat interface to Claude + MCP retail server.
Claude uses the MCP tools to answer natural language questions about the retail database.

Run locally:
    pip install fastapi uvicorn anthropic httpx python-dotenv
    uvicorn backend:app --reload --port 8000

Environment variables:
    ANTHROPIC_API_KEY   - your Anthropic API key
    MCP_SERVER_URL      - your Cloud Run MCP server base URL
    MCP_BEARER_TOKEN    - bearer token for MCP server auth
"""

import os
import json
import httpx
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL    = os.environ.get("MCP_SERVER_URL", "https://retail-mcp-server-630538663455.us-central1.run.app")
MCP_BEARER_TOKEN  = os.environ.get("MCP_BEARER_TOKEN", "")

app = FastAPI(title="Retail DB Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []


SYSTEM_PROMPT = """You are a retail operations assistant with direct access to a live PostgreSQL database 
containing inventory, supplier, promotion, and product embedding data for a retail confectionery business.

You have 6 tools available:
- search_similar_products: semantic similarity search over product embeddings
- get_inventory_status: check stock levels by product, warehouse, or status
- get_low_stock_alerts: find all LOW_STOCK or OUT_OF_STOCK products
- get_supplier_risk: identify critical raw material supply chain risks
- get_active_promotions: see live and upcoming promotional campaigns
- get_promotion_stock_risk: find promoted products that are running low on stock

Always use the tools to get real data before answering. Be specific, cite actual product names, 
quantities, and supplier names from the results. If a question spans multiple tools, call them all.
Format numbers clearly and highlight urgent situations (OUT_OF_STOCK, high lead times) prominently."""


async def get_mcp_tools() -> list[dict]:
    """Fetch available tools from the MCP server."""
    headers = {"Content-Type": "application/json"}
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
        resp = await http.post(
            f"{MCP_SERVER_URL}/mcp/",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        data = resp.json()
        tools = data.get("result", {}).get("tools", [])

    # Convert MCP tool schema to Anthropic tool format
    anthropic_tools = []
    for t in tools:
        anthropic_tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
        })
    return anthropic_tools


async def call_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """Call a specific tool on the MCP server."""
    headers = {"Content-Type": "application/json"}
    if MCP_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_BEARER_TOKEN}"

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        resp = await http.post(
            f"{MCP_SERVER_URL}/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_input},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", json.dumps(result))
    return json.dumps(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        tools = await get_mcp_tools()
    except Exception as e:
        logger.error(f"Failed to fetch MCP tools: {e}")
        tools = []

    messages = req.history + [{"role": "user", "content": req.message}]
    tool_calls_log = []

    # Agentic loop — Claude calls tools until it has enough info
    for _ in range(10):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools if tools else [],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            reply = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return ChatResponse(reply=reply, tool_calls=tool_calls_log)

        if response.stop_reason == "tool_use":
            # Add assistant message with tool use blocks
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"Calling tool: {block.name} with {block.input}")
                    tool_calls_log.append({"tool": block.name, "input": block.input})
                    try:
                        result = await call_mcp_tool(block.name, block.input)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                        logger.error(f"Tool {block.name} failed: {e}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    return ChatResponse(reply="I reached the maximum reasoning steps. Please try a more specific question.", tool_calls=tool_calls_log)


@app.get("/health")
async def health():
    return {"status": "ok", "mcp_server": MCP_SERVER_URL}


# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")