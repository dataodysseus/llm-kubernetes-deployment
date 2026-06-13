#!/usr/bin/env python3
"""
MCP Handshake Client — retail-postgres-mcp
Uses the official MCP Python client library to perform the full
Streamable HTTP handshake, discover tools, then find out-of-stock products.
"""

import asyncio
import json
import sys
import textwrap

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# ── Config ─────────────────────────────────────────────────────────────────
MCP_URL   = "https://retail-mcp-server-630538663455.us-central1.run.app/mcp"
MCP_TOKEN = "retlmcp-s3cur3-t0k3n-2026"
HEADERS   = {"Authorization": f"Bearer {MCP_TOKEN}"}


# ── Pretty printers ─────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print('─' * 62)


def print_tool_list(tools):
    for t in tools:
        desc = textwrap.shorten(t.description or "", width=70, placeholder="…")
        print(f"  • {t.name}")
        print(f"    {desc}")


def print_inventory(rows: list[dict]):
    if not rows:
        print("  (no results)")
        return
    w = {"name": 34, "region": 20, "qty": 5, "status": 14}
    header = (
        f"  {'Item Name':<{w['name']}}"
        f"{'Region':<{w['region']}}"
        f"{'Qty':>{w['qty']}}"
        f"  {'Status':<{w['status']}}"
    )
    print(header)
    print("  " + "─" * (sum(w.values()) + 4))
    for row in rows:
        print(
            f"  {str(row.get('item_name',''))[:w['name']]:<{w['name']}}"
            f"{str(row.get('warehouse_region',''))[:w['region']]:<{w['region']}}"
            f"{str(row.get('quantity_on_hand','')):>{w['qty']}}"
            f"  {str(row.get('status','')):<{w['status']}}"
        )


def extract_json(result) -> list[dict]:
    """Pull the list[dict] out of an MCP tool CallToolResult."""
    for block in result.content:
        if block.type == "text":
            try:
                data = json.loads(block.text)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, AttributeError):
                pass
    return []


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    section("STEP 1 — MCP Streamable HTTP Handshake  (initialize)")
    print(f"  → Connecting to: {MCP_URL}")

    async with streamablehttp_client(MCP_URL, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:

            # ── initialize ────────────────────────────────────────────────
            init = await session.initialize()
            print(f"  ✓ Server      : {init.serverInfo.name} v{init.serverInfo.version}")
            print(f"  ✓ Protocol    : {init.protocolVersion}")
            caps = list(init.capabilities.model_fields_set)
            print(f"  ✓ Capabilities: {caps or list(vars(init.capabilities).keys())}")

            # ── tools/list ────────────────────────────────────────────────
            section("STEP 2 — Discover Tools  (tools/list)")
            tools_resp = await session.list_tools()
            tools = tools_resp.tools
            print(f"  ✓ {len(tools)} tools discovered:\n")
            print_tool_list(tools)

            # ── tools/call: get_inventory_status (OUT_OF_STOCK) ──────────
            section("STEP 3 — tools/call  →  get_inventory_status (OUT_OF_STOCK)")
            print("  → Calling get_inventory_status with status='OUT_OF_STOCK' …\n")
            inv_result = await session.call_tool(
                "get_inventory_status",
                {"status": "OUT_OF_STOCK", "limit": 100},
            )
            out_of_stock = extract_json(inv_result)
            print(f"  Products with status OUT_OF_STOCK: {len(out_of_stock)}\n")
            print_inventory(out_of_stock)

            # ── tools/call: get_low_stock_alerts ─────────────────────────
            section("STEP 4 — tools/call  →  get_low_stock_alerts (all regions)")
            print("  → Calling get_low_stock_alerts …\n")
            alert_result = await session.call_tool("get_low_stock_alerts", {})
            alerts = extract_json(alert_result)
            oos    = [a for a in alerts if a.get("status") == "OUT_OF_STOCK"]
            low    = [a for a in alerts if a.get("status") == "LOW_STOCK"]
            print(f"  Total alerts : {len(alerts)}")
            print(f"  OUT_OF_STOCK : {len(oos)}")
            print(f"  LOW_STOCK    : {len(low)}\n")
            print_inventory(alerts[:40])

    section("DONE — MCP handshake and tool calls completed successfully")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        raise
