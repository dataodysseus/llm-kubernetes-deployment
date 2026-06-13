# ============================================================
# MCP Server: PostgreSQL Sales & Vector Store Tools
# ============================================================
# Transport  : Streamable HTTP (MCP spec 2025-03-26)
# Auth       : Bearer token (static, set via MCP_BEARER_TOKEN env var)
# Framework  : FastAPI + mcp[server]>=1.25
# Database   : GCP PostgreSQL (appdb)
# Deploy to  : Cloud Run
# ============================================================

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg2
import psycopg2.pool
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
PG_HOST        = os.environ.get("PG_HOST", "")
PG_PORT        = int(os.environ.get("PG_PORT", "5432"))
PG_DB          = os.environ.get("PG_DB", "appdb")
PG_USER        = os.environ.get("PG_USER", "appuser")
PG_PASSWORD    = os.environ.get("PG_PASSWORD", "")
# Static bearer token — set this as a Cloud Run env var secret
# Databricks connection uses this as the Bearer token value
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info(f"Creating DB pool -> {PG_HOST}:{PG_PORT}/{PG_DB}")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=10,
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
            connect_timeout=10,
        )
    return _pool


def query(sql: str, params: tuple = ()) -> list[dict]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        pool.putconn(conn)


def embedding_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(round(x, 8)) for x in embedding) + "]"


def verify_token(request: Request):
    """Validate Bearer token if MCP_BEARER_TOKEN is set."""
    if not MCP_BEARER_TOKEN:
        return  # no token configured — allow all (for testing)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if auth[7:] != MCP_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")


# ── MCP server ─────────────────────────────────────────────
mcp = FastMCP(
    name="retail-postgres-mcp",
    stateless_http=True,  # Required for Claude Code + Claude Desktop compatibility
    streamable_http_path="/",  # Handler lives at root of the /mcp mount point
    # Disable DNS rebinding protection: Cloud Run is behind Google's load balancer
    # so the Host header will never be 127.0.0.1. Auth is enforced via Bearer token.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    instructions=(
        "You have access to a retail PostgreSQL database containing "
        "inventory, suppliers, Bill of Materials, promotions, and "
        "semantic product embeddings. Use these tools to answer "
        "questions about stock levels, supply chain risk, active "
        "promotions, and product similarity."
    ),
)


@mcp.tool(description="Search for retail products semantically similar to a query. Uses pgvector cosine similarity on 1024d embeddings. Returns top-k results with similarity scores.")
def search_similar_products(query_embedding: list[float], top_k: int = 5, source_filter: str = "databricks_item_table") -> list[dict]:
    top_k = min(top_k, 20)
    vec_str = embedding_to_pgvector(query_embedding)
    return query(
        """
        SELECT metadata->>'item_id' AS item_id, metadata->>'item_name' AS item_name,
               metadata->>'item_description' AS description, content,
               ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similarity
        FROM vectors.documents WHERE metadata->>'source' = %s
        ORDER BY embedding <=> %s::vector LIMIT %s
        """,
        (vec_str, source_filter, vec_str, top_k),
    )


@mcp.tool(description="Get current inventory levels across warehouses. Filter by item_id, warehouse_region, or status (IN_STOCK, LOW_STOCK, OUT_OF_STOCK).")
def get_inventory_status(item_id: str | None = None, warehouse_region: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    conditions, params = [], []
    if item_id:
        conditions.append("item_id = %s"); params.append(item_id)
    if warehouse_region:
        conditions.append("warehouse_region ILIKE %s"); params.append(f"%{warehouse_region}%")
    if status:
        conditions.append("status = %s"); params.append(status.upper())
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(min(limit, 200))
    return query(f"SELECT item_id, item_name, warehouse_id, warehouse_name, warehouse_region, quantity_on_hand, reorder_level, unit_cost, last_restocked, status FROM sales.inventory {where} ORDER BY status DESC, quantity_on_hand ASC LIMIT %s", tuple(params))


@mcp.tool(description="Get all LOW_STOCK or OUT_OF_STOCK products. Optionally filter by warehouse_region. Critical for reorder decisions.")
def get_low_stock_alerts(warehouse_region: str | None = None) -> list[dict]:
    conditions, params = ["status IN ('LOW_STOCK', 'OUT_OF_STOCK')"], []
    if warehouse_region:
        conditions.append("warehouse_region ILIKE %s"); params.append(f"%{warehouse_region}%")
    where = "WHERE " + " AND ".join(conditions)
    return query(f"SELECT item_name, warehouse_id, warehouse_region, quantity_on_hand, reorder_level, reorder_quantity, status, last_restocked, CURRENT_DATE - last_restocked AS days_since_restock FROM sales.inventory {where} ORDER BY status DESC, days_since_restock DESC", tuple(params))


@mcp.tool(description="Identify supply chain risk — critical raw materials from single suppliers and which finished products are affected.")
def get_supplier_risk(supplier_id: str | None = None, critical_only: bool = True, min_products_affected: int = 1) -> list[dict]:
    conditions, params = [], []
    if supplier_id:
        conditions.append("b.supplier_id = %s"); params.append(supplier_id)
    if critical_only:
        conditions.append("b.is_critical = TRUE")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(min_products_affected)
    return query(f"SELECT s.supplier_id, s.supplier_name, s.country, s.lead_time_days, s.reliability_score, b.raw_material_id, b.raw_material_name, b.raw_material_category, b.is_critical, COUNT(DISTINCT b.product_id) AS products_affected, STRING_AGG(DISTINCT b.product_name, ', ' ORDER BY b.product_name) AS affected_products FROM sales.bom b JOIN sales.suppliers s ON s.supplier_id = b.supplier_id {where} GROUP BY s.supplier_id, s.supplier_name, s.country, s.lead_time_days, s.reliability_score, b.raw_material_id, b.raw_material_name, b.raw_material_category, b.is_critical HAVING COUNT(DISTINCT b.product_id) >= %s ORDER BY products_affected DESC, s.lead_time_days DESC", tuple(params))


@mcp.tool(description="Get currently active promotions with discount values, applicable products, regional restrictions, and usage stats.")
def get_active_promotions(region: str | None = None, include_upcoming: bool = False) -> list[dict]:
    date_cond = "start_date <= CURRENT_DATE + INTERVAL '30 days'" if include_upcoming else "start_date <= CURRENT_DATE AND end_date >= CURRENT_DATE"
    conditions, params = [f"is_active = TRUE AND {date_cond}"], []
    if region:
        conditions.append("(applicable_region ILIKE %s OR applicable_region = 'ALL')"); params.append(f"%{region}%")
    where = "WHERE " + " AND ".join(conditions)
    return query(f"SELECT promotion_id, promotion_name, promotion_type, discount_value, min_order_amount, applicable_items, applicable_region, start_date, end_date, end_date - CURRENT_DATE AS days_remaining, times_used, usage_limit, CASE WHEN usage_limit IS NULL THEN NULL ELSE ROUND(times_used * 100.0 / usage_limit, 1) END AS pct_used FROM sales.promotions {where} ORDER BY end_date ASC", tuple(params))


@mcp.tool(description="Find products on active promotions that are LOW_STOCK or OUT_OF_STOCK — critical business risk where demand is being driven for unfulfillable products.")
def get_promotion_stock_risk(warehouse_region: str | None = None) -> list[dict]:
    conditions = ["p.is_active = TRUE", "p.start_date <= CURRENT_DATE", "p.end_date >= CURRENT_DATE", "i.status IN ('LOW_STOCK', 'OUT_OF_STOCK')"]
    params: list[Any] = []
    if warehouse_region:
        conditions.append("i.warehouse_region ILIKE %s"); params.append(f"%{warehouse_region}%")
    where = "WHERE " + " AND ".join(conditions)
    return query(f"SELECT DISTINCT p.promotion_name, p.promotion_type, p.end_date, p.end_date - CURRENT_DATE AS days_remaining, i.item_name, i.item_id, i.warehouse_region, i.quantity_on_hand, i.status, s.supplier_name, s.lead_time_days, s.reliability_score FROM sales.promotions p CROSS JOIN LATERAL UNNEST(p.applicable_items) AS pi(item_id) JOIN sales.inventory i ON i.item_id = pi.item_id LEFT JOIN sales.supplier_items si ON si.item_id = pi.item_id AND si.is_primary = TRUE LEFT JOIN sales.suppliers s ON s.supplier_id = si.supplier_id {where} ORDER BY p.end_date ASC, i.status DESC, i.quantity_on_hand ASC", tuple(params))


# ── FastAPI app ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MCP server starting...")
    logger.info(f"PG_HOST={PG_HOST} PG_DB={PG_DB} PG_USER={PG_USER}")
    logger.info(f"Bearer token auth: {'enabled' if MCP_BEARER_TOKEN else 'disabled'}")
    async with mcp.session_manager.run():
        yield
    global _pool
    if _pool:
        _pool.closeall()


app = FastAPI(
    title="Retail PostgreSQL MCP Server",
    description="MCP tools over sales + pgvector tables in appdb",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
)


@app.get("/health")
async def health():
    try:
        query("SELECT 1 AS ok")
        return {"status": "healthy", "db": "connected"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "degraded", "db": str(e)}


@app.get("/")
async def root():
    return {
        "service": "retail-postgres-mcp",
        "mcp_endpoint": "/mcp",
        "health": "/health",
        "auth": "Bearer token (set MCP_BEARER_TOKEN env var)",
    }


# ── Direct REST tool endpoints ────────────────────────────
# Exposes each MCP tool as a plain POST /tools/<name> endpoint.
# This is what the chat UI backend calls — no MCP protocol needed.

TOOL_REGISTRY = {
    "search_similar_products":  search_similar_products,
    "get_inventory_status":     get_inventory_status,
    "get_low_stock_alerts":     get_low_stock_alerts,
    "get_supplier_risk":        get_supplier_risk,
    "get_active_promotions":    get_active_promotions,
    "get_promotion_stock_risk": get_promotion_stock_risk,
}


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call any registered tool by name. Accepts JSON body as tool arguments."""
    if MCP_BEARER_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != MCP_BEARER_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    if tool_name not in TOOL_REGISTRY:
        return JSONResponse(status_code=404, content={"detail": f"Tool '{tool_name}' not found. Available: {list(TOOL_REGISTRY.keys())}"})

    try:
        body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    except Exception:
        body = {}

    try:
        result = TOOL_REGISTRY[tool_name](**body)
        return {"tool": tool_name, "result": result}
    except Exception as e:
        logger.error(f"Tool {tool_name} error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {"tools": list(TOOL_REGISTRY.keys())}


# Register MCP server at /mcp — uses a direct Route (not mount) so both
# /mcp and /mcp/ work without Starlette's mount trailing-slash requirement.
_mcp_sub_app = mcp.streamable_http_app()
_mcp_handler = _mcp_sub_app.routes[0].endpoint  # StreamableHTTPASGIApp instance
app.router.routes.insert(0, Route("/mcp", endpoint=_mcp_handler))


@app.middleware("http")
async def bearer_token_middleware(request: Request, call_next):
    """Validate Bearer token on /mcp routes only."""
    if request.url.path.startswith("/mcp") and MCP_BEARER_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != MCP_BEARER_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing Bearer token"})
    return await call_next(request)