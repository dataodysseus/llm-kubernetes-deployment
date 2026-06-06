# ============================================================
# MCP Server: PostgreSQL Sales & Vector Store Tools
# ============================================================
# Transport  : Streamable HTTP (MCP spec 2025-03-26)
# Framework  : FastAPI + mcp[server]
# Database   : GCP PostgreSQL (appdb)
# Deploy to  : Cloud Run
#
# Tools exposed:
#   - search_similar_products     (pgvector semantic search)
#   - get_inventory_status        (warehouse stock levels)
#   - get_low_stock_alerts        (LOW_STOCK + OUT_OF_STOCK)
#   - get_supplier_risk           (critical material + supplier)
#   - get_active_promotions       (live promotions)
#   - get_promotion_stock_risk    (promos vs low inventory)
# ============================================================

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg2
import psycopg2.pool
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Database connection pool ───────────────────────────────
PG_HOST     = os.environ["PG_HOST"]
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_DB       = os.environ.get("PG_DB", "appdb")
PG_USER     = os.environ.get("PG_USER", "appuser")
PG_PASSWORD = os.environ["PG_PASSWORD"]

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            connect_timeout=10,
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query and return rows as list of dicts."""
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


# ── MCP server ─────────────────────────────────────────────
mcp = FastMCP(
    name="retail-postgres-mcp",
    instructions=(
        "You have access to a retail PostgreSQL database containing "
        "inventory, suppliers, Bill of Materials, promotions, and "
        "semantic product embeddings. Use these tools to answer "
        "questions about stock levels, supply chain risk, active "
        "promotions, and product similarity."
    ),
)


# ── Tool 1: Semantic product search ───────────────────────
@mcp.tool(
    description=(
        "Search for retail products semantically similar to a query string. "
        "Uses pgvector cosine similarity on BGE-large-en 1024d embeddings. "
        "Returns top-k most relevant products with similarity scores and metadata."
    )
)
def search_similar_products(
    query_embedding: list[float],
    top_k: int = 5,
    source_filter: str = "databricks_item_table",
) -> list[dict]:
    """
    Args:
        query_embedding: 1024-dimensional float list from BGE-large-en
        top_k: number of results to return (default 5, max 20)
        source_filter: filter by metadata source field
    """
    top_k = min(top_k, 20)
    vec_str = embedding_to_pgvector(query_embedding)
    rows = query(
        """
        SELECT
            metadata->>'item_id'          AS item_id,
            metadata->>'item_name'        AS item_name,
            metadata->>'item_description' AS description,
            content,
            ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similarity
        FROM vectors.documents
        WHERE metadata->>'source' = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, source_filter, vec_str, top_k),
    )
    return rows


# ── Tool 2: Inventory status ───────────────────────────────
@mcp.tool(
    description=(
        "Get current inventory levels for products across warehouses. "
        "Filter by item, warehouse region, or stock status. "
        "Returns quantity on hand, reorder levels, and status per warehouse."
    )
)
def get_inventory_status(
    item_id: str | None = None,
    warehouse_region: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Args:
        item_id: optional product ID filter (e.g. 'p-1')
        warehouse_region: optional region filter (e.g. 'Northeast USA')
        status: optional status filter — IN_STOCK, LOW_STOCK, OUT_OF_STOCK
        limit: max rows to return (default 50)
    """
    conditions = []
    params: list[Any] = []

    if item_id:
        conditions.append("item_id = %s")
        params.append(item_id)
    if warehouse_region:
        conditions.append("warehouse_region ILIKE %s")
        params.append(f"%{warehouse_region}%")
    if status:
        conditions.append("status = %s")
        params.append(status.upper())

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(min(limit, 200))

    return query(
        f"""
        SELECT item_id, item_name, warehouse_id, warehouse_name,
               warehouse_region, quantity_on_hand, reorder_level,
               unit_cost, last_restocked, status
        FROM sales.inventory
        {where}
        ORDER BY status DESC, quantity_on_hand ASC
        LIMIT %s
        """,
        tuple(params),
    )


# ── Tool 3: Low stock alerts ───────────────────────────────
@mcp.tool(
    description=(
        "Get all products that are LOW_STOCK or OUT_OF_STOCK across warehouses. "
        "Includes reorder quantities and days since last restock. "
        "Essential for supply chain triage and reorder decisions."
    )
)
def get_low_stock_alerts(warehouse_region: str | None = None) -> list[dict]:
    """
    Args:
        warehouse_region: optional filter by region (e.g. 'West Coast USA')
    """
    conditions = ["status IN ('LOW_STOCK', 'OUT_OF_STOCK')"]
    params: list[Any] = []

    if warehouse_region:
        conditions.append("warehouse_region ILIKE %s")
        params.append(f"%{warehouse_region}%")

    where = "WHERE " + " AND ".join(conditions)

    return query(
        f"""
        SELECT item_name, warehouse_id, warehouse_region,
               quantity_on_hand, reorder_level, reorder_quantity,
               status, last_restocked,
               CURRENT_DATE - last_restocked AS days_since_restock
        FROM sales.inventory
        {where}
        ORDER BY status DESC, days_since_restock DESC
        """,
        tuple(params),
    )


# ── Tool 4: Supplier risk analysis ────────────────────────
@mcp.tool(
    description=(
        "Identify supply chain risk by finding critical raw materials "
        "that come from a single supplier. Shows which finished products "
        "are affected, the supplier's lead time and reliability score, "
        "and current inventory status of those products."
    )
)
def get_supplier_risk(
    supplier_id: str | None = None,
    critical_only: bool = True,
    min_products_affected: int = 1,
) -> list[dict]:
    """
    Args:
        supplier_id: optional filter to a specific supplier (e.g. 'SUP-009')
        critical_only: if True, only return is_critical=TRUE BOM lines
        min_products_affected: minimum number of products a supplier must
                                affect to appear in results (default 1)
    """
    conditions = []
    params: list[Any] = []

    if supplier_id:
        conditions.append("b.supplier_id = %s")
        params.append(supplier_id)
    if critical_only:
        conditions.append("b.is_critical = TRUE")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    params.append(min_products_affected)

    return query(
        f"""
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.country,
            s.lead_time_days,
            s.reliability_score,
            b.raw_material_id,
            b.raw_material_name,
            b.raw_material_category,
            b.is_critical,
            COUNT(DISTINCT b.product_id)           AS products_affected,
            STRING_AGG(DISTINCT b.product_name, ', '
                ORDER BY b.product_name)           AS affected_products
        FROM sales.bom b
        JOIN sales.suppliers s ON s.supplier_id = b.supplier_id
        {where}
        GROUP BY s.supplier_id, s.supplier_name, s.country,
                 s.lead_time_days, s.reliability_score,
                 b.raw_material_id, b.raw_material_name,
                 b.raw_material_category, b.is_critical
        HAVING COUNT(DISTINCT b.product_id) >= %s
        ORDER BY products_affected DESC, s.lead_time_days DESC
        """,
        tuple(params),
    )


# ── Tool 5: Active promotions ──────────────────────────────
@mcp.tool(
    description=(
        "Get all currently active promotions including type, discount value, "
        "applicable products, regional restrictions, and usage statistics. "
        "Use this to understand what deals are live and how they are performing."
    )
)
def get_active_promotions(
    region: str | None = None,
    include_upcoming: bool = False,
) -> list[dict]:
    """
    Args:
        region: optional region filter (e.g. 'Northeast USA' or 'ALL')
        include_upcoming: if True, also include promotions starting in future
    """
    if include_upcoming:
        date_condition = "start_date <= CURRENT_DATE + INTERVAL '30 days'"
    else:
        date_condition = "start_date <= CURRENT_DATE AND end_date >= CURRENT_DATE"

    conditions = [f"is_active = TRUE AND {date_condition}"]
    params: list[Any] = []

    if region:
        conditions.append("(applicable_region ILIKE %s OR applicable_region = 'ALL')")
        params.append(f"%{region}%")

    where = "WHERE " + " AND ".join(conditions)

    return query(
        f"""
        SELECT
            promotion_id,
            promotion_name,
            promotion_type,
            discount_value,
            min_order_amount,
            applicable_items,
            applicable_region,
            start_date,
            end_date,
            end_date - CURRENT_DATE          AS days_remaining,
            times_used,
            usage_limit,
            CASE WHEN usage_limit IS NULL THEN NULL
                 ELSE ROUND(times_used * 100.0 / usage_limit, 1)
            END                              AS pct_used
        FROM sales.promotions
        {where}
        ORDER BY end_date ASC
        """,
        tuple(params),
    )


# ── Tool 6: Promotion × inventory risk ────────────────────
@mcp.tool(
    description=(
        "Find products that are currently on an active promotion but "
        "are simultaneously LOW_STOCK or OUT_OF_STOCK. This is a critical "
        "business risk — a live promotion driving demand for a product "
        "that cannot be fulfilled. Returns promotion details, stock levels, "
        "and the primary supplier lead time for each at-risk product."
    )
)
def get_promotion_stock_risk(warehouse_region: str | None = None) -> list[dict]:
    """
    Args:
        warehouse_region: optional filter to a specific warehouse region
    """
    conditions = [
        "p.is_active = TRUE",
        "p.start_date <= CURRENT_DATE",
        "p.end_date >= CURRENT_DATE",
        "i.status IN ('LOW_STOCK', 'OUT_OF_STOCK')",
    ]
    params: list[Any] = []

    if warehouse_region:
        conditions.append("i.warehouse_region ILIKE %s")
        params.append(f"%{warehouse_region}%")

    where = "WHERE " + " AND ".join(conditions)

    return query(
        f"""
        SELECT DISTINCT
            p.promotion_name,
            p.promotion_type,
            p.end_date,
            p.end_date - CURRENT_DATE        AS days_remaining,
            i.item_name,
            i.item_id,
            i.warehouse_region,
            i.quantity_on_hand,
            i.status,
            s.supplier_name,
            s.lead_time_days,
            s.reliability_score
        FROM sales.promotions p
        CROSS JOIN LATERAL UNNEST(p.applicable_items) AS pi(item_id)
        JOIN sales.inventory    i ON i.item_id     = pi.item_id
        LEFT JOIN sales.supplier_items si
               ON si.item_id   = pi.item_id
              AND si.is_primary = TRUE
        LEFT JOIN sales.suppliers s ON s.supplier_id = si.supplier_id
        {where}
        ORDER BY p.end_date ASC, i.status DESC, i.quantity_on_hand ASC
        """,
        tuple(params),
    )


# ── FastAPI app wiring ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MCP server — pre-warming DB pool...")
    get_pool()
    logger.info("Ready")
    yield
    global _pool
    if _pool:
        _pool.closeall()
        logger.info("DB pool closed")


app = FastAPI(
    title="Retail PostgreSQL MCP Server",
    description="MCP tools over sales + pgvector tables in appdb",
    version="1.0.0",
    lifespan=lifespan,
)


# Health check — Cloud Run requires this
@app.get("/health")
async def health():
    try:
        rows = query("SELECT 1 AS ok")
        return {"status": "healthy", "db": "connected", "ok": rows[0]["ok"] == 1}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


# MCP endpoint — Streamable HTTP transport
# Databricks supervisor agent registers this URL
@app.post("/mcp")
@app.get("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    transport = StreamableHTTPServerTransport(mcp_session_id=None)
    async with transport.connect():
        await mcp.run(transport)
    return transport.response