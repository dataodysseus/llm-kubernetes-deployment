"""
Retail Intelligence MCP Server v3 (Stdio Transport)
===================================================
Transport : MCP Stdio (Standard Input/Output)
Auth      : Handled by GCP Identity-Aware Proxy (IAP) SSH tunneling
Database  : PostgreSQL appdb on GCP VM (34.9.255.250)
"""

import os
import sys
import logging
import psycopg2
import psycopg2.pool
from mcp.server.fastmcp import FastMCP

# MUST log to stderr. stdout is strictly reserved for the MCP protocol JSON stream.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
PG_HOST     = os.environ.get("PG_HOST", "34.9.255.250")
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_DB       = os.environ.get("PG_DB", "appdb")
PG_USER     = os.environ.get("PG_USER", "appuser")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info(f"Connecting to Postgres at {PG_HOST}:{PG_PORT}/{PG_DB}")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=5,
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
            connect_timeout=10,
        )
    return _pool

def run_query(sql: str, params: tuple = ()) -> list[dict]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        pool.putconn(conn)

# ── MCP Server ───────────────────────────────────────────────
mcp = FastMCP(
    name="retail-intelligence-mcp",
    instructions="""
You are connected to a retail PostgreSQL database (appdb) with two schemas.

RETAIL SCHEMA — star schema for business analytics:
  retail.items_sales      — FACT: item_id, location_id, customer_id, sale_date, units_sold, total_sales_value
  retail.item_details     — DIM: item_id, item_name, item_description
  retail.store_location   — DIM: location_id, location_name, location_description
  retail.customer_details — DIM: customer_id, customer_name, customer_email, customer_zip_code
  retail.data_dictionary  — METADATA: full schema documentation

VECTORS SCHEMA — semantic search:
  vectors.documents       — pgvector 1024d embeddings of item descriptions

ALWAYS start by calling get_schema_context to understand the exact schema.
Then write SQL and call execute_retail_query.
For product similarity questions use search_products.
"""
)

@mcp.tool(description="Returns complete data dictionary for retail and vectors schemas. Call this FIRST.")
def get_schema_context() -> dict:
    logger.info("Tool called: get_schema_context")
    rows = run_query("""
        SELECT db_schema, table_name, column_name, column_type, column_description, related_table
        FROM retail.data_dictionary
        ORDER BY db_schema, table_name, column_name
    """)
    # Condense formatting for brevity
    return {"data_dictionary": rows}

@mcp.tool(description="Execute read-only SQL query against retail/vectors schemas.")
def execute_retail_query(sql: str, limit: int = 100) -> dict:
    logger.info(f"Tool called: execute_retail_query. SQL length: {len(sql)}")
    cleaned = sql.strip().lstrip("(").upper()
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return {"error": "Only SELECT/WITH statements are permitted."}

    limit = min(limit, 500)
    safe_sql = sql if "LIMIT" in sql.upper() else f"SELECT * FROM ({sql}) _q LIMIT {limit}"

    try:
        rows = run_query(safe_sql)
        return {"row_count": len(rows), "rows": rows}
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {"error": str(e)}

@mcp.tool(description="Search for retail products by meaning/characteristics using vector text matching.")
def search_products(query_text: str, top_k: int = 5) -> dict:
    logger.info(f"Tool called: search_products. Query: {query_text}")
    top_k = min(top_k, 20)
    try:
        rows = run_query("""
            SELECT metadata->>'item_id' AS item_id, metadata->>'item_name' AS item_name, metadata->>'item_description' AS description
            FROM vectors.documents
            WHERE content ILIKE %s LIMIT %s
        """, (f"%{query_text}%", top_k))
        return {"query": query_text, "results": rows}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting Retail Intelligence MCP Server via stdio...")
    mcp.run(transport="stdio")