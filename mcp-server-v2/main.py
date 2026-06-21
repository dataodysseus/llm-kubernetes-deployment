"""
Retail Intelligence MCP Server v2
===================================
Transport  : MCP Streamable HTTP (stateless_http=True)
Auth       : Bearer token via MCP_BEARER_TOKEN env var
Database   : PostgreSQL appdb on GCP VM (postgres-vm: 34.9.255.250)
Deploy on  : mcp-vm (35.253.0.40)

3 general-purpose tools:
  get_schema_context    - returns full data dictionary so the AI client
                          understands schema, relationships and business rules
                          before writing any SQL

  execute_retail_query  - runs any read-only SQL Claude writes dynamically
                          answers ANY business question without needing new tools

  search_products       - pgvector text search on item descriptions
                          finds products by meaning not exact name
"""

import os
import logging
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.pool
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
PG_HOST          = os.environ.get("PG_HOST", "34.9.255.250")
PG_PORT          = int(os.environ.get("PG_PORT", "5432"))
PG_DB            = os.environ.get("PG_DB", "appdb")
PG_USER          = os.environ.get("PG_USER", "appuser")
PG_PASSWORD      = os.environ.get("PG_PASSWORD", "")
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info(f"Connecting to {PG_HOST}:{PG_PORT}/{PG_DB}")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=5,
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
            connect_timeout=10,
        )
        logger.info("Connection pool ready")
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
    stateless_http=True,
    instructions="""
You are connected to a retail PostgreSQL database (appdb) with two schemas.

RETAIL SCHEMA — star schema for business analytics:
  retail.items_sales      — FACT: one row per sale transaction
                            columns: item_id, location_id, customer_id,
                                     sale_date, units_sold, total_sales_value
  retail.item_details     — DIM: product master
                            columns: item_id, item_name, item_description
  retail.store_location   — DIM: store master
                            columns: location_id, location_name, location_description
  retail.customer_details — DIM: customer master
                            columns: customer_id, customer_name,
                                     customer_email, customer_zip_code
  retail.data_dictionary  — METADATA: full schema documentation

VECTORS SCHEMA — semantic search:
  vectors.documents       — pgvector 1024d embeddings of item descriptions
                            columns: id, content, embedding, metadata (jsonb)
                            metadata keys: item_id, item_name, item_description, source

ALWAYS start by calling get_schema_context to understand the exact schema.
Then write SQL and call execute_retail_query.
For product similarity questions use search_products.

STANDARD JOIN PATTERN:
  FROM retail.items_sales f
  JOIN retail.customer_details c ON c.customer_id = f.customer_id
  JOIN retail.store_location   s ON s.location_id = f.location_id
  JOIN retail.item_details     i ON i.item_id     = f.item_id
""",
)


@mcp.tool(
    description="""Call this FIRST before writing any SQL.
Returns the complete data dictionary for all tables in the retail and vectors schemas.
Provides: column names, data types, business rules, relationships, and sample values.
Without this context you cannot know the exact column names needed for queries."""
)
def get_schema_context() -> dict:
    rows = run_query("""
        SELECT db_schema, table_name, table_type, table_description,
               column_name, column_type, column_description,
               is_key, is_measure, is_nullable,
               related_table, business_rules, sample_values
        FROM retail.data_dictionary
        ORDER BY db_schema, table_name,
                 CASE WHEN column_name IS NULL THEN 0 ELSE 1 END,
                 column_name
    """)

    context: dict = {"schemas": {}, "join_guide": {
        "fact_table": "retail.items_sales",
        "dimensions": {
            "retail.item_details":     "ON item_details.item_id = items_sales.item_id",
            "retail.store_location":   "ON store_location.location_id = items_sales.location_id",
            "retail.customer_details": "ON customer_details.customer_id = items_sales.customer_id",
        },
        "vector_join": "vectors.documents.metadata->>'item_id' = retail.item_details.item_id",
        "example_query": (
            "SELECT c.customer_name, COUNT(*) AS visits, "
            "SUM(f.total_sales_value) AS total_spent "
            "FROM retail.items_sales f "
            "JOIN retail.customer_details c ON c.customer_id = f.customer_id "
            "JOIN retail.store_location s ON s.location_id = f.location_id "
            "WHERE s.location_name ILIKE '%Austin%' "
            "GROUP BY c.customer_id, c.customer_name "
            "ORDER BY visits DESC LIMIT 5"
        ),
    }}

    for row in rows:
        schema = row["db_schema"]
        table  = row["table_name"]
        if schema not in context["schemas"]:
            context["schemas"][schema] = {}
        if table not in context["schemas"][schema]:
            context["schemas"][schema][table] = {
                "table_type": None, "description": None,
                "business_rules": None, "columns": [],
            }
        t = context["schemas"][schema][table]
        if row["column_name"] is None:
            t["table_type"]     = row["table_type"]
            t["description"]    = row["table_description"]
            t["business_rules"] = row["business_rules"]
        else:
            t["columns"].append({
                "name":          row["column_name"],
                "type":          row["column_type"],
                "description":   row["column_description"],
                "is_key":        row["is_key"],
                "is_measure":    row["is_measure"],
                "related_table": row["related_table"],
                "business_rules": row["business_rules"],
                "sample_values": row["sample_values"],
            })

    return context


@mcp.tool(
    description="""Execute any read-only SQL query against the retail and vectors schemas.
Use AFTER calling get_schema_context to know the exact column names.

This single tool can answer ANY business question by letting Claude write
the appropriate SQL — no need for separate tools per question type.

Examples of questions this answers:
  'Top 5 most frequent customers at the Austin store'
  'Bottom 3 stores by revenue last quarter'
  'Which items have zero sales this month'
  'Customers who spent over $500 and live in Texas'
  'Daily revenue trend for the last 30 days'
  'Which store sells the most chocolate items'

Args:
  sql   : A read-only SELECT or WITH...SELECT statement
  limit : Max rows returned (default 100, max 500)

Safety: Only SELECT statements allowed. INSERT/UPDATE/DELETE are rejected."""
)
def execute_retail_query(sql: str, limit: int = 100) -> dict:
    # Safety check — only SELECT
    cleaned = sql.strip().lstrip("(").upper()
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return {
            "error": "Only SELECT and WITH...SELECT statements are permitted.",
            "received": sql[:200],
        }

    limit = min(limit, 500)
    safe_sql = sql if "LIMIT" in sql.upper() else f"SELECT * FROM ({sql}) _q LIMIT {limit}"

    try:
        rows = run_query(safe_sql)
        return {
            "row_count": len(rows),
            "columns":   list(rows[0].keys()) if rows else [],
            "rows":      rows,
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {
            "error": str(e),
            "hint": "Call get_schema_context to verify table/column names.",
        }


@mcp.tool(
    description="""Search for retail products by meaning using text matching on descriptions.
Use when the question involves finding products by characteristics, not exact names.

Examples:
  'Find chocolatey products'
  'Show me fruity and tropical items'
  'What products are nutty or crunchy'

Returns item_id, item_name, description — use item_ids in execute_retail_query
to get sales data for the matching products.

Args:
  query_text : Natural language description of the product characteristics
  top_k      : Number of results to return (default 5, max 20)"""
)
def search_products(query_text: str, top_k: int = 5) -> dict:
    top_k = min(top_k, 20)
    try:
        rows = run_query("""
            SELECT
                metadata->>'item_id'          AS item_id,
                metadata->>'item_name'        AS item_name,
                metadata->>'item_description' AS description,
                content
            FROM vectors.documents
            WHERE content ILIKE %s
               OR metadata->>'item_description' ILIKE %s
               OR metadata->>'item_name' ILIKE %s
            LIMIT %s
        """, (f"%{query_text}%", f"%{query_text}%", f"%{query_text}%", top_k))

        if not rows:
            # Fallback — return all items so client still gets useful data
            rows = run_query("""
                SELECT metadata->>'item_id' AS item_id,
                       metadata->>'item_name' AS item_name,
                       metadata->>'item_description' AS description
                FROM vectors.documents
                WHERE metadata->>'source' = 'databricks_item_table'
                LIMIT %s
            """, (top_k,))
            return {
                "query": query_text,
                "count": len(rows),
                "results": rows,
                "note": "No text match found — showing sample items. Try broader search terms.",
                "tip": "Use item_id values in execute_retail_query to get sales data.",
            }

        return {
            "query": query_text,
            "count": len(rows),
            "results": rows,
            "tip": "Use item_id values in execute_retail_query to get sales data.",
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e), "query": query_text}


# ── FastAPI ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Retail Intelligence MCP Server v2 starting ===")
    logger.info(f"PostgreSQL : {PG_HOST}:{PG_PORT}/{PG_DB}")
    logger.info(f"Auth       : {'ENABLED' if MCP_BEARER_TOKEN else 'DISABLED'}")
    try:
        result = run_query("SELECT COUNT(*) AS n FROM retail.data_dictionary")
        logger.info(f"DB connected — {result[0]['n']} data dictionary entries")
    except Exception as e:
        logger.warning(f"DB not reachable at startup: {e}")
    yield
    if _pool:
        _pool.closeall()


app = FastAPI(
    title="Retail Intelligence MCP Server",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    try:
        run_query("SELECT 1")
        return {"status": "healthy", "db": f"{PG_HOST}/{PG_DB}", "tools": ["get_schema_context", "execute_retail_query", "search_products"]}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/mcp") and MCP_BEARER_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != MCP_BEARER_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing Bearer token"})
    return await call_next(request)


app.mount("/mcp", mcp.streamable_http_app())