# Databricks notebook source
# ============================================================
# Notebook  : Item Embeddings → pgvector (GCP PostgreSQL)
# Purpose   : Generate BGE-large-en embeddings for the retail
#             item table and write them to vectors.documents
#             in your GCP PostgreSQL pgvector store via JDBC.
#
# Pre-req   :
#   - BGE-large-en endpoint running in Databricks
#   - pgvector installed in appdb (Workflow 5 completed)
#   - PostgreSQL JDBC driver available in cluster
#     (or install via: %pip install psycopg2-binary)
#   - GCP firewall allows port 5432 from Databricks NAT IP
#
# Cluster   : Any Databricks cluster with MLflow + pandas_udf
# ============================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install dependencies

# COMMAND ----------

# MAGIC %pip install psycopg2-binary
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration
# MAGIC Update these values to match your environment.

# COMMAND ----------

# ── PostgreSQL connection ──────────────────────────────────
PG_HOST     = "34.9.255.250"       # your static IP from Workflow 4
PG_PORT     = 5432
PG_DB       = "appdb"
PG_USER     = "appuser"
PG_PASSWORD = dbutils.secrets.get(scope="pgvector", key="pg_password")

# ── Embedding config ───────────────────────────────────────
EMBEDDING_ENDPOINT = "databricks-bge-large-en"
EMBEDDING_DIM      = 1024
MAX_BATCH          = 150

# ── Source table ───────────────────────────────────────────
# Either use the in-notebook item data or point to an existing table
USE_EXISTING_TABLE = False         # set True if item table already in Databricks
EXISTING_TABLE     = "catalog.schema.item_details"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Create item DataFrame
# MAGIC Skip this cell and set USE_EXISTING_TABLE=True if your item table already exists.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM catalog.sales_analysis.item_details LIMIT 5;

# COMMAND ----------

import random
from pyspark.sql import SparkSession

item_df = spark.table("catalog.sales_analysis.item_details")

# Combine name + description as the text to embed
# This gives richer semantic signal than description alone
from pyspark.sql.functions import concat_ws, col
item_df = item_df.withColumn(
    "embed_text",
    concat_ws(" - ", col("item_name"), col("item_description"))
)

display(item_df)
print(f"Total items: {item_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Generate embeddings using BGE-large-en

# COMMAND ----------

import pandas as pd
import mlflow.deployments
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType

@pandas_udf(ArrayType(FloatType()))
def get_embedding(contents: pd.Series) -> pd.Series:
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def embed_batch(texts: list) -> list:
        try:
            response = deploy_client.predict(
                endpoint=EMBEDDING_ENDPOINT,
                inputs={"input": texts},
            )
            return [e["embedding"] for e in response.data]
        except Exception as e:
            print(f"[WARN] Embedding batch failed: {e}")
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    all_embeddings = []
    for i in range(0, len(contents), MAX_BATCH):
        all_embeddings += embed_batch(contents.iloc[i: i + MAX_BATCH].tolist())
    return pd.Series(all_embeddings)


# Apply UDF to generate embeddings
embedded_df = item_df.withColumn("embedding", get_embedding(col("embed_text")))

display(embedded_df.select("item_id", "item_name", "embed_text", "embedding"))
print(f"Embeddings generated: {embedded_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Validate embedding dimensions

# COMMAND ----------

from pyspark.sql.functions import size

dim_check = embedded_df.select(size("embedding").alias("dim")).distinct().collect()
print(f"Embedding dimensions found: {[r['dim'] for r in dim_check]}")

assert all(r["dim"] == EMBEDDING_DIM for r in dim_check), \
    f"Dimension mismatch! Expected {EMBEDDING_DIM}, got {[r['dim'] for r in dim_check]}"
print(f"Dimension check passed: all embeddings are {EMBEDDING_DIM}d")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Write embeddings to pgvector via psycopg2
# MAGIC
# MAGIC We use psycopg2 directly (not JDBC) because pgvector's
# MAGIC vector type needs the `[x,y,z]` string format which
# MAGIC JDBC doesn't handle natively without extra casting.

# COMMAND ----------

import psycopg2
import json
from pyspark.sql.functions import to_json, struct

# Collect to driver — 50 rows is tiny, safe to collect
rows = embedded_df.select(
    "item_id", "item_name", "item_description", "embed_text", "embedding"
).collect()

print(f"Collected {len(rows)} rows to driver")

# COMMAND ----------

def embedding_to_pgvector(embedding_list: list) -> str:
    """Convert Python list of floats to pgvector string format: [0.1,0.2,...]"""
    return "[" + ",".join(str(round(x, 8)) for x in embedding_list) + "]"


def write_to_pgvector(rows: list, pg_conn_params: dict, db_name: str, table: str = "vectors.documents"):
    """
    Write embedded rows to pgvector table using psycopg2.
    Uses INSERT ... ON CONFLICT DO UPDATE to make it idempotent —
    safe to re-run without creating duplicate rows.
    """
    conn = psycopg2.connect(**pg_conn_params)
    cursor = conn.cursor()

    inserted = 0
    updated  = 0
    errors   = 0

    for row in rows:
        try:
            vec_str  = embedding_to_pgvector(row["embedding"])
            metadata = json.dumps({
                "item_id":          row["item_id"],
                "item_name":        row["item_name"],
                "item_description": row["item_description"],
                "source":           "databricks_item_table"
            })

            cursor.execute(
                """
                INSERT INTO vectors.documents (content, embedding, metadata)
                VALUES (%s, %s::vector, %s::jsonb)
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                (row["embed_text"], vec_str, metadata)
            )

            result = cursor.fetchone()
            if result:
                inserted += 1
            else:
                updated += 1

        except Exception as e:
            print(f"[ERROR] Failed to insert {row['item_id']}: {e}")
            conn.rollback()
            errors += 1
            continue

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Write complete — inserted: {inserted}, skipped (already exists): {updated}, errors: {errors}")
    return inserted, updated, errors


# ── Connection params ──────────────────────────────────────
pg_conn_params = {
    "host":     PG_HOST,
    "port":     PG_PORT,
    "dbname":   PG_DB,
    "user":     PG_USER,
    "password": PG_PASSWORD,
    "connect_timeout": 10
}

# ── Test connection first ──────────────────────────────────
print("Testing PostgreSQL connection...")
try:
    test_conn = psycopg2.connect(**pg_conn_params)
    test_conn.close()
    print("Connection successful")
except Exception as e:
    raise Exception(f"Cannot connect to PostgreSQL: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Execute the write

# COMMAND ----------

inserted, skipped, errors = write_to_pgvector(rows, pg_conn_params, PG_DB)

if errors > 0:
    print(f"Warning: {errors} rows failed to insert. Check logs above.")
else:
    print(f"All {inserted + skipped} rows processed successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Verify in PostgreSQL

# COMMAND ----------

conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()

# Row count
cursor.execute("SELECT COUNT(*) FROM vectors.documents WHERE metadata->>'source' = 'databricks_item_table'")
count = cursor.fetchone()[0]
print(f"Rows in vectors.documents from item table: {count}")

# Sample rows
cursor.execute("""
    SELECT id,
           content,
           metadata->>'item_id'   AS item_id,
           metadata->>'item_name' AS item_name,
           LEFT(embedding::text, 60) || '...' AS embedding_preview
    FROM vectors.documents
    WHERE metadata->>'source' = 'databricks_item_table'
    ORDER BY id
    LIMIT 5
""")
print("\nSample rows:")
print(f"{'id':<5} {'item_id':<8} {'item_name':<20} {'content':<45} {'embedding_preview'}")
print("-" * 120)
for r in cursor.fetchall():
    print(f"{r[0]:<5} {r[3]:<8} {r[4]:<20} {r[1][:43]:<45} {r[4]}")

cursor.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Test semantic similarity search

# COMMAND ----------

# MAGIC %md
# MAGIC Now run a real similarity search — embed a query and find the most similar items.

# COMMAND ----------

def search_similar_items(query_text: str, top_k: int = 5) -> list:
    """
    Embed a query string using BGE-large-en and find the
    top-k most similar items in the pgvector store.
    """
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    # Embed the query
    response = deploy_client.predict(
        endpoint=EMBEDDING_ENDPOINT,
        inputs={"input": [query_text]},
    )
    query_embedding = response.data[0]["embedding"]
    query_vec_str   = embedding_to_pgvector(query_embedding)

    # Query pgvector
    conn   = psycopg2.connect(**pg_conn_params)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            metadata->>'item_name'        AS item_name,
            metadata->>'item_description' AS description,
            1 - (embedding <=> %s::vector) AS similarity
        FROM vectors.documents
        WHERE metadata->>'source' = 'databricks_item_table'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec_str, query_vec_str, top_k)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


# ── Run sample searches ────────────────────────────────────
test_queries = [
    "something with chocolate and nuts",
    "fruity and tropical flavor",
    "coffee or espresso taste",
]

for query in test_queries:
    print(f"\nQuery : '{query}'")
    print("-" * 60)
    results = search_similar_items(query, top_k=3)
    for i, (name, desc, score) in enumerate(results, 1):
        print(f"  {i}. {name:<25} (similarity: {score:.4f})  — {desc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inventory

# COMMAND ----------

import psycopg2
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales.inventory (
        inventory_id      SERIAL PRIMARY KEY,
        item_id           VARCHAR(20)    NOT NULL,
        item_name         VARCHAR(100)   NOT NULL,
        warehouse_id      VARCHAR(10)    NOT NULL,
        warehouse_name    VARCHAR(100)   NOT NULL,
        warehouse_region  VARCHAR(50)    NOT NULL,
        quantity_on_hand  INT            NOT NULL CHECK (quantity_on_hand >= 0),
        reorder_level     INT            NOT NULL,
        reorder_quantity  INT            NOT NULL,
        unit_cost         NUMERIC(10,2)  NOT NULL,
        last_restocked    DATE           NOT NULL,
        status            VARCHAR(20)    NOT NULL
                          CHECK (status IN ('IN_STOCK','LOW_STOCK','OUT_OF_STOCK')),
        created_at        TIMESTAMP      DEFAULT NOW(),
        UNIQUE (item_id, warehouse_id)
    );
""")
 
cursor.execute("""
    GRANT ALL PRIVILEGES ON sales.inventory TO appuser;
    GRANT ALL PRIVILEGES ON SEQUENCE sales.inventory_inventory_id_seq TO appuser;
""")
 
conn.commit()
cursor.close()
conn.close()
print("sales.inventory table created")

# COMMAND ----------

import random
from datetime import date, timedelta

import random
from datetime import date, timedelta
 
# ── 5 warehouses across different regions ─────────────────
warehouses = [
    {"id": "WH-001", "name": "Northeast Distribution Center", "region": "Northeast USA"},
    {"id": "WH-002", "name": "Southeast Fulfillment Hub",     "region": "Southeast USA"},
    {"id": "WH-003", "name": "Midwest Logistics Center",      "region": "Midwest USA"},
    {"id": "WH-004", "name": "West Coast Warehouse",          "region": "West Coast USA"},
    {"id": "WH-005", "name": "Central Distribution Hub",      "region": "Central USA"},
]
 
# ── 50 products (matches your item table) ─────────────────
item_names = [
    "Choco Bliss", "Nutty Crunch", "Caramel Dream", "Minty Fresh", "Berry Burst",
    "Peanut Delight", "Crispy Joy", "Fudge Fantasy", "Toffee Twist", "Coconut Charm",
    "Almond Supreme", "Hazel Heaven", "Marshmallow Magic", "Cookie Craze", "Sugar Rush",
    "Golden Nugget", "Jelly Gem", "Cocoa Swirl", "Vanilla Velvet", "Maple Munch",
    "Cherry Chew", "Orange Zest", "Lemon Drop", "Gummy Glow", "Rainbow Ribbons",
    "S'mores Sensation", "Mocha Melt", "Cinnamon Swirl", "Pecan Pleasure", "Honey Hug",
    "Strawberry Sizzle", "Banana Bonanza", "Apple Aroma", "Grape Gala", "Bubble Bliss",
    "Espresso Edge", "Salted Caramel", "Truffle Treat", "Pumpkin Pop", "Raspberry Ripple",
    "Mango Magic", "Pistachio Punch", "Cranberry Crunch", "Apricot Adventure", "Plum Passion",
    "Tropical Tango", "Lime Lush", "Blueberry Bash", "Butterscotch Burst", "Walnut Whirl"
]
 
random.seed(42)  # fixed seed for reproducibility
 
def random_restock_date() -> date:
    """Random date within last 90 days"""
    days_ago = random.randint(1, 90)
    return date.today() - timedelta(days=days_ago)
 
def derive_status(qty: int, reorder: int) -> str:
    if qty == 0:
        return "OUT_OF_STOCK"
    elif qty <= reorder:
        return "LOW_STOCK"
    else:
        return "IN_STOCK"
 
def generate_inventory_rows() -> list:
    rows = []
    for i, item_name in enumerate(item_names):
        item_id   = f"p-{i+1}"
        unit_cost = round(random.uniform(1.50, 12.00), 2)
 
        for wh in warehouses:
            # Not every product is stocked in every warehouse
            # ~15% chance a product is absent from a given warehouse
            if random.random() < 0.15:
                continue
 
            # Realistic stock levels — some products intentionally low/out
            roll = random.random()
            if roll < 0.08:
                qty = 0                                      # out of stock
            elif roll < 0.20:
                qty = random.randint(1, 25)                  # low stock
            else:
                qty = random.randint(50, 500)                # healthy stock
 
            reorder_level    = random.randint(20, 50)
            reorder_quantity = random.randint(100, 300)
            status           = derive_status(qty, reorder_level)
            last_restocked   = random_restock_date()
 
            rows.append((
                item_id,
                item_name,
                wh["id"],
                wh["name"],
                wh["region"],
                qty,
                reorder_level,
                reorder_quantity,
                unit_cost,
                last_restocked,
                status,
            ))
    return rows
 
inventory_rows = generate_inventory_rows()
print(f"Generated {len(inventory_rows)} inventory rows across {len(warehouses)} warehouses")
 
# Preview as Spark DataFrame
inventory_df = spark.createDataFrame(
    inventory_rows,
    schema=[
        "item_id", "item_name", "warehouse_id", "warehouse_name",
        "warehouse_region", "quantity_on_hand", "reorder_level",
        "reorder_quantity", "unit_cost", "last_restocked", "status"
    ]
)
display(inventory_df)

# COMMAND ----------

def write_inventory(rows: list, pg_conn_params: dict):
    conn   = psycopg2.connect(**pg_conn_params)
    cursor = conn.cursor()
 
    inserted = 0
    skipped  = 0
    errors   = 0
 
    for row in rows:
        try:
            cursor.execute(
                """
                INSERT INTO sales.inventory (
                    item_id, item_name, warehouse_id, warehouse_name,
                    warehouse_region, quantity_on_hand, reorder_level,
                    reorder_quantity, unit_cost, last_restocked, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (item_id, warehouse_id) DO NOTHING
                RETURNING inventory_id
                """,
                row
            )
            result = cursor.fetchone()
            if result:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[ERROR] {row[0]} / {row[2]}: {e}")
            conn.rollback()
            errors += 1
            continue
 
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inventory write complete — inserted: {inserted}, skipped: {skipped}, errors: {errors}")
    return inserted, skipped, errors
 
 
write_inventory(inventory_rows, pg_conn_params)

# COMMAND ----------

conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
# ── 1. Row count by warehouse ──────────────────────────────
print("=== Inventory count by warehouse ===")
cursor.execute("""
    SELECT warehouse_name, warehouse_region,
           COUNT(*)                    AS products_stocked,
           SUM(quantity_on_hand)       AS total_units,
           ROUND(SUM(quantity_on_hand * unit_cost)::numeric, 2) AS stock_value_usd
    FROM sales.inventory
    GROUP BY warehouse_name, warehouse_region
    ORDER BY total_units DESC
""")
for r in cursor.fetchall():
    print(f"  {r[1]:<20} | {r[0]:<35} | products: {r[2]:>3} | units: {r[3]:>6} | value: ${r[4]:>10}")
 
# ── 2. Low stock and out-of-stock alerts ───────────────────
print("\n=== Low stock / out-of-stock alerts ===")
cursor.execute("""
    SELECT item_name, warehouse_id, warehouse_region,
           quantity_on_hand, reorder_level, status
    FROM sales.inventory
    WHERE status IN ('LOW_STOCK', 'OUT_OF_STOCK')
    ORDER BY status DESC, quantity_on_hand ASC
    LIMIT 10
""")
for r in cursor.fetchall():
    print(f"  [{r[5]:<13}] {r[0]:<25} @ {r[2]:<20} — qty: {r[3]:>4} (reorder at {r[4]})")
 
# ── 3. Top 5 products by total stock across all warehouses ─
print("\n=== Top 5 products by total units across all warehouses ===")
cursor.execute("""
    SELECT item_name,
           SUM(quantity_on_hand)  AS total_units,
           COUNT(warehouse_id)    AS warehouses_stocked,
           ROUND(AVG(unit_cost)::numeric, 2) AS avg_unit_cost
    FROM sales.inventory
    GROUP BY item_name
    ORDER BY total_units DESC
    LIMIT 5
""")
for r in cursor.fetchall():
    print(f"  {r[0]:<25} | total units: {r[1]:>5} | in {r[2]} warehouses | avg cost: ${r[3]}")
 
# ── 4. Products missing from 2 or more warehouses ─────────
print("\n=== Products stocked in fewer than 4 warehouses ===")
cursor.execute("""
    SELECT item_name,
           COUNT(warehouse_id) AS warehouse_count,
           STRING_AGG(warehouse_id, ', ' ORDER BY warehouse_id) AS warehouses
    FROM sales.inventory
    GROUP BY item_name
    HAVING COUNT(warehouse_id) < 4
    ORDER BY warehouse_count ASC
    LIMIT 8
""")
for r in cursor.fetchall():
    print(f"  {r[0]:<25} | stocked in {r[1]} warehouse(s): {r[2]}")
 
cursor.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Suppliers

# COMMAND ----------

import psycopg2
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales.suppliers (
        supplier_id        VARCHAR(10)    PRIMARY KEY,
        supplier_name      VARCHAR(150)   NOT NULL,
        contact_name       VARCHAR(100),
        contact_email      VARCHAR(150),
        phone              VARCHAR(30),
        country            VARCHAR(60)    NOT NULL,
        region             VARCHAR(60),
        payment_terms      VARCHAR(50),
        lead_time_days     INT            NOT NULL CHECK (lead_time_days > 0),
        reliability_score  NUMERIC(3,1)   CHECK (reliability_score BETWEEN 1.0 AND 5.0),
        is_active          BOOLEAN        DEFAULT TRUE,
        created_at         TIMESTAMP      DEFAULT NOW()
    );
""")
 
# Junction table: which supplier provides which product
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales.supplier_items (
        supplier_id        VARCHAR(10)    REFERENCES sales.suppliers(supplier_id),
        item_id            VARCHAR(20)    NOT NULL,
        unit_cost          NUMERIC(10,2)  NOT NULL,
        min_order_qty      INT            NOT NULL,
        is_primary         BOOLEAN        DEFAULT FALSE,
        PRIMARY KEY (supplier_id, item_id)
    );
""")
 
cursor.execute("""
    GRANT ALL PRIVILEGES ON sales.suppliers TO appuser;
    GRANT ALL PRIVILEGES ON sales.supplier_items TO appuser;
""")
 
conn.commit()
cursor.close()
conn.close()
print("sales.suppliers and sales.supplier_items tables created")

# COMMAND ----------

import random
random.seed(42)
 
# ── 10 realistic suppliers ─────────────────────────────────
suppliers = [
    ("SUP-001", "Sweet Valley Foods Inc.",       "James Carter",    "j.carter@sweetvalley.com",    "+1-212-555-0101", "USA",         "Northeast",  "Net 30",  7,  4.8),
    ("SUP-002", "Global Confections Ltd.",        "Maria Santos",    "m.santos@globalconf.co.uk",   "+44-20-7946-001", "UK",          "London",     "Net 45",  14, 4.5),
    ("SUP-003", "Alpine Sweets GmbH",             "Hans Mueller",    "h.mueller@alpinesweets.de",   "+49-89-555-0303", "Germany",     "Bavaria",    "Net 30",  21, 4.6),
    ("SUP-004", "Pacific Rim Treats Co.",         "Yuki Tanaka",     "y.tanaka@pacificrim.jp",      "+81-3-5555-0404", "Japan",       "Tokyo",      "Net 60",  28, 4.3),
    ("SUP-005", "Southern Harvest LLC",           "Bobby Williams",  "b.williams@southernhvst.com", "+1-404-555-0505", "USA",         "Southeast",  "Net 30",  5,  4.7),
    ("SUP-006", "Tropical Delights S.A.",         "Carlos Mendez",   "c.mendez@tropdelights.mx",    "+52-55-5555-060", "Mexico",      "Mexico City","Net 45",  10, 4.2),
    ("SUP-007", "Nordic Naturals AB",             "Ingrid Larsson",  "i.larsson@nordicnat.se",      "+46-8-555-0707",  "Sweden",      "Stockholm",  "Net 30",  18, 4.9),
    ("SUP-008", "Midwest Grain & Sugar Co.",      "Tom Kowalski",    "t.kowalski@mwgrainsugar.com", "+1-312-555-0808", "USA",         "Midwest",    "Net 15",  3,  4.6),
    ("SUP-009", "Cocoa Origins Brazil Ltda.",     "Ana Ferreira",    "a.ferreira@cocoaorigins.br",  "+55-11-5555-090", "Brazil",      "São Paulo",  "Net 60",  35, 4.4),
    ("SUP-010", "West Coast Premium Ingredients", "Lisa Chang",      "l.chang@wcpremium.com",       "+1-310-555-1010", "USA",         "West Coast", "Net 30",  7,  4.7),
]
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
inserted_sup = 0
for s in suppliers:
    cursor.execute("""
        INSERT INTO sales.suppliers (
            supplier_id, supplier_name, contact_name, contact_email,
            phone, country, region, payment_terms, lead_time_days,
            reliability_score
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (supplier_id) DO NOTHING
        RETURNING supplier_id
    """, s)
    if cursor.fetchone():
        inserted_sup += 1
 
conn.commit()
print(f"Suppliers inserted: {inserted_sup}")
 
# ── Assign suppliers to products ──────────────────────────
# Each product gets 1-2 suppliers; one is marked primary
item_names = [
    "Choco Bliss","Nutty Crunch","Caramel Dream","Minty Fresh","Berry Burst",
    "Peanut Delight","Crispy Joy","Fudge Fantasy","Toffee Twist","Coconut Charm",
    "Almond Supreme","Hazel Heaven","Marshmallow Magic","Cookie Craze","Sugar Rush",
    "Golden Nugget","Jelly Gem","Cocoa Swirl","Vanilla Velvet","Maple Munch",
    "Cherry Chew","Orange Zest","Lemon Drop","Gummy Glow","Rainbow Ribbons",
    "S'mores Sensation","Mocha Melt","Cinnamon Swirl","Pecan Pleasure","Honey Hug",
    "Strawberry Sizzle","Banana Bonanza","Apple Aroma","Grape Gala","Bubble Bliss",
    "Espresso Edge","Salted Caramel","Truffle Treat","Pumpkin Pop","Raspberry Ripple",
    "Mango Magic","Pistachio Punch","Cranberry Crunch","Apricot Adventure","Plum Passion",
    "Tropical Tango","Lime Lush","Blueberry Bash","Butterscotch Burst","Walnut Whirl"
]
 
supplier_ids = [s[0] for s in suppliers]
inserted_si  = 0
 
for i, item_name in enumerate(item_names):
    item_id = f"p-{i+1}"
 
    # Pick 1-2 suppliers per product
    num_suppliers = random.choices([1, 2], weights=[0.4, 0.6])[0]
    chosen = random.sample(supplier_ids, num_suppliers)
 
    for idx, sup_id in enumerate(chosen):
        unit_cost     = round(random.uniform(0.80, 8.00), 2)
        min_order_qty = random.choice([50, 100, 200, 500])
        is_primary    = (idx == 0)
 
        cursor.execute("""
            INSERT INTO sales.supplier_items
                (supplier_id, item_id, unit_cost, min_order_qty, is_primary)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (supplier_id, item_id) DO NOTHING
            RETURNING supplier_id
        """, (sup_id, item_id, unit_cost, min_order_qty, is_primary))
        if cursor.fetchone():
            inserted_si += 1
 
conn.commit()
cursor.close()
conn.close()
print(f"Supplier-item links inserted: {inserted_si}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promotions

# COMMAND ----------

conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales.promotions (
        promotion_id       VARCHAR(15)    PRIMARY KEY,
        promotion_name     VARCHAR(150)   NOT NULL,
        promotion_type     VARCHAR(50)    NOT NULL
                           CHECK (promotion_type IN (
                               'PERCENTAGE_DISCOUNT','FIXED_DISCOUNT',
                               'BUY_X_GET_Y','FREE_SHIPPING','BUNDLE'
                           )),
        discount_value     NUMERIC(6,2),
        min_order_amount   NUMERIC(10,2)  DEFAULT 0,
        applicable_items   TEXT[],
        applicable_region  VARCHAR(60)    DEFAULT 'ALL',
        start_date         DATE           NOT NULL,
        end_date           DATE           NOT NULL,
        usage_limit        INT,
        times_used         INT            DEFAULT 0,
        is_active          BOOLEAN        DEFAULT TRUE,
        created_by         VARCHAR(100),
        created_at         TIMESTAMP      DEFAULT NOW(),
        CHECK (end_date >= start_date)
    );
""")
 
cursor.execute("""
    GRANT ALL PRIVILEGES ON sales.promotions TO appuser;
""")
 
conn.commit()
cursor.close()
conn.close()
print("sales.promotions table created")

# COMMAND ----------

from datetime import date, timedelta
 
today      = date.today()
last_month = today - timedelta(days=30)
next_month = today + timedelta(days=30)
last_qtr   = today - timedelta(days=90)
 
# Mix of active, upcoming, and expired promotions
promotions = [
    # Active promotions
    (
        "PROMO-2026-001", "Spring Chocolate Festival",
        "PERCENTAGE_DISCOUNT", 15.00, 25.00,
        ["p-1","p-7","p-8","p-18","p-27"],
        "ALL",
        last_month, next_month,
        500, 312, True, "sales_team"
    ),
    (
        "PROMO-2026-002", "Healthy Snack Bundle Deal",
        "BUNDLE", 10.00, 50.00,
        ["p-2","p-6","p-11","p-29","p-42"],
        "ALL",
        last_month, next_month,
        200, 87, True, "marketing_team"
    ),
    (
        "PROMO-2026-003", "West Coast Weekend Special",
        "FIXED_DISCOUNT", 5.00, 30.00,
        ["p-33","p-34","p-35","p-46","p-47"],
        "West Coast USA",
        today - timedelta(days=5), today + timedelta(days=9),
        150, 43, True, "regional_manager"
    ),
    (
        "PROMO-2026-004", "Free Shipping on Tropical Range",
        "FREE_SHIPPING", 0.00, 20.00,
        ["p-41","p-43","p-46","p-47","p-48"],
        "ALL",
        last_month, next_month,
        None, 215, True, "sales_team"
    ),
    (
        "PROMO-2026-005", "Buy 2 Get 1 Free — Fruit Flavors",
        "BUY_X_GET_Y", 33.33, 0.00,
        ["p-22","p-23","p-24","p-31","p-32"],
        "ALL",
        today - timedelta(days=3), today + timedelta(days=27),
        300, 129, True, "marketing_team"
    ),
    # Upcoming promotions
    (
        "PROMO-2026-006", "Summer Refresh Campaign",
        "PERCENTAGE_DISCOUNT", 20.00, 40.00,
        ["p-4","p-22","p-23","p-46","p-47"],
        "ALL",
        today + timedelta(days=5), today + timedelta(days=35),
        400, 0, True, "marketing_team"
    ),
    (
        "PROMO-2026-007", "Northeast Loyalty Reward",
        "FIXED_DISCOUNT", 8.00, 35.00,
        ["p-1","p-2","p-3","p-4","p-5"],
        "Northeast USA",
        today + timedelta(days=10), today + timedelta(days=40),
        100, 0, True, "regional_manager"
    ),
    # Expired promotions (historical data)
    (
        "PROMO-2025-001", "Holiday Season Mega Sale",
        "PERCENTAGE_DISCOUNT", 25.00, 50.00,
        ["p-1","p-7","p-8","p-13","p-26"],
        "ALL",
        last_qtr - timedelta(days=30), last_qtr,
        1000, 892, False, "sales_team"
    ),
    (
        "PROMO-2025-002", "New Year Bundle Bonanza",
        "BUNDLE", 15.00, 60.00,
        ["p-2","p-6","p-11","p-37","p-38"],
        "ALL",
        last_qtr - timedelta(days=15), last_qtr + timedelta(days=15),
        500, 347, False, "marketing_team"
    ),
    (
        "PROMO-2025-003", "Valentine's Sweet Deal",
        "FIXED_DISCOUNT", 6.00, 25.00,
        ["p-1","p-8","p-9","p-30","p-37"],
        "ALL",
        last_qtr + timedelta(days=10), last_qtr + timedelta(days=24),
        250, 198, False, "sales_team"
    ),
]
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
inserted_promo = 0
for p in promotions:
    cursor.execute("""
        INSERT INTO sales.promotions (
            promotion_id, promotion_name, promotion_type,
            discount_value, min_order_amount, applicable_items,
            applicable_region, start_date, end_date,
            usage_limit, times_used, is_active, created_by
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (promotion_id) DO NOTHING
        RETURNING promotion_id
    """, p)
    if cursor.fetchone():
        inserted_promo += 1
 
conn.commit()
cursor.close()
conn.close()
print(f"Promotions inserted: {inserted_promo}")

# COMMAND ----------

conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
# ── 1. Active promotions right now ────────────────────────
print("=== Currently active promotions ===")
cursor.execute("""
    SELECT promotion_name, promotion_type,
           discount_value, applicable_region,
           start_date, end_date,
           times_used,
           CASE WHEN usage_limit IS NULL THEN 'unlimited'
                ELSE CONCAT(times_used, ' / ', usage_limit)
           END AS usage
    FROM sales.promotions
    WHERE is_active = TRUE AND start_date <= CURRENT_DATE AND end_date >= CURRENT_DATE
    ORDER BY end_date ASC
""")
for r in cursor.fetchall():
    print(f"  [{r[1]:<20}] {r[0]:<35} | {r[3]:<15} | {r[4]} → {r[5]} | usage: {r[7]}")
 
# ── 2. Best performing promotions (by times used) ─────────
print("\n=== Top promotions by usage (all time) ===")
cursor.execute("""
    SELECT promotion_name, promotion_type, times_used,
           is_active,
           CASE WHEN usage_limit IS NULL THEN NULL
                ELSE ROUND(times_used * 100.0 / usage_limit, 1)
           END AS pct_used
    FROM sales.promotions
    ORDER BY times_used DESC
    LIMIT 5
""")
for r in cursor.fetchall():
    active = "ACTIVE" if r[3] else "EXPIRED"
    pct    = f"{r[4]}%" if r[4] else "unlimited"
    print(f"  [{active}] {r[0]:<35} | type: {r[1]:<20} | used: {r[2]:>4} ({pct})")
 
# ── 3. Supplier reliability ranking ───────────────────────
print("\n=== Supplier reliability ranking ===")
cursor.execute("""
    SELECT s.supplier_name, s.country, s.lead_time_days,
           s.reliability_score, s.payment_terms,
           COUNT(si.item_id) AS products_supplied
    FROM sales.suppliers s
    LEFT JOIN sales.supplier_items si ON si.supplier_id = s.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.country,
             s.lead_time_days, s.reliability_score, s.payment_terms
    ORDER BY s.reliability_score DESC
""")
for r in cursor.fetchall():
    print(f"  {r[3]} ⭐  {r[0]:<40} | {r[1]:<10} | lead: {r[2]:>3}d | terms: {r[4]:<8} | products: {r[5]}")
 
# ── 4. Products with no primary supplier ──────────────────
print("\n=== Products with no primary supplier assigned ===")
cursor.execute("""
    SELECT i.item_id, i.item_name
    FROM sales.inventory i
    WHERE NOT EXISTS (
        SELECT 1 FROM sales.supplier_items si
        WHERE si.item_id = i.item_id AND si.is_primary = TRUE
    )
    GROUP BY i.item_id, i.item_name
    ORDER BY i.item_id
    LIMIT 10
""")
rows = cursor.fetchall()
if rows:
    for r in rows:
        print(f"  {r[0]:<10} {r[1]}")
else:
    print("  All products have a primary supplier assigned")
 
# ── 5. Low stock items with supplier info ─────────────────
print("\n=== Low/out-of-stock items with primary supplier ===")
cursor.execute("""
    SELECT inv.item_name, inv.warehouse_region,
           inv.quantity_on_hand, inv.status,
           s.supplier_name, s.lead_time_days,
           s.reliability_score
    FROM sales.inventory inv
    JOIN sales.supplier_items si ON si.item_id = inv.item_id AND si.is_primary = TRUE
    JOIN sales.suppliers s       ON s.supplier_id = si.supplier_id
    WHERE inv.status IN ('LOW_STOCK','OUT_OF_STOCK')
    ORDER BY inv.status DESC, s.lead_time_days ASC
    LIMIT 10
""")
for r in cursor.fetchall():
    print(f"  [{r[3]:<13}] {r[0]:<25} @ {r[1]:<20} | qty: {r[2]:>4} | supplier: {r[4]} ({r[5]}d lead, {r[6]}⭐)")
 
cursor.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bill of Material (BOM)

# COMMAND ----------

import psycopg2
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales.bom (
        bom_id              SERIAL         PRIMARY KEY,
        product_id          VARCHAR(20)    NOT NULL,
        product_name        VARCHAR(100)   NOT NULL,
        raw_material_id     VARCHAR(15)    NOT NULL,
        raw_material_name   VARCHAR(100)   NOT NULL,
        raw_material_category VARCHAR(60)  NOT NULL,
        supplier_id         VARCHAR(10)    REFERENCES sales.suppliers(supplier_id),
        quantity_required   NUMERIC(10,4)  NOT NULL CHECK (quantity_required > 0),
        unit_of_measure     VARCHAR(20)    NOT NULL,
        unit_cost           NUMERIC(10,4)  NOT NULL,
        is_critical         BOOLEAN        DEFAULT FALSE,
        notes               TEXT,
        created_at          TIMESTAMP      DEFAULT NOW(),
        UNIQUE (product_id, raw_material_id)
    );
""")
 
cursor.execute("""
    GRANT ALL PRIVILEGES ON sales.bom TO appuser;
    GRANT ALL PRIVILEGES ON SEQUENCE sales.bom_bom_id_seq TO appuser;
""")
 
conn.commit()
cursor.close()
conn.close()
print("sales.bom table created")

# COMMAND ----------

import random
random.seed(42)
 
# ── Raw material master (id, name, category, supplier_id, unit_cost, uom) ──
raw_materials = [
    # Base ingredients
    ("RM-001", "Cocoa Powder",          "Base Ingredient",  "SUP-009", 4.50,  "kg"),
    ("RM-002", "Milk Chocolate Coating","Base Ingredient",  "SUP-001", 6.20,  "kg"),
    ("RM-003", "White Sugar",           "Base Ingredient",  "SUP-008", 0.80,  "kg"),
    ("RM-004", "Glucose Syrup",         "Base Ingredient",  "SUP-008", 1.20,  "kg"),
    ("RM-005", "Whole Milk Powder",     "Base Ingredient",  "SUP-001", 3.40,  "kg"),
    ("RM-006", "Wheat Flour",           "Base Ingredient",  "SUP-008", 0.60,  "kg"),
    ("RM-007", "Vegetable Fat",         "Base Ingredient",  "SUP-010", 2.10,  "kg"),
    ("RM-008", "Cocoa Butter",          "Base Ingredient",  "SUP-009", 8.90,  "kg"),
 
    # Flavoring agents
    ("RM-009", "Vanilla Extract",       "Flavoring",        "SUP-005", 12.00, "L"),
    ("RM-010", "Caramel Paste",         "Flavoring",        "SUP-001", 5.50,  "kg"),
    ("RM-011", "Peppermint Oil",        "Flavoring",        "SUP-007", 18.00, "L"),
    ("RM-012", "Strawberry Concentrate","Flavoring",        "SUP-005", 9.00,  "kg"),
    ("RM-013", "Mango Puree",           "Flavoring",        "SUP-006", 7.50,  "kg"),
    ("RM-014", "Espresso Extract",      "Flavoring",        "SUP-003", 22.00, "L"),
    ("RM-015", "Maple Syrup",           "Flavoring",        "SUP-005", 14.00, "L"),
    ("RM-016", "Coconut Cream",         "Flavoring",        "SUP-006", 4.80,  "kg"),
    ("RM-017", "Raspberry Puree",       "Flavoring",        "SUP-007", 8.20,  "kg"),
    ("RM-018", "Orange Zest Extract",   "Flavoring",        "SUP-003", 11.00, "L"),
 
    # Nuts & inclusions
    ("RM-019", "Roasted Almonds",       "Nuts & Inclusions","SUP-010", 9.50,  "kg"),
    ("RM-020", "Dry Roasted Peanuts",   "Nuts & Inclusions","SUP-005", 4.20,  "kg"),
    ("RM-021", "Hazelnut Paste",        "Nuts & Inclusions","SUP-003", 11.00, "kg"),
    ("RM-022", "Pecan Pieces",          "Nuts & Inclusions","SUP-005", 13.50, "kg"),
    ("RM-023", "Walnut Halves",         "Nuts & Inclusions","SUP-010", 10.80, "kg"),
    ("RM-024", "Pistachio Kernels",     "Nuts & Inclusions","SUP-004", 16.00, "kg"),
    ("RM-025", "Shredded Coconut",      "Nuts & Inclusions","SUP-006", 3.60,  "kg"),
    ("RM-026", "Crispy Rice",           "Nuts & Inclusions","SUP-004", 2.80,  "kg"),
    ("RM-027", "Marshmallow Pieces",    "Nuts & Inclusions","SUP-001", 5.10,  "kg"),
 
    # Packaging
    ("RM-028", "Foil Wrapper",          "Packaging",        "SUP-010", 0.05,  "units"),
    ("RM-029", "Cardboard Box",         "Packaging",        "SUP-010", 0.12,  "units"),
    ("RM-030", "Printed Label",         "Packaging",        "SUP-008", 0.03,  "units"),
]
 
# ── BOM map: product_id → list of (raw_material_id, qty, is_critical) ─────
# Each product gets 3-5 raw materials reflecting its flavor profile
bom_map = {
    "p-1":  [("RM-001",0.120,"Base Ingredient",True),  ("RM-002",0.180,"Base Ingredient",True),  ("RM-009",0.005,"Flavoring",False),      ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-2":  [("RM-002",0.150,"Base Ingredient",True),  ("RM-020",0.080,"Nuts & Inclusions",False),("RM-003",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-3":  [("RM-010",0.200,"Flavoring",True),        ("RM-002",0.120,"Base Ingredient",True),  ("RM-004",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-4":  [("RM-011",0.008,"Flavoring",True),        ("RM-002",0.150,"Base Ingredient",True),  ("RM-003",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-5":  [("RM-012",0.100,"Flavoring",False),       ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-6":  [("RM-020",0.120,"Nuts & Inclusions",False),("RM-002",0.100,"Base Ingredient",True), ("RM-003",0.040,"Base Ingredient",False), ("RM-007",0.030,"Base Ingredient",False),("RM-028",1.0,"Packaging",False)],
    "p-7":  [("RM-026",0.080,"Nuts & Inclusions",False),("RM-002",0.150,"Base Ingredient",True), ("RM-006",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-8":  [("RM-001",0.100,"Base Ingredient",True),  ("RM-008",0.080,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-004",0.040,"Base Ingredient",False),("RM-028",1.0,"Packaging",False)],
    "p-9":  [("RM-010",0.180,"Flavoring",True),        ("RM-003",0.080,"Base Ingredient",False), ("RM-004",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-10": [("RM-025",0.120,"Nuts & Inclusions",False),("RM-016",0.080,"Flavoring",False),      ("RM-002",0.100,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-11": [("RM-019",0.100,"Nuts & Inclusions",True), ("RM-002",0.150,"Base Ingredient",True), ("RM-009",0.005,"Flavoring",False),       ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-12": [("RM-021",0.120,"Nuts & Inclusions",True), ("RM-002",0.130,"Base Ingredient",True), ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-13": [("RM-027",0.150,"Nuts & Inclusions",False),("RM-002",0.100,"Base Ingredient",True), ("RM-003",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-14": [("RM-026",0.060,"Nuts & Inclusions",False),("RM-002",0.120,"Base Ingredient",True), ("RM-001",0.080,"Base Ingredient",True),  ("RM-006",0.050,"Base Ingredient",False),("RM-028",1.0,"Packaging",False)],
    "p-15": [("RM-003",0.120,"Base Ingredient",False),  ("RM-004",0.080,"Base Ingredient",False), ("RM-002",0.150,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-16": [("RM-010",0.160,"Flavoring",True),        ("RM-003",0.060,"Base Ingredient",False), ("RM-002",0.100,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-17": [("RM-012",0.080,"Flavoring",False),       ("RM-004",0.060,"Base Ingredient",False), ("RM-002",0.120,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-18": [("RM-001",0.130,"Base Ingredient",True),  ("RM-008",0.060,"Base Ingredient",True),  ("RM-004",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-19": [("RM-009",0.010,"Flavoring",False),       ("RM-005",0.080,"Base Ingredient",False), ("RM-002",0.150,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-20": [("RM-015",0.015,"Flavoring",True),        ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-21": [("RM-012",0.090,"Flavoring",False),       ("RM-002",0.110,"Base Ingredient",True),  ("RM-003",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-22": [("RM-018",0.008,"Flavoring",False),       ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-23": [("RM-003",0.100,"Base Ingredient",False),  ("RM-004",0.060,"Base Ingredient",False), ("RM-002",0.080,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-24": [("RM-004",0.120,"Base Ingredient",False),  ("RM-012",0.060,"Flavoring",False),       ("RM-002",0.080,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-25": [("RM-003",0.080,"Base Ingredient",False),  ("RM-004",0.060,"Base Ingredient",False), ("RM-002",0.100,"Base Ingredient",True),  ("RM-007",0.030,"Base Ingredient",False),("RM-028",1.0,"Packaging",False)],
    "p-26": [("RM-027",0.100,"Nuts & Inclusions",False),("RM-001",0.100,"Base Ingredient",True),  ("RM-002",0.120,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-27": [("RM-014",0.012,"Flavoring",True),        ("RM-001",0.080,"Base Ingredient",True),  ("RM-002",0.120,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-28": [("RM-002",0.130,"Base Ingredient",True),  ("RM-003",0.070,"Base Ingredient",False), ("RM-009",0.006,"Flavoring",False),       ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-29": [("RM-022",0.100,"Nuts & Inclusions",False),("RM-002",0.120,"Base Ingredient",True), ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-30": [("RM-015",0.012,"Flavoring",False),       ("RM-002",0.130,"Base Ingredient",True),  ("RM-003",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-31": [("RM-012",0.100,"Flavoring",False),       ("RM-002",0.110,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-32": [("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-004",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-33": [("RM-002",0.110,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-009",0.005,"Flavoring",False),       ("RM-028",1.0,"Packaging",False)],
    "p-34": [("RM-004",0.100,"Base Ingredient",False),  ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-35": [("RM-002",0.150,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-005",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-36": [("RM-014",0.015,"Flavoring",True),        ("RM-001",0.090,"Base Ingredient",True),  ("RM-002",0.110,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-37": [("RM-010",0.180,"Flavoring",True),        ("RM-008",0.060,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-38": [("RM-001",0.120,"Base Ingredient",True),  ("RM-008",0.080,"Base Ingredient",True),  ("RM-009",0.006,"Flavoring",False),       ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-39": [("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-009",0.005,"Flavoring",False),       ("RM-028",1.0,"Packaging",False)],
    "p-40": [("RM-017",0.090,"Flavoring",False),       ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-41": [("RM-013",0.100,"Flavoring",False),       ("RM-002",0.110,"Base Ingredient",True),  ("RM-003",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-42": [("RM-024",0.090,"Nuts & Inclusions",True), ("RM-002",0.130,"Base Ingredient",True), ("RM-003",0.050,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-43": [("RM-002",0.110,"Base Ingredient",True),  ("RM-017",0.070,"Flavoring",False),       ("RM-026",0.050,"Nuts & Inclusions",False),("RM-028",1.0,"Packaging",False)],
    "p-44": [("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-004",0.040,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-45": [("RM-002",0.130,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-007",0.030,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False)],
    "p-46": [("RM-016",0.080,"Flavoring",False),       ("RM-013",0.060,"Flavoring",False),       ("RM-002",0.100,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-47": [("RM-002",0.110,"Base Ingredient",True),  ("RM-003",0.050,"Base Ingredient",False), ("RM-018",0.006,"Flavoring",False),       ("RM-028",1.0,"Packaging",False)],
    "p-48": [("RM-017",0.100,"Flavoring",False),       ("RM-002",0.120,"Base Ingredient",True),  ("RM-003",0.060,"Base Ingredient",False), ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
    "p-49": [("RM-010",0.140,"Flavoring",True),        ("RM-005",0.060,"Base Ingredient",False), ("RM-002",0.110,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False)],
    "p-50": [("RM-023",0.090,"Nuts & Inclusions",False),("RM-002",0.120,"Base Ingredient",True), ("RM-001",0.080,"Base Ingredient",True),  ("RM-028",1.0,"Packaging",False),  ("RM-030",1.0,"Packaging",False)],
}
 
# Build raw material lookup
rm_lookup = {rm[0]: rm for rm in raw_materials}
 
item_names = [
    "Choco Bliss","Nutty Crunch","Caramel Dream","Minty Fresh","Berry Burst",
    "Peanut Delight","Crispy Joy","Fudge Fantasy","Toffee Twist","Coconut Charm",
    "Almond Supreme","Hazel Heaven","Marshmallow Magic","Cookie Craze","Sugar Rush",
    "Golden Nugget","Jelly Gem","Cocoa Swirl","Vanilla Velvet","Maple Munch",
    "Cherry Chew","Orange Zest","Lemon Drop","Gummy Glow","Rainbow Ribbons",
    "S'mores Sensation","Mocha Melt","Cinnamon Swirl","Pecan Pleasure","Honey Hug",
    "Strawberry Sizzle","Banana Bonanza","Apple Aroma","Grape Gala","Bubble Bliss",
    "Espresso Edge","Salted Caramel","Truffle Treat","Pumpkin Pop","Raspberry Ripple",
    "Mango Magic","Pistachio Punch","Cranberry Crunch","Apricot Adventure","Plum Passion",
    "Tropical Tango","Lime Lush","Blueberry Bash","Butterscotch Burst","Walnut Whirl"
]
 
conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
inserted = 0
for i, item_name in enumerate(item_names):
    product_id = f"p-{i+1}"
    materials  = bom_map.get(product_id, [])
 
    for rm_id, qty, category, is_critical in materials:
        rm = rm_lookup[rm_id]
        cursor.execute("""
            INSERT INTO sales.bom (
                product_id, product_name, raw_material_id, raw_material_name,
                raw_material_category, supplier_id, quantity_required,
                unit_of_measure, unit_cost, is_critical
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (product_id, raw_material_id) DO NOTHING
            RETURNING bom_id
        """, (
            product_id, item_name,
            rm_id, rm[1], category,
            rm[3], qty, rm[5], rm[4],
            is_critical
        ))
        if cursor.fetchone():
            inserted += 1
 
conn.commit()
cursor.close()
conn.close()
print(f"BOM rows inserted: {inserted}")

# COMMAND ----------

conn   = psycopg2.connect(**pg_conn_params)
cursor = conn.cursor()
 
# ── 1. BOM for a specific product ─────────────────────────
print("=== Bill of Materials: Choco Bliss (p-1) ===")
cursor.execute("""
    SELECT b.raw_material_id, b.raw_material_name,
           b.raw_material_category, b.quantity_required,
           b.unit_of_measure, b.unit_cost,
           b.is_critical, s.supplier_name, s.lead_time_days
    FROM sales.bom b
    JOIN sales.suppliers s ON s.supplier_id = b.supplier_id
    WHERE b.product_id = 'p-1'
    ORDER BY b.is_critical DESC, b.raw_material_category
""")
for r in cursor.fetchall():
    critical = "CRITICAL" if r[6] else "        "
    print(f"  [{critical}] {r[0]} {r[1]:<25} | {r[3]:>6} {r[4]:<6} @ ${r[5]:>6}/unit | {r[7]} ({r[8]}d)")
 
# ── 2. Raw material cost to produce 100 units ─────────────
print("\n=== Material cost to produce 100 units (top 10 products by cost) ===")
cursor.execute("""
    SELECT product_name,
           ROUND(SUM(quantity_required * unit_cost * 100)::numeric, 2) AS cost_100_units,
           COUNT(raw_material_id) AS num_materials
    FROM sales.bom
    GROUP BY product_id, product_name
    ORDER BY cost_100_units DESC
    LIMIT 10
""")
for r in cursor.fetchall():
    print(f"  {r[0]:<25} | cost for 100 units: ${r[1]:>8} | {r[2]} materials")
 
# ── 3. Critical material risk — which materials affect most products ───
print("\n=== Critical raw materials (single-source risk) ===")
cursor.execute("""
    SELECT b.raw_material_id, b.raw_material_name,
           b.raw_material_category, s.supplier_name,
           s.country, s.lead_time_days,
           COUNT(b.product_id) AS products_affected
    FROM sales.bom b
    JOIN sales.suppliers s ON s.supplier_id = b.supplier_id
    WHERE b.is_critical = TRUE
    GROUP BY b.raw_material_id, b.raw_material_name,
             b.raw_material_category, s.supplier_name,
             s.country, s.lead_time_days
    ORDER BY products_affected DESC
    LIMIT 10
""")
for r in cursor.fetchall():
    print(f"  {r[0]} {r[1]:<25} | {r[5]:>3}d lead | {r[3]:<35} ({r[4]}) | affects {r[6]} products")
 
# ── 4. Supplier dependency — which suppliers are most relied on ───────
print("\n=== Supplier dependency (total BOM lines per supplier) ===")
cursor.execute("""
    SELECT s.supplier_name, s.country, s.reliability_score,
           COUNT(b.bom_id)                              AS total_bom_lines,
           COUNT(DISTINCT b.product_id)                 AS products_dependent,
           SUM(CASE WHEN b.is_critical THEN 1 ELSE 0 END) AS critical_lines
    FROM sales.suppliers s
    JOIN sales.bom b ON b.supplier_id = s.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.country, s.reliability_score
    ORDER BY total_bom_lines DESC
""")
for r in cursor.fetchall():
    print(f"  {r[0]:<40} | {r[1]:<10} | {r[2]}⭐ | BOM lines: {r[3]:>3} | products: {r[4]:>3} | critical: {r[5]:>2}")
 
# ── 5. If a supplier goes down — impact analysis ──────────
print("\n=== Impact if SUP-009 (Cocoa Origins Brazil) goes down ===")
cursor.execute("""
    SELECT b.product_name, b.raw_material_name,
           b.is_critical, inv.quantity_on_hand, inv.status
    FROM sales.bom b
    LEFT JOIN sales.inventory inv ON inv.item_id = b.product_id
    WHERE b.supplier_id = 'SUP-009'
    ORDER BY b.is_critical DESC, inv.quantity_on_hand ASC
    LIMIT 15
""")
for r in cursor.fetchall():
    critical = "CRITICAL" if r[2] else "        "
    qty      = r[3] if r[3] is not None else "N/A"
    status   = r[4] if r[4] is not None else "N/A"
    print(f"  [{critical}] {r[0]:<25} needs {r[1]:<25} | stock: {str(qty):>5} ({status})")
 
cursor.close()
conn.close()