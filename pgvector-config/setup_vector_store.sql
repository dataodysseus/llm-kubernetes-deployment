-- pgvector setup script
-- Run against appdb as postgres superuser
-- Dimensions and user are substituted by the workflow via sed

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS vectors AUTHORIZATION PGVECTOR_USER;
GRANT USAGE ON SCHEMA vectors TO PGVECTOR_USER;

CREATE TABLE IF NOT EXISTS vectors.documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   vector(PGVECTOR_DIMS),
    metadata    JSONB       DEFAULT '{}',
    created_at  TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON vectors.documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

GRANT ALL PRIVILEGES ON SCHEMA vectors TO PGVECTOR_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA vectors TO PGVECTOR_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA vectors TO PGVECTOR_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA vectors
    GRANT ALL PRIVILEGES ON TABLES TO PGVECTOR_USER;

SELECT e.extname AS extension, e.extversion AS version
FROM pg_extension e WHERE e.extname = 'vector';

SELECT schemaname, tablename, tableowner
FROM pg_tables WHERE schemaname = 'vectors';
