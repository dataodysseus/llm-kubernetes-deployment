-- pgvector setup script
-- Dimensions and user are substituted by the workflow via sed
-- Extension requires superuser (postgres); table/index are owned by PGVECTOR_USER

-- Must run as postgres superuser
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop and recreate schema so re-runs are idempotent
DROP SCHEMA IF EXISTS vectors CASCADE;
CREATE SCHEMA vectors AUTHORIZATION PGVECTOR_USER;

-- Switch to appuser context: all DDL below is owned by PGVECTOR_USER
SET SESSION AUTHORIZATION PGVECTOR_USER;

CREATE TABLE vectors.documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   vector(PGVECTOR_DIMS),
    metadata    JSONB       DEFAULT '{}',
    created_at  TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX documents_embedding_hnsw_idx
    ON vectors.documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Return to superuser
RESET SESSION AUTHORIZATION;

-- Verify extension and ownership
SELECT e.extname AS extension, e.extversion AS version
FROM pg_extension e WHERE e.extname = 'vector';

SELECT schemaname, tablename, tableowner
FROM pg_tables WHERE schemaname = 'vectors';
