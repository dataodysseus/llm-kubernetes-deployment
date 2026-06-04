-- pgvector sanity test
-- Inserts 3 dummy documents with placeholder vectors
-- Real embeddings will replace these when you connect an embedding model

INSERT INTO vectors.documents (content, embedding, metadata) VALUES
    ('Alice Johnson is a customer from the USA',
     PGVECTOR_ZERO_VEC,
     '{"source": "customers", "country": "USA"}'),
    ('Bob Smith is a customer from Canada',
     PGVECTOR_ZERO_VEC,
     '{"source": "customers", "country": "Canada"}'),
    ('PostgreSQL is a powerful open source database',
     PGVECTOR_ZERO_VEC,
     '{"source": "knowledge_base", "category": "technology"}')
ON CONFLICT DO NOTHING;

SELECT id, content, metadata FROM vectors.documents;
