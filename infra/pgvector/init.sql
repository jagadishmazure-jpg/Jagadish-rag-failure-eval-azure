-- Local pgvector schema for the offline-ish store path. The embedding width matches
-- BAAI/bge-small-en-v1.5 (384), the same vectors the rest of the eval committed.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL,
    title text,
    text text,
    status text,
    doc_type text,
    district text,
    topic text,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_status ON chunks (status);
CREATE INDEX IF NOT EXISTS chunks_district ON chunks (district);
