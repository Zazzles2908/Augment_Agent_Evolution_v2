-- Enable required extensions (safe if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create document_vectors table for per-chunk embeddings
CREATE TABLE IF NOT EXISTS augment_agent.document_vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id UUID NOT NULL,
  chunk_id TEXT NOT NULL,
  page_no INT,
  title TEXT,
  text_excerpt TEXT,
  embedding VECTOR(2000) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_document_vectors_doc FOREIGN KEY (doc_id)
    REFERENCES augment_agent.documents(id) ON DELETE CASCADE
);

-- Basic index for lookups
CREATE INDEX IF NOT EXISTS idx_document_vectors_doc_chunk ON augment_agent.document_vectors(doc_id, chunk_id);

-- ANN index (adjust lists/ef_search via session config)
-- Note: ivfflat requires a parameter for lists. Choose based on dataset size (e.g., 100)
CREATE INDEX IF NOT EXISTS idx_document_vectors_embedding_ivfflat
  ON augment_agent.document_vectors USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

