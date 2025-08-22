-- RPC to match documents using cosine similarity over vector(2000)
-- Mirrors scripts/migrations/supabase_match_documents.sql but scoped to augment_agent schema

CREATE OR REPLACE FUNCTION augment_agent.match_documents(
  query_embedding vector(2000),
  match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.3
)
RETURNS TABLE (
  id uuid,
  doc_id uuid,
  chunk_id text,
  title text,
  text_excerpt text,
  metadata jsonb,
  similarity float
)
LANGUAGE sql STABLE AS $$
  SELECT v.id,
         v.doc_id,
         v.chunk_id,
         v.title,
         v.text_excerpt,
         v.metadata,
         1 - (v.embedding <=> query_embedding) AS similarity
  FROM augment_agent.document_vectors v
  WHERE 1 - (v.embedding <=> query_embedding) >= similarity_threshold
  ORDER BY v.embedding <=> query_embedding
  LIMIT match_count;
$$;

