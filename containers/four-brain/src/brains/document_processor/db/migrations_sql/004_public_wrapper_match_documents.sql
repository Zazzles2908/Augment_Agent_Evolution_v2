-- Public wrapper to expose RPC via PostgREST/Supabase
-- Supabase REST RPC endpoint: /rest/v1/rpc/match_documents

CREATE OR REPLACE FUNCTION public.match_documents(
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
  SELECT * FROM augment_agent.match_documents(query_embedding, match_count, similarity_threshold);
$$;

