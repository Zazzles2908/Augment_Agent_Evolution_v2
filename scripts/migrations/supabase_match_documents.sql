-- Supabase RPC: match_documents
-- Usage: psql -f scripts/migrations/supabase_match_documents.sql

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(2000) not null,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops);

create or replace function match_documents(
  query_embedding vector(2000),
  match_count int default 10,
  similarity_threshold float default 0.3
)
returns table (id uuid, content text, metadata jsonb, similarity float)
language sql stable as $$
  select d.id, d.content, d.metadata,
         1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  where 1 - (d.embedding <=> query_embedding) >= similarity_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

