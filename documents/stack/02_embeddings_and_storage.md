# Embeddings and Storage (Qwen3-4B + Supabase + Redis)

- Embedding model: Qwen3-4B served via Triton (NVFP4 plan)
- Output dimension: 2000 (aligns with Supabase pgvector HNSW limit)
- Cache: Redis for deduping repeated texts
- Store: Supabase table with text, embedding (vector(2000)), and metadata

Supabase schema (example)

```sql
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(2000) not null,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);
create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops);
```

Embedding flow
1) Check Redis cache by hash(content)
2) If miss, call Triton model qwen3_embedding_trt
3) Store embedding in Redis and upsert into Supabase

