# GLM-4.5 Implementation Recommendations – 02_embeddings_and_storage.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Embedding Service (Triton) Setup
- Triton with FP8 plan for Qwen3-4B embeddings; NVFP4 for reranker & generator
- Ensure 2000-dim output vector compatibility with pgvector
- Preprocess inputs (tokenisation, attention masks)
- Configure Triton model repository and config.pbtxt
- Health check endpoint for monitoring

## 2. Supabase Schema & HNSW Indexing
```sql
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(2000) not null,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  constraint unique_content unique (content)
);
create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops)
  with (m=16, ef_construction=64);
```
- Unique content constraint to reduce duplicates
- Tune HNSW parameters (m, ef_construction)
- Consider partial indexes for hot metadata fields

## 3. Redis Cache & Deduplication Design
- SHA-256 content hash as key; TTL ~7 days
- Use pipelines/transactions for atomic check+set
- Cache warming for hot docs; monitor hit/miss to tune TTL/memory

## 4. RPC: match_documents (SQL + Usage)
```sql
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
```
- Batch embeddings in client; add pagination if needed

## 5. Throughput and Cost Optimisations
- Batch embedding generation (10–50 items)
- Connection pooling for Supabase/Redis; async pipelines
- Compress large text before storage
- Tune similarity thresholds for precision/recall balance

## 6. Validation & Monitoring
- Metrics: embedding time, cache hit rate, query latency
- Validate with known similar pairs; monitor index performance
- Alerts on error spikes or perf degradation; test with realistic loads

