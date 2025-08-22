Title: Data Integration — Supabase pgvector + Redis Working Memory

Goals
- Persist semantic memory in Supabase (pgvector)
- Use Redis for working memory, queues, and hot embedding cache
- Avoid recomputation with smart caching and usage tracking

1) Supabase pgvector
- Table: knowledge_chunks(id, doc_id, chunk_idx, text, embedding vector, metadata jsonb, created_at)
- Index: ivfflat or hnsw on embedding
- API: service key from server-side only; client uses REST/RPC
- Similarity search: SELECT ... ORDER BY embedding <=> query_vec LIMIT k

2) Redis Working Memory
- Keys:
  - embed:sha256(text) → vector bytes (ttl)
  - usage:model:{name}:count → increment on use
  - queues: ingestion, embedding, reranking
- Patterns: batch embedding by time/size windows; use streams for multi-consumer

3) Embedding Pipeline
- Pre-check Redis cache; on miss, call qwen3_embedding_trt
- Store vector in Supabase and Redis
- Maintain checksum to avoid duplicates; update metadata (source, doc_id)

4) Reranking Pipeline
- Given candidates, call qwen3_reranker_trt if budget allows
- Cache pairwise scores with a composite key

5) Observability
- Metrics: cache hit rate, Supabase latency, queue lengths
- Alerts on db failures → degrade gracefully (operate cache-only for reads)

See also: 10_hrm_processing_flow.md and 12_docling_integration.md.

