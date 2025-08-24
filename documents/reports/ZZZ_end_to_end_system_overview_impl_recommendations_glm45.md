# GLM-4.5 Implementation Recommendations – ZZZ_end_to_end_system_overview.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. System Architecture (ASCII)
```
Docling → Chunking → Triton(Embedding) → Supabase(pgvector) → Reranker(Triton) → GLM-4.5(Triton) → Response
```

## 2. Component Contracts (Triton, Redis, Supabase)
- Triton: `/v2/health/ready`, `/v2/models/{model}/infer`, explicit model control, metrics `:8002/metrics`
- Redis: keys `embeddings:{hash}`, `reranked:{query_hash}`; TTLs: embeddings ~1h, reranked ~5m; pooled connections
- Supabase: RPC `match_documents`, HNSW index on `embedding VECTOR(2000)`; PostgREST exposure

## 3. Data Flow & Schemas
- Ingestion: PDF → Docling → chunks → 2000-d embeddings → `document_vectors`
- Query: query → cache → vector search → top-k → rerank → generate → response with citations
- Schema core:
```sql
document_vectors (
  id UUID PRIMARY KEY,
  content TEXT,
  embedding VECTOR(2000),
  metadata JSONB,
  created_at TIMESTAMPTZ
)
```

## 4. Error Handling & Fallbacks
- Circuit breakers per model; retries with backoff
- Cache miss → DB; Redis down → bypass
- Rerank unavailable → use vector results; OOM → reduce batch/context, unload idle model

## 5. Observability & SLOs
- Metrics: latency p50/p95/p99 per model, cache hit/miss, VRAM, error rates, QPS
- Alerts: OOM > 0 (5m), errors > 2% (10m), cache hit < 80% (30m), p95 > 2× baseline (15m)
- SLOs: 99.9% availability; p95 embed < 500ms; rerank < 200ms; generate < 2s; error < 2%; cache hit > 80%

## 6. Readiness Checklist
- Models built/placed; configs validated; explicit load/unload tested; single-batch infer passes
- Docling ingestion CLI; Redis cache integrated; E2E tokenised test; failure paths validated
- Grafana dashboards + alerts; Loki log; baseline metrics captured
- Perf test at target load; backup strategy; capacity plan; ops docs updated

