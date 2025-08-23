# GLM-4.5 Implementation Recommendations – 00_stack_overview.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Recommended Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Documents  │───▶│   Docling    │───▶│   Chunks    │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Queries   │───▶│  Embedding   │    │   Redis     │
└─────────────┘    │  (Qwen3-4B)  │◀───┤   Cache     │
                   └──────────────┘    └─────────────┘
                         │
                         ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  Supabase    │◀───┤   Vector    │
                   │ (pgvector)   │    │   Search    │
                   └──────────────┘    └─────────────┘
                         │
                         ▼
                   ┌──────────────┐    ┌─────────────┐
                   │   Rerank     │    │   Results   │
                   │ (Qwen3-0.6B) │───▶│   Top-k     │
                   └──────────────┘    └─────────────┘
                         │
                         ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ Generator    │───▶│ Response    │
                   │ (GLM-4.5 Air)│    │ + Citations│
                   └──────────────┘    └─────────────┘
```

## 2. Model Selection, Quantisation, and Triton Config

- Models: Qwen3-4B for embeddings, Qwen3-0.6B for reranking, GLM-4.5 Air for generation
- Quantisation: FP8 for Qwen3-4B embeddings; NVFP4 for Qwen3-0.6B reranker, Docling, and GLM-4.5 Air (fits 16GB VRAM)
- Triton configuration:
  - Dynamic batching: `max_queue_delay_ms=50`
  - `instance_group` on GPU; set `preferred_memory_space_type=0`
  - Enable/disable models based on demand (model control API)
  - Expose Prometheus metrics on `:8002/metrics`

## 3. Data & Storage (Chunking, Metadata, pgvector/HNSW)

- Chunking: Docling 512–1024 tokens; preserve section headings and page refs
- Metadata: source path/URI, page numbers, chunk id, checksum, timestamp
- Supabase/pgvector schema example:

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY,
  content TEXT NOT NULL,
  embedding VECTOR(2000) NOT NULL,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX documents_embedding_hnsw
  ON documents USING hnsw (embedding vector_cosine_ops);
```

- Redis: TTL-based cache for embeddings (e.g., 24h) to reduce recomputation

## 4. Query Pipeline (Retrieval, Rerank, Generation)

- Retrieval: cosine similarity search, fetch top-20
- Reranking: Qwen3-0.6B to score and select top-5
- Generation: prompt GLM-4.5 Air with reranked chunks + query; enforce citation format
- Optimisations: query caching for repeated queries; prefetch likely hot documents

## 5. Performance & Capacity Planning (VRAM/RAM Budgets)

- VRAM (approx.): Qwen3-4B ~8GB; Qwen3-0.6B ~2GB; GLM-4.5 Air ~5GB; buffer ~1GB
- RAM allocation: Supabase 8GB; Redis 4GB; OS/system buffers 4GB
- Concurrency: limit to ≤3 concurrent active pipelines (one per model)
- Batch: enable dynamic batching on Triton for embeddings and rerank paths

## 6. Monitoring & Ops (Dashboards, Alerts, SLOs)

- Dashboards: latency percentiles (p50/p90/p99), throughput, error rates, GPU/CPU/RAM utilisation
- Alerts: p90 latency > 500ms; error rate > 5%; GPU utilisation > 90% sustained
- SLOs: 99.9% uptime; p90 end-to-end latency < 1000ms; ingestion success > 99.5%
- Logs/Tracing: Loki for logs; OpenTelemetry traces; Grafana panels per model

## 7. Security & Hygiene (Secrets, Repos, CI)

- Secrets via environment variables or secret manager; never commit keys
- `.gitignore` large artifacts (weights, cache, logs); commit only configs/templates
- CI: smoke tests for ingestion/retrieval/generation; schema migration checks; lint/type checks

## 8. Phased Rollout Plan (MVP → Hardening → Prod)

- MVP (Weeks 1–2): Docling ingestion, embeddings write to Supabase, basic retrieval API
- Hardening (Weeks 3–4): add reranking + generation; implement monitoring + alerts
- Production (Week 5): tune batching/concurrency; resilience (retries, backoff); full documentation and runbooks

