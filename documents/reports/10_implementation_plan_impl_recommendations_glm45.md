# GLM-4.5 Implementation Recommendations – 10_implementation_plan.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Roadmap Phases
- MVP (Weeks 1–2): ingestion (Docling), embeddings → pgvector, basic retrieval API
- Hardening (Weeks 3–4): rerank + generation, monitoring + alerts, caching, error handling
- Production (Week 5): performance tuning, resilience (circuit breakers), documentation and runbooks

## 2. Detailed Task Breakdown
- Ingestion: DoclingLoader + SemanticChunker, metadata schema, table handling
- Embeddings: Triton Qwen3-4B NVFP4, Redis caching, Supabase schema + HNSW index
- Retrieval: `match_documents` RPC, threshold tuning, pagination
- Rerank: Qwen3-0.6B endpoint, scoring, thresholding, top-k
- Generation: GLM-4.5 Air prompts, citation format, output constraints
- Monitoring: Prometheus/Grafana/Loki, GPU + latency panels, alert rules
- Ops: deployment scripts, explicit model control, health checks, rollbacks

## 3. Resource Plan
- Single RTX 16GB GPU; sequential model load/unload (embed → unload → rerank → unload → generate)
- RAM: Redis 4–8GB, Postgres 8GB, monitoring 2–4GB
- Quantise models (NVFP4) to fit VRAM; use dynamic batching and cooldowns

## 4. Acceptance Criteria & Test Plan
- Unit tests: ingestion, schema, RPC, cache
- Integration: E2E pipeline returns answer with citations < 1.5s p95
- Load test: 10 RPS sustained for 10 min; error rate < 2%
- Failover: degrade gracefully on model overload; fall back to cached results

## 5. Risks, Mitigations, Go/No-Go Gates
- Risk: VRAM pressure → Mitigate with sequential load/unload; reduce batch sizes
- Risk: 2000-dim index perf → Tune HNSW (m, ef_construction); consider ANN params
- Risk: Triton endpoint failures → Circuit breakers; retries with backoff; cached fallback
- Go/No-Go: all health checks pass; e2e p95 < 1.5s; error rate < 2%; dashboards green

