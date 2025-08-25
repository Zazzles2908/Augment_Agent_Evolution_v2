# Main System – Current Implementation Stage

Scope
- Docling ingestion/chunking, Triton models (qwen3_4b_embedding 2000‑d, qwen3_0_6b_reranking, glm45_air), Redis caching, monitoring
- Supabase and Zen MCP integrations are paused; focus on main system readiness

Status Summary
- Triton configs present for all three models; engines require build/validation
- Ingestion CLI (services/ingestion/docling_ingest.py) exists with naive chunking and Triton embedding; needs Redis caching and decoupling from Supabase for now
- Query path not fully implemented as a single service chaining embed → retrieve → rerank → generate
- Monitoring configs (Prom/Grafana/Loki/Alloy) exist; need wiring and dashboards validation

Gaps to Close (before integrating sub-systems)
1) Build and validate TensorRT engines (FP8/NVFP4) and Triton endpoints
   - Verify /v2/repository load/unload and /v2/models/{name}/infer for all models
2) Add Redis caching
   - Cache embeddings by content hash; optional cache for reranked results
3) Minimal Retrieval Placeholder
   - Implement a local store (e.g., JSONL or SQLite) with cosine similarity for development until Supabase MCP resumes
4) End-to-end Query Service
   - New service that chains embed → retrieve (placeholder) → rerank → generate, with timing and error logging
5) Monitoring
   - Expose per-step timings and counters; ensure Triton metrics appear in Grafana; add simple alerts
6) Performance Targets
   - Tune batch sizes, timeouts, and caching to achieve P95 < 1.5s on target hardware

Deliverables
- Smoke tests for Triton load/unload and inference
- CLI/HTTP service for end-to-end query flow
- Redis-enabled ingestion and query paths
- Monitoring dashboards minimally covering latency, throughput, GPU/VRAM, cache hit rate

Engine smoke scripts checklist (expected)
- scripts/smoke/triton_repository.sh (load/unload models)
- scripts/smoke/embed_infer.py (qwen3_4b_embedding)
- scripts/smoke/rerank_infer.py (qwen3_0_6b_reranking)
- scripts/smoke/generate_infer.py (glm45_air)

