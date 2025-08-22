# End-to-End Implementation Plan (Authoritative)

Goal
- Deliver a clean, user-focused, professional system: Docling → Qwen3-4B (embed 2000-dim) → Supabase/pgvector + Redis → Qwen3-0.6B rerank → GLM-4.5 Air generate → Triton/TensorRT serving → Prom/Grafana/Loki/Alloy monitoring.

Authoritative Versions & Naming
- OS: Ubuntu 24.04 LTS; CUDA: 13.x; TensorRT: 10.13.x; Triton: 25.07+
- Canonical model names: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air
- Embedding dimension: 2000 (pgvector vector(2000))

Phase 0 — Clean Slate & Versions
- Confirm OS: Ubuntu 24.04 LTS (fresh)
- Install drivers/CUDA/TRT/Triton per 08_ubuntu_24_04_clean_setup.md
- Validate with 09_environment_validation.md and scripts/env/validate_stack.sh
- Normalize Triton model repo names: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air; remove hrm_* folders
- Ensure .gitignore excludes model artifacts/logs

Phase 1 — Triton Model Repo & Serving
- Prepare /models repo with config.pbtxt and plan files
- Start Triton (25.07-py3) with explicit model control
- Smoke test: scripts/validation/triton_load_unload.ps1 (or Linux equivalent) load/unload models

Phase 2 — Data Layer
- Start Redis 7.4.x and Postgres 16.x (pgvector >= 0.7)
- Supabase CLI v2: init and start local stack; confirm vector ops
- Create documents table and HNSW index; add RPC (match_documents)

Phase 3 — Ingestion & Embeddings
- Implement Docling ingestion pipeline; persistent storage of chunks
- Embedding via Triton (qwen3_4b_embedding); Redis cache
- Upsert to Supabase with metadata

Phase 4 — Query Pipeline
- Query embedding → vector search (RPC) → rerank via qwen3_0_6b_reranking → generate via glm45_air
- Return answer + cited sources; add optional caching

Phase 5 — Monitoring & Ops
- Prometheus scrape Triton :8002/metrics; dashboards in Grafana (GPU, latency, throughput)
- Loki + Alloy for logs and OTel; alerts for failures and GPU pressure

Phase 6 — Personalization & Improvement
- User feedback capture (per-question rating)
- Lightweight profile vectors influencing retrieval/reranking
- Privacy & storage policies via Supabase

Phase 7 — QA & Stabilization
- Integration tests across embedding/rerank/generation paths
- Load tests and memory pressure tests; verify ResourceManager behavior
- Documentation polish and runbooks finalized

Deliverables
- Working services under Docker and/or native
- Scripts for environment validation and smoke tests
- Documentation in documents/stack/* kept up to date
- Documentation in documents/stack/* is the single source of truth; keep it updated

