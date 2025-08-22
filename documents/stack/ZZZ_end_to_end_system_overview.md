# End-to-End System Overview (Zen MCP Generated)

This document explains the system end-to-end and consolidates Zen MCP outputs (aug_chat, aug_thinkdeep, aug_consensus) with our repo state. It also details how Zen MCP works and which functions we used.

## 1) System at a Glance
- Ingestion: Docling to extract and chunk
- Embeddings: Qwen3-4B via Triton (2000-d)
- Storage: Supabase/Postgres + pgvector; Redis cache
- Reranking: Qwen3-0.6B via Triton
- Generation: GLM-4.5 Air via Triton
- Observability: Prometheus, Grafana, Loki, Alloy
- Platform: Ubuntu 24.04, CUDA 13.x, TensorRT 10.13.x, Triton 25.07+, explicit model control
- Canonical model names: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air

## 2) Architecture & Data Flow
```
Docling → Chunking → Triton(qwen3_4b_embedding) → Supabase(pgvector 2000-d)
Query → Redis → (miss) Supabase vector search → Triton(qwen3_0_6b_reranking) → Triton(glm45_air) → Response
```

## 3) Deployment & Model Repository
- Triton explicit model control with models under /models/<name>/1/model.plan
- Configs committed; engines built locally and placed under 1/
- Prometheus scraping Triton at :8002; Alloy/Loki for logs

## 4) Database Essentials
- vector(2000) embeddings
- RPC: augment_agent.match_documents + public wrapper for PostgREST
- Migrations located at containers/four-brain/src/brains/document_processor/db/migrations_sql

## 5) VRAM Strategy & Performance
- Load only the models you need; prefer sequential phases when memory is tight
- Dynamic batching for throughput; alert on OOM/alloc failures

## 6) Zen MCP Outputs

### 6.1 aug_chat (Architecture Narrative)
<details>
<summary>Output</summary>

[Truncated for brevity in this file; see repo operations log]

Key takeaways:
- Clear component delineation and data flow
- Concrete model repo structure and explicit control guidance
- DB schema with vector(2000) and RPC
- Observability best practices
</details>

### 6.2 aug_thinkdeep (Execution Plan)
<details>
<summary>Output</summary>

- Phase 1: Finalize scope, resolve blockers, implement core features
- Phase 2: Integrate components, full testing, UAT
- Phase 3: Staged deployment, docs, post-launch monitoring
- Risks: TRT compatibility, tokenizer/shape mismatches, pgvector tuning
</details>

### 6.3 aug_consensus (Prioritization)
<details>
<summary>Output</summary>

Tool encountered a provider parsing error when consulting models; we captured the intended prompt and synthesized prioritization using aug_chat/thinkdeep outcomes.

Top ROI tasks:
1) Build/Place TRT engines (NVFP4/FP16) for all three models; verify load/unload
2) Implement Docling ingestion pipeline
3) E2E demo runbook and script with real tokenization
4) Grafana dashboard pack (GPU/latency/error rate); alert rules
5) Redis caching layer for embeddings and reranked sets
</details>

## 7) Current Repo State (Validated)
- Triton configs present for: glm45_air, qwen3_4b_embedding, qwen3_0_6b_reranking
- Supabase auto-migrate: 001 schema, 002 vectors/index, 003 RPC, 004 public wrapper
- Prometheus includes Triton scrape job
- Smoke script at scripts/smoke/e2e_query.py
- Pending: Docling ingestion scaffold; archive models/hrm_simplified_real_fp16.pt

## 8) Required Steps to Finish (with Acceptance Criteria)
1) Build engines and verify serving
   - AC: All models load via /v2/repository; readiness OK; single-batch inference succeeds
2) Docling ingestion pipeline
   - AC: CLI ingests PDFs to augment_agent.document_vectors; metadata persisted; idempotent retries
3) End-to-end demo with real tokenization
   - AC: Query → match → rerank → generate returns answer + citations; error rate <2%
4) Observability dashboards + alerts
   - AC: Grafana shows p50/p95/p99 per model; GPU VRAM; alert on OOM > 0 for 5m
5) Redis caching
   - AC: Cache hit ratio visible; configurable TTL; fallback to DB on cache miss
6) Hygiene
   - AC: Legacy model archived; .gitignore excludes heavy artifacts

## 9) How Zen MCP Works (and what we used)
- Zen MCP is a meta-coordination layer providing specialized reasoning tools:
  - **aug_chat**: conversational synthesis—used to generate the architecture narrative and concise end-to-end description
  - **aug_thinkdeep**: structured step-by-step planning—used to derive the project completion plan, risks, and mitigations
  - **aug_consensus**: multi-model consultation—designed to compare model opinions and produce a ranked prioritization (we hit a provider parsing edge case; we documented the prompt and used the other tools’ outputs to synthesize a stable prioritization)
- Typical workflow: use aug_chat to draft, aug_thinkdeep to decompose and order work, aug_consensus to validate/rank approaches. We embedded these outputs here and aligned the repo accordingly.

## 10) Next Actions I Will Take
- Implement Docling ingestion scaffold under services/ingestion/docling_ingest.py (CLI + batching)
- Add generation client helper for glm45_air and extend smoke script beyond placeholder
- Prepare Grafana dashboards JSON and Alert rules; wire into configs
- Add Redis caching helpers and integrate into query path
- Submit a change to archive models/hrm_simplified_real_fp16.pt (pending approval)

---
This file is the single, generated overview for the project, aligned with documents/stack and current repo state. Update it as tasks complete.
