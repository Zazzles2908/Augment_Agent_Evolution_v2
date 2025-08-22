# Orchestrated Stack Overview (Docling + Qwen3 + Supabase + Redis + Triton/TensorRT + GLM-4.5 Air)


Goal
- Define the end-to-end system replacing HRM with a clean, efficient pipeline suitable for local RTX 5070 Ti 16GB + 64GB RAM.

Components
- Docling: document extraction + chunking
- Qwen3-4B (NVFP4): embeddings via Triton
- Supabase (pgvector 2000-dim) + Redis: storage and caching
- Qwen3-0.6B (NVFP4): reranker via Triton
- GLM-4.5 Air (NVFP4): generation via Triton
- Triton: multi-model serving and metrics
- Prometheus + Grafana + Loki + Alloy: observability

Dataflow
1) Ingest: Docling -> chunks with metadata
2) Embed: Qwen3-4B -> 2000-dim vectors (cache in Redis)
3) Store: upsert text + embeddings into Supabase (HNSW index)
4) Query: embed query -> vector search -> rerank with Qwen3-0.6B
5) Generate: GLM-4.5 Air produces final answer with cited context

Key decisions
- 2000-dim embeddings to match pgvector index limit (Supabase docs)
- NVFP4 quantization for GPU fit and speed; Triton for concurrency
- Strong monitoring: collect metrics from Triton at :8002/metrics

Next
- See 01_document_ingestion_docling.md, 02_embeddings_and_storage.md, 03_query_pipeline_rerank_generate.md, 04_triton_model_repository.md, 05_monitoring_observability.md, 06_deployment_runbook.md, 07_memory_and_performance.md


Canonical Source
- This folder (documents/stack) is the single authoritative specification for architecture, versions, naming, and runbooks. All other docs must align with these files.

Platform & Versions
- OS: Ubuntu 24.04 LTS
- GPU: NVIDIA RTX 5070 Ti 16GB VRAM; System RAM: 64GB
- Drivers/SDK: NVIDIA driver 550+, CUDA 13.x, TensorRT 10.13.x
- Serving: NVIDIA Triton Inference Server 25.07+ (explicit model control)

Canonical Model Names (Triton model_repository)
- qwen3_4b_embedding (embeddings, 2000-dim)
- qwen3_0_6b_reranking (reranker)
- glm45_air (generator)

Repository Hygiene
- Do not commit heavy artifacts (ONNX/PLAN/weights) — version only config.pbtxt and templates
- .gitignore must exclude engines, weights, logs, and large transient outputs

Document Map
- 01_document_ingestion_docling.md — ingestion & chunking
- 02_embeddings_and_storage.md — 2000-dim embeddings, pgvector schema, Redis cache
- 03_query_pipeline_rerank_generate.md — retrieval → rerank → generation
- 04_triton_model_repository.md — repository layout, explicit control, naming, dtypes
- 05_monitoring_observability.md — Prom/Grafana/Loki/Alloy
- 06_deployment_runbook.md — step-by-step run instructions
- 07_memory_and_performance.md — VRAM budget, unloading, fallbacks
- 08_ubuntu_24_04_clean_setup.md — clean setup baseline
- 09_environment_validation.md — smoke/health checks
- 10_implementation_plan.md — phased plan and deliverables
- 11_docker_and_git_hygiene.md — cleanup and hygiene

