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

