Title: Augment Code playbook — Four-Brain HRM System with Triton/TensorRT, NVFP4, and Orchestrator

Purpose
- Provide step-by-step, non-disruptive guidance to implement a resource-managed four-brain HRM architecture
- Cover TensorRT engine build/quantization (NVFP4/FP16) for 8B models, Triton explicit model control with dynamic loading, and on-demand Docling GPU
- Implement orchestrator-based coordination (replaces legacy k2-hub architecture)
- Unify memory/resource management, persistent semantic memory (Supabase pgvector), and working memory (Redis streams)
- Keep Git/MCP integration safe without committing large artifacts

How to use this folder
- Follow the docs roughly in numeric order when triaging/implementing
- Prefer quick health checks and smoke tests before heavy operations

Index
1) 01_triton_explicit_mode_load_infer.md — Load models (explicit mode) and run a minimal working inference
2) 02_qwen3_embedding_request_examples.md — Known-good curl and tritonclient examples
3) 03_tensorrt_trt_engine_swap.md — Build TRT engines (NVFP4/FP16) and swap Triton models
4) 04_quantization_checklist_nvfp4_fp8.md — Quantization pipeline checks (NVFP4/FP8)
5) 05_mcp_sync_plan.md — Sync MCP servers between VSCode and CLI
6) 06_github_integration_main_brain.md — Connect main brain to GitHub safely
7) 07_architecture_hrm_multi_brain.md — Architecture overview, models, budgets, phases
8) 08_triton_config_multi_models.md — Triton explicit control, dynamic batching, load/unload
9) 09_resource_manager_design.md — VRAM tracker, LRU unloading, pressure handling
10) 10_hrm_processing_flow.md — H-Module planning, L-Module execution, iterative loop
11) 11_data_integration_supabase_redis.md — Supabase pgvector + Redis working memory
12) 12_docling_integration.md — On-demand Docling GPU + doc→embedding pipeline
13) 13_monitoring_logging.md — Health, logging, telemetry, graceful degradation

Non-goals
- Do not modify live containers while another agent is running; use guidance docs and staged commands
- Do not commit model weights or engines; see .gitignore rules already present in repo

References in repo
- Triton repo: containers/four-brain/triton/model_repository
- ONNX export: scripts/export_embedding_onnx.py
- TRT build: scripts/build_trt_engine.ps1
- Triton client: containers/four-brain/src/brains/embedding_service/modules/triton_client.py
- Docker compose (Triton explicit mode): containers/four-brain/docker/docker-compose.yml

Four-Brain System Overview
- **Brain 1**: Embedding Service (Qwen3 8B NVFP4) - containers/four-brain/src/brains/embedding_service/
- **Brain 2**: Reranking Service (Qwen3 8B NVFP4) - containers/four-brain/src/brains/reranker_service/
- **Brain 3**: Intelligence Service (HRM Manager) - containers/four-brain/src/brains/intelligence_service/
- **Brain 4**: Document Processor (Docling) - containers/four-brain/src/brains/document_processor/
- **Orchestrator Hub**: Central coordination - containers/four-brain/src/orchestrator_hub/

Checklist before you start
- Ensure Triton container is healthy (HTTP 200 on /v2/health/ready)
- Confirm model files exist under the mounted model_repository
- Know the exact model name you will call (e.g., hrm_h_trt, hrm_l_trt, qwen3_embedding_8b_trt, qwen3_reranker_8b_trt)
- Validate available VRAM (~16GB on RTX 5070 Ti) and budgets before loading additional models
- Verify orchestrator hub is running and coordinating brain services
