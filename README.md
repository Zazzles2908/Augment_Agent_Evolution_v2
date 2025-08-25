# Augment Agent Evolution v2

Main system focus: Docling → Triton (Qwen3‑4B embeddings 2000‑d, Qwen3‑0.6B reranker, GLM‑4.5 Air) → Redis caching → Monitoring.

Status
- Supabase and Zen MCP integrations are paused until main system completion
- 2000‑dim vectors, P95 latency target < 1.5s

Repo Structure
- containers/ — Triton model_repository, monitoring configs, four-brain services
- services/ — ingestion and service code
- examples/ — helper clients (Redis, Supabase client examples)
- documents/
  - stack/ — authoritative architecture and runbooks
  - implementation/ — current stage, integration plans, testing plan
  - integration/ — system/subsystem integration guides
  - reports/ — Zen-generated recommendations
- zen-mcp-server/ — external assistance tools (paused for integration into main)
- supabase-mcp-server/ — planned MCP server (paused)

Getting Started (Main System)
- Build TensorRT engines (scripts/tensorrt/config/*) and place plan files under containers/four-brain/triton/model_repository/*/1/
- Start Triton in explicit model-control mode and validate /v2/repository and /infer
- Implement Redis caching in ingestion and query paths
- Create a placeholder retrieval to enable end-to-end testing (replace with Supabase MCP later)

Docs
- See documents/stack/* for specs and runbooks
- See documents/implementation/* for current stage and detailed plans

