# Repository Structure (Authoritative)

- README.md — Top-level quickstart and structure
- containers/
  - four-brain/
    - triton/model_repository/{qwen3_4b_embedding,qwen3_0_6b_reranking,glm45_air}/
    - config/monitoring/{prometheus,grafana,loki}/
- services/
  - ingestion/ (Docling ingestion and chunking)
  - (planned) query/ (embed → retrieve (placeholder) → rerank → generate)
- examples/
  - utils/ (redis_client.py, supabase helpers)
  - clients/ (supabase_mcp_python_client.py)
- documents/
  - stack/ (authoritative specs and runbooks)
  - implementation/ (current stage, integration plans, testing)
  - integration/ (integration overviews and contracts)
  - reports/ (Zen recommendations)
- zen-mcp-server/ (external tools; integration paused)
- supabase-mcp-server/ (planned; paused)

Naming Conventions
- Triton model names: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air
- Embedding dimension: 2000
- Keep helpers and examples under examples/

