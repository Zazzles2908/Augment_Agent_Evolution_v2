# Query Pipeline: Embed → Vector Search → Rerank → Generate

1) Query embedding: Qwen3-4B (FP8) via Triton
2) Vector search: Placeholder/local store for now; Supabase RPC (pgvector HNSW) will be integrated via Supabase MCP later
3) Rerank: Qwen3-0.6B (NVFP4) via Triton on (query, candidates)
4) Generate: GLM-4.5 Air (NVFP4) via Triton with reranked context

Response policy
- Return answer + supporting sources (chunk ids/metadata)
- Cache results if appropriate for fast recall
- Optional: compress context after rerank (top‑k) before generation to reduce latency

Integration phase order
- Phase 1: Main-system only with placeholder retrieval (current)
- Phase 2: Zen MCP validation loops (planning/QA) — no runtime coupling
- Phase 3: Replace retrieval with Supabase MCP vectors.match_documents and refactor ingestion to vectors.upsert_document_vectors

