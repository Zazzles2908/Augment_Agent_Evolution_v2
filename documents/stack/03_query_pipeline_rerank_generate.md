# Query Pipeline: Embed → Vector Search → Rerank → Generate

1) Query embedding: Qwen3-4B (FP8) via Triton
2) Vector search: Supabase RPC over pgvector HNSW (top-k)
3) Rerank: Qwen3-0.6B (NVFP4) via Triton on (query, candidates)
4) Generate: GLM-4.5 Air (NVFP4) via Triton with reranked context

Response policy
- Return answer + supporting sources (chunk ids/metadata)
- Cache results if appropriate for fast recall

