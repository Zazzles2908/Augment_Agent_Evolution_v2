# Testing the Entire System

Scope
- Covers main system end-to-end tests initially; expands to include Supabase MCP once integrated

Test Layers
- Unit: tokenization adapters, Redis helpers, config loaders
- Component: Triton inference (embedding/rerank/generate) with golden tests
- Integration: end-to-end query flow (embed → retrieve placeholder → rerank → generate)
- Performance: P95 latency < 1.5s; GPU VRAM budget adherence
- Observability: Metrics presence and sane values

Initial Test Plan (Main System Only)
1) Triton Load/Unload and Infer smoke for all models
2) Embedding cache hit/miss behavior (Redis)
3) Reranker scores monotonicity and range sanity
4) Generation basic sanity (non-empty outputs, token lengths)
5) End-to-end query returns answer+citations with placeholder retrieval

Extend When Sub-systems Are Added
- Replace placeholder retrieval with Supabase MCP and re-run integration tests
- Add MCP tool smoke tests (upsert/match)
- Validate tool list (16 public) and no MCP errors/drops

