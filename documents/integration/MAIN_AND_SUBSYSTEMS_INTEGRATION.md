# Main and Sub‑systems Integration (Detailed)

Purpose
- Define how the Main System integrates with Zen MCP and Supabase MCP, with precise interfaces and responsibilities.

Components
- Main System: Docling, Triton (Qwen3‑4B, Qwen3‑0.6B, GLM‑4.5 Air), Redis, Supabase
- Zen MCP: external assistance tools (planning, QA, diagnostics); helper tools hidden to keep 16 public
- Supabase MCP (planned): stdio MCP server exposing tools for vector upsert/query and agent messages

Supabase MCP Tools (MVP)
- vectors.upsert_document_vectors(rows: [{doc_id, chunk_id, title?, text_excerpt, embedding[2000], metadata?}]) -> {status}
- vectors.match_documents(query_embedding[2000], match_count=10, similarity_threshold=0.3) -> [{id, doc_id, chunk_id, title, text_excerpt, metadata, similarity}]
- agents.upsert_message(agent_id, role, content, embedding[2000], metadata?) -> {status}
- agents.match_messages(query_embedding[2000], match_count=10, similarity_threshold=0.3) -> [{id, agent_id, role, content, metadata, similarity}]

Data Contracts
- Embeddings: 2000‑dim (MRL); pgvector vector(2000)
- RPCs: augment_agent.match_documents + public wrapper
- Indexes: HNSW vector_cosine_ops (tuned)

Integration Points
1) Ingestion Service
   - Calls Supabase MCP vectors.upsert_document_vectors instead of direct REST table insert
2) Query Service
   - Calls Supabase MCP vectors.match_documents to fetch candidates, then Triton reranker and generator
3) Agent Comms
   - Optionally store and search agent messages via agents.* tools for vectorized agent comms

Security & Config
- .env (server‑side): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SCHEMA=augment_agent
- RLS policies: strict; service role key only in MCP server
- Redis TTL caching hints for query results

Validation & Tooling
- Auggie CLI: listmodels, version, self‑check, targeted MCP smoke
- Observability: Triton metrics; logs for MCP servers

Roadmap
- Phase 1: Implement Supabase MCP (stdio to PostgREST), add smoke tests
- Phase 2: Refactor ingestion/query to MCP tools; consolidate SQL migrations and RLS
- Phase 3: Repo restructuring (Option B) after all green checks

