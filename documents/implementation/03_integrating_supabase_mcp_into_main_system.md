# Integrating Supabase MCP into Main System (Future)

Assumption
- Integration will resume after main-system readiness is achieved

Planned Tools (MVP)
- vectors.upsert_document_vectors(rows)
- vectors.match_documents(query_embedding, match_count, similarity_threshold)
- agents.upsert_message(...)
- agents.match_messages(...)

Steps
1) Implement supabase-mcp-server (stdio -> PostgREST), read keys from .env (server-side only)
2) Replace placeholder retrieval in query service with vectors.match_documents
3) Update ingestion to call vectors.upsert_document_vectors
4) Add Redis TTL hints inside MCP for common results
5) Consolidate SQL migrations and RLS policies (augment_agent schema + public wrapper)

Validation
- Auggie CLI self-check; MCP smoke tests for upsert/match
- Ensure exactly 16 public tools remain visible (helpers hidden)

