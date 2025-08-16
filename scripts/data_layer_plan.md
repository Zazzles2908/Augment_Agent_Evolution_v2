# Data Layer Strategy: Redis + Supabase (pgvector)

## Decision
- Use Supabase local for Postgres + pgvector (preferred dev parity) on port 5432.
- Remove/disable the internal pgvector Postgres service in compose OR change its mapped host port to avoid conflict when Supabase is active.

## Actions
1. Pick one runtime:
   - A) Supabase local up: point DATABASE_URL/POSTGRES_URL to Supabase; stop compose `postgres`.
   - B) Compose Postgres up: map host port to 5433 to avoid conflicts; keep Supabase off.
2. Ensure `augment_agent` schema exists; enable `vector` extension.
3. Confirm embedding dimension; adjust VECTOR(N) accordingly (current D=2000 from MRL truncation).
4. Cache policy: Redis for fast lookups/session; Supabase for persistent storage.

## Migration Commands (SQL)
- CREATE SCHEMA IF NOT EXISTS augment_agent;
- CREATE EXTENSION IF NOT EXISTS vector;
- ALTER TABLE ... ALTER COLUMN embedding TYPE VECTOR(<D>);

## Testing
- Write/read few embeddings to Supabase and Redis; compare latencies.

