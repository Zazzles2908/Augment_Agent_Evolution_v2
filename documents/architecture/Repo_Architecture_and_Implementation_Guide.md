# Repository Architecture and Implementation Guide (Kimi/GLM + MCP)

## 1. Recommended Structure (non‑disruptive)

Root (this repo: Augment_Agent_Evolution_v2)
- zen-mcp-server/ (current MCP; providers: Kimi, GLM; optional: OpenRouter, Custom)
- supabase-mcp-server/ (future MCP; pgvector, auth policies, RPCs) [planned]
- containers/ (Triton, Redis, Docling, workers)
- documents/
  - stack/
  - reports/
  - architecture/ (this file)
- tools/ (top‑level devops helpers, CI, scripts that span multiple components)
- .github/

Notes:
- Keep zen-mcp-server stable; add supabase-mcp-server as a sibling to avoid churn
- Avoid moving entrypoints that users depend on (run-server.sh/ps1) unless necessary
- Scripts inside zen-mcp-server → zen-mcp-server/scripts/ (done)

## 2. MCP components

- zen-mcp-server
  - Providers: Kimi/GLM (+ OpenRouter, Custom optional)
  - 16 public tools (helper tools hidden)
  - Uses .zen_venv Python; mcp-config.json absolute command
- supabase-mcp-server (planned)
  - Tooling: vector upsert/query, match_documents, migrations, RLS checkers
  - Models: Qwen3-embed (2000‑dim MRL), reranker, Docling
  - Focus: Safety (RLS), low‑latency RPCs, TTL cache hints for Redis

## 3. Migration plan (safe)

- Phase 1: Prepare skeleton supabase-mcp-server/
  - Copy minimal server skeleton from zen‑mcp‑server
  - Add README + roadmap + placeholder tools
  - No coupling to zen‑mcp‑server internals
- Phase 2: Implement basic tools
  - list-tables, show-policies, query-vector, upsert-docs
  - Integration tests with a local Supabase project
- Phase 3: Wire Auggie CLI
  - Separate mcp-config.json; avoid conflicts; explicit command path

## 4. User actions required

- Confirm .env secrets for KIMI_API_KEY and GLM_API_KEY (already present) and keep OPENROUTER_API_KEY empty unless needed
- Confirm keeping run-server.sh/ps1 at repo/zen-mcp-server root
- Approve creating supabase-mcp-server/ with a minimal skeleton
- Optionally approve hiding recommend/activity in production (already hidden) and adding coverage reporting

## 5. Coverage reporting (optional)

Value:
- Quantifies test exercise of critical paths (providers, tools registry)
- Helps prevent regressions when optimizing for latency
How:
- pip install pytest-cov
- Run: pytest --maxfail=1 --disable-warnings -q --cov=zen-mcp-server --cov-report=term-missing
- Add scripts/coverage.{sh,ps1} wrappers later
When to add:
- After basic stability; useful before large refactors

## 6. Auggie default model and Kimi/GLM

- Auggie CLI’s own LLM (for CLI-only reasoning) can be Claude or GPT (config‑dependent). This does not change the MCP server’s providers.
- For MCP tool calls that need models, zen‑mcp‑server uses Kimi/GLM per .env and tool inputs
- No configuration changes are needed for Kimi/GLM due to Auggie’s default LLM choice

## 7. Cleanliness rules

- Keep helper tools internal (not listed to clients)
- Keep provider set minimal (Kimi/GLM) unless requested
- Keep scripts under scripts/; only main entrypoints at root
- Document any path changes; validate with Auggie after edits

## 8. Next steps

- I can scaffold supabase-mcp-server/ with a README and a minimal mcp-config.json (skeleton created)
- Add optional coverage wrappers and CI job (GitHub Actions) after approval
- Continue MCP cleanup guided by Zen, verify with Auggie CLI

