Goal
- Connect the main brain to your GitHub account without syncing large model artifacts

Safeguards already present
- .gitignore excludes models/, *.onnx, *.plan, numeric version dirs under model_repository, and other large artifacts

Plan
1) Clarify which component is the “main brain” that should talk to GitHub
- If Brain‑3 (intelligence) needs repo access, prefer using a GitHub MCP server or GitHub API client with PAT
- Alternative: integrate a minimal Git workflow client for read-only operations (list issues/PRs) inside orchestrator-hub

2) Recommended: GitHub MCP server
- Add to .qwen/settings.json under mcpServers (see 05_mcp_sync_plan.md)
- Provide GH token via env (e.g., GITHUB_TOKEN)
- VSCode Augment and CLI will both get uniform GitHub tools (create issue, comment on PR, etc.)

3) Optional: direct API client
- For Python services, store token in environment (never in repo)
- Use GitHub REST API via requests or PyGithub for read-only ops

4) Git hygiene
- Do not commit model_repository versioned files; verify .gitignore patterns are effective
- Use a .gitattributes with filter/lfs only if you later decide to track specific large files (not recommended here)

Verification
- Run /mcp (or equivalent) to confirm GitHub server tools are available
- Attempt a harmless read: list open issues in your repo

