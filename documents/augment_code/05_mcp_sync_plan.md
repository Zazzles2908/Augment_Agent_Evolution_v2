Goal
- Synchronize MCP servers between VSCode extension and CLI so both agents have the same tool surface without installing anything inside this agent

What exists in repo
- MCP support in qwen-code CLI/core; docs show mcpServers config schema
- Example internal MCP server: containers/four-brain/config/monitoring/alloy/run_alloy_mcp.py

Plan
1) Define a single source of truth file for MCP servers
- Create .qwen/settings.json (or reuse your existing) with an mcpServers block
- Include servers you use in VSCode (e.g., GitHub MCP server, Alloy MCP server)
- Example minimal settings.json snippet:
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": ["dist/server.js"],
      "cwd": "./mcp/github-mcp-server",
      "timeout": 15000,
      "trust": true
    },
    "alloy-four-brain": {
      "command": "python",
      "args": ["containers/four-brain/config/monitoring/alloy/run_alloy_mcp.py"],
      "timeout": 10000
    }
  }
}

2) VSCode Augment
- Point the VSCode extension to the same settings.json so it auto-discovers the same MCPs

3) CLI usage
- When running the CLI, pass --settings or ensure it reads from the same .qwen/settings.json
- Use /mcp to list and verify servers

Notes
- This agent cannot import/install VSCode’s MCPs “into itself” directly; instead we unify configuration so both agents call the same MCP servers
- Keep secrets in env vars; reference via ${ENV_VAR} in settings.json if supported

