# VS Code + Zen MCP: Best‑Practice Setup for Zazzles's Agent

Goal: Fast startup, reliable MCP wiring, easy tool prompts (/zen:chat, planner, codereview, precommit), safe‑by‑default execution, and a single place for the Master Prompt so every session starts right.

## TL;DR
- Configure VS Code User settings.json to register the zen MCP server (stdio).
- Optionally add Workspace settings.json with conventional keys that point to your Master Prompt and default operating mode.
- Validate with simple zen commands (listmodels, version), then follow the Plan → Implement → Precommit → PR loop.

## 1) Prerequisites
- Run zen MCP setup once (creates venv and .env):
  - PowerShell: `./zen-mcp-server/run-server.ps1`
  - Optional: `./zen-mcp-server/run-server.ps1 -Config` to print client config hints
- Ensure `.env` has at least one valid provider key: KIMI_API_KEY or GLM_API_KEY (or OPENROUTER_API_KEY / CUSTOM_API_URL if you prefer those).
- Decide launch mode:
  - Python venv (recommended for VS Code): `zen-mcp-server/.zen_venv/` + `server.py`
  - Docker-run (portable): prebuild `zen-mcp-server:latest` or let compose build it

## 2) VS Code User settings.json (register MCP server)
Path (Windows): `%APPDATA%/Code/User/settings.json`

Preferred: Python venv (stdio)
```json
{
  "mcp": {
    "servers": {
      "zen": {
        "command": "C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.zen_venv/Scripts/python.exe",
        "args": [
          "C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/server.py"
        ],
        "type": "stdio"
      }
    }
  }
}
```

Alternative: Docker-run (stdio)
```json
{
  "mcp": {
    "servers": {
      "zen": {
        "command": "docker",
        "args": [
          "run", "--rm", "-i",
          "--env-file", "C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.env",
          "zen-mcp-server:latest", "python", "server.py"
        ],
        "type": "stdio"
      }
    }
  }
}
```

## 3a) Migration from previous MCP entry (v1 → v2)
If your client had a previous “zen” tool configured, update it to v2 paths and split Command/Args as below.

- Name: zen
- Command: `C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.zen_venv/Scripts/python.exe`
- Args: `C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/server.py`
- PythonPath (if the UI has it): `C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server`
- ENV_FILE (if the UI has it): `C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.env`

Notes
- You can also use `mcp_server_wrapper.py` in place of `server.py` if your client needs a wrapper, but `server.py` is the recommended target.
- Keep client metadata (Default_model=auto, Locale=en-AU, Log_level=INFO) consistent with `.env` below.

Note: If you relocate the repo, update the paths. Restart VS Code after editing.

## 3b) Clients with a single “Command” field (no separate Args)
Some MCP clients only provide a single Command text box. In that case, put both the Python executable and the server script on one line, quoted:

"C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.zen_venv/Scripts/python.exe" "C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/server.py"

Also set environment variables in the client UI (if available):
- ENV_FILE = C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/.env
- Default_model = auto
- Locale = en-AU
- Log_level = INFO
- Auggie_CLI = true (optional)
- Auggie_config = C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/auggie-config.json (optional)

Restart the VS Code window after editing.


## 3) VS Code Workspace settings.json (conventions)
Path: `.vscode/settings.json` in your repo. These keys are conventions for you/your tooling to read; VS Code itself won’t enforce them.
```json
{
  "Zazzles's Agent.masterPromptPath": "${workspaceFolder}/documents/stack/Augment_Code_Master_Guidelines_v2.md",
  "Zazzles's Agent.mode": "Auto",
  "Zazzles's Agent.safeByDefault": true,
  "Zazzles's Agent.maxRuntimeMinutes": 10,
  "Zazzles's Agent.consultationModel": "auto"
}
```
Tip: Keep the Master Prompt under version control. When you update it, append a note to CHANGES.md.

## 4) Boot and Validate
1. Open this repo in VS Code, restart the window.
2. In the chat panel or command palette (depending on client), run one of:
   - `/zen:listmodels Use zen to list available models`
   - `/zen:version What version of zen do I have`
3. Expected: You see “(MCP) [tool]” and a compact response. If not, see Troubleshooting.

## 5) Daily Workflow (Plan → Implement → Precommit → PR)
- Plan (create/continue a plan thread)
  - `/zen:planner Draft a minimal plan to add XYZ, 3 steps max; continue from prior plan if one exists`
- Analyze before edit (optional for multi-file work)
  - `/zen:analyze Identify files and patterns relevant to XYZ and risks`
- Implement small slice
  - Edit code; keep changes small and revertable.
- Validate (safe-by-default)
  - `/zen:precommit Validate changed files only; run lint, type-check, unit tests, build; show blockers only`
- Code review (structured feedback)
  - `/zen:codereview Review my latest diffs for correctness, security, performance; summarize critical → low`
- PR
- `zen-mcp-server/.env` (v2 recommended entries; server reads these):
```
DEFAULT_MODEL=glm-4.5-flash
LOCALE=en-AU
LOG_LEVEL=INFO
# Choose ONE provider path below:
# Option A: Native keys
#   KIMI_API_KEY=...
#   GLM_API_KEY=...
#   (Optionally) KIMI_API_URL=https://api.moonshot.ai/v1
#                GLM_API_URL=https://api.z.ai/api/paas/v4
# Option B: Unified provider
#   OPENROUTER_API_KEY=...
#   # OR DIAL_API_KEY=...
# Option C: Custom OpenAI-compatible endpoint
#   CUSTOM_API_URL=http://localhost:11434/v1
#   CUSTOM_API_KEY=
#   CUSTOM_MODEL_NAME=kimi-k2-0711-preview
DEFAULT_THINKING_MODE_THINKDEEP=high


## 10) Using Auggie CLI with Zen MCP

See also: documents/stack/Auggie_CLI_Terminal_Defaults_Guide.md for setting GPT‑5 as the terminal default orchestrator (PowerShell profile or cross-shell wrapper).
Auggie wrappers are already integrated in the server. To enable CLI-optimized tools (aug_chat, aug_thinkdeep, aug_consensus):

1. Ensure your `.env` has:
```
AUGGIE_CLI=true
AUGGIE_CONFIG=C:/Project/Augment_Agent_Evolution_v2/zen-mcp-server/auggie-config.json
```
2. Optional: tune auggie-config.json
   - wrappers.compact_output=true
   - wrappers.show_progress=true
   - wrappers.error_detail="detailed"
3. Validate from your MCP client:
   - `/zen:listmodels` (sanity)
   - Call Auggie tools directly:
     - `/zen:aug_chat Prompt text...`
     - `/zen:aug_thinkdeep step:"Summarize progress"`
     - `/zen:aug_consensus step:"Evaluate approach A vs B"`
4. If you use the auggie CLI binary separately:
   - Point it at the same repo and ensure the server is resolvable via stdio
   - The wrappers are invoked server-side; no extra installation needed
5. Troubleshooting:
   - If aug_* tools don’t appear, restart the client and check `logs/mcp_server.log`
   - Verify AUGGIE_CLI and AUGGIE_CONFIG are set in the server environment

## 9) Post-migration verification
- Restart VS Code window
- Run quick probes:
  - `/zen:listmodels Use zen to list available models`
  - `/zen:version What version of zen do I have`
- Start with the Master Prompt:
  - `Load and follow the Master Prompt at ${Zazzles's Agent.masterPromptPath}. Mode: ${config:Zazzles's Agent.mode}. Begin with planner to implement <task>.`
- If models are missing: check .env provider keys (KIMI_API_KEY/GLM_API_KEY, or OPENROUTER_API_KEY, or CUSTOM_API_URL/CUSTOM_MODEL_NAME)
- If locale isn’t applied, confirm LOCALE=en-AU is set in .env

  - Summarize changes + risks; open PR with clear title and checklist.

Common quick prompts
- `/zen:debug Help diagnose intermittent failure in <file:line>, gather evidence step-by-step`
- `/zen:testgen Create focused unit tests for <function> covering edge cases`
- `/zen:docgen Update docs for <module>, include complexity and gotchas`

## 6) Performance & Output
Prefer compact, actionable outputs by default. Configure on the server side:
- `zen-mcp-server/.env` – set these (examples):
```
DEFAULT_MODEL=glm-4.5-flash
LOG_LEVEL=INFO
DEFAULT_THINKING_MODE_THINKDEEP=high
```
- `zen-mcp-server/auggie-config.json` (if using Auggie CLI wrappers):
  - `wrappers.compact_output=true`, `wrappers.show_progress=true`, `wrappers.error_detail="detailed"`

Client tips
- Keep chat messages concise; ask for “expand details” only when needed.
- Favor targeted tool runs (e.g., codereview on a subset) to minimize cost/latency.

## 7) Troubleshooting
- Server not detected in VS Code:
  - Reopen window after editing settings.json.
  - Paths must be absolute and correct for your machine.
  - Run `./zen-mcp-server/run-server.ps1 -Follow` in a terminal to watch logs.
- Tools don’t execute:
  - Ensure .env has a valid provider or CUSTOM_API_URL.
  - Try the Python venv option first (fewer moving parts than Docker).
- Output too verbose/noisy:
  - Use compact prompts; set `wrappers.compact_output=true` in auggie-config.json (if applicable) and `LOG_LEVEL=INFO` in .env.
- Token/context errors:
  - Use “precision” tools (tracer/analyze) on specific files; avoid broad directory globs.

## 8) Master Prompt Placement
- Keep the canonical policy at: `documents/stack/Augment_Code_Master_Guidelines_v2.md`.
- Store its path in Workspace settings under `Zazzles's Agent.masterPromptPath` so you (and scripts) can reference it.
- When starting a new thread, you can instruct your agent: “Load and follow the Master Prompt at ${Zazzles's Agent.masterPromptPath}.”

—
This guide aligns with the Zen MCP prompt syntax and the client integration patterns used by the setup scripts. Keep iterating on Workspace settings to match your team’s defaults and budget.

