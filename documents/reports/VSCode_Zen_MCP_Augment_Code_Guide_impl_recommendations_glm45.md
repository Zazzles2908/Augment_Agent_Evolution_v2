# GLM-4.5 Implementation Recommendations – VSCode_Zen_MCP_Augment_Code_Guide.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. VSCode Setup (Extensions, Settings, Tasks)
- Install: Python, Markdown All in One, GitLens
- Settings: `editor.formatOnSave=true`, Python Black formatter
- Tasks: `.vscode/tasks.json` with a `run_mcp_tool` shell task

## 2. MCP Tool Usage Patterns (Investigation Gates)
- Enforce investigation between tool calls; new evidence each step
- Choose thinking modes based on complexity (low/medium/high/max)
- Keep steps numbered; no recursion without new analysis

## 3. Model Routing (cost_profile=balanced, locale=en-AU)
- .env:
```
COST_AWARE_ROUTING_ENABLED=true
FREE_TIER_PREFERENCE_ENABLED=true
COST_PROFILE=balanced
LOCALE=en-AU
MAX_COST_PER_REQUEST=5.0
```
- Routing: fast → glm-4.5-flash; balanced → glm-4.5-air; extended → kimi-k2-0711-preview

## 4. Auggie CLI Usage (Commands & Examples)
```
auggie init
auggie tool analyze --scope src/
auggie tool debug --model kimi-k2-0711-preview
auggie tool testgen --scope auth/ --thinking_mode medium
auggie model list
auggie version
```

## 5. Troubleshooting & Red-dot Resolution
- Check encoding (UTF-8) and line endings (LF)
- Ensure MCP server running: `auggie version`
- Reduce context or split requests for timeouts
- Use specific scopes to keep costs low

