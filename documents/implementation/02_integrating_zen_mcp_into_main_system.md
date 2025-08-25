# Integrating Zen MCP into Main System

Goal
- Use Zen MCP for diagnostics, planning, validation, and automated checks without altering main-system runtime paths

Integration Points
- Planning/QA workflows (Auggie CLI) to run: listmodels, version, self-check
- Diagnostics tools for repo hygiene (16 public tools; helper tools hidden)
- Codegen/QA loops for tests and runbooks

Process
1) Add Zen-driven validation steps to CI/local scripts (smoke, style, doc sync)
2) Use Zen for YAML/JSON config validation and MCP client configuration
3) Keep Zen MCP as a separate stdio server; no tight coupling inside main service

Outputs
- Automated reports in documents/reports/*
- Validated configs (mcp-config.json, VSCode settings) per VSCode_Zen_MCP_Augment_Code_Guide.md

