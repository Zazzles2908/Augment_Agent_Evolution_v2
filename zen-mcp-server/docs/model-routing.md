# Zen MCP Model Routing & Cost-Aware Selection (GLM + Kimi)

This document explains how Zen MCP selects and routes models intelligently across GLM and Kimi providers, balancing cost, latency, and task complexity. It also covers ThinkDeep and Consensus usage, observability, and integration hooks.

## Goals
- Keep default behavior cheap (free/low-cost) while remaining effective
- Escalate reasoning capacity only when necessary (ThinkDeep, BALANCED→EXTENDED)
- Provide explicit gates for costly/rare models (glm-4.5-x, vision)
- Offer operator overrides and budget caps

## Models and costs
Use output $/MTok as single-number cost for ordering/capping. Configure in `.env`:

```
MODEL_COSTS_JSON={"glm-4.5-flash":0.0,"glm-4.5-air":1.1,"glm-4.5":2.2,"glm-4.5-airx":4.5,"glm-4.5v":1.8,"glm-4.5-x":8.9,"kimi-k2-turbo-preview":2.0,"kimi-k2-0711-preview":2.5,"kimi-k2-thinking":2.5}
MAX_COST_PER_REQUEST=5.0
FREE_TIER_PREFERENCE_ENABLED=true
FREE_MODEL_LIST=glm-4.5-flash
```

Restrict for predictability:
```
GLM_ALLOWED_MODELS=glm-4.5-flash,glm-4.5-air,glm-4.5,glm-4.5-airx
KIMI_ALLOWED_MODELS=kimi-k2-0711-preview,kimi-k2-turbo-preview,kimi-k2-thinking
```

## Category → model fallbacks
- FAST_RESPONSE: [glm-4.5-flash, glm-4.5-air, glm-4.5]
- BALANCED: [glm-4.5-air, glm-4.5, kimi-k2-turbo-preview]
- EXTENDED_REASONING: [kimi-k2-thinking, glm-4.5-airx, kimi-k2-0711-preview] (glm-4.5-x only on explicit override)
- Vision: glm-4.5v only when explicitly selected

## Context-aware selection
Zen inspects tool category and can factor prompt hints (keywords like "image", "diagram", "vision", "think step-by-step", "chain of thought") to bias selection within the allowed/cost-capped candidates. Keep a cheap-first bias; escalate when:
- user forces a model, or
- tool is EXTENDED_REASONING, or
- prompt contains reasoning/vision hints and cap allows escalation.

## ThinkDeep & Consensus
- DEFAULT_THINKING_MODE_THINKDEEP=high in `.env`
- Use ThinkDeep for hard problems; run a Consensus pass (glm-4.5-air vs kimi-k2-thinking) for high-stakes decisions.

## Observability & costs
- Enable metrics with `PROMETHEUS_ENABLED=true`; server exposes per-model telemetry.
- Registry now estimates cost per model using accumulated usage tokens (see utils/costs.py). Exported via `ModelProviderRegistry.get_telemetry()`.

## Integration hooks
- Vector DB: implement a StorageBackend that proxies to Supabase/pgvector (see utils/storage_backend.py for interface) and enable via env switch.
- MCP bridge templates exist under docs and utils for adding new tools/providers.

## Operator overrides
- Force model in tool call: `model="kimi-k2-thinking"` or `model="glm-4.5-airx"`
- Vision: `model="glm-4.5v"`
- Raise cap temporarily: `MAX_COST_PER_REQUEST=10.0` then restart
- Keep default cheap: leave free-first flags and restrictions as above

