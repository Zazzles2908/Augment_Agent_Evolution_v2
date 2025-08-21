# Four-Brain Orchestrator Ops Guide

This document explains the current state, how to run/maintain the orchestrator hub, and how to continue from here. It captures the HRM high/low adjustments and the container cleaning we performed.

## High-level
- Service: Orchestrator Hub (FastAPI), coordinates Triton models and messaging for Four-Brain system.
- Triton: External container at http://triton:8000.
- Redis and Postgres: Infra containers managed via compose.
- Status: Orchestrator is up and reachable; health is "degraded" until K2/Moonshot API key is provided.

## What changed recently
1) Fixed module imports so server boots
- Replaced legacy `k2_vector_hub.*` imports with `orchestrator_hub.*` in `containers/four-brain/src/orchestrator_hub/hub_service.py`.

2) Fixed CUDA 13 validator dataclass usage
- Updated `containers/four-brain/src/shared/system_validation/config_validator.py` to use `ValidationResult` fields (`title`, `description`, `expected`, `actual`, `passed`, `metadata`, `timestamp`) instead of `message`/`details`.

3) HRM high & low always-on
- Updated `containers/four-brain/src/shared/resource_manager/triton_resource_manager.py` to preload both `hrm_h_trt` and `hrm_l_trt` by default.

4) Compose defaults and smoke mode
- Set `PHASE1_SMOKE=0` in compose to initialize all components.
- Added `.env` and `.env.example` under `containers/four-brain/docker/`.

5) Docker cleanup
- Removed unused images to reclaim SSD space using `docker image prune -a -f`.

## Directory layout (relevant)
- containers/four-brain/docker
  - docker-compose.yml
  - .env (local, secrets)
  - .env.example
  - OPERATIONS_README.md (this file)
- containers/four-brain/src
  - orchestrator_hub/...
  - shared/resource_manager/triton_resource_manager.py
  - shared/system_validation/config_validator.py

## Configuration
- Create `containers/four-brain/docker/.env` (use `.env.example` as template).
- Required for full health:
  - K2_API_KEY = <your Moonshot/Kimi API key>
- Optional but recommended:
  - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY
- Optional: POSTGRES_URL if hub connects directly.

## Running
- From `containers/four-brain/docker`:
  - Build (if needed): `docker compose build orchestrator-hub`
  - Start: `docker compose up -d orchestrator-hub`
  - Health: `curl http://localhost:9018/health`

Expected: `status` becomes `healthy` after K2_API_KEY is set and dependencies initialize.

## Triton models (HRM)
- Ensure model repository contains `hrm_h_trt` and `hrm_l_trt` (and others):
  - Path (host): `containers/four-brain/triton/model_repository`
  - Hub will preload both HRM H/L on start.
- Admin endpoints (examples):
  - `/admin/models/status`
  - `/admin/models/load?model=hrm_l_trt`

## Cleaning SSD / Docker
- We already ran a safe bulk cleanup:
  - `docker image prune -a -f`
- To further clean (with caution):
  - Remove stopped containers: `docker container prune -f`
  - Remove unused volumes: `docker volume prune -f`
  - Remove networks: `docker network prune -f`
  - Everything at once: `docker system prune -a --volumes -f`

Note: Only run global prune if you understand it may remove images/volumes used by other projects.

## Next tasks for a coding agent
1) Secrets and initialization
- Insert `K2_API_KEY` into `.env` and restart orchestrator to reach full health.
- If Supabase is required, populate SUPABASE_* and verify connectivity in logs/health.

2) Triton model repository
- Verify HRM model artifacts exist and match config.pbtxt.
- Add missing models/runtimes per HRM_high&low_module.md.

3) CI and tests
- Add unit tests for validator and resource manager policies.
- Add a healthcheck integration test that mocks Moonshot and Redis.

4) Observability
- Wire basic logs/metrics to your preferred stack (Prometheus/Grafana if retained).

5) Documentation
- Keep `OPERATIONS_README.md` updated with any new env vars or services.

## Troubleshooting
- Health stays degraded due to `api_key_configured=false`:
  - Set `K2_API_KEY` in `.env` and `docker compose up -d orchestrator-hub`.
- Triton load failures:
  - Check model repo path bind and configs under `model_repository`.
- Import errors on boot:
  - Confirm orchestrator_hub package layout and PYTHONPATH inside container (`/workspace/src`).

