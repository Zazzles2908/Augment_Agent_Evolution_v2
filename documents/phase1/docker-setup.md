# Phase 1 Docker Compose Environment

This compose stack provides Triton, Redis, Prometheus, and Grafana with preconfigured dashboards. Use it to validate the main system with Zen guardrails.

## Prerequisites
- NVIDIA GPU with drivers and NVIDIA Container Toolkit installed
- Docker (24+) and docker compose plugin
- Ports available by default: 8000/8001/8002 (Triton), 6379 (Redis), 9090 (Prometheus), 3000 (Grafana)

## Environment variables
You can override defaults via environment or a .env file in repo root:
- MODEL_REPO=./containers/four-brain/triton/model_repository
- TRITON_HTTP_PORT=8000, TRITON_GRPC_PORT=8001, TRITON_METRICS_PORT=8002
- REDIS_PORT=6379, PROMETHEUS_PORT=9090, GRAFANA_PORT=3000
- TZ=UTC
- GRAFANA_USER=admin, GRAFANA_PASSWORD=admin

## Start/stop
- Start: `docker compose up -d`
- Stop/remove: `docker compose down -v`
- Logs: `docker compose logs -f <service>` (triton|redis|prometheus|grafana|dcgm|redis_exporter)

## Services
- Triton: http://localhost:${TRITON_HTTP_PORT}/v2/health/ready, metrics http://localhost:${TRITON_METRICS_PORT}/metrics
- Redis: redis://localhost:${REDIS_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Grafana: http://localhost:${GRAFANA_PORT} (default admin/admin)

## Dashboards
Grafana auto-loads dashboards from containers/four-brain/config/monitoring/grafana/dashboards/ including:
- triton_redis_gpu.json: p50/p95/p99 latency, GPU utilization, Redis hit rate, Triton RPS

Interpretation tips:
- p95 target: < 1.5s across end-to-end queries. Investigate spikes: look at GPU duty cycle and Triton queue.
- Redis hit rate: should trend upward post-warmup; low hit rates suggest caching keys/TTLs need tuning.

## Zen validation integration
Run Make targets with DRY_RUN=0 to execute with Zen tracing/analysis:
- `make compose-up` then `make smoke-all DRY_RUN=0` to generate traces
- `make zen-report` to produce documents/reports/phase1_validation_report.md

Windows PowerShell equivalent:
- `powershell -File scripts/phase1/Run-Phase1.ps1 -DryRun:$false -UseCompose:$true`

## Model repository mount
By default, the compose file mounts `${MODEL_REPO}` to `/models` in the Triton container. Ensure your TensorRT engine plan files exist under:
- qwen3_4b_embedding/1/model.plan
- qwen3_0_6b_reranking/1/model.plan
- glm45_air/1/model.plan

## Notes
- DCGM exporter exposes GPU metrics for Prometheus; ensure drivers support it.
- Redis exporter exposes Redis metrics for Grafana panels (hit/miss rate, memory).
- You can customize Prometheus scrape config at containers/four-brain/config/monitoring/prometheus/prometheus.compose.yml

