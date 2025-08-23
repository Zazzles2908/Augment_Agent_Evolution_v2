# GLM-4.5 Implementation Recommendations – 06_deployment_runbook.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Prereqs & System Prep
- Ubuntu 24.04 LTS; NVIDIA Driver 550+; CUDA 13.x; TensorRT 10.13.x
- Validate:
```bash
# OS
lsb_release -a
# GPU + driver
nvidia-smi
# Docker (optional)
docker --version
```

## 2. Services Bring-up (Triton, Redis, Supabase client, Monitoring)
```bash
# Redis
redis-server --port 6379 --daemonize yes

# Supabase local dev (or ensure remote is reachable)
supabase start  # or configure env vars for remote

# Triton (explicit control)
tritonserver \
  --model-repository=/models \
  --model-control-mode=explicit \
  --http-port=8000 \
  --metrics-port=8002 \
  --log-verbose=0 &

# Monitoring stack (example)
prometheus --config.file=/etc/prometheus/prometheus.yml &
grafana-server --homepath /usr/share/grafana &
# Loki/Alloy as per configs
```

## 3. Model Load & Validation
```bash
# Load models
curl -s -X POST localhost:8000/v2/repository/models/qwen3_4b_embedding/load
curl -s -X POST localhost:8000/v2/repository/models/qwen3_0_6b_reranking/load
curl -s -X POST localhost:8000/v2/repository/models/glm45_air/load

# Health checks
curl -s localhost:8000/v2/health/ready | jq .

# Functional smoke checks (pseudo)
# embedding infer (adjust to your model I/O schema)
curl -s -X POST localhost:8000/v2/models/qwen3_4b_embedding/infer -d '{"inputs":[{"name":"input_ids","shape":[1,8],"datatype":"INT32","data":[101,7592,102,0,0,0,0,0]}]}'
```

## 4. E2E Validation (embed → search → rerank → generate)
```bash
# 1) Embed query (cache if applicable)
# 2) RPC to Supabase match_documents
psql "$SUPABASE_URL" -c "select * from match_documents($$[0.1, ... 2000 dims ...]$$, 10, 0.3);"
# 3) Rerank top-k via Triton qwen3_0_6b_reranking
# 4) Generate with glm45_air using reranked context
```
- Record timings per step; target e2e P95 < 1.5s

## 5. Rollback & Recovery
- Unload problematic model:
```bash
curl -s -X POST localhost:8000/v2/repository/models/<model>/unload
```
- Restart Triton/Redis services if unstable
- If DB corruption suspected, restore from latest backup snapshot
- Document clear runbook steps for common failures (model load fail, OOM, RPC fail)

## 6. Daily Operations Checklist
- Triton ready: `curl -s localhost:8000/v2/health/ready`
- GPU utilisation and VRAM headroom via Grafana
- Redis memory usage and eviction rate
- Supabase RPC latency p50/p95; error rates < 2%
- Logs: scan Triton and app logs for ERROR spikes (Loki dashboards)

