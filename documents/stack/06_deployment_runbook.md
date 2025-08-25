# Deployment Runbook (Ubuntu 24.04)

Prerequisites
- Ubuntu 24.04 LTS, NVIDIA driver 550+, CUDA 13.x, TensorRT 10.13.x
- Docker 26+ and NVIDIA Container Toolkit, or native installs

Start services
- redis-server --port 6379
- tritonserver --model-repository=/models --http-port=8000 --metrics-port=8002
- prometheus --config.file=prometheus.yml
- grafana-server --config=grafana.ini
- loki --config.file=loki-config.yaml
- alloy run alloy-config.yaml

Initialize DB
- supabase db start

Sanity checks
- Triton: curl http://localhost:8000/v2/health/ready
- Metrics: curl http://localhost:8002/metrics | head
- Tests: see documents/implementation/04_testing_full_system.md for smoke and acceptance criteria

Notes
- Ensure model_repository is mounted into Triton and names match scripts
- Validate GPU visibility: nvidia-smi
- Verify Redis and DB reachable before embedding loads


Step-by-step (minimal local run)
1) Start Redis, Postgres/Supabase
2) Ensure /models contains qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air with config.pbtxt and plan files. If using a Docling model in Triton, ensure the model name is exactly "docling" for consistency with scripts.
3) Start Triton in explicit mode:
   - tritonserver \
     --model-repository=/models \
     --model-control-mode=explicit \
     --http-port=8000 \
     --metrics-port=8002
4) Load models via HTTP (use curl or tritonclient) and verify /v2/models/{name}
5) Run smoke test embedding → search (placeholder) → rerank → generate (replace placeholder with Supabase MCP later)
6) Validate metrics in Grafana; check logs in Loki

Sanity curl examples
- Readiness: curl http://localhost:8000/v2/health/ready
- Metrics:   curl http://localhost:8002/metrics | head

