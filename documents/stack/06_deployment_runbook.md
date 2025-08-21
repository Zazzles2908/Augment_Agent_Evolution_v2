# Deployment Runbook (Ubuntu 24.04)

Prerequisites
- Ubuntu 24.04 LTS, NVIDIA driver 550+, CUDA 12.x, TensorRT 10.x
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

Notes
- Ensure model_repository is mounted into Triton and names match scripts
- Validate GPU visibility: nvidia-smi
- Verify Redis and DB reachable before embedding loads

