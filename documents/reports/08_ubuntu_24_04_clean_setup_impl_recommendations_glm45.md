# GLM-4.5 Implementation Recommendations – 08_ubuntu_24_04_clean_setup.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. NVIDIA Stack (Driver ↔ CUDA ↔ TensorRT ↔ Triton)
- Driver: 570+ (or 575.xx) on Ubuntu 24.04
- CUDA: 13.0 recommended (12.8 acceptable)
- TensorRT: 10.8+
- Triton: 25.07+ (py3)

## 2. System Packages & Docker
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git curl ca-certificates gnupg lsb-release unzip jq python3 python3-venv python3-pip
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER
# NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 3. Datastores (Redis, Postgres/pgvector)
```bash
sudo apt install -y redis-server && sudo systemctl enable --now redis-server
sudo apt install -y postgresql postgresql-contrib
sudo -u postgres createdb augment_db
sudo -u postgres psql -d augment_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 4. Triton Install & Model Repo Layout
```bash
docker pull nvcr.io/nvidia/tritonserver:25.08-py3
mkdir -p /models/{embedding,reranker,generation}
# Start Triton (explicit control)
docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /models:/models nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit &
```
Repo layout:
```
/models/{embedding,reranker,generation}/
  ├── 1/
  │   └── model.plan
  └── config.pbtxt
```

## 5. Python Env & Clients
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install redis supabase psycopg[binary] pydantic fastapi uvicorn httpx tritonclient[all] docling langchain-docling
```

## 6. Smoke Tests (GPU, Triton, DB, Cache)
```bash
# GPU
nvidia-smi && nvcc --version
# Triton health
curl -s http://localhost:8000/v2/health/ready && curl -s http://localhost:8002/metrics | head -5
# Redis
redis-cli PING
# Postgres
psql -d augment_db -c "select version();"
```

