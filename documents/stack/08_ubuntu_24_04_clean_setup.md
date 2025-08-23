# Ubuntu 24.04 Clean Setup (as of 2025-08-21)

Purpose
- Start from a clean slate on Ubuntu 24.04 LTS for the stack: Triton/TensorRT, Qwen models, Docling, GLM‑4.5 Air, Supabase/Postgres(pgvector), Redis, Prometheus/Grafana/Loki/Alloy.
- Align versions known to be compatible as of Aug 21, 2025.

Hardware
- NVIDIA RTX 5070 Ti (16GB) assumed. Adjust as needed.

1) Base OS hygiene
- Fresh install Ubuntu 24.04 LTS
- Update system: sudo apt update && sudo apt full-upgrade -y && sudo reboot
- Essentials: sudo apt install -y build-essential git curl ca-certificates software-properties-common gnupg lsb-release unzip jq python3 python3-venv python3-pip
- Set timezone/locale as needed (timedatectl set-timezone, locale-gen)

2) NVIDIA driver
- Recommended driver series: 575.xx (or 570.124.04+) for 40/50-series; prefer proprietary over open when using TensorRT
- Install (Option A: Ubuntu Additional Drivers GUI) or (Option B: CLI):
  - sudo add-apt-repository ppa:graphics-drivers/ppa -y
  - sudo apt update && sudo apt install -y nvidia-driver-575
  - Reboot and verify: nvidia-smi

3) CUDA Toolkit (13.0 preferred; 12.8 supported)
- CUDA 13.0: Latest major as of 2025-08-21; backwards-compatible with CUDA 12-built apps
- CUDA 12.8: Stable baseline that pairs with many current containers
- Install (NVIDIA repo):
  - sudo apt install -y linux-headers-$(uname -r)
  - wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  - sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
  - EITHER: sudo apt install -y cuda-toolkit-13-0
    OR:     sudo apt install -y cuda-toolkit-12-8
  - echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
  - echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
  - source /etc/profile.d/cuda.sh && nvcc --version

4) TensorRT (supports CUDA 12.x and 13.x)
- Install via NVIDIA repo (matches CUDA repo); choose latest TensorRT 10.8+ (or newer if available)
  - sudo apt install -y tensorrt python3-libnvinfer libnvinfer-dev
  - Verify libraries under /usr/lib/x86_64-linux-gnu and python import if needed

5) Docker Engine + NVIDIA Container Toolkit
- Docker Engine: 27.x (latest stable)
- NVIDIA Container Toolkit: >= 1.17.8 (fixes CVE-2025-23266)
- Install Docker (official docs) then:
  - distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  - curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  - curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - sudo apt update && sudo apt install -y nvidia-container-toolkit
  - sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  - Test (CUDA 13): docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi
    or (CUDA 12.8): docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi

6) Triton Inference Server
- Version: 25.08-py3 (align with PyTorch 25.08 images)
- Recommended: Use NGC container (py3 variant)
  - docker pull nvcr.io/nvidia/tritonserver:25.08-py3
  - Run (example):
    docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
      -v /models:/models nvcr.io/nvidia/tritonserver:25.08-py3 \
      tritonserver --model-repository=/models --model-control-mode=explicit

7) Datastores and caching
- Redis: 7.4.x (apt: redis-server)
- PostgreSQL: 16.x (Ubuntu 24.04) with pgvector >= 0.7.x
  - sudo apt install -y postgresql postgresql-contrib
  - Install pgvector (package or from source). On Supabase local, pgvector is bundled.
- Supabase CLI: v2.x (latest)
  - curl -fsSL https://cli.supabase.com/install/linux | sh

8) Monitoring stack
- Prometheus: 2.5x.x (latest)
- Grafana: 11.x
- Loki: 3.x
- Grafana Alloy: latest (OTel collector distribution)
- Install via official tarballs or containers; keep Prometheus scraping Triton :8002/metrics

9) Python environment and libraries
- Python 3.12 (Ubuntu 24.04 default)
- Create venv: python3 -m venv .venv && source .venv/bin/activate
- pip install --upgrade pip
- Core libs (examples):
  - pip install redis supabase==2.* psycopg[binary] pydantic uvicorn fastapi httpx
  - pip install docling langchain-docling # document processing
  - For client stubs to Triton: pip install tritonclient[all]

10) Models
- Embedding: Qwen3-4B (export to ONNX/TRT; target embedding dim 2000)
- Reranker: Qwen3-0.6B (export to ONNX/TRT)
- Generation: GLM-4.5 Air (TensorRT plan)
- Store plans under /models with proper config.pbtxt (see 04_triton_model_repository.md)

11) Post-install validation
- nvidia-smi; nvcc --version; dpkg -l | grep TensorRT
- docker run --gpus all ... nvidia-smi
- curl http://localhost:8000/v2/health/ready; curl http://localhost:8002/metrics | head
- redis-cli PING; psql --version; supabase --version

Notes
- Exact minor versions change frequently; prefer official docs for latest point releases; maintain compatibility triad (Driver ↔ CUDA ↔ TensorRT ↔ Triton).

