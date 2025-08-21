# Environment Validation Checklist

GPU/Driver/CUDA/TRT
- nvidia-smi shows driver >= 570/575 line
- nvcc --version shows CUDA 12.8
- dpkg -l | grep TensorRT shows 10.8 packages

Container runtime
- docker --version (27.x)
- nvidia-ctk --version (>= 1.17.8) and runtime configured
- docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi

Triton
- Container runs: nvcr.io/nvidia/tritonserver:25.07-py3
- Health: curl http://localhost:8000/v2/health/ready
- Metrics: curl http://localhost:8002/metrics | head

Datastores
- redis-server --version (7.4.x)
- psql --version (Postgres 16.x)
- supabase --version (v2.x)

Python
- python3 --version (3.12)
- pip list contains: tritonclient, redis, supabase, psycopg, docling

Models
- /models contains glm45_air, qwen3_4b_embedding, qwen3_0_6b_reranking with config.pbtxt and plan files

Docs
- See 08_ubuntu_24_04_clean_setup.md for install steps

