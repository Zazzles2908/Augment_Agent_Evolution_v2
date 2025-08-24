# GLM-4.5 Implementation Recommendations – 09_environment_validation.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. GPU/CUDA/TensorRT/Triton
```bash
# GPU driver
nvidia-smi
# CUDA
nvcc --version
# TensorRT packages
dpkg -l | grep tensorrt
# Triton health
curl -s http://localhost:8000/v2/health/ready
```
Pass: Driver ≥ 570/575; CUDA 12.8+; TensorRT 10.8+; Triton READY

## 2. Docker/NVIDIA Runtime
```bash
docker --version && docker info | grep -i nvidia
# GPU visible in container
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi | head -5
```
Pass: Docker 27.x; GPU visible via NVIDIA runtime

## 3. Datastores (Redis, Postgres/Supabase)
```bash
redis-cli PING  # PONG
psql -U postgres -c "SELECT version();"
supabase --version
```
Pass: Redis responds; Postgres 16.x; Supabase CLI available

## 4. Python Env & Packages
```bash
python3 --version  # 3.12
python3 -m venv .venv && source .venv/bin/activate
pip install -q tritonclient[all] redis supabase psycopg[binary] docling
pip list | grep -E "(tritonclient|redis|supabase|psycopg|docling)"
```
Pass: Packages installed; venv active

## 5. Model Repository Checks
```bash
find /models -maxdepth 2 -name config.pbtxt
find /models -type f -name "*.plan" -size +1k -printf "%p %k KB\n"
# Config validation (dry run)
tritonserver --model-repository=/models --model-control-mode=explicit --exit-timeout-secs=15 &
sleep 10 && pkill -f tritonserver
```
Pass: All config/plan files present; Triton logs show no config errors

## 6. End-to-End Smoke Test
```bash
# Embed → search → rerank → generate (pseudo endpoints)
# 1) Embed
echo "hello world" | python embed.py  # produces 2000-dim vector
# 2) Search
psql -d augment_db -c "select id, similarity from match_documents('[...]', 5, 0.3);"
# 3) Rerank
python rerank.py --query "hello world" --ids 1 2 3
# 4) Generate
python generate.py --query "hello world" --context ids=1,2
```
Pass: All stages return without errors within target latency

