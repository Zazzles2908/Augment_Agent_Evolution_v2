# Build Instructions: Local AI Stack

Run one script to build and demo the stack.

## Prerequisites
- Ubuntu 24.04 with CUDA 13 (nvidia-smi shows CUDA Version: 13.x)
- NVIDIA Driver ~580.x
- TensorRT 10.8+ (trtexec in PATH)
- Docker Engine installed (with NVIDIA runtime)
- Python 3.12
- Internet access to download models
- Optional: Supabase URL/Key (for RPC demo)

## One-command build
```
chmod +x scripts/build_full_stack.sh
bash scripts/build_full_stack.sh
```
- Logs written to build.log
- Uses model repo at containers/four-brain/triton/model_repository by default
- Builds NVFP4 engines (falls back to FP16 if needed)
- Starts Triton and runs demo

## What the script does
1) Validates CUDA/TensorRT/Docker/Python
2) Creates Python venv and installs deps
3) Renames Triton models to final names
4) Builds TensorRT engines (NVFP4 → FP16 fallback)
5) Starts Triton and checks /v2/health/ready
6) Applies Supabase migration if possible
7) Runs end-to-end demo

## Expected outputs
- Engine files at: {model}/1/model.plan
- Triton health ready (HTTP 200)
- Demo prints similar documents, reranker scores, and generator logits

## Troubleshooting
- trtexec missing: Install TensorRT and ensure trtexec on PATH
- VRAM exceeded: Script auto-falls back to FP16; reduce builder.workspace_gb in scripts/tensorrt/config/*.yaml
- Network timeouts: Build script retries 3 times; re-run if persistent
- Triton not ready: Ensure ports 8000/8001/8002 are free; check build.log
- Supabase errors: Set SUPABASE_URL and SUPABASE_ANON_KEY; run psql migration manually

## Verify success
- curl http://localhost:8000/v2/health/ready returns 200
- Engines exist in model_repository
- Demo completes without errors

