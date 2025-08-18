# Inference Stack — TensorRT + Triton Configuration (CUDA 13)

This guide details the model optimization and serving stack tuned for RTX 5070 Ti 16GB.

## Prerequisites
- Windows 11 Pro + Docker Desktop (WSL2 backend)
- NVIDIA driver + WSL2 NVIDIA Container Toolkit
- CUDA 13.x runtime via Triton container; TensorRT 10.13.x

## Models & Targets
- Qwen3-8B Embedding — TensorRT FP8
- Qwen3-8B Reranker — TensorRT NVFP4
- Docling Model — TensorRT NVFP4

> Tip: If VRAM is tight, test 4B variants or FP16 fallback.

## Engine Building (WSL2/Linux shell)
```bash
# Example directory layout
models/
  qwen3-8b-embed/
  qwen3-8b-rerank/
  docling/
triton/model_repository/
  qwen_embed/1/model.plan
  qwen_rerank/1/model.plan
  docling/1/model.plan

# Convert to ONNX (if needed) or use provided ONNX
python export_to_onnx.py --model models/qwen3-8b-embed --out qwen3-embed.onnx

# Build FP8 engine (embedding)
trtexec --onnx=qwen3-embed.onnx \
  --fp8 --saveEngine=triton/model_repository/qwen_embed/1/model.plan \
  --device=0 --memPoolSize=workspace:4096 --buildOnly

# Build NVFP4 engine (reranker)
trtexec --onnx=qwen3-rerank.onnx \
  --nvfp4 --saveEngine=triton/model_repository/qwen_rerank/1/model.plan \
  --device=0 --memPoolSize=workspace:4096 --buildOnly

# Build NVFP4 engine (Docling)
trtexec --onnx=docling.onnx \
  --nvfp4 --saveEngine=triton/model_repository/docling/1/model.plan \
  --device=0 --memPoolSize=workspace:4096 --buildOnly
```

## Triton Model Configs (config.pbtxt)
```protobuf
name: "qwen_embed"
platform: "tensorrt_plan"
max_batch_size: 32
instance_group [{ kind: KIND_GPU, count: 1 }]
dynamic_batching { preferred_batch_size: [4, 8, 16], max_queue_delay_microseconds: 5000 }
input [ { name: "INPUT_0", data_type: TYPE_FP16, dims: [ -1 ] } ]
output [ { name: "OUTPUT_0", data_type: TYPE_FP16, dims: [ 2000 ] } ]
```

```protobuf
name: "qwen_rerank"
platform: "tensorrt_plan"
max_batch_size: 16
instance_group [{ kind: KIND_GPU, count: 1 }]
dynamic_batching { preferred_batch_size: [2,4,8], max_queue_delay_microseconds: 5000 }
input [ { name: "PAIR_INPUT", data_type: TYPE_FP16, dims: [ -1 ] } ]
output [ { name: "SCORES", data_type: TYPE_FP16, dims: [ -1 ] } ]
```

```protobuf
name: "docling"
platform: "tensorrt_plan"
max_batch_size: 4
instance_group [{ kind: KIND_GPU, count: 1 }]
dynamic_batching { preferred_batch_size: [1,2], max_queue_delay_microseconds: 20000 }
input [ { name: "IMAGE_TOKENS", data_type: TYPE_FP16, dims: [ -1, -1 ] } ]
output [ { name: "STRUCTURED_JSON", data_type: TYPE_BYTES, dims: [ 1 ] } ]
```

## Triton Runtime
- Start Triton container (see docker-compose.yml) with model_repository mounted.
- Use explicit model control endpoints to load/unload models:
  - POST /v2/repository/models/{model}/load
  - POST /v2/repository/models/{model}/unload
- Monitor: GET /v2/health/ready, /metrics

## Orchestrator Integration
- Orchestrator calls Triton load/unload around HRM plans
- Cache hot models; evict idle ones (LRU)

## Validation
- Smoke test with `perf_client`:
```bash
perf_client -m qwen_embed -b 8 -p 5000 --concurrency-range 1:2
```

## Troubleshooting
- If FP8/NVFP4 flags not supported, fallback to `--fp16`.
- If OOM, reduce `max_batch_size` or `instance_group.count`.
- Ensure ONNX opsets are compatible with TensorRT; re-export if needed.



## Blackwell (SM_120) Optimization Notes
- Target architecture: SM_120 (Blackwell). Ensure NVIDIA driver and CUDA 13.x match Triton/TensorRT images.
- Precision strategy:
  - Embedding: FP8 (output dims 2000)
  - Reranker: NVFP4 (fallback FP16)
  - Docling: NVFP4 (fallback FP16)
- Concurrency on 16GB VRAM: 1 instance per model; orchestrator hot-load/unload to fit budget.

### TensorRT Builder Config (cheat‑sheet)
- Use per‑model optimization profiles to bound input sizes and reduce activations.
- Example trtexec flags:
```bash
trtexec --onnx=model.onnx \
  --fp8 \
  --device=0 \
  --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=5 \
  --skipInference \
  --precisionConstraints=prefer \
  --hardwareCompatibilityLevel=0 \
  --enableLayerNormPlugin \
  --enableQAPlugin
```
- For NVFP4 (Blackwell): replace --fp8 with --nvfp4.
- Recommended: verify SM target (automatically detected in recent TRT); profile with Nsight to confirm DPX/TMA utilization.

### Triton Dynamic Batching Presets
- Embedding (qwen_embed): preferred_batch_size [4,8,16], queue_delay 2–5ms
- Reranker (qwen_rerank): preferred_batch_size [2,4,8], queue_delay ~5ms
- Docling: small batches [1,2], queue_delay ~20ms

### Eviction Strategy
- Orchestrator loads required model before call; unloads idle models when VRAM high-watermark is reached. Track per‑model VRAM and last-access timestamps.
