# Triton config.pbtxt Templates — Canonical Models (CUDA 13 / Blackwell SM_120)

Target stack: Qwen3‑4B Embedding (2000‑dim), Qwen3‑0.6B Reranking, GLM‑4.5 Air Generation, optional Docling GPU.
Precision flags in Triton express intent; FP8/NVFP4 require TensorRT‑built engines (TRT 10.13.x). Start with FP16 while calibrating.

## Model repository layout (canonical)
- containers/four-brain/triton/model_repository/qwen3_4b_embedding/1/model.plan
- containers/four-brain/triton/model_repository/qwen3_0_6b_reranking/1/model.plan
- containers/four-brain/triton/model_repository/glm45_air/1/model.plan
- containers/four-brain/triton/model_repository/docling/1/model.onnx (optional)

Note: Build .plan inside Linux (Ubuntu 24.04) with TRT 10.13.x matching Triton 25.07+.

## Qwen3‑4B Embedding (NVFP4 or FP16)
```
name: "qwen3_4b_embedding"
platform: "tensorrt_plan"
max_batch_size: 0
input: [
  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1, -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1, -1 ] }
]
output: [ { name: "embedding", data_type: TYPE_FP32, dims: [ -1, 2000 ] } ]
instance_group [ { kind: KIND_GPU, count: 1 } ]
dynamic_batching { preferred_batch_size: [4,8] max_queue_delay_microseconds: 100 }
```

## Qwen3‑0.6B Reranking (NVFP4 or FP16)
```
name: "qwen3_0_6b_reranking"
platform: "tensorrt_plan"
max_batch_size: 0
input: [
  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1, -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1, -1 ] }
]
output: [ { name: "scores", data_type: TYPE_FP32, dims: [ -1 ] } ]
instance_group [ { kind: KIND_GPU, count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 200 }
```

## GLM‑4.5 Air Generation (FP16)
```
name: "glm45_air"
platform: "tensorrt_plan"
max_batch_size: 0
input: [
  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1, -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1, -1 ] }
]
output: [ { name: "logits", data_type: TYPE_FP32, dims: [ -1, -1, -1 ] } ]
instance_group [ { kind: KIND_GPU, count: 1 } ]
dynamic_batching { preferred_batch_size: [1] max_queue_delay_microseconds: 100 }
```

## Docling (optional; FP16 or NVFP4 plan)
```
name: "docling"
backend: "tensorrt"  # or platform: "onnxruntime_onnx" if serving ONNX
max_batch_size: 4
input: [ { name: "page_tensor", data_type: TYPE_FP16, dims: [ -1, -1, -1 ] } ]
output: [ { name: "layout", data_type: TYPE_FP16, dims: [ -1 ] } ]
instance_group [ { kind: KIND_GPU, count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
```

### Orchestration
- Application orchestrates: Docling → qwen3_4b_embedding → Supabase/Redis → qwen3_0_6b_reranking → glm45_air.
- Prefer explicit model control; no models preloaded by default.

### Profiles and VRAM discipline (suggested)
- Embedding/Reranker: min 1×128, opt 4×256, max 4×512
- GLM‑4.5 Air: min 1×128, opt 1×512, max 1×1024
- Tune preferred_batch_size and queue delay to maintain headroom on 16GB VRAM.
