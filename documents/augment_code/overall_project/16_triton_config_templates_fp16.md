# Triton config.pbtxt Templates — Four‑Brain (CUDA 13 / Blackwell SM_120)

This reflects your target precisions and single shared embedding service. Precision flags in Triton config express intent; FP8/NVFP4 require engines built or calibrated via TensorRT (TRT 10.13.x). Use FP16 baseline while calibrating.

## Model repository layout (Windows path → Triton mount)
- containers/four-brain/triton/model_repository/brain1_embedding_fp8/1/model.onnx
- containers/four-brain/triton/model_repository/brain2_reranker_nvfp4/1/model.onnx
- containers/four-brain/triton/model_repository/hrm_high_fp16/1/model.onnx
- containers/four-brain/triton/model_repository/hrm_low_fp8/1/model.onnx
- containers/four-brain/triton/model_repository/docling_nvfp4/1/model.onnx

Note: Build .plan inside Linux Triton on first load. For FP8/NVFP4, either prebuild .plan with TRT (PTQ/QAT) or provide calibrations; Triton config alone won’t quantize to FP8/FP4.

## Brain 1 — Qwen3‑8B Embedding (FP8 target, FP16 fallback)
```
name: "brain1_embedding_fp8"
backend: "tensorrt"
max_batch_size: 16
input [
  { name: "input_ids"      data_type: TYPE_INT64 dims: [-1] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }
]
output [ { name: "embedding" data_type: TYPE_FP16 dims: [-1] } ]
instance_group [ { kind: KIND_GPU count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
parameters: { key: "precision_mode" value: { string_value: "FP16" } }
# FP8: supply a prebuilt FP8 plan or TRT calibration; keep output FP16 for downstream stability.
```

## Brain 2 — Qwen3‑8B Reranker (NVFP4 target, FP16 fallback)
```
name: "brain2_reranker_nvfp4"
backend: "tensorrt"
max_batch_size: 8
input [
  { name: "input_ids"      data_type: TYPE_INT64 dims: [-1] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }
]
output [ { name: "scores" data_type: TYPE_FP16 dims: [-1] } ]
instance_group [ { kind: KIND_GPU count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
parameters: { key: "precision_mode" value: { string_value: "FP16" } }
# NVFP4: requires TRT quantization/prebuilt plan; Triton will serve the FP4 plan, outputs surfaced as FP16.
```

## Brain 3 — HRM High Module (FP16)
```
name: "hrm_high_fp16"
backend: "tensorrt"
max_batch_size: 8
input [
  { name: "input_ids"      data_type: TYPE_INT64 dims: [-1] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }
]
output [ { name: "logits" data_type: TYPE_FP16 dims: [-1] } ]
instance_group [ { kind: KIND_GPU count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
parameters: { key: "precision_mode" value: { string_value: "FP16" } }
```

## Brain 3 — HRM Low Module (FP8 target, FP16 fallback)
```
name: "hrm_low_fp8"
backend: "tensorrt"
max_batch_size: 8
input [
  { name: "input_ids"      data_type: TYPE_INT64 dims: [-1] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [-1] }
]
output [ { name: "logits" data_type: TYPE_FP16 dims: [-1] } ]
instance_group [ { kind: KIND_GPU count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
parameters: { key: "precision_mode" value: { string_value: "FP16" } }
# FP8: prebuilt/calibrated engine recommended; keep outputs FP16.
```

## Brain 4 — Docling (NVFP4 target, FP16 fallback)
```
name: "docling_nvfp4"
backend: "tensorrt"
max_batch_size: 4
input [ { name: "page_tensor" data_type: TYPE_FP16 dims: [-1,-1,-1] } ]
output [ { name: "layout" data_type: TYPE_FP16 dims: [-1] } ]
instance_group [ { kind: KIND_GPU count: 1 } ]
dynamic_batching { preferred_batch_size: [2,4,8] max_queue_delay_microseconds: 2000 }
parameters: { key: "precision_mode" value: { string_value: "FP16" } }
# NVFP4: serve a prebuilt FP4 plan for peak efficiency; FP16 fallback acceptable initially.
```

### Orchestration / Avoiding duplication
- HRM does NOT embed internally. HRM calls Brain 1 via Triton (HTTP/gRPC) for embeddings, then proceeds with HRM High/Low.
- Optionally migrate to Triton ensembles later; start with application‑level orchestration for clarity and control.

### Profiles and VRAM discipline (suggested starting shapes)
- Embedding/Reranker:
  - min: 1×128, opt: 4×256, max: 4×512
- HRM High/Low:
  - min: 1×128, opt: 4×256, max: 8×512 (tighten if VRAM pressure)
- Tune preferred_batch_size and queue delay to maintain headroom for 2 concurrent engines on 16GB.

