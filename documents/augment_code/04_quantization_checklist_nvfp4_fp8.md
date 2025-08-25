Purpose
- Provide a sanity checklist for NVFP4 / FP8 quantization aligned with Triton/TensorRT and Blackwell GPUs

Quick guidance
- Policy: inference runs on Triton; clients are thin. Quantize upstream and export ONNX/TRT.

Checklist
1) Choose quantization target
- NVFP4 (Blackwell): nvidia-modelopt (torch) configs; best throughput on Blackwell; primary target
- FP8: modelopt FP8_DEFAULT_CFG; good speed/accuracy tradeoff
- INT4 AWQ: weight-only; smaller memory but may require custom kernels

Target Models (HRM + Qwen3 Architecture)
- qwen3_embedding_8b_trt (NVFP4) — Qwen/Qwen3-Embedding-8B
- qwen3_reranker_8b_trt (NVFP4) — Qwen/Qwen3-Reranker-8B
- hrm_high_level_trt (FP16) — HRM high-level module for critical reasoning
- hrm_low_level_trt (FP8) — HRM low-level module for fast execution
- docling_gpu_trt (NVFP4) — Document processing with GPU acceleration

2) Apply quantization and validate
- Use modelopt.torch.quantization configs (see docs/06-reports/technical/tensorrt-model-optimizer-integration-analysis-20250808.md)
- Calibrate/run forward loop with representative data
- Validate accuracy vs FP16 baseline

3) Export path
- If serving with ONNXRuntime on Triton:
  - Export ONNX of quantized model and ensure runtime kernels exist
  - Keep input dtypes as in tokenizer (ids/mask int64)
- If serving with TensorRT:
  - export_tensorrt_llm_checkpoint or build via trtexec with correct shapes
  - Ensure config.pbtxt matches dtypes for the TRT engine (often INT32 inputs)

4) Repository hygiene
- Store large artifacts in containers/.../model_repository/<model>/1 locally only
- Do not commit *.onnx, *.plan (already ignored)

5) Performance tips
- Use timing cache in trtexec for repeatable builds
- Utilize dynamic batching in Triton for ONNX; for explicit-batch TRT, set max_batch_size: 0
