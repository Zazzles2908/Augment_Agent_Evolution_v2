Title: Triton Configuration — Explicit Control, Dynamic Batching, On-Demand Models

Goals
- Run Triton in explicit model-control-mode
- Preload HRM H-Module; load/unload other models on demand
- Enable dynamic batching tailored per model

1) Server Flags
- --model-control-mode=explicit
- --repository=/models (mounted to containers/four-brain/triton/model_repository)
- --http-port=8000 --grpc-port=8001 --metrics-port=8002

2) Model Repository Layout (examples)
- hrm_h_trt/
  - 1/model.plan  (FP16)
  - config.pbtxt
- hrm_l_trt/
  - 1/model.plan  (NVFP4)
  - config.pbtxt
- qwen3_embedding_trt/
  - 1/model.plan  (NVFP4)
  - config.pbtxt
- qwen3_reranker_trt/
  - 1/model.plan  (NVFP4)
  - config.pbtxt
- docling_gpu/
  - 1/model.onnx or model.plan
  - config.pbtxt

3) Config Highlights per Model
- Set max_batch_size: 0 for explicit-batch TensorRT; use dynamic_batching section for ONNXRuntime
- Inputs/outputs must match engine dtypes/shapes (INT32 for TRT embeddings; INT64 typical for ONNX)
- Instance groups: prefer kind: KIND_GPU, count: 1 unless testing concurrency
- Example dynamic batching:
  dynamic_batching {
    preferred_batch_size: [2, 4, 8]
    max_queue_delay_microseconds: 2000
  }

4) Explicit Load/Unload API
- List: GET  /v2/repository/index
- Load:  POST /v2/repository/models/{model}/load
- Unload: POST /v2/repository/models/{model}/unload
- Health: GET  /v2/health/ready, GET /v2/models/{model}

5) Startup Behavior
- POST /v2/repository/models/hrm_h_trt/load during bootstrapping
- Keep hrm_h_trt resident; track it as non-evictable in ResourceManager

6) Client Dtype/Shape Discipline
- Match client payloads to model config; promote INT32→INT64 or cast as required
- Maintain per-model adapters in client code

7) Monitoring
- Scrape /metrics; alert on model load failures and queue time spikes
- Log repository actions and keep timing cache for TRT rebuilds

See also: 09_resource_manager_design.md for how load/unload is decided at runtime.

