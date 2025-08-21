# Validation Playbook — Health, Load/Unload, 1‑Sample Inference

Purpose: Provide repeatable smoke tests and evidence capture to enforce zero‑fabrication across Four‑Brain services.

## Triton Health & Repository
- GET http://localhost:8000/v2/health/ready ⇒ 200
- GET http://localhost:8000/v2/repository/index ⇒ models listed
- POST /v2/repository/models/hrm_h_trt/load on startup; others on demand

## Load/Unload Tests (Explicit Mode)
- For each on-demand model (hrm_l_trt, qwen3_embedding_trt, qwen3_reranker_trt, docling_gpu):
  - POST /v2/repository/models/{name}/load ⇒ READY
  - POST /v2/models/{name}/infer ⇒ minimal request by dtype (INT32 for TRT, INT64 for ONNX)
  - POST /v2/repository/models/{name}/unload ⇒ success

## Sample Inference Payloads
- Embedding TRT (INT32 explicit-batch): see docs/augment_code/02_qwen3_embedding_request_examples.md
- ONNX variant (INT64): see docs/augment_code/01_triton_explicit_mode_load_infer.md

## ResourceManager Checks
- Verify HRM H resident; ensure_loaded() loads needed models; observe LRU evictions under pressure
- Soft threshold 75%; hard 85%; log decisions with reasons

## Evidence Capture
- Save trtexec logs, Triton metrics snapshots, and response JSONs under docs/06-reports/technical/ (date‑stamped)
- Link evidence from any doc claiming performance or accuracy

## Acceptance Criteria
- ALL models load/unload/infer with minimal requests
- Residency respects budgets; no OOM at test shapes
- Metrics show healthy queue times and no error spikes


