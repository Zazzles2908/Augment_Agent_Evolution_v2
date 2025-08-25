# Phase 1 — Zen‑integrated Validation Plan

Goal: Build a real main system (Docling → Triton → Redis → Monitoring) with P95 < 1.5s and 2000‑dim embeddings, while Supabase and Zen MCP runtime integrations remain paused. Zen MCP is used as a false‑win detector across all steps.

## 1) Model conversion & validation

Commands (run locally on a GPU host):
- Analyze conversion scripts before each build
  - analyze_zen --target scripts/tensorrt/
- Build TensorRT engines (examples)
  - python scripts/tensorrt/convert_model.py --repo containers/four-brain/triton/model_repository --model qwen3_4b_embedding --config containers/four-brain/triton/config/embed.yaml --precision nvfp4
  - python scripts/tensorrt/convert_model.py --repo containers/four-brain/triton/model_repository --model qwen3_0_6b_reranking --config containers/four-brain/triton/config/rerank.yaml --precision nvfp4
  - python scripts/tensorrt/convert_model.py --repo containers/four-brain/triton/model_repository --model glm45_air --config containers/four-brain/triton/config/generate.yaml --precision nvfp4
- Validate model loading
  - tracer_zen --command "tritonserver --model-repository=/models --model-control-mode=explicit" --output traces/model_loading

Zen checks:
- Precision flags: NVFP4 prefers FP8; INT8 only when builder.int8=true
- Inputs/outputs match docs/configs: embedding(2000), score(1), logits(b,s,v)
- No hard‑coded success flags; proper error handling

## 2) Smoke testing with Zen guardrails

Adversarial cases:
- testgen_zen --target scripts/smoke/ --mode adversarial --coverage edge_cases

Run under tracing:
- tracer_zen --command "bash scripts/smoke/triton_repository.sh" --output traces/model_ops
- tracer_zen --command "python scripts/smoke/embed_infer.py" --output traces/embed
- tracer_zen --command "python scripts/smoke/rerank_infer.py" --output traces/rerank
- tracer_zen --command "python scripts/smoke/generate_infer.py" --output traces/generate

Zen flags on:
- <50ms test runtime (stub suspicion)
- Identical outputs across runs (canned responses)
- Missing GPU memory utilization
- Absent tensor computations in call flows

## 3) Redis cache validation

- analyze_zen --function cache_embedding (ingestion)
  - SHA256 keys are computed; no short‑circuit
  - setex TTL is set (e.g., 7d)
  - Key pattern emb:v1:{sha256}
- tracer_zen (query service)
  - Lookup before inference; write after inference
  - No cache bypasses

## 4) Anti‑falsification measures

- Reject mock responses, hard‑coded OKs, skipped validations
- Correlate perf metrics with GPU utilization
- docgen_zen --source traces/ --output documents/reports/phase1_validation_report.md
  - Expected vs actual tensor ops
  - End‑to‑end call‑flow verification
  - Redis cache hit/miss with logs
  - GPU utilization during inference

## 5) Exit criteria

- Conversions show real tensor ops
- Smoke tests demonstrate real inference
- Redis operations verifiable
- No fabricated success indicators
- Metrics correlate with resource usage

Notes:
- All commands above are executed locally (not in CI) to avoid GPU/driver constraints.
- Keep Supabase paused; use local placeholder retrieval during Phase 1.

