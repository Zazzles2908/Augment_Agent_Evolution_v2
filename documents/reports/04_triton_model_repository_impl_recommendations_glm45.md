# GLM-4.5 Implementation Recommendations – 04_triton_model_repository.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Repository Structure & Conventions
- Root `models/` with subdirs: `glm45_air/`, `qwen3_4b_embedding/`, `qwen3_0_6b_reranking/`, `docling/`
- Use `1/` version dir only; add README.md per model with I/O specs and dtypes

## 2. config.pbtxt Templates (embedding, reranker, generator)
Embedding – qwen3_4b_embedding:
```
name: "qwen3_4b_embedding"
platform: "tensorrt_plan"
max_batch_size: 64
input [{ name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] }]
output [{ name: "embedding" data_type: TYPE_FP32 dims: [ 2000 ] }]
instance_group [{ count: 1 kind: KIND_GPU }]
```
Reranker – qwen3_0_6b_reranking:
```
name: "qwen3_0_6b_reranking"
platform: "tensorrt_plan"
max_batch_size: 32
input [{ name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] }]
output [{ name: "scores" data_type: TYPE_FP32 dims: [ -1 ] }]
instance_group [{ count: 1 kind: KIND_GPU }]
```
Generator – glm45_air:
```
name: "glm45_air"
platform: "tensorrt_plan"
max_batch_size: 8
input [{ name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] }]
output [{ name: "output_ids" data_type: TYPE_INT32 dims: [ -1 ] }]
instance_group [{ count: 1 kind: KIND_GPU }]
```

## 3. Explicit Model Control & Autoscaling
Start Triton:
```
tritonserver --model-repository=/models --model-control-mode=explicit --http-port=8000 --metrics-port=8002
```
- Controller monitors queues + GPU mem; load/unload via `/v2/repository/models/{model}/(load|unload)`
- Apply cooldown between changes to avoid thrashing

## 4. Batching and instance_group Sizing
- Use `preferred_batch_size` ≈ 50% of `max_batch_size`; set `max_queue_delay_ms`
- Single-GPU: `KIND_GPU`, `count: 1` for all models; adjust based on metrics

## 5. Health Checks, Metrics, Alerts
- Health: `/v2/health/ready` every 30s
- Metrics: gpu utilisation, inference success, duration, model load time
- Alerts: GPU > 90%, errors > 1%, load time > 30s

## 6. Safe Rollout & Rollback
- Canary: load new versions in parallel dir; route small % traffic
- Auto-rollback on error rate > 2%
- Keep version history + rollback scripts; document procedures

