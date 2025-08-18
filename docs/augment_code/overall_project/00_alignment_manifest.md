# Alignment Manifest — Authoritative Decisions and Versions

This document declares the docs/augment_code folder as the master plan and lists binding decisions all other docs must align with.

## Canonical Sources
- Master plan: docs/augment_code/*.md (07, 08, 09, 10 are binding)
- Visuals: docs/diagrams/*.mmd
- Architecture overview: docs/02-architecture/architecture-overview.md (must not contradict augment_code)

## Binding Decisions
- Orchestrator-based Four-Brain HRM architecture (H-Module FP16 always resident; L-Module NVFP4 on-demand)
- Triton Inference Server: explicit model-control-mode; dynamic batching per model
- TensorRT engines optimized for Blackwell (SM_120); NVFP4 primary; FP8 (no streaming) allowed for comparison; FP4 gated by accuracy validation
- ResourceManager: ~8 GB dynamic pool on 16 GB GPU; LRU evictions; non-evictable HRM H
- Data: Supabase pgvector (persistent) + Redis (working memory/cache)
- Observability: Prometheus/Grafana metrics, Loki logs, Triton /metrics
- Hygiene: Do not commit *.onnx/*.plan or model_repository contents

## Versions and Environment
- Host: Windows 11 + WSL2 (Ubuntu 24.04), Docker Desktop (WSL2)
- NVIDIA Driver r580+, CUDA 13 host; Triton 25.06 (CUDA 12.9 runtime); TensorRT 10.13.2

## Naming and Contracts
- Keep model/service names generic to enable swaps (examples: qwen3_embedding_trt)
- Dtype/shape discipline: TRT INT32 explicit-batch; ONNX often INT64

## Change Control
- Any doc claiming performance/accuracy must include evidence links (logs, benchmarks, engine paths)
- Deviations from this manifest require explicit approval and associated evidence

