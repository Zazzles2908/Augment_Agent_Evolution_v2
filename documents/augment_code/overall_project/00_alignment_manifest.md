# Alignment Manifest — Authoritative Decisions and Versions

This document declares the docs/augment_code folder as the master plan and lists binding decisions all other docs must align with.

## Canonical Sources
- Master plan: docs/augment_code/*.md (07, 08, 09, 10 are binding)
- Visuals: docs/diagrams/*.mmd
- Architecture overview: docs/02-architecture/architecture-overview.md (must not contradict augment_code)

## Binding Decisions
- Target stack: Docling → Qwen3‑4B Embedding (2000‑dim) → Supabase/pgvector + Redis → Qwen3‑0.6B Reranking → GLM‑4.5 Air Generation
- Triton Inference Server: explicit model-control-mode; dynamic batching per model
- TensorRT engines optimized for Blackwell (SM_120); NVFP4 or FP16 acceptable; calibrate FP4/FP8 only with evidence
- ResourceManager: ~8 GB dynamic pool on 16 GB GPU; LRU evictions; no always-on legacy HRM modules
- Data: Supabase pgvector (persistent) + Redis (working memory/cache)
- Observability: Prometheus/Grafana metrics, Loki logs, Triton /metrics
- Hygiene: Do not commit *.onnx/*.plan or model_repository contents

## Versions and Environment
- Host: Windows 11 + WSL2 (Ubuntu 24.04), Docker Desktop (WSL2)
- NVIDIA Driver r580+; CUDA 13.x; Triton 25.07+ (CUDA 13 runtime); TensorRT 10.13.x

## Naming and Contracts
- Canonical model names: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air, docling
- Dtype/shape discipline: use INT64 for text inputs across TRT engines in this repo

## Change Control
- Any doc claiming performance/accuracy must include evidence links (logs, benchmarks, engine paths)
- Deviations from this manifest require explicit approval and associated evidence

