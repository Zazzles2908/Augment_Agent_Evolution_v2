# Model Precision and Weights Assignment (Four‑Brain, CUDA 13 / TRT 10.13.x / SM_120)

This document records why each model uses a specific precision and how weights are assigned, with ZEN analysis guiding trade‑offs for 16GB VRAM and two concurrent engines.

## Context
- Hardware: RTX 5070 Ti (Blackwell, SM_120), 16GB VRAM
- Runtime: Triton + TensorRT 10.13.x (CUDA 13), explicit model control, dynamic batching [2,4,8] @ 2000µs
- Goal: Sub‑100ms p50 for interactive batches ≤4; preserve headroom for 2 engines
- Architecture: HRM calls Brain 1 (embedding) via Triton (no duplicate embedding)

## Assignments

### Brain 1 — Qwen3‑8B Embedding (FP8 target, FP16 fallback)
- Rationale: 
  - Embeddings are frequent and amenable to FP8 with minimal accuracy impact
  - FP8 reduces memory bandwidth and activation footprint, enabling larger max shapes
- Weights: 
  - Serve FP8‑quantized plan when available (PTQ/QAT via TensorRT Model Optimizer)
  - Fallback to FP16 engine if FP8 not validated (accuracy regression ≤1%)
- Profiles (ZEN agreed): min 1×128, opt 4×256, max 8×512

### Brain 2 — Qwen3‑8B Reranker (NVFP4 target, FP16 fallback)
- Rationale: 
  - Pairwise scoring tolerates FP4 well; NVFP4 is optimized for Blackwell throughput
  - Keeps headroom for HRM and Docling while supporting 4×512 bursts
- Weights: 
  - Serve NVFP4 plan (TensorRT Model Optimizer) when validated
  - Fallback to FP16 engine if FP4 accuracy drops >1%
- Profiles (ZEN agreed): min 1×128, opt 4×256, max 4×512

### Brain 3 — HRM High Module (FP16)
- Rationale: 
  - High‑fidelity control logic merits FP16 for stability
  - Constrains opt to 2 for latency/VRAM protection
- Weights: FP16 plan
- Profiles (ZEN agreed): min 1×128, opt 2×256, max 4×512

### Brain 3 — HRM Low Module (FP8 target, FP16 fallback)
- Rationale: 
  - Fast execution path benefits from FP8 memory and speed
  - Absorbs bursts up to 8×512 without starving other engines
- Weights: FP8 plan when validated; fallback to FP16 if needed
- Profiles (ZEN agreed): min 1×128, opt 4×256, max 8×512

### Brain 4 — Docling (NVFP4 target, FP16 fallback)
- Rationale: 
  - Vision pipeline gains most from aggressive quantization of activations/weights
  - Caps batch to ≤2 at high resolution to avoid OOM alongside 8B models
- Weights: NVFP4 plan when validated; FP16 fallback
- Profiles (ZEN agreed, C×H×W): min 1×768×768, opt 2×1024×1024, max 2×1280×1280

## Why these weights and profiles
- Two‑engine headroom: Profiles cap opt/max to fit two concurrent engines on 16GB
- Precision by role: FP8/NVFP4 where accuracy is resilient (embedding, rerank, vision); FP16 where stability matters (HRM High)
- SDPA compliance: Ensures TRT attention fusions; avoids custom FlashAttention ops in ONNX
- Upgrade path: Start FP16 baseline, then enable quantized plans per model with accuracy gates (≤1% regression)

## ZEN notes
- Trade‑offs: Increasing max shapes boosts throughput but raises VRAM and may evict engines; if pressure observed, reduce to 4×384 (text) and 1×1024×1024 (Docling)
- Latency: 2000µs queue delay fits interactive p50; consider adaptive delays for batch workloads
- Monitoring: Set alerts at 75%/85% VRAM; persist TRT tactic cache

## Next
- Build/validate quantized plans (NVFP4/FP8) with TensorRT Model Optimizer or PTQ/QAT
- Keep outputs FP16 for downstream stability even for quantized engines
- Implement HRM→Brain1 Triton client call (no internal embedding); integrate ResourceManager for LRU unload

