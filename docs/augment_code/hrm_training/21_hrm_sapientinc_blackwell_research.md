# SapientInc HRM (Hierarchical Reasoning Model) — Deep Research and Blackwell Integration Plan

Purpose
- Collect authoritative information on Sapient Intelligence's HRM, how to obtain raw model/code/checkpoints, and how to best build and deploy it on NVIDIA Blackwell (SM_120) within our Four‑Brain HRM architecture on Ubuntu 24.04.
- Align findings with our precision/weights strategy and Triton/TensorRT plan; document contingencies where TensorRT is not yet practical.

## TL;DR
- HRM is an open-source hierarchical recurrent reasoning architecture (Apache‑2.0) with two interdependent modules (High/Low), ~27M params, SOTA results on ARC/Sudoku/Maze with very small data.
- Primary sources confirm: official GitHub repo (sapientinc/HRM), arXiv paper (2506.21734), and Hugging Face checkpoints.
- Best Blackwell path now: FP16 baseline in PyTorch with TorchInductor/CUDA Graphs and FlashAttention (where applicable), while exploring FP8/NVFP4 via TensorRT only if we can export or re-implement ops. Keep HRM High in FP16 for stability; HRM Low targets FP8 when feasible.

## Primary Sources (Authoritative)
- GitHub (official): https://github.com/sapientinc/HRM  (Apache‑2.0)
- arXiv: https://arxiv.org/abs/2506.21734
- Hugging Face checkpoints (from README):
  - ARC‑AGI‑2: https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2
  - Sudoku 9x9 Extreme: https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme
  - Maze 30x30 Hard: https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard
- Press/secondary: community posts and articles exist but are not authoritative. We rely on the three above for implementation.

## What HRM Is (from paper/repo)
- Architecture: hierarchical, recurrent; two modules:
  - High-level: slow, abstract planning; handles macro steps
  - Low-level: fast, detailed computation; executes micro steps
- Execution: single forward pass performs multi-step reasoning without explicit supervision of intermediate steps
- Scale: ~27M parameters (paper)
- Data efficiency: strong performance with ~1k training samples, no CoT pretraining required
- Tasks: ARC‑AGI, Sudoku, Maze; repo provides data builders, training, evaluation, W&B integration

## How to Get the Raw Model/Code/Checkpoints

## Training From Scratch — Zen MCP Aligned Plan

Principle
- Start simple and controlled, then scale. Use HRM’s official small-sample recipes to earn correctness and reproducibility before integrating full Four‑Brain infrastructure.

Phase 0 — Reproduce Official Baselines (single GPU)
- Goal: Verify our environment and know-good behavior before adding complexity.
- Steps:
  - Build one dataset (e.g., Sudoku-Extreme 1k) with the repo script
  - Train with pretrain.py using the README hyperparameters and W&B tracking
  - Evaluate with evaluate.py and confirm metrics close to reported ranges
- Why: This validates CUDA extensions, PyTorch/FlashAttention/SDPA, and HRM code on our hardware.

Phase 1 — Project-Specific Micro-Tasks (single GPU)
- Goal: Align HRM with our domain via small, high-signal datasets (1k–10k).
- Steps:
  - Define 2–4 micro-task datasets mirroring our HRM H/L roles (e.g., tool routing, short DAG planning, rerank justification, memory retrieval critique)
  - Use the repo’s training loop; tune only lr and batch size initially
  - Early stopping once training accuracy approaches 100% to avoid instability (per README notes)
- Outputs: Checkpoints + eval scripts; accuracy/variance tracked in W&B.

Phase 2 — Multi-GPU Scaling (optional)
- Goal: Run the provided torchrun multi-GPU configs for larger data/longer schedules.
- Steps: Follow README distributed examples (nproc-per-node 8). Keep exact seeds for reproducibility.

Phase 3 — Integrate with Four‑Brain (carefully, not all at once)
- Goal: Keep training loops decoupled from live infra until models are stable.
- Steps:
  - Serve stable HRM checkpoints via a minimal PyTorch microservice (FastAPI)
  - Orchestrator interacts with HRM over HTTP/gRPC on curated eval tasks
  - Log results to Redis for working memory; store metrics/artifacts in Supabase for persistence
  - Only after stability: consider online data generation or active learning loops
- Why not end-to-end from day 1: Coupled pipelines hide where problems originate. We get faster signal by isolating training first.

Phase 4 — Blackwell Optimizations and Precision
- Start PyTorch FP16 with torch.compile (Inductor) + CUDA Graphs
- Profile FA3 on SM_120; if not mature, use SDPA/TE kernels
- Explore Torch-TensorRT subgraph compile for HRM Low; keep FP16 outputs; add accuracy gates (≤1% degradation)

Data Strategy
- Begin with small clean datasets (1k–10k) with high label fidelity; augment as per repo examples
- Add project-specific micro-tasks that directly exercise HRM’s hierarchical reasoning
- Maintain a data registry (source, version, augmentations, license) and store in object storage; index metadata in Supabase

Hyperparameter Defaults (from README, adapted)
- Small-sample: global_batch_size 384, lr ~7e-5 to 1e-4; long epochs with eval_interval 2k; early stop near 100% training acc
- Full Sudoku-Hard: example uses global_batch_size 2304, lr 3e-4, weight_decay 0.1, L_cycles=8, halt_max_steps=8
- Keep H-module stability priority: modest LR, deterministic flags on, gradient clipping if needed

Infra Guidance
- Training: containers separate from Triton; Ubuntu 24.04, PyTorch CUDA 12.6 or 13 when stable
- Serving: HRM as PyTorch service; other brains via Triton (explicit mode) as per our plan
- Observability: W&B for training, Prometheus/Grafana for serving; persist eval sets in Supabase; short-term working memory in Redis

Decision: Do not build the entire end-to-end system and “throw data” at it on day 1
- Instead, iterate through Phases 0→1→3 with tight loops
- This maximizes learning speed, isolates issues, and prevents infra churn from blocking model progress

1) Clone official repo
```
git clone https://github.com/sapientinc/HRM.git
cd HRM
```
2) Environment (Ubuntu 24.04)
- CUDA: repo targets CUDA 12.6 (per README); ensure driver supports Blackwell (CUDA 13 capable). For dev, we can keep CUDA 12.6 userland via PyTorch wheel if kernels build; else migrate to CUDA 13 when upstream libs support it.
- PyTorch: install CUDA‑enabled PyTorch (cu126 wheel per repo, or cu13 once available)
- FlashAttention:
  - Hopper path uses FA3 (flash-attention/hopper); for Blackwell, FA3 support is emerging. If FA3 is not yet SM_120‑ready, use FA2 or disable FA and rely on PyTorch SDPA.
3) Checkpoints

## Phase 0 (Proposed): HRM Validation with Sudoku on CUDA 13 + TensorRT 10.13.x (Blackwell)

Goal
- Verify HRM implementation and a feasible TensorRT conversion path in a controlled task (Sudoku‑Extreme), with FP16 baseline and exploratory FP8 where safe.

High‑level concerns
- HRM is a hierarchical recurrent model; not all ops may have direct ONNX/TRT coverage.
- FP8 generally requires PTQ/QAT workflows (TensorRT Model Optimizer) to meet accuracy gates.
- Best path: start with PyTorch FP16 baseline; attempt Torch‑TensorRT subgraph compile; in parallel, try ONNX→TRT for the largest supported subgraph. Preserve FP16 outputs.

Steps
1) Environment (Ubuntu 24.04; CUDA 13; TRT 10.13.x)
   - Prefer building/running inside containers that match TRT minor versions to avoid ABI issues.
   - Example base images:
     - Training: nvcr.io/nvidia/pytorch:25.06-py3 (CUDA 13)
     - Conversion/serving: nvcr.io/nvidia/tritonserver:25.06-py3 (TensorRT 10.13.x)
   - Ensure PyTorch SDPA is enabled; FA3 on SM_120 only if confirmed stable; else SDPA fallback.

2) Train HRM on Sudoku (single GPU; per README)
   - Build dataset and run pretrain.py as documented; log to W&B; retain final checkpoint.

3) Convert to TensorRT (two tracks)
   A) Torch‑TensorRT (incremental, pragmatic)
   - Compile supported subgraphs to TRT; allow fallback to PyTorch for unsupported nodes.
   - Precision: start with FP16; try enabling FP8 only after accuracy checks.
   - Example (conceptual):
     - import torch_tensorrt as trt
     - trt.compile(model, inputs=[trt.Input((B,S,...), dtype=torch.half)], enabled_precisions={torch.half})
   B) ONNX → TensorRT (ambitious)
   - Export with opset ≥19; ensure model uses SDPA paths (avoid custom FA kernels during export).
   - Use trtexec with min/opt/max shapes appropriate for Sudoku input tensors.
   - Start FP16 ("--fp16"); explore FP8 ("--fp8") only with validation/calo/quantization support and accuracy gates.

4) Benchmark
   - Throughput/latency: trtexec (for engine) and Python microbench (PyTorch vs TRT subgraph/full engine).
   - VRAM/engine size: compare PyTorch FP16 vs TRT FP16/FP8.

5) Validate Reasoning Quality
   - Run evaluate.py on a held‑out Sudoku set for both PyTorch baseline and TRT path; compare exact_accuracy.
   - Gate FP8 enablement at ≤1% accuracy regression.

Success Criteria
- Accuracy: Match or exceed PyTorch FP16 on Sudoku; FP8 only if ≤1% regression.
- Performance: Demonstrated latency/throughput gain with TRT vs PyTorch FP16.
- Stability: FP16 outputs with consistent dtypes; deterministic runs under fixed seeds.

Fallbacks / Mitigations
- If full‑graph ONNX export fails, keep mixed Torch‑TRT (partial graph) for wins without blocking integration.
- If FP8 is unstable numerically, remain on FP16 for HRM and focus FP8 on embedding/reranker/docling where support is mature.

- Download from Hugging Face links above.
- Evaluate with evaluate.py (see repo README) to reproduce paper metrics.

## Build/Runtime Dependencies (from README)
- CUDA 12.6 toolkits and environment variables (if building CUDA extensions)
- PyTorch + torchvision + torchaudio (matching CUDA variant)
- Packaging toolchain: packaging, ninja, wheel, setuptools, setuptools‑scm
- FlashAttention 2/3 depending on GPU generation; for SM_120, confirm FA3 support. If unavailable, use PyTorch SDPA kernels on CUDA 13 + TE where possible.
- W&B optional for experiment tracking

## Training/Evaluation Highlights
- Small‑sample regimes: 1k examples with augmentation, long schedules (e.g., 20k epochs) but small model size – feasible on a single modern GPU
- Distributed examples in README (torchrun with 8 GPUs) for full‑scale runs
- Tasks include ARC‑AGI‑2, Sudoku‑Extreme, Maze; datasets constructed via included scripts

## Export and Blackwell Inference Strategies
Key question: can HRM be exported to TensorRT easily?
- Risks:
  - Custom recurrent structure; potential custom ops; FlashAttention kernels may not export cleanly to ONNX
  - TensorRT plugins may be needed; ONNX graph may require simplification or re‑implementation of attention/positional encodings
- Strategies:
  1) PyTorch‑native serving (recommended baseline)
     - Precision: FP16 (bf16 optional if supported), stable outputs
     - Optimizations: torch.compile (Inductor), CUDA Graphs, SDPA fused attention, enable cudnn heuristics
     - Blackwell features: test FP8 scaled quantization via TE for attention blocks if re‑implemented; otherwise keep FP16
     - Deployment: containerized service behind Triton or separate microservice (HTTP/gRPC), with orchestrator coordination
  2) Torch‑TensorRT (experimental)
     - Try scripting/FX tracing to compile supported subgraphs to TRT, fall back to PyTorch for unsupported nodes
     - Gains: partial TRT acceleration on Blackwell (NVFP4/FP8 where possible)
     - Caveat: complexity and maintenance; good for incremental speedups
  3) ONNX → TensorRT (advanced)
     - Attempt ONNX export with opset ≥17, SDPA paths; replace FA ops with SDPA before export
     - Implement/enable plugins if required; set dynamic shapes and optimization profiles
     - Precision: FP16 baseline; explore FP8 once numerics validated; ensure FP16 outputs
     - Caveat: more engineering; only pursue if measurable benefits over PyTorch baseline

## Precision/Weights Mapping to Our Four‑Brain Plan
- HRM High (H‑Module)
  - Role: control logic and planning – prioritize stability
  - Precision: FP16 baseline (our docs already set this)
  - Accelerator: PyTorch first; evaluate TRT later if exportable without regressions
- HRM Low (L‑Module)
  - Role: fast execution path – prioritize throughput
  - Precision: FP8 target on Blackwell; FP16 fallback
  - Path: begin PyTorch FP16 + Inductor; explore TE/FA optimizations; attempt Torch‑TRT/ONNX‑TRT when stable
- Embedding, reranker, docling
  - As already planned: FP8/NVFP4 targets under TensorRT with FP16 outputs, explicit Triton control

## Ubuntu 24.04 Integration Path (Concrete)
1) Reproduce HRM locally (dev)
```
# Ubuntu 24.04 with NVIDIA driver supporting Blackwell
conda create -n hrm python=3.11 -y && conda activate hrm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # or cu13 when available
pip install -r requirements.txt
# Optional: FlashAttention (if SM_120 supported), else skip and rely on SDPA
```
2) Evaluate a checkpoint
```
OMP_NUM_THREADS=8 torchrun --nproc-per-node 1 evaluate.py checkpoint=/path/to/HRM-checkpoint-sudoku-extreme
```
3) Package as a microservice (baseline)
- Build a small FastAPI/uvicorn wrapper exposing /infer for HRM‑High and HRM‑Low
- Precision: autocast to fp16; ensure deterministic modes for stability
- Containerize on Ubuntu 24.04 base with CUDA userland matching PyTorch wheel
4) Optional hybrid acceleration
- Try torch_tensorrt.compile on segments; A/B test latency on SM_120
- Attempt ONNX export with SDPA; use TensorRT where graphs are supported

## What We Can Do With HRM in This Project
- As our H/L modules:
  - H‑Module: embed HRM‑High recurrent planner for long‑horizon task decomposition; keep FP16 for stability; integrate with ResourceManager
  - L‑Module: HRM‑Low executor for rapid micro‑reasoning; aim FP8 on Blackwell as maturity allows
- Compose with Four‑Brain services:
  - No internal embedding duplication; call Brain‑1 via Triton for shared embeddings
  - Reranker and Docling continue as NVFP4 targets for throughput; HRM focuses on reasoning core
- Training/data strategy:
  - Fine‑tune HRM on project‑specific micro‑tasks (program synthesis fragments, tool routing mini‑tasks, small DAG planning) with small curated datasets (1k‑10k)
  - Use W&B for metric tracking; enforce early stopping to avoid Q‑learning instability noted in README
- Reliability/scaling:
  - Start PyTorch FP16; measure; add TorchInductor/CUDA Graphs
  - Gradually pilot Torch‑TRT or ONNX‑TRT on L‑Module; retain FP16 outputs and tighten optimization profiles to preserve two‑engine concurrency on 16GB

## Gaps / Open Questions
- FlashAttention 3 support for Blackwell SM_120: confirm upstream readiness; otherwise rely on SDPA/TE
- ONNX export fidelity: verify graph export without custom kernels; identify missing ops early
- TensorRT plugin needs: assess whether HRM blocks require custom plugins for parity
- Precision gates: define ≤1% accuracy regression threshold for enabling FP8 on L‑Module

## References
- GitHub: https://github.com/sapientinc/HRM
- arXiv: https://arxiv.org/abs/2506.21734
- HF checkpoints: see links above
- Our integration docs: 16_triton_config_templates_fp16.md, 17_model_precision_and_weights.md, 20_ubuntu24_trt_build_alignment.md, 18_smoke_validation_explicit_mode.md

