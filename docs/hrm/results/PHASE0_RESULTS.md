# Phase 0 — Warm-up Probe Report (RTX 5070 Ti, CUDA 13)

Run links:
- W&B warm-up (latest): https://wandb.ai/jajireen1-hrm_project/Sudoku-extreme-1k-aug-1000%20ACT-torch/runs/n3zevn3d
- W&B earlier warm-up: https://wandb.ai/jajireen1-hrm_project/Sudoku-extreme-1k-aug-1000%20ACT-torch/runs/rr93nie8

## 1) Setup summary
- Hardware
  - GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell, SM 12.0), 16 GB
- Software
  - Base image: nvcr.io/nvidia/pytorch:25.06-py3 (CUDA 13)
  - PyTorch: 2.8.0 (NV build), SDPA attention path
  - Optimizer: AdamW (adam-atan2 not yet SM_120 compatible)
  - WANDB: enabled with step/system metrics
- Data/Model
  - Model: HRM Phase 0 (HierarchicalReasoningModel_ACTV1)
  - Dataset: Sudoku-extreme-1k-aug-1000 (generated in-container)
- Command
  - `epochs=200 eval_interval=20 global_batch_size=256 lr=7e-5`

## 2) Methodology
- Live cockpit: W&B per-step metrics (train/step_time_s, train/steps_per_second, system/gpu_util_percent, system/memory_used_gb)
- Warm-up probe: let training run to steady state, then sample ~10–15 minutes
- We kept only one training process to avoid contention
- Analysis script (optional): see `perf/WANDB_ANALYSIS.md` for extracting medians

## 3) Results (steady-state snapshot)
- GPU
  - Utilization: ~98% median during steady state
  - Memory used: ~5.9 GB / 16.3 GB at batch 256 (ample headroom)
- Throughput (from W&B charts; see run)
  - train/steps_per_second: ~4.8–5.2 steps/sec (visual estimate)
  - train/step_time_s: ~0.19–0.21 s/step
  - Implied epoch time (781 steps): ~2.6 minutes/epoch
- Stability
  - No OOMs or stability issues observed
  - Occasional minor variance due to data and kernel scheduling; small

## 4) Comparison context (Blackwell vs Ada/Lovelace)
- Expectation
  - FP16 SDPA training uplift vs Ada/Lovelace: ~15–25% conservatively; up to ~30–40% with fully tuned kernels
- Caveats
  - Early-arch kernel maturity (SM_120) can skew results
  - HRM is not a standard transformer; proxy benchmarks (BERT/GPT) provide directional guidance
  - Ensure apples-to-apples: same precision, batch, sequence/data shape, and dataloader parameters
- Action
  - We will extract median steps/sec directly via W&B API for this run and any Ada/Lovelace baselines you have (or from published references if methodology matches)

## 5) Improvement opportunities
- Batch size scaling
  - Try 320 → 384 (monitor memory headroom and steps/sec)
- Data loader tuning
  - Use persistent_workers=True, pin_memory=True, increase num_workers
  - Consider prefetch_factor=3–4 if not already
- Precision settings
  - Validate AMP functionality and SDPA kernel path; optionally test BF16
  - Confirm TF32 is off for matmul/cudnn if comparing pure FP16 goals
- Attention kernels
  - Prefer SDPA until FlashAttention officially supports SM_120 optimally
  - Revisit FlashAttention later for potential boost
- Optimizer revisit
  - Attempt building adam-atan2 with SM_120; compare stability/throughput vs AdamW

## 6) Proposed next steps
1. Throughput probe @ batch=320 for 10–15 minutes, record median steps/sec and GPU stats (if stable, try 384)
2. Append metrics to this file with W&B links
3. Choose best batch size; run a 30–45 minute quality probe and perform a quick evaluate
4. Optional: try TensorRT (Torch-TensorRT partial) on a checkpoint for inference uplift
5. Optional: compile adam-atan2 for SM_120 and re-test

---

Appendix A — How to reproduce metrics extraction
- See `perf/WANDB_ANALYSIS.md` for a small snippet to pull medians from the W&B API.

Appendix B — Raw observations
- Container level: GPU reported consistently high util (~98%) during steady state; memory headroom was ample.
- Multiple training processes initially were cleaned to a single process to avoid contention and to ensure accurate throughput sampling.

