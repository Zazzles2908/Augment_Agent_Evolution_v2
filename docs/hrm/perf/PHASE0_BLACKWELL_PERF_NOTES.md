# Blackwell (RTX 5070 Ti) — Performance Notes for HRM Phase 0

This page explains what to expect performance‑wise on Blackwell (SM 12.0), what we saw in the smoke run, and how to measure uplift properly.

## 1) What we saw in the smoke run
- Environment verified: PyTorch 2.8 (NV), CUDA 13, GPU recognized (SM 12.0)
- Training completed end‑to‑end with AdamW (adam-atan2 kernel not SM_120 ready)
- GPU showed high utilization (your observation of ~100%) during compute phases
- We did not log tokens/sec or steps/sec in this first smoke run; it was too short to benchmark throughput reliably

## 2) Expected uplift on Blackwell vs Ada/Lovelace
- FP16 training conservative: +15–25%
- Optimistic (with tuned attention kernels/backends): +30–40%
- Influencers:
  - SDPA backend maturity for SM_120
  - Batch size / sequence length
  - Data pipeline and CPU/disk bottlenecks
  - Driver/CUDA version

## 3) How to measure properly (next run)
We’ll log per‑step performance to W&B for a 10–15 minute run to get steady‑state numbers:
- `train/step_time`, `train/steps_per_second`
- `train/samples_per_second` (and `tokens_per_second` if we fix an effective seq length)
- `system/gpu_util`, `system/memory_used_gb`

Suggested command:
- `docker exec -it hrm-phase0 bash -lc "cd /workspace/hrm_phase0/HRM && OMP_NUM_THREADS=8 python3 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=200 eval_interval=20 global_batch_size=384 lr=7e-5"`

Compare:
- Against your own prior GPU runs (if available)
- Against published references with similar batch/seq configs (interpret with caution)

## 4) Pitfalls & workarounds
- FlashAttention for SM_120 may lag initially — prefer SDPA
- Custom CUDA ops (like adam-atan2) may lack SM_120 kernels — use AdamW temporarily or compile SM_120 wheels
- Windows/WSL2 file mounts can be slower — ensure batch/data pipeline isn’t IO‑bound

## 5) Next steps
- Enable W&B live metrics logging (if you agree) so your W&B becomes the main live dashboard
- Run the 10–15 minute perf pass
- Summarize results in `PHASE0_RESULTS.md` with tokens/sec, steps/sec, GPU util, and any tuning notes

