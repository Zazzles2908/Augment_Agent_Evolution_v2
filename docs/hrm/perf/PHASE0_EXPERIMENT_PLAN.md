# Phase 0 — Experiment Plan (Blackwell RTX 5070 Ti)

Goal: Determine stable throughput and quality baselines, then scale.

## A) Defaults (for consistency)
- Optimizer: AdamW (adam-atan2 reserved for follow-up once SM_120 build is ready)
- Precision: FP16 (PyTorch autocast/SDPA)
- Global batch size (starting point): 256–384 (we’ll probe)
- Sequence length: fixed per-task (Sudoku is grid-like; not tokenized sequences; for tokens/sec, we’ll log samples/sec instead)
- Data: Sudoku Extreme 1k (for quick iterations)

## B) Live cockpit metrics (W&B)
Log each step:
- train/step_time_s, train/steps_per_second, train/samples_per_second (derived)
- system/gpu_util_percent, system/memory_used_gb
- train/lr, train/lm_loss

## C) Phased runs
1) Warm-up probe (5–10 minutes)
   - Command: `epochs=200 eval_interval=20 global_batch_size=256`
   - Observe GPU util, step_time stability, OOM risk
2) Throughput probe (10–15 minutes)
   - Increase `global_batch_size` to 320 or 384 if memory allows
   - Choose the highest batch with stable steps/sec and no OOM
3) Quality probe (30–45 minutes)
   - Use best batch size from step 2, collect loss curve and a quick evaluate

## D) Decision points
- If GPU util < 80% and CPU is headroom: increase batch size
- If OOM occurs: drop batch by one step (e.g., 384 -> 320 -> 288)
- If step_time variance is high: check data loading; consider persistent workers/prefetch

## E) Next (optional)
- Try Torch-TensorRT partial compile on checkpoint to gauge inference uplift
- Attempt adam-atan2 SM_120 build and re-run probe for stability/perf

## F) Recording results
- After each run, append to `PHASE0_RESULTS.md`:
  - Run date/time, command, batch size, epochs window
  - Median steps/sec, samples/sec; avg GPU util; any OOM or instability
  - Loss trend and eval snapshot

