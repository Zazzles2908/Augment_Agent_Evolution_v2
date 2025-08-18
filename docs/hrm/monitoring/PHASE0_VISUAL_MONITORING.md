# HRM Phase 0 — Visual Live Monitoring

This page shows multiple ways to watch training live on Windows + WSL2 + Docker, without adding heavy new tools.

## Option A — Minimal and effective (Terminal + W&B)
Use your terminal for progress bars, W&B for live charts, and a second terminal for GPU utilization.

1) Run the training in your terminal (interactive):
- PowerShell
  - `docker exec -it hrm-phase0 /workspace/scripts/hrm/phase0_inside_container.sh`

2) Open a second terminal to watch GPU:
- PowerShell (via WSL)
  - `wsl watch -n 1 "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv"`

3) Watch the training log file (optional):
- PowerShell
  - `docker exec -it hrm-phase0 bash -lc "tail -f /workspace/docs/augment_code/hrm_training/results/PHASE0_TRAIN_LOG.txt"`

4) Open the W&B project in your browser (live charts):
- https://wandb.ai/jajireen1-hrm_project/Sudoku-extreme-1k-aug-1000%20ACT-torch
- Click the latest run to see metrics updating during training.

Why Docker Desktop “Logs” are empty: our container’s main process is `sleep`. We launch training with `docker exec`, so the live output is in your terminal and the log file.

## Option B — Add a simple GPU monitor loop (still lightweight)
Keep a record of your GPU stats during the run.

- PowerShell
  - `docker exec -it hrm-phase0 bash -lc 'while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,nounits >> /workspace/docs/augment_code/hrm_training/results/gpu_monitor.log; sleep 1; done'`

Open `docs/augment_code/hrm_training/results/gpu_monitor.log` afterwards to correlate GPU behavior with W&B metrics.

## Option C — Enrich W&B (best visual cockpit, no heavy infra)
We can add a tiny patch to log per-step performance and system stats so the W&B dashboard becomes your live console:
- What we’d log per step:
  - `train/step_time`, `train/steps_per_second`, `train/samples_per_second`
  - `system/gpu_util`, `system/memory_used_gb`
- If you want this, say “enable W&B live metrics” and we’ll patch HRM’s training code before the next run.

## Quick “what am I looking at?”
- Terminal (TQDM): live progress, steps
- W&B: loss curves, learning rate, run metadata, system metrics
- nvidia-smi: GPU % utilization, memory, power
- PHASE0_TRAIN_LOG.txt: the full textual record of the run

## Troubleshooting visibility
- No output in Docker Desktop “Logs”: expected (see above)
- W&B not updating: check WANDB_API_KEY in `containers/hrm/.env.local` and that you’re online
- PHASE0_TRAIN_LOG.txt missing: ensure the script ran; verify the path `/workspace/docs/augment_code/hrm_training/results`

