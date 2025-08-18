# HRM Phase 0 — Hands‑On User Guide (Windows + WSL2 + Docker)

This guide explains, step by step, how to build, run, observe, and understand the HRM Phase 0 training workflow on your machine. It includes where files live, how to get container details, how to view logs, and how to find our Weights & Biases (W&B) runs.

## 1) What you just ran (high level)
- Image: `hrm-train:25.06` (based on NVIDIA PyTorch 25.06, CUDA 13)
- Container: `hrm-phase0` (started via docker compose)
- Mount: The repo on Windows is mounted into the container at `/workspace`
- Auto W&B login: Entry point reads WANDB_API_KEY from `containers/hrm/.env.local` and silently logs in
- Training: A small “smoke test” training run on the Sudoku Extreme (1k) dataset
- Tracking: Metrics/logs are pushed to W&B; text logs are written under the repo

Why you didn’t see `docker logs` output: The container’s main process is `sleep infinity`. We invoke training using `docker exec`, so the output goes to the exec session (and to a file inside the repo), not to `docker logs`.

## 2) Prerequisites already set up
- Compose file: `containers/hrm/docker-compose.yml`
- Local env file: `containers/hrm/.env.local` (contains WANDB_API_KEY)
- Entry point: `containers/hrm/entrypoint.sh` (auto‑login to W&B)
- Image built: `hrm-train:25.06`

## 3) Start/stop and inspect the container
From a PowerShell in the repo root (C:\Project\Augment_Agent_Evolution):

- Start
  - `docker compose -f containers/hrm/docker-compose.yml up -d`
- Stop
  - `docker compose -f containers/hrm/docker-compose.yml down`
- Status (ID and image)
  - `docker ps --filter name=hrm-phase0 --format "table {{.Names}}\t{{.ID}}\t{{.Image}}\t{{.Status}}"`
- GPU check inside container
  - `docker exec hrm-phase0 nvidia-smi | head -n 20`

If the container shows as running but you see no logs: that’s expected (main process is `sleep`). Use the log file or W&B sections below.

## 4) Where files and logs live (host <-> container)
- Host repo root: `C:\Project\Augment_Agent_Evolution` (Windows)
- Container mount: `/workspace` (Linux)
- Training script path (host): `scripts/hrm/phase0_inside_container.sh`
- Training logs directory (host): `docs/augment_code/hrm_training/results`
  - Same directory inside container: `/workspace/docs/augment_code/hrm_training/results`

To see the logs from Windows after a run:
- Open `C:\Project\Augment_Agent_Evolution\docs\augment_code\hrm_training\results`

To tail logs live during a run from inside the container:
- `docker exec -it hrm-phase0 bash -lc "tail -f /workspace/docs/augment_code/hrm_training/results/PHASE0_TRAIN_LOG.txt"`

## 5) Rerun the Phase 0 smoke test
What it does: captures environment info, clones HRM (if missing), builds Sudoku Extreme (1k) dataset, runs a 1‑epoch smoke train, and (if a checkpoint exists) evaluates it.

Run it:
- `docker exec -it hrm-phase0 /workspace/scripts/hrm/phase0_inside_container.sh`

Where to look after it finishes:
- Logs: `/workspace/docs/augment_code/hrm_training/results` (inside container) or the mapped Windows path above
- W&B: see the next section

## 6) Weights & Biases (W&B)
- Auto login: Happens at container start if `WANDB_API_KEY` is present
- Project we used: `Sudoku-extreme-1k-aug-1000 ACT-torch`
- Your account: `jajireen1` (based on the configured key)

Quick links from our smoke run:
- Project: https://wandb.ai/jajireen1-hrm_project/Sudoku-extreme-1k-aug-1000%20ACT-torch
- Latest run (example): the script prints the run URL at the end (e.g., `/runs/zg0bx1eb`).

How to find runs if you didn’t copy the link:
1) Go to https://wandb.ai and log in
2) In the top nav, click your workspace `jajireen1-hrm_project`
3) Open the project `Sudoku-extreme-1k-aug-1000 ACT-torch`
4) You’ll see a list of runs (latest on top). Click one to view metrics, system stats, logs, and config

CLI commands inside the container (optional):
- `docker exec -it hrm-phase0 bash` then:
  - `wandb status` (shows local config)
  - `wandb online` / `wandb offline` (toggle syncing)
  - `wandb login --relogin $WANDB_API_KEY` (force relogin if ever needed)

## 7) Understanding the smoke test output
- Dataset build: Creates `data/sudoku-extreme-1k-aug-1000` inside `/workspace/hrm_phase0/HRM`
- Training (smoke): 1 epoch with small batch size; prints progress bars and writes to `PHASE0_TRAIN_LOG.txt`
- Optimizer: AdamW (we used AdamW for compatibility with RTX 5070 Ti SM_120)
- Checkpoints: If created, they appear under `/workspace/hrm_phase0/HRM/checkpoints`
- W&B: Shows metrics like loss, learning rate, and run metadata; links are printed at the end of the run

## 8) Common pitfalls and why you didn’t see logs in Docker Desktop
- `docker logs hrm-phase0` shows only the container’s main process output (we run `sleep` as the main process)
- The training process runs via `docker exec`, so its output isn’t captured by `docker logs`
- Use:
  - The run’s W&B page for rich dashboards and logs
  - The on-disk training log file `PHASE0_TRAIN_LOG.txt`
  - Or run an interactive shell: `docker exec -it hrm-phase0 bash` and inspect files directly

## 9) Inspecting what’s running
- Container process table (inside container):
  - `docker exec -it hrm-phase0 bash -lc "ps aux | head -n 30"`
- Python processes (inside container):
  - `docker exec -it hrm-phase0 bash -lc "pgrep -a python || true"`
- GPU activity sample (host):
  - `nvidia-smi` (Windows or WSL2)

## 10) Start/Stop cheat sheet
- Start container: `docker compose -f containers/hrm/docker-compose.yml up -d`
- Stop container: `docker compose -f containers/hrm/docker-compose.yml down`
- Enter shell: `docker exec -it hrm-phase0 bash`
- Rerun smoke test: `docker exec -it hrm-phase0 /workspace/scripts/hrm/phase0_inside_container.sh`
- Tail training log: `docker exec -it hrm-phase0 bash -lc "tail -f /workspace/docs/augment_code/hrm_training/results/PHASE0_TRAIN_LOG.txt"`

## 11) Next steps (beyond smoke)
- Longer training (example):
  - `docker exec -it hrm-phase0 bash -lc "cd /workspace/hrm_phase0/HRM && OMP_NUM_THREADS=8 python3 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=200 eval_interval=20 global_batch_size=384 lr=7e-5"`
- Evaluation (after a checkpoint appears):
  - `docker exec -it hrm-phase0 bash -lc "cd /workspace/hrm_phase0/HRM && OMP_NUM_THREADS=8 python3 evaluate.py checkpoint=$(ls -1t checkpoints/*.pt | head -n1)"`
- TensorRT path (optional, after a stable checkpoint): see `docs/augment_code/hrm_training/22_hrm_phase0_runbook.md`

## 12) Quick reference: key paths
- Windows host repo: `C:\Project\Augment_Agent_Evolution`
- Container mount: `/workspace`
- Compose file: `containers/hrm/docker-compose.yml`
- Env file (W&B key): `containers/hrm/.env.local`
- Phase 0 script: `scripts/hrm/phase0_inside_container.sh`
- HRM working dir (inside container): `/workspace/hrm_phase0/HRM`
- Logs (host): `docs/augment_code/hrm_training/results`
- Logs (container): `/workspace/docs/augment_code/hrm_training/results`

---
If anything is unclear, ping me in this thread with what you’re trying to see (container info, logs, W&B, etc.), and I’ll walk you through it live or add more examples.

