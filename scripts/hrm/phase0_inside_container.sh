#!/usr/bin/env bash
set -euo pipefail

# Phase 0 inside the hrm-train:25.06 container
# Assumes: CUDA 13 + PyTorch 25.06 base, GPU visible
# Writes logs into /workspace/docs/augment_code/hrm_training/results

LOG_DIR=/workspace/docs/augment_code/hrm_training/results
mkdir -p "$LOG_DIR"

# 0) Environment capture
{
  echo "== ENV ==";
  date;
  nvidia-smi --query-gpu=name,compute_cap --format=csv || true;
  python3 - <<'PY'
import torch, platform, os
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
  print('Device:', torch.cuda.get_device_name(0))
  print('SM capability:', torch.cuda.get_device_capability(0))
print('Python:', platform.python_version())
print('CWD:', os.getcwd())
PY
} | tee -a "$LOG_DIR/PHASE0_ENV_CHECK.md"

# 1) Get HRM repo
WORK=/workspace/hrm_phase0
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d HRM ]; then
  git clone https://github.com/sapientinc/HRM.git
fi
cd HRM

# 2) Install deps (avoid overwriting container torch)
export PIP_BREAK_SYSTEM_PACKAGES=1
if [ -f requirements.txt ]; then
  grep -vE '^(torch|torchvision|torchaudio)\b' requirements.txt > /tmp/reqs.txt || true
  pip install --no-cache-dir -r /tmp/reqs.txt || true
fi

# 3) Build Sudoku dataset (small sample)
python3 dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000

# 4) Train (smoke: 1 epoch). Increase epochs for real run
OMP_NUM_THREADS=8 python3 pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=1 eval_interval=1 global_batch_size=64 lr=7e-5 \
  | tee -a "$LOG_DIR/PHASE0_TRAIN_LOG.txt"

# 5) Evaluate (best/latest checkpoint)
if [ -d checkpoints ]; then
  CKPT=$(ls -1t checkpoints/*.pt 2>/dev/null | head -n1 || true)
  if [ -n "${CKPT:-}" ]; then
    python3 evaluate.py checkpoint="$CKPT" | tee -a "$LOG_DIR/PHASE0_EVAL_PYTORCH.txt"
  fi
fi

echo "Phase 0 smoke run complete. Check $LOG_DIR for logs."
