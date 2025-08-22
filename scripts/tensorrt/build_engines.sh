#!/usr/bin/env bash
set -euo pipefail

# TensorRT Engine Build Orchestrator
# Usage:
#   ./scripts/tensorrt/build_engines.sh -r /absolute/path/to/model_repository \
#       [-c scripts/tensorrt/config] [-p nvfp4|fp8|fp16] [--dry-run]
#
# Requirements:
# - Ubuntu 24.04, CUDA 13.0, TensorRT 10.8+, trtexec in PATH
# - ONNX models present in each model dir's 'onnx' or '1/model.onnx'
# - Config YAML describes inputs/outputs and builder options

REPO=""
CONFIG_DIR="scripts/tensorrt/config"
PRECISION="nvfp4"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--repo) REPO="$2"; shift; shift ;;
    -c|--config-dir) CONFIG_DIR="$2"; shift; shift ;;
    -p|--precision) PRECISION="$2"; shift; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "Missing -r / --repo path to Triton model_repository" >&2
  exit 1
fi

models=(qwen3_4b_embedding qwen3_0_6b_reranking glm45_air)

build_one() {
  local model="$1" cfg="$2" prec="$3"
  echo "\n=== Building $model (precision=$prec) ==="
  if ! $DRY_RUN; then
    python3 scripts/tensorrt/vram_check.py || { echo "[ERROR] VRAM pre-check failed"; return 100; }
  fi
  if $DRY_RUN; then
    echo "DRY RUN: python3 scripts/tensorrt/convert_model.py --repo '$REPO' --model '$model' --config '$cfg' --precision '$prec'"
    return 0
  fi
  # Retry loop for transient network issues during HF download/export
  local attempts=0 max_attempts=3
  until python3 scripts/tensorrt/convert_model.py --repo "$REPO" --model "$model" --config "$cfg" --precision "$prec"; do
    attempts=$((attempts+1))
    echo "[WARN] Build failed for $model (attempt $attempts/$max_attempts). Retrying in 10s..."
    sleep 10
    if (( attempts >= max_attempts )); then
      echo "[ERROR] Build failed after $attempts attempts for $model (precision=$prec)"
      return 1
    fi
  done
  python3 scripts/tensorrt/vram_check.py || { echo "[ERROR] VRAM post-check failed"; return 101; }
  echo "=== Done: $model ===\n"
}

for m in "${models[@]}"; do
  cfg="$CONFIG_DIR/$m.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi
  if ! build_one "$m" "$cfg" "$PRECISION"; then
    echo "[WARN] Falling back to FP16 for $m"
    if ! build_one "$m" "$cfg" "fp16"; then
      echo "[FATAL] Unable to build $m with NVFP4 or FP16"
      exit 1
    fi
  fi
done

echo "All builds completed. Ensure Triton uses --model-control-mode=explicit and loads from: $REPO"

