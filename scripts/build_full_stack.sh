#!/usr/bin/env bash
set -euo pipefail

LOG=build.log
exec > >(tee -a "$LOG") 2>&1

echo "[INFO] Starting full stack build at $(date)"

# --- Helpers ---
err() { echo "[ERROR] $*"; }
warn() { echo "[WARN] $*"; }
info() { echo "[INFO] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { err "$1 is required"; exit 1; }; }
cleanup() {
  warn "Cleanup on failure"
  # Try to stop Triton container if we named it
  docker rm -f "${TRITON_NAME:-triton-server}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- Load .env if present ---
if [[ -f containers/four-brain/.env ]]; then
  info "Loading environment from containers/four-brain/.env (safe parser)"
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      key="${line%%=*}"; val="${line#*=}"
      export "$key"="$val"
    fi
  done < containers/four-brain/.env
  # Map HF_TOKEN to HUGGINGFACE_HUB_TOKEN if present
  if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi
fi

# --- Preflight ---
require_cmd nvidia-smi
require_cmd python3
require_cmd docker

# CUDA/TensorRT checks
info "Checking CUDA/TensorRT"
if ! nvidia-smi | grep -q "CUDA Version: 13."; then
  warn "CUDA 13 not detected; proceeding but builds may fail"
fi
if ! command -v trtexec >/dev/null 2>&1; then
  warn "trtexec not in PATH (TensorRT CLI). Please install TensorRT 10.8+"
fi

# --- Python env ---
info "Setting up Python virtual environment"
PYVENV=.venv
python3 -m venv "$PYVENV"
source "$PYVENV/bin/activate"
pip install --upgrade pip
pip install -q transformers onnx onnxruntime onnxscript tritonclient redis supabase pyyaml torch

# --- Model repository & names ---
REPO=${1:-"$(pwd)/containers/four-brain/triton/model_repository"}
info "Using model_repository: $REPO"

bash scripts/triton/rename_models.sh "$REPO" || true

# --- Build TensorRT engines (NVFP4 with fallback to FP16) ---
info "Building TensorRT engines"
bash scripts/tensorrt/build_engines.sh -r "$REPO" -p nvfp4 || {
  err "Engine build failed"
  exit 1
}

# --- Start Triton ---
info "Starting Triton server"
(set -x; bash scripts/triton/start_triton.sh "$REPO" &)
sleep 8
if ! curl -fsS http://localhost:8000/v2/health/ready >/dev/null; then
  err "Triton did not become ready"
  exit 1
fi
info "Triton is healthy"

# --- Supabase migration ---
if [[ -n "${SUPABASE_URL:-}" && -n "${SUPABASE_ANON_KEY:-}" ]]; then
  info "Applying Supabase RPC migration"
  if command -v psql >/dev/null 2>&1; then
    psql -f scripts/migrations/supabase_match_documents.sql || warn "psql migration failed; ensure local DB"
  else
    warn "psql not found; skip direct migration. Apply via Supabase SQL editor."
  fi
else
  warn "SUPABASE_URL/ANON_KEY not set; skipping Supabase checks"
fi

# --- Run end-to-end demo ---
info "Running end-to-end demo"
python3 examples/end_to_end_demo.py --config examples/config/config.yaml \
  --document tests/fixtures/sample.txt \
  --question "What is this document about?" || { err "Demo failed"; exit 1; }

info "[SUCCESS] Full stack build and demo completed"

