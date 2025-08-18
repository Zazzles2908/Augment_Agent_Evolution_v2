#!/usr/bin/env bash
set -euo pipefail

# Prepare working directories
mkdir -p /workspace /root/hrmwork /workspace/docs/augment_code/hrm_training/results || true

# Auto-login to Weights & Biases if API key is provided
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  NETRC_FILE="/root/.netrc"
  {
    echo "machine api.wandb.ai"
    echo "  login user"
    echo "  password ${WANDB_API_KEY}"
  } >"${NETRC_FILE}"
  chmod 600 "${NETRC_FILE}"
  if command -v wandb >/dev/null 2>&1; then
    # Non-interactive login; ignore errors if already logged in
    WANDB_MODE=${WANDB_MODE:-online} wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
  fi
fi

# Execute the requested command (default comes from CMD or docker-compose)
exec "$@"

