#!/usr/bin/env bash
set -euo pipefail
TRITON_URL=${TRITON_URL:-http://localhost:8000}
models=(qwen3_4b_embedding qwen3_0_6b_reranking glm45_air)
for m in "${models[@]}"; do
  echo "Loading $m"; curl -sf -X POST "$TRITON_URL/v2/repository/models/$m/load" || true
  sleep 1
  echo "Model $m status:"; curl -sf "$TRITON_URL/v2/models/$m" || true
  echo "Unloading $m"; curl -sf -X POST "$TRITON_URL/v2/repository/models/$m/unload" || true
  sleep 1
  echo "Reloading $m"; curl -sf -X POST "$TRITON_URL/v2/repository/models/$m/load" || true
  echo
  sleep 1
done

