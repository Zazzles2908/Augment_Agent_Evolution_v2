#!/usr/bin/env bash
set -euo pipefail

REPO=${1:-/models}
IMAGE=${IMAGE:-nvcr.io/nvidia/tritonserver:25.07-py3}

# Start Triton with explicit model control and metrics
TRITON_NAME=${TRITON_NAME:-triton-server}
exec docker run --rm --gpus all --name "$TRITON_NAME" \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "${REPO}:/models" "$IMAGE" \
  tritonserver --model-repository=/models \
               --model-control-mode=explicit \
               --http-thread-count=4 \
               --log-verbose=1

