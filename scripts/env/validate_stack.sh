#!/usr/bin/env bash
set -euo pipefail

red() { echo -e "\e[31m$*\e[0m"; }
green() { echo -e "\e[32m$*\e[0m"; }
yellow() { echo -e "\e[33m$*\e[0m"; }

pass() { green "[PASS] $*"; }
fail() { red "[FAIL] $*"; }
info() { yellow "[INFO] $*"; }

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then pass "$1 present: $($1 --version | head -n1)"; else fail "$1 missing"; fi
}

info "Validating GPU/Driver/CUDA/TRT"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  pass "nvidia-smi available"
else
  fail "nvidia-smi missing"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | head -n3
  pass "CUDA toolkit present"
else
  fail "CUDA toolkit missing"
fi

dpkg -l | grep -i nvinfer || info "TensorRT packages not found via dpkg (ok if using containers)"

info "Validating containers"
check_cmd docker
if command -v nvidia-ctk >/dev/null 2>&1; then
  pass "nvidia-ctk present"
else
  info "nvidia-ctk not found; ensure NVIDIA Container Toolkit installed if using GPU in Docker"
fi

info "Quick GPU test in container"
if docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  pass "GPU accessible in Docker"
else
  info "GPU in Docker test failed; verify NVIDIA Container Toolkit setup"
fi

info "Validating Triton endpoints (if running)"
if curl -fsS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then pass "Triton HTTP ready"; else info "Triton not reachable"; fi
if curl -fsS http://localhost:8002/metrics >/dev/null 2>&1; then pass "Triton metrics reachable"; else info "Triton metrics not reachable"; fi

info "Validating datastores"
check_cmd redis-server || true
check_cmd psql || true
check_cmd supabase || true

info "Validating Python env"
check_cmd python3 || true
check_cmd pip || true

info "Validation complete"

