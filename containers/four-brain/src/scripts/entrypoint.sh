#!/bin/bash
# Enhanced Four-Brain System - Container Entrypoint Script
# Unified intelligent system with shared components and coordinated learning
# Generated: 2025-07-24 AEST

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log "🚀 Starting Enhanced Four-Brain System"
log "🧠 Unified intelligent system with coordinated learning"

# Set RTX 5070 Ti optimizations
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:512"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ENABLE_4BIT_QUANTIZATION=${ENABLE_4BIT_QUANTIZATION:-true}
export ENABLE_8BIT_QUANTIZATION=${ENABLE_8BIT_QUANTIZATION:-false}
export ENABLE_FLASH_ATTENTION=${ENABLE_FLASH_ATTENTION:-true}

# Performance optimizations
export TORCH_COMPILE=${TORCH_COMPILE:-1}
export CUDA_GRAPHS=${CUDA_GRAPHS:-1}
export TORCHDYNAMO_FULLGRAPH=${TORCHDYNAMO_FULLGRAPH:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

# Model cache paths - Use writable cache volume
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/.cache/transformers}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}

# Create necessary directories in writable locations
log "📁 Creating required directories..."
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/transformers
mkdir -p /workspace/.cache/torch
mkdir -p /workspace/logs
mkdir -p /workspace/data

# Set permissions for writable directories only
chmod -R 755 /workspace/.cache
chmod -R 755 /workspace/logs
chmod -R 755 /workspace/data
chmod -R 755 /workspace/logs
chmod -R 755 /workspace/data

# Run pre-flight checks
log "🔍 Running pre-flight health checks..."
if python /workspace/health/pre_flight_check.py; then
    success "✅ Pre-flight checks passed"
else
    warn "⚠️  Pre-flight checks had warnings - continuing startup"
fi

# Wait for dependencies
log "📊 Waiting for Redis and other dependencies..."
sleep 5

# Start the Enhanced Four-Brain System
log "🎬 Starting Enhanced Four-Brain System..."
cd /workspace/src

# Check if this is the K2-Hub container
if [ "$BRAIN_ROLE" = "k2_hub" ]; then
    log "🏛️ Starting K2-Vector-Hub Service (The Mayor's Office)..."
    exec python k2_hub_main.py
else
    log "🧠 Starting Brain Service..."
    exec python main.py
fi
