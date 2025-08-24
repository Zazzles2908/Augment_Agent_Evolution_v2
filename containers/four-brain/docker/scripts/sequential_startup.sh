#!/bin/bash

# Sequential Startup Script for Four-Brain System
# Prevents simultaneous model loading that causes 100% resource usage

set -e

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to check if a service is healthy
check_service_health() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    log "🔍 Checking health of $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service_name | grep -q "healthy"; then
            log "✅ $service_name is healthy"
            return 0
        fi
        
        info "⏳ Waiting for $service_name to be healthy (attempt $attempt/$max_attempts)..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    error "❌ $service_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to implement startup delays based on brain role
implement_startup_delay() {
    local brain_role=$1
    
    case $brain_role in
        "brain1")
            log "🧠 Brain1 (Embedding) - Starting immediately as primary model"
            ;;
        "brain2")
            log "🧠 Brain2 (Reranker) - Waiting 120s for Brain1 model loading..."
            sleep 120
            ;;
        "brain3")
            log "🧠 Brain3 (Zazzles's Agent) - Waiting 240s for Brain1+Brain2 model loading..."
            sleep 240
            ;;
        "brain4")
            log "🧠 Brain4 (Docling) - Waiting 360s for all previous models..."
            sleep 360
            ;;
        *)
            log "🔧 Infrastructure service - No delay needed"
            ;;
    esac
}

# Main startup logic
main() {
    log "🚀 Starting Four-Brain System Sequential Startup"
    
    # Get brain role from environment
    BRAIN_ROLE=${BRAIN_ROLE:-"unknown"}
    
    log "📋 Brain Role: $BRAIN_ROLE"
    
    # Implement startup delay based on brain role
    implement_startup_delay $BRAIN_ROLE
    
    # Check GPU memory allocation
    if [ ! -z "$CUDA_MEMORY_FRACTION" ]; then
        log "🎯 GPU Memory Allocation: ${CUDA_MEMORY_FRACTION} ($(echo "$CUDA_MEMORY_FRACTION * 16" | bc)GB of 16GB)"
    fi
    
    # Check system resources before starting
    log "💾 System Resources:"
    log "   - Available Memory: $(free -h | awk '/^Mem:/ {print $7}')"
    log "   - Available Disk: $(df -h / | awk 'NR==2 {print $4}')"
    
    # Start the main application
    log "🎯 Starting $BRAIN_ROLE application..."
    
    # Execute the original entrypoint
    exec "$@"
}

# Run main function with all arguments
main "$@"
