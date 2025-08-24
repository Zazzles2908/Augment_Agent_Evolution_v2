"""
Brain 1 FastAPI Service
Main FastAPI application for Brain 1 embedding service.

Created: 2025-07-16 AEST
Author: Zazzles's Agent - Brain Architecture Standardization
"""

import asyncio
import logging
import signal
import sys
import uvicorn
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from enum import Enum
from typing import Dict, Any
# Prometheus metrics integrated through FastAPI endpoints

from brains.embedding_service.api.endpoints import router
from brains.embedding_service.api import endpoints
from brains.embedding_service.embedding_manager import Brain1Manager
from brains.embedding_service.config.settings import brain1_settings
# Redis Streams client for vector-bus
sys.path.append('/workspace/src')
from shared.redis_client import RedisStreamsClient
from shared.streams import StreamNames, EmbeddingResult

# Import VRAM management
sys.path.append('/workspace/src')
from shared.gpu.vram_manager import initialize_vram_management, get_vram_manager

# Import Brain 1 metrics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Brain 1 metrics available through /metrics endpoint in api/endpoints.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global settings
settings = brain1_settings

# Global service state management
class ServiceState(Enum):
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ServiceStateManager:
    """Thread-safe service state management for proper health checks"""
    def __init__(self):
        self._state = ServiceState.STARTING
        self._lock = threading.Lock()
        self._model_loading_start = None
        self._model_loading_end = None
        self._error_message = None

    def set_state(self, state: ServiceState, error_message: str = None):
        with self._lock:
            self._state = state
            if state == ServiceState.LOADING and self._model_loading_start is None:
                self._model_loading_start = time.time()
            elif state == ServiceState.READY and self._model_loading_end is None:
                self._model_loading_end = time.time()
            elif state == ServiceState.ERROR:
                self._error_message = error_message

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            status = {
                "state": self._state.value,
                "service": "brain1-embedding",
                "model_loaded": self._state == ServiceState.READY,
                "timestamp": time.time()
            }

            if self._model_loading_start:
                if self._model_loading_end:
                    status["model_loading_time"] = self._model_loading_end - self._model_loading_start
                else:
                    status["model_loading_duration"] = time.time() - self._model_loading_start

            if self._error_message:
                status["error"] = self._error_message

            return status

    def is_ready(self) -> bool:
        with self._lock:
            return self._state == ServiceState.READY

# Global state manager
state_manager = ServiceStateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper state tracking"""

    # Startup
    logger.info("🚀 Starting Brain 1 application with proper state management...")
    state_manager.set_state(ServiceState.STARTING)

    try:
        # Initialize VRAM management first
        logger.info("🎮 Initializing VRAM management for Brain 1 (35% allocation)...")
        vram_manager = initialize_vram_management('embedding', start_monitoring=True)
        logger.info(f"✅ VRAM management initialized: {vram_manager.allocated_vram_gb:.1f}GB allocated")

        # Metrics available through FastAPI /metrics endpoint
        logger.info("📊 Brain 1 metrics available at /metrics endpoint")

        # Initialize Brain 1 manager (modular)
        logger.info("🧠 Initializing Brain 1 manager (modular)...")
        manager = Brain1Manager(settings)

        # Set loading state before starting model loading
        state_manager.set_state(ServiceState.LOADING)
        logger.info("🔄 Starting Brain 1 manager...")

        success = await manager.start()
        if not success:
            error_msg = "Failed to start Brain 1 Manager (modular)"
            logger.error(f"❌ {error_msg}")
            state_manager.set_state(ServiceState.ERROR, error_msg)
            raise RuntimeError(error_msg)

        logger.info("✅ Brain 1 Manager started successfully")
        endpoints.brain1_manager = manager
        state_manager.set_state(ServiceState.READY)

        # Register Redis Streams embedding consumer is handled by Brain1Manager via StreamsHandler
        redis_client = None
        try:
            redis_client = RedisStreamsClient(redis_url=settings.redis_url, brain_id="brain1")
            await redis_client.connect()
            logger.info("ℹ️ Redis Streams client connected; handlers are managed within Brain1Manager")
        except Exception as e:
            logger.warning(f"⚠️ Redis Streams client not connected: {e}")

        # Metrics tracking handled by Brain1Manager internally
        logger.info("✅ Brain 1 application started successfully")

        yield

        # On shutdown, stop redis client
        if redis_client:
            await redis_client.disconnect()

    except Exception as e:
        error_msg = f"Failed to start Brain 1 application: {e}"
        logger.error(error_msg)
        state_manager.set_state(ServiceState.ERROR, error_msg)
        raise

    finally:
        # Shutdown
        logger.info("🛑 Shutting down Brain 1 application...")
        state_manager.set_state(ServiceState.SHUTDOWN)

        if endpoints.brain1_manager:
            # Cleanup if needed
            logger.info("Brain 1 manager cleanup complete")

        logger.info("✅ Brain 1 application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Brain 1 - Qwen3-4B Embedding Service",
    description="Vector embedding generation service using Qwen3-4B model for Four-Brain AI System",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Add Prometheus metrics endpoint
try:
    from brains.embedding_service.api.metrics_endpoint import metrics_router
    app.include_router(metrics_router, tags=["metrics"])
    logger.info("✅ Brain 1 metrics endpoint integrated")
except ImportError as e:
    logger.warning(f"⚠️ Brain 1 metrics endpoint not available: {e}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Brain 1 - Qwen3-4B Embedding",
        "version": "1.0.0",
        "status": "operational",
        "model": "Qwen3-4B-Embedding",
        "capabilities": [
            "text_embedding",
            "batch_processing", 
            "mrl_truncation",
            "vector_normalization"
        ],
        "endpoints": {
            "health": "/api/v1/health",
            "embed": "/api/v1/embed",
            "model": "/api/v1/model",
            "metrics": "/metrics",
            "docs": "/docs" if settings.debug else None
        },
        "zero_fabrication": True
    }

@app.get("/health")
async def health_check():
    """Proper health check endpoint that verifies model loading status"""
    status = state_manager.get_status()

    # Return appropriate HTTP status code based on service state
    if status["state"] == "ready":
        return status
    elif status["state"] == "error":
        raise HTTPException(status_code=503, detail=status)
    else:
        # Service is starting or loading
        raise HTTPException(status_code=503, detail=status)

@app.get("/health/simple")
async def simple_health():
    """Simple health check for basic load balancer compatibility"""
    if state_manager.is_ready():
        return {"status": "healthy", "service": "brain1-embedding"}
    else:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "service": "brain1-embedding"})

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "brain1_embedding.brain1_service:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=True
    )
