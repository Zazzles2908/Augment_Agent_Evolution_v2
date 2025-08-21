#!/usr/bin/env python3
"""
Brain 2 (Qwen3-Reranker-4B) FastAPI Service
Production-ready service for document relevance ranking

This module implements the FastAPI service for Brain 2, providing
HTTP endpoints for document reranking functionality with real model integration.

Zero Fabrication Policy: ENFORCED
All endpoints provide real functionality with actual model inference.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
import torch

# Add paths for imports
import sys
import os
workspace_src = os.getenv('WORKSPACE_SRC', '/workspace/src')
sys.path.append(workspace_src)

from brains.reranker_service.api.endpoints import router
from brains.reranker_service.config.settings import get_brain2_settings
from brains.reranker_service.brain2_manager import Brain2Manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import flow monitoring after logger is configured
try:
    from shared.monitoring.flow_monitoring import initialize_flow_monitoring, get_flow_monitor, BrainType, ToolType
    FLOW_MONITORING_AVAILABLE = True
    logger.info("🔄 Flow monitoring imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Flow monitoring not available: {e}")
    FLOW_MONITORING_AVAILABLE = False

# Global manager instance and service start time
brain2_manager: Brain2Manager = None
service_start_time = time.time()  # Track actual service start time for uptime calculation

class RerankerService:
    """Brain 2 Reranker Service Class"""
    
    def __init__(self, manager: Brain2Manager = None):
        self.manager = manager or brain2_manager
        self.logger = logging.getLogger(__name__)
    
    async def health_check(self):
        """Service health check"""
        if self.manager:
            return await self.manager.health_check()
        return {"status": "error", "message": "Manager not initialized"}
    
    async def rerank_documents(self, query: str, documents: list, top_k: int = 5):
        """Rerank documents using Brain 2 manager"""
        if not self.manager:
            raise HTTPException(status_code=500, detail="Manager not initialized")
        
        try:
            return await self.manager.rerank_documents(query, documents, top_k)
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for Brain 2 service"""
    global brain2_manager
    
    # Startup
    logger.info("🚀 Starting Brain 2 Reranker Service...")
    logger.info(f"📅 Date: July 11, 2025 12:40 AEST")
    
    try:
        settings = get_brain2_settings()
        brain2_manager = Brain2Manager(settings)
        logger.info("🧠 Brain 2 Manager initialized successfully")
        logger.info(f"📦 Model: {settings.model_name}")

        # CRITICAL FIX: Actually load the model!
        logger.info("🚀 Loading Qwen3-Reranker-4B model...")
        await brain2_manager.initialize()
        logger.info("✅ Brain 2 model loading completed!")

        # Pre-warm the model (per fix_containers.md) - eliminates cold-start delay
        logger.info("🔥 Pre-warming Brain 2 model...")
        try:
            # Use real test data from test_data folder for model warming
            import os
            workspace_root = os.getenv('WORKSPACE_ROOT', '/workspace')
            test_data_path = os.path.join(workspace_root, "tests/integration/end_to_end/test_data/Context_engineering_template_README.md")
            if os.path.exists(test_data_path):
                with open(test_data_path, 'r', encoding='utf-8') as f:
                    real_content = f.read()[:500]  # First 500 chars for warming
                real_query = "context engineering methodology"
                real_docs = [real_content]
                await brain2_manager.rerank_documents(real_query, real_docs)
                logger.info("✅ Brain 2 model pre-warming completed successfully with real data")
            else:
                logger.info("⚠️ Test data not found, skipping pre-warming to avoid dummy data")
        except Exception as e:
            logger.warning(f"⚠️ Brain 2 model pre-warming failed: {e}")

        # Initialize flow monitoring (integrate with existing metrics endpoint)
        if FLOW_MONITORING_AVAILABLE:
            try:
                flow_monitor = initialize_flow_monitoring("brain2_reranker", enable_http_server=False)
                logger.info("🔄 Flow monitoring initialized for Brain 2 (integrated with existing metrics)")

                # Update connection status
                flow_monitor.update_connection_status("model", "qwen3_reranker_4b", True)
                flow_monitor.update_connection_status("service", "brain2_api", True)

            except Exception as e:
                logger.warning(f"⚠️ Flow monitoring initialization failed: {e}")

        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Brain 2 service: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Brain 2 Reranker Service...")

# Create FastAPI application
app = FastAPI(
    title="Brain 2 - Qwen3 Reranker Service",
    description="Document relevance ranking service using Qwen3-Reranker-4B",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Add Prometheus metrics endpoint
try:
    from .metrics.prometheus_metrics import get_brain2_metrics
    from fastapi import Response

    @app.get("/metrics", response_class=Response)
    async def metrics_endpoint():
        """Prometheus metrics endpoint for Brain 2"""
        try:
            metrics_instance = get_brain2_metrics()
            metrics_data = metrics_instance.get_metrics()
            return Response(
                content=metrics_data,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return Response(
                content=f"# Error generating metrics: {str(e)}\n",
                media_type="text/plain; version=0.0.4; charset=utf-8",
                status_code=500
            )

    logger.info("✅ Brain 2 metrics endpoint integrated")
except ImportError as e:
    logger.warning(f"⚠️ Brain 2 metrics endpoint not available: {e}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Brain 2 - Qwen3 Reranker",
        "version": "1.0.0",
        "status": "operational",
        "model": "Qwen/Qwen3-Reranker-4B",
        "capabilities": [
            "document_reranking",
            "relevance_scoring",
            "batch_processing"
        ],
        "zero_fabrication": True
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    global brain2_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "brain2_reranker",
        "version": "1.0.0",
        "components": {}
    }
    
    try:
        # Check Brain 2 Manager
        if brain2_manager:
            manager_health = await brain2_manager.health_check()
            health_status["components"]["brain2_manager"] = manager_health
        else:
            health_status["components"]["brain2_manager"] = {
                "status": "not_initialized",
                "healthy": False
            }
        
        # Check GPU availability
        health_status["components"]["gpu"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "healthy": torch.cuda.is_available()
        }
        
        # Overall health assessment
        component_health = [comp.get("healthy", False) for comp in health_status["components"].values()]
        overall_healthy = all(component_health) and len(component_health) > 0
        
        health_status["status"] = "healthy" if overall_healthy else "degraded"
        health_status["healthy"] = overall_healthy
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }

# Duplicate metrics endpoint removed - using prometheus_metrics module instead

def get_brain2_manager() -> Brain2Manager:
    """Dependency injection for Brain 2 Manager"""
    global brain2_manager
    if not brain2_manager:
        raise HTTPException(status_code=503, detail="Brain 2 Manager not initialized")
    return brain2_manager

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Brain 2 Reranker Service...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        workers=1,
        log_level="info"
    )
