#!/usr/bin/env python3
"""
Enhanced Four-Brain System - Main Application Entry Point
Unified intelligent system with shared components and coordinated learning.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any
from contextlib import asynccontextmanager

# Add src to Python path
sys.path.insert(0, '/workspace/src')
sys.path.insert(0, '/workspace')

# Fix deprecated environment variables
os.environ["HF_HOME"] = "/workspace/cache/huggingface"
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]

# Check if packages are available and provide fallback
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    FASTAPI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FastAPI/uvicorn not available: {e}")
    FASTAPI_AVAILABLE = False
    # Create minimal fallback classes
    class FastAPI:
        def __init__(self, **kwargs): pass
        def on_event(self, event): return lambda f: f
        def get(self, path): return lambda f: f
        def post(self, path): return lambda f: f

    class HTTPException(Exception):
        def __init__(self, status_code, detail): pass

    class JSONResponse:
        def __init__(self, content): self.content = content

# Import Enhanced Four-Brain System modules
try:
    from shared.redis_client import RedisStreamsClient
    from shared.streams import StreamDefinitions
    from shared.memory_store import MemoryStore
    from shared.self_grading import SelfGradingSystem
    from shared.cli_executor import CLIExecutor
    from shared.self_improvement import SelfImprovementEngine
    from shared.message_flow import MessageFlowOrchestrator as MessageFlowCoordinator
    from shared.model_verification import ModelVerificationSystem, run_comprehensive_verification
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import Enhanced Four-Brain modules: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan event handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    # Startup
    logger.info("🚀 Starting Enhanced Four-Brain System...")
    await initialize_system()
    await start_background_tasks()

    yield

    # Shutdown
    logger.info("🛑 Shutting down Enhanced Four-Brain System...")
    if embedding_manager:
        # EmbeddingManager doesn't have async shutdown, just cleanup
        logger.info("🛑 Shutting down Embedding Manager...")
    if reranker_manager:
        await reranker_manager.shutdown()
        logger.info("🛑 Shutting down Reranker Manager...")
    if intelligence_manager:
        await intelligence_manager.shutdown()
        logger.info("🛑 Shutting down Intelligence Manager...")
    if document_manager:
        await document_manager.shutdown()
        logger.info("🛑 Shutting down Document Manager...")
    if redis_client:
        await redis_client.disconnect()

# FastAPI application
app = FastAPI(
    title="Enhanced Four-Brain System",
    description="Unified intelligent system with shared components and coordinated learning",
    version="6.0.0",
    lifespan=lifespan
)

# Import and include service routers
# FIXED: Import embedding router separately to avoid import errors from other services
try:
    from brains.embedding_service.api.endpoints import router as embedding_router
    app.include_router(embedding_router)
    logger.info("✅ Embedding service router included successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import embedding router: {e}")
except Exception as e:
    logger.error(f"❌ Error including embedding router: {e}")

# Try to import other service routers (non-critical for embedding functionality)
try:
    from brains.reranker_service.api.endpoints import router as reranker_router
    from brains.document_processor.api.documents import router as document_router

    app.include_router(reranker_router)
    app.include_router(document_router, prefix="/document", tags=["Document Processor"])
    logger.info("✅ Additional service routers included successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import additional service routers: {e}")
except Exception as e:
    logger.warning(f"⚠️ Error including additional routers: {e}")

# Global system components
redis_client = None
memory_store = None
self_grading = None
cli_executor = None
self_improvement = None
message_flow = None
embedding_manager = None
reranker_manager = None
intelligence_manager = None
document_manager = None
system_status = {
    "status": "initializing",
    "start_time": datetime.now(timezone.utc).isoformat(),
    "components": {},
    "health": {}
}

async def initialize_system():
    """Initialize all Enhanced Four-Brain System components."""
    global redis_client, memory_store, self_grading
    global cli_executor, self_improvement, message_flow, system_status
    global embedding_manager, reranker_manager, intelligence_manager, document_manager

    logger.info("🚀 Initializing Enhanced Four-Brain System...")

    try:
        # Run model verification first
        logger.info("🔍 Running model verification...")
        verification_result = run_comprehensive_verification()

        if verification_result["models_failed"] > 0:
            logger.warning(f"⚠️ Some models failed verification - failed count: {verification_result['models_failed']}")
            for model, result in verification_result["model_results"].items():
                if not result["passed"]:
                    logger.error(f"Model {model} verification failed - errors: {result['errors']}")
        else:
            logger.info("✅ All models verified successfully")

        system_status["model_verification"] = verification_result
        # Initialize Redis client
        logger.info("📡 Initializing Redis client...")
        redis_client = RedisStreamsClient()
        await redis_client.connect()
        system_status["components"]["redis"] = "connected"
        logger.info("✅ Redis client initialized")
        
        # Initialize Streams (already done in redis_client.connect())
        logger.info("🌊 Streams initialized via Redis client")
        system_status["components"]["streams"] = "initialized"
        logger.info("✅ Stream system ready")
        
        # Initialize Memory Store
        logger.info("🧠 Initializing Memory Store...")
        memory_store = MemoryStore()  # MemoryStore creates its own Redis connection
        await memory_store.connect()  # Ensure connection is established
        system_status["components"]["memory"] = "initialized"
        logger.info("✅ Memory Store initialized and connected")

        # Initialize Self-Grading System
        logger.info("📊 Initializing Self-Grading System...")
        self_grading = SelfGradingSystem(redis_client, memory_store)  # SelfGradingSystem needs both redis_client and memory_store
        system_status["components"]["self_grading"] = "initialized"
        logger.info("✅ Self-Grading System initialized")

        # Initialize CLI Executor
        logger.info("⚡ Initializing CLI Executor...")
        cli_executor = CLIExecutor()  # CLIExecutor uses default workspace_dir="/workspace"
        system_status["components"]["cli_executor"] = "initialized"
        logger.info("✅ CLI Executor initialized")

        # Initialize Self-Improvement System with proper dependency injection
        logger.info("🔄 Initializing Self-Improvement System...")
        self_improvement = SelfImprovementEngine(memory_store, self_grading)
        system_status["components"]["self_improvement"] = "initialized"

        # Verify dependencies are properly injected
        if self_improvement.memory_store is None:
            logger.error("❌ Memory store not properly injected into self-improvement engine")
            raise Exception("Self-improvement engine dependency injection failed")
        if self_improvement.grading_engine is None:
            logger.error("❌ Grading engine not properly injected into self-improvement engine")
            raise Exception("Self-improvement engine dependency injection failed")

        logger.info("✅ Self-Improvement System initialized with verified dependencies")
        
        # Initialize Message Flow Coordinator
        logger.info("🎯 Initializing Message Flow Coordinator...")
        message_flow = MessageFlowCoordinator(
            redis_client, memory_store,
            self_grading, self_improvement
        )
        system_status["components"]["message_flow"] = "initialized"
        logger.info("✅ Message Flow Coordinator initialized")

        # Initialize Brain Managers based on BRAIN_ROLE
        brain_role = os.getenv("BRAIN_ROLE", "unknown")

        if brain_role == "embedding":
            logger.info("🧠 Initializing Embedding Service Manager...")
            try:
                from brains.embedding_service.core.brain1_manager import Brain1Manager
                from brains.embedding_service.api import endpoints

                embedding_manager = Brain1Manager()
                success = embedding_manager.initialize_brain1_only()
                if success:
                    endpoints.brain1_manager = embedding_manager
                    system_status["components"]["embedding_manager"] = "initialized"
                    logger.info("✅ Embedding Service Manager initialized and started")
                else:
                    raise Exception("Failed to load Qwen3-4B model")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Embedding Manager: {e}")
                system_status["components"]["embedding_manager"] = "failed"

        elif brain_role == "reranker":
            logger.info("🧠 Initializing Reranker Service Manager...")
            try:
                from brains.reranker_service.reranker_manager import Brain2Manager
                from brains.reranker_service.api import endpoints as reranker_endpoints

                reranker_manager = Brain2Manager()
                success = await reranker_manager.initialize()
                if success:
                    # FIXED: Use proper setter to avoid lazy loading issues
                    reranker_endpoints.set_brain2_manager(reranker_manager)
                    system_status["components"]["reranker_manager"] = "initialized"
                    logger.info("✅ Reranker Service Manager initialized and started at startup")
                else:
                    raise Exception("Failed to load reranker model")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Reranker Manager: {e}")
                system_status["components"]["reranker_manager"] = "failed"

        elif brain_role == "intelligence":
            logger.info("🧠 Initializing Intelligence Service Manager...")
            try:
                from brains.intelligence_service.intelligence_manager import Brain3Manager
                from brains.intelligence_service.api import endpoints as intelligence_endpoints

                intelligence_manager = Brain3Manager()
                success = await intelligence_manager.initialize()
                if success:
                    intelligence_endpoints.brain3_manager = intelligence_manager
                    system_status["components"]["intelligence_manager"] = "initialized"
                    logger.info("✅ Intelligence Service Manager initialized and started")
                else:
                    raise Exception("Failed to initialize Intelligence Agent")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Intelligence Manager: {e}")
                system_status["components"]["intelligence_manager"] = "failed"

        elif brain_role == "document":
            logger.info("🧠 Initializing Document Processor Manager...")
            try:
                from brains.document_processor.document_manager import Brain4Manager
                from brains.document_processor.config.settings import settings as document_settings

                document_manager = Brain4Manager(document_settings)
                await document_manager.start()
                system_status["components"]["document_manager"] = "initialized"
                logger.info("✅ Document Processor Manager initialized and started")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Document Manager: {e}")
                system_status["components"]["document_manager"] = "failed"
        else:
            logger.info(f"🤖 No specific service manager for role: {brain_role}")
            document_manager = None

        system_status["status"] = "running"
        logger.info("🎉 Enhanced Four-Brain System fully initialized!")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        system_status["status"] = "failed"
        system_status["error"] = str(e)
        raise

async def start_background_tasks():
    """Start background tasks for the Enhanced Four-Brain System."""
    logger.info("🔄 Starting background tasks...")

    # Start message flow coordination
    if message_flow:
        try:
            message_flow_task = asyncio.create_task(message_flow.start_monitoring())
            logger.info("✅ Message flow coordination task started")
        except Exception as e:
            logger.error(f"❌ Failed to start message flow coordination: {e}")
    else:
        logger.warning("⚠️ Message flow coordinator not available")

    # Start self-improvement loop with enhanced error handling
    if self_improvement:
        try:
            improvement_task = asyncio.create_task(self_improvement.start_improvement_loop())
            logger.info("✅ Self-improvement loop task started")

            # Verify the loop is actually running by checking after a short delay
            await asyncio.sleep(2)
            if not improvement_task.done():
                logger.info("🔄 Self-improvement loop confirmed running")
            else:
                # Task completed unexpectedly, check for errors
                try:
                    await improvement_task
                except Exception as e:
                    logger.error(f"❌ Self-improvement loop failed immediately: {e}")
                    # Restart the loop
                    logger.info("🔄 Attempting to restart self-improvement loop...")
                    improvement_task = asyncio.create_task(self_improvement.start_improvement_loop())
        except Exception as e:
            logger.error(f"❌ Failed to start self-improvement loop: {e}")
    else:
        logger.warning("⚠️ Self-improvement engine not available")

    # Start health monitoring
    try:
        health_task = asyncio.create_task(monitor_system_health())
        logger.info("✅ Health monitoring task started")
    except Exception as e:
        logger.error(f"❌ Failed to start health monitoring: {e}")

    logger.info("✅ Background tasks startup completed")

async def monitor_system_health():
    """Monitor system health and update status."""
    while True:
        try:
            # Check Redis connection
            if redis_client:
                redis_health = await redis_client.ping()
                system_status["health"]["redis"] = "healthy" if redis_health else "unhealthy"
            
            # Check component status
            system_status["health"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            system_status["health"]["uptime"] = time.time() - time.mktime(
                datetime.fromisoformat(system_status["start_time"]).timetuple()
            )
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(60)

# Event handlers replaced by lifespan function above

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enhanced Four-Brain System",
        "version": "6.0.0",
        "status": system_status["status"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content=system_status)

@app.get("/status")
async def system_status_endpoint():
    """Detailed system status."""
    return JSONResponse(content={
        "system": system_status,
        "components": {
            "redis": "connected" if redis_client else "disconnected",
            "memory": "active" if memory_store else "inactive",
            "self_grading": "active" if self_grading else "inactive",
            "cli_executor": "active" if cli_executor else "inactive",
            "self_improvement": "active" if self_improvement else "inactive",
            "message_flow": "active" if message_flow else "inactive"
        }
    })

@app.post("/api/process")
async def process_request(request: Dict[str, Any]):
    """Process a request through the Enhanced Four-Brain System."""
    try:
        if not message_flow:
            raise HTTPException(status_code=503, detail="System not ready")
        
        result = await message_flow.process_request(request)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        if not memory_store:
            raise HTTPException(status_code=503, detail="Memory store not ready")

        metrics = await memory_store.get_system_metrics()
        return JSONResponse(content=metrics)

    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def prometheus_metrics():
    """Standard Prometheus metrics endpoint."""
    try:
        if not memory_store:
            raise HTTPException(status_code=503, detail="Memory store not ready")

        metrics = await memory_store.get_system_metrics()
        # Convert to Prometheus format (simplified)
        prometheus_format = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                prometheus_format.append(f"four_brain_{key} {value}")

        return Response(content="\n".join(prometheus_format), media_type="text/plain")

    except Exception as e:
        logger.error(f"Prometheus metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/custom-metrics")
async def custom_metrics():
    """Custom Four-Brain specific metrics."""
    try:
        brain_role = os.getenv("BRAIN_ROLE", "unknown")
        metrics = {
            "brain_role": brain_role,
            "system_status": system_status["status"],
            "components_active": len([k for k, v in system_status["components"].items() if v == "connected"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if memory_store:
            memory_metrics = await memory_store.get_system_metrics()
            metrics.update({
                "memory_store_connected": memory_metrics.get("redis_connected", False),
                "scores_stored": memory_metrics.get("scores_stored", 0),
                "patterns_matched": memory_metrics.get("patterns_matched", 0)
            })

        return JSONResponse(content=metrics)

    except Exception as e:
        logger.error(f"Custom metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grading-metrics")
async def grading_metrics():
    """Self-grading system metrics."""
    try:
        metrics = {
            "grading_system_active": self_grading is not None,
            "brain_role": os.getenv("BRAIN_ROLE", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if memory_store:
            memory_metrics = await memory_store.get_system_metrics()
            metrics.update({
                "patterns_matched": memory_metrics.get("patterns_matched", 0),
                "cache_hit_rate": memory_metrics.get("cache_hit_rate", 0.0)
            })

        return JSONResponse(content=metrics)

    except Exception as e:
        logger.error(f"Grading metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory-metrics")
async def memory_metrics():
    """Memory store specific metrics."""
    try:
        if not memory_store:
            raise HTTPException(status_code=503, detail="Memory store not ready")

        health = await memory_store.health_check()
        return JSONResponse(content=health)

    except Exception as e:
        logger.error(f"Memory metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/self-improvement-status")
async def self_improvement_status():
    """Self-improvement engine status and statistics."""
    try:
        if not self_improvement:
            raise HTTPException(status_code=503, detail="Self-improvement engine not ready")

        stats = self_improvement.get_statistics()

        # Add runtime status
        stats.update({
            "engine_active": self_improvement is not None,
            "memory_store_connected": self_improvement.memory_store is not None,
            "grading_engine_connected": self_improvement.grading_engine is not None,
            "timestamp": time.time()
        })

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Self-improvement status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-verification")
async def model_verification_status():
    """Model verification status and results."""
    try:
        # Get verification results from system status
        verification_result = system_status.get("model_verification", {})

        if not verification_result:
            # Run verification if not available
            logger.info("Running on-demand model verification...")
            verification_result = run_comprehensive_verification()

        return JSONResponse(content=verification_result)

    except Exception as e:
        logger.error(f"Model verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Four-Brain System...")

    # Check system readiness
    if not ENHANCED_MODULES_AVAILABLE:
        logger.error("❌ Enhanced Four-Brain modules not available - cannot start system")
        sys.exit(1)

    if not FASTAPI_AVAILABLE:
        logger.warning("⚠️  FastAPI not available - running in minimal mode")
        # Run basic initialization without web server
        import asyncio
        async def minimal_startup():
            await initialize_system()
            await start_background_tasks()
            logger.info("✅ Enhanced Four-Brain System running in minimal mode")
            # Keep running
            while True:
                await asyncio.sleep(60)

        asyncio.run(minimal_startup())
    else:
        # Configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        # Run the application
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
