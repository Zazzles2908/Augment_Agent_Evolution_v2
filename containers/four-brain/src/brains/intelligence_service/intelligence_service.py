#!/usr/bin/env python3
"""
Brain 3 (Zazzles's Agent Integration) FastAPI Service
Production-ready service for Zazzles's Agent integration

This module implements the FastAPI service for Brain 3, providing
HTTP endpoints for Zazzles's Agent functionality with real integration.

Zero Fabrication Policy: ENFORCED
All endpoints provide real functionality with actual Zazzles's Agent integration.
"""

import time
import logging
import threading
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware

# Add paths for imports
import sys
sys.path.append('/workspace/src')
sys.path.append('/workspace/src/brains')
sys.path.append('/workspace/src/brains/intelligence_service')

from brains.intelligence_service.api.endpoints import router
from brains.intelligence_service.config.settings import get_brain3_settings
from brains.intelligence_service.brain3_manager import Brain3Manager
# HRM removed from project; intelligence service no longer initializes HRM

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# K2 Vector Bridge removed - deprecated module

# Import flow monitoring after logger is configured
try:
    from shared.monitoring.flow_monitoring import initialize_flow_monitoring, get_flow_monitor, BrainType, ToolType
    FLOW_MONITORING_AVAILABLE = True
    logger.info("🔄 Flow monitoring imported successfully for Brain 3")
except ImportError as e:
    logger.warning(f"⚠️ Flow monitoring not available: {e}")
    FLOW_MONITORING_AVAILABLE = False

# Global manager instance
brain3_manager: Brain3Manager = None
# HRM removed
# hrm_orchestrator removed
# blackwell_optimizer removed

class AugmentService:
    """Brain 3 Zazzles's Agent Service Class"""
    
    def __init__(self, manager: Brain3Manager = None):
        self.manager = manager or brain3_manager
        self.logger = logging.getLogger(__name__)
    
    async def health_check(self):
        """Service health check"""
        if self.manager:
            return await self.manager.health_check()
        return {"status": "error", "message": "Manager not initialized"}
    
    async def process_augment_request(self, request_data: dict):
        """Process Zazzles's Agent request using Brain 3 manager"""
        if not self.manager:
            raise HTTPException(status_code=500, detail="Manager not initialized")
        
        try:
            return await self.manager.process_augment_request(request_data)
        except Exception as e:
            self.logger.error(f"Zazzles's Agent processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for Brain 3 service"""
    global brain3_manager
    
    # Startup
    logger.info("🚀 Starting Brain 3 Zazzles's Agent Service...")
    from datetime import datetime
    import pytz

    # Initialize VRAM management first
    logger.info("🎮 Initializing VRAM management for Brain 3 (15% allocation)...")
    import sys
    sys.path.append('/workspace/src')
    from shared.gpu.vram_manager import initialize_vram_management
    vram_manager = initialize_vram_management('intelligence', start_monitoring=True)
    logger.info(f"✅ VRAM management initialized: {vram_manager.allocated_vram_gb:.1f}GB allocated")

    # Get current date in AEST timezone
    aest = pytz.timezone('Australia/Sydney')
    current_date = datetime.now(aest).strftime("%B %d, %Y %Z")
    logger.info(f"📅 Date: {current_date}")

    try:
        settings = get_brain3_settings()
        brain3_manager = Brain3Manager()
        logger.info("🧠 Brain 3 Manager initialized successfully")
        logger.info(f"🔗 Integration Mode: {settings.integration_mode}")
        
        # Initialize the manager
        init_result = await brain3_manager.initialize()
        if init_result:
            logger.info("✅ Brain 3 initialization completed successfully")
        else:
            logger.warning("⚠️ Brain 3 initialization completed with warnings")

        # Wait for Redis and Supabase to be fully ready
        logger.info("⏳ Waiting for services to be fully ready...")
        await asyncio.sleep(5)

        # HRM removed: no HRM initialization in Intelligence Service

        # Initialize flow monitoring (integrate with existing metrics endpoint)
        if FLOW_MONITORING_AVAILABLE:
            try:
                flow_monitor = initialize_flow_monitoring("brain3_augment", enable_http_server=False)
                logger.info("🔄 Flow monitoring initialized for Brain 3 (integrated with existing metrics)")

                # Update connection status
                flow_monitor.update_connection_status("service", "augment_agent", True)
                flow_monitor.update_connection_status("service", "brain3_api", True)

            except Exception as e:
                logger.warning(f"⚠️ Flow monitoring initialization failed: {e}")

        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Brain 3 service: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Brain 3 Zazzles's Agent Service...")
        # Cleanup communicator if needed
        if brain3_manager and hasattr(brain3_manager, 'communicator') and brain3_manager.communicator:
            if hasattr(brain3_manager.communicator, 'disconnect'):
                await brain3_manager.communicator.disconnect()

# Create FastAPI application
app = FastAPI(
    title="Brain 3 - Zazzles's Agent Integration Service",
    description="Zazzles's Agent integration service for Four-Brain Architecture",
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
    from .metrics.prometheus_metrics import get_brain3_metrics
    from fastapi import Response

    @app.get("/metrics", response_class=Response)
    async def metrics_endpoint():
        """Prometheus metrics endpoint for Brain 3"""
        try:
            metrics_instance = get_brain3_metrics()
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

    logger.info("✅ Brain 3 metrics endpoint integrated")
except ImportError as e:
    logger.warning(f"⚠️ Brain 3 metrics endpoint not available: {e}")

# K2 Vector Bridge removed - deprecated functionality

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Brain 3 - Zazzles's Agent Integration (The Concierge)",
        "version": "1.0.0",
        "status": "operational",
        "integration": "Zazzles's Agent via Supabase",
        "port": 8013,
        "role": "Orchestrator and Front Door",
        "capabilities": [
            "conversation_analysis",
            "task_management",
            "code_generation",
            "system_integration",
            "documentation_creation",
            "problem_solving",
            "workflow_orchestration",
            "sequential_thinking",
            "tool_execution",
            "brain_coordination",
            "k2_vector_hub_integration"
        ],
        "zero_fabrication": True
    }

@app.post("/ask")
async def ask_question(request_data: dict):
    """
    Main orchestration endpoint - The Concierge front door
    Implements the workflow: User → Brain 3 → K2-Vector-Hub → All Brains → Response
    """
    global brain3_manager

    try:
        if not brain3_manager:
            raise HTTPException(status_code=503, detail="Brain 3 Manager not initialized")

        # Extract question from request
        question = request_data.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        logger.info(f"🎯 Brain 3 Concierge received question: {question[:100]}...")

        # Step 1: Post question to Redis channel 'vector_jobs'
        if brain3_manager.communicator and brain3_manager.communicator.connected:
            try:
                job_id = await brain3_manager.communicator.publish_vector_job(
                    question=question,
                    user_context=request_data.get("context", {})
                )
                logger.info(f"📤 Published vector job {job_id} to Redis")

                # Step 2: Wait for K2-Vector-Hub strategy response
                logger.info(f"⏳ Waiting for K2-Vector-Hub strategy for job {job_id}")
                strategy_plan = await brain3_manager.communicator.wait_for_strategy_plan(job_id, timeout_seconds=30)

                if strategy_plan:
                    # Step 3: Coordinate with other brains based on strategy
                    logger.info(f"🎯 Executing strategy plan for job {job_id}")
                    coordination_results = await brain3_manager.communicator.coordinate_brain_execution(strategy_plan)

                    return {
                        "status": "completed",
                        "message": "Brain 3 Concierge successfully orchestrated your request",
                        "question": question,
                        "job_id": job_id,
                        "strategy_plan": strategy_plan,
                        "coordination_results": coordination_results,
                        "workflow_stage": "completed",
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "timeout",
                        "message": "K2-Vector-Hub strategy timeout - request may still be processing",
                        "question": question,
                        "job_id": job_id,
                        "workflow_stage": "strategy_timeout",
                        "timestamp": time.time()
                    }

            except Exception as redis_error:
                logger.error(f"❌ Redis workflow error: {redis_error}")
                return {
                    "status": "error",
                    "message": f"Redis communication error: {str(redis_error)}",
                    "question": question,
                    "workflow_stage": "redis_error",
                    "timestamp": time.time()
                }
        else:
            return {
                "status": "error",
                "message": "Redis communication not available - Brain 3 not properly initialized",
                "question": question,
                "workflow_stage": "initialization_error",
                "timestamp": time.time()
            }

    except Exception as e:
        logger.error(f"❌ Ask endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    global brain3_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "brain3_augment",
        "version": "1.0.0",
        "components": {}
    }
    
    try:
        # Check Brain 3 Manager
        if brain3_manager:
            manager_health = await brain3_manager.health_check()
            health_status["components"]["brain3_manager"] = manager_health
        else:
            health_status["components"]["brain3_manager"] = {
                "status": "not_initialized",
                "healthy": False
            }
        
        # Check Supabase connectivity
        if brain3_manager and hasattr(brain3_manager, 'supabase_connected'):
            health_status["components"]["supabase"] = {
                "connected": brain3_manager.supabase_connected,
                "healthy": brain3_manager.supabase_connected
            }
        
        # Check Redis communication
        if brain3_manager and hasattr(brain3_manager, 'communicator') and brain3_manager.communicator:
            if hasattr(brain3_manager.communicator, 'connected'):
                health_status["components"]["redis"] = {
                    "connected": brain3_manager.communicator.connected,
                    "healthy": brain3_manager.communicator.connected
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

@app.get("/metrics")
async def get_metrics():
    """Service metrics endpoint in Prometheus format"""
    global brain3_manager

    try:
        # Get current timestamp
        current_time = time.time()

        # Default values
        total_requests = 0
        avg_processing_time = 0
        uptime_seconds = 0
        capabilities_count = 0
        active_tasks = 0
        messages_sent = 0
        messages_received = 0
        connected = 0

        # Agent metrics
        if brain3_manager:
            status = brain3_manager.get_status()
            total_requests = status.get("total_requests", 0)
            avg_processing_time = status.get("average_processing_time_ms", 0)
            uptime_seconds = status.get("uptime_seconds", 0)
            capabilities_count = len(status.get("capabilities", []))
            active_tasks = status.get("task_orchestrator", {}).get("active_tasks", 0)

        # Communication metrics
        if brain3_manager and hasattr(brain3_manager, 'communicator') and brain3_manager.communicator:
            if hasattr(brain3_manager.communicator, 'get_stats'):
                comm_stats = brain3_manager.communicator.get_stats()
                messages_sent = comm_stats.get("messages_sent", 0)
                messages_received = comm_stats.get("messages_received", 0)
                connected = 1 if comm_stats.get("connected", False) else 0

        # Prometheus format metrics
        metrics = f"""# HELP brain3_uptime_seconds Service uptime in seconds
# TYPE brain3_uptime_seconds gauge
brain3_uptime_seconds {uptime_seconds}

# HELP brain3_requests_total Total number of requests processed
# TYPE brain3_requests_total counter
brain3_requests_total {total_requests}

# HELP brain3_processing_time_ms Average processing time in milliseconds
# TYPE brain3_processing_time_ms gauge
brain3_processing_time_ms {avg_processing_time}

# HELP brain3_capabilities_count Number of available capabilities
# TYPE brain3_capabilities_count gauge
brain3_capabilities_count {capabilities_count}

# HELP brain3_active_tasks Number of active tasks
# TYPE brain3_active_tasks gauge
brain3_active_tasks {active_tasks}

# HELP brain3_messages_sent_total Total messages sent to other brains
# TYPE brain3_messages_sent_total counter
brain3_messages_sent_total {messages_sent}

# HELP brain3_messages_received_total Total messages received from other brains
# TYPE brain3_messages_received_total counter
brain3_messages_received_total {messages_received}

# HELP brain3_connected Connection status to other brains (1=connected, 0=disconnected)
# TYPE brain3_connected gauge
brain3_connected {connected}
"""

        # Add flow monitoring metrics if available
        if FLOW_MONITORING_AVAILABLE:
            try:
                from prometheus_client import generate_latest, REGISTRY
                flow_metrics = generate_latest(REGISTRY).decode('utf-8')
                metrics += "\n# Flow Monitoring Metrics\n" + flow_metrics
            except Exception as e:
                logger.debug(f"Flow monitoring metrics not available: {e}")

        return Response(content=metrics, media_type="text/plain")

    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        error_metrics = f"""# HELP brain3_error Error status
# TYPE brain3_error gauge
brain3_error 1
"""
        return Response(content=error_metrics, media_type="text/plain")

def get_brain3_manager() -> Brain3Manager:
    """Dependency injection for Brain 3 Manager"""
    global brain3_manager
    if not brain3_manager:
        raise HTTPException(status_code=503, detail="Brain 3 Manager not initialized")
    return brain3_manager

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Brain 3 Zazzles's Agent Service...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8013,
        reload=False,
        workers=1,
        log_level="info"
    )
