"""
Coordination API for Four-Brain System v2
RESTful API interface for brain coordination management

Created: 2025-07-30 AEST
Purpose: Provide unified API access to all coordination components
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

# Import coordination components
from .brain_coordinator import brain_coordinator, initialize_brain_coordinator
from .load_balancer import load_balancer, initialize_load_balancer
from .task_scheduler import task_scheduler, initialize_task_scheduler, TaskPriority
from .failover_manager import failover_manager, initialize_failover_manager
from .brain_health_monitor import brain_health_monitor, initialize_brain_health_monitor
from .coordination_metrics import coordination_metrics, initialize_coordination_metrics
from .result_aggregator import result_aggregator, initialize_result_aggregator, AggregationStrategy, ResultType
from .coordination_config import coordination_config_manager, initialize_coordination_config_manager
from .brain_discovery import brain_discovery, initialize_brain_discovery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Four-Brain Coordination API",
    description="Unified API for managing brain coordination in Four-Brain System v2",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TaskSubmissionRequest(BaseModel):
    task_type: str
    input_data: Dict[str, Any]
    priority: str = "normal"
    estimated_duration: float = 60.0
    deadline: Optional[str] = None
    dependencies: List[str] = []
    metadata: Dict[str, Any] = {}

class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: str
    message: str

class BrainRegistrationRequest(BaseModel):
    brain_id: str
    brain_type: str
    endpoint: str
    port: int
    capabilities: List[str]
    version: str = "1.0.0"
    metadata: Dict[str, Any] = {}

class AggregationRequest(BaseModel):
    task_id: str
    result_type: str
    strategy: str = "confidence_weighted"
    weights: Optional[Dict[str, float]] = None
    quality_requirements: Dict[str, float] = {}
    timeout_seconds: float = 30.0

class ConfigUpdateRequest(BaseModel):
    config_id: str
    value: Any
    scope: str = "global"
    updated_by: str = "api"

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

# Global state
coordination_initialized = False

async def initialize_coordination_system():
    """Initialize all coordination components"""
    global coordination_initialized
    
    if coordination_initialized:
        return
    
    try:
        logger.info("🚀 Initializing Four-Brain Coordination System...")
        
        # Initialize all coordination components
        await initialize_brain_coordinator()
        await initialize_load_balancer()
        await initialize_task_scheduler()
        await initialize_failover_manager()
        await initialize_brain_health_monitor()
        await initialize_coordination_metrics()
        await initialize_result_aggregator()
        await initialize_coordination_config_manager()
        await initialize_brain_discovery()
        
        coordination_initialized = True
        logger.info("✅ Four-Brain Coordination System initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize coordination system: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize coordination system on startup"""
    await initialize_coordination_system()

# Health and Status Endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Get system health status"""
    try:
        components = {
            "brain_coordinator": "healthy",
            "load_balancer": "healthy",
            "task_scheduler": "healthy",
            "failover_manager": "healthy",
            "health_monitor": "healthy",
            "metrics_collector": "healthy",
            "result_aggregator": "healthy",
            "config_manager": "healthy",
            "brain_discovery": "healthy"
        }
        
        # Get basic metrics
        metrics = {
            "active_brains": len(await brain_discovery.get_discovered_brains()),
            "pending_tasks": (await task_scheduler.get_scheduler_metrics())["queue_length"],
            "system_uptime": "operational"
        }
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get metrics from all components
        brain_metrics = await brain_coordinator.get_coordination_metrics()
        scheduler_metrics = await task_scheduler.get_scheduler_metrics()
        health_metrics = await brain_health_monitor.get_health_summary()
        discovery_metrics = await brain_discovery.get_discovery_metrics()
        
        return {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "brain_coordination": brain_metrics,
            "task_scheduling": scheduler_metrics,
            "health_monitoring": health_metrics,
            "brain_discovery": discovery_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Brain Management Endpoints
@app.post("/brains/register")
async def register_brain(request: BrainRegistrationRequest):
    """Register a new brain instance"""
    try:
        # Convert string brain_type to enum
        from .brain_coordinator import BrainType
        brain_type = BrainType(request.brain_type)
        
        success = await brain_coordinator.register_brain(
            request.brain_id,
            brain_type,
            request.endpoint,
            request.port,
            request.capabilities,
            request.version,
            request.metadata
        )
        
        if success:
            return {"status": "success", "message": f"Brain {request.brain_id} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Brain registration failed")
            
    except Exception as e:
        logger.error(f"❌ Brain registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/brains")
async def list_brains():
    """List all discovered and registered brains"""
    try:
        discovered_brains = await brain_discovery.get_discovered_brains()
        
        brains = []
        for brain in discovered_brains:
            brains.append({
                "brain_id": brain.brain_id,
                "brain_type": brain.brain_type.value,
                "host": brain.host,
                "port": brain.port,
                "status": brain.status.value,
                "capabilities": brain.capabilities,
                "last_seen": brain.last_seen.isoformat(),
                "discovery_method": brain.discovery_method.value
            })
        
        return {"brains": brains, "total": len(brains)}
        
    except Exception as e:
        logger.error(f"❌ Brain listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/brains/{brain_id}")
async def get_brain_details(brain_id: str):
    """Get detailed information about a specific brain"""
    try:
        brain = await brain_discovery.get_brain_by_id(brain_id)
        if not brain:
            raise HTTPException(status_code=404, detail="Brain not found")
        
        # Get health information
        health_profile = await brain_health_monitor.get_brain_health(brain_id)
        
        return {
            "brain_id": brain.brain_id,
            "brain_type": brain.brain_type.value,
            "host": brain.host,
            "port": brain.port,
            "status": brain.status.value,
            "capabilities": brain.capabilities,
            "version": brain.version,
            "endpoints": brain.endpoints,
            "discovered_at": brain.discovered_at.isoformat(),
            "last_seen": brain.last_seen.isoformat(),
            "health_profile": health_profile.__dict__ if health_profile else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Brain details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Task Management Endpoints
@app.post("/tasks/submit", response_model=TaskSubmissionResponse)
async def submit_task(request: TaskSubmissionRequest):
    """Submit a new task for processing"""
    try:
        # Convert priority string to enum
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT,
            "critical": TaskPriority.CRITICAL
        }
        priority = priority_map.get(request.priority.lower(), TaskPriority.NORMAL)
        
        # Parse deadline if provided
        deadline = None
        if request.deadline:
            deadline = datetime.fromisoformat(request.deadline)
        
        # Submit task
        task_id = await task_scheduler.schedule_task(
            request.task_type,
            request.input_data,
            priority,
            request.estimated_duration,
            deadline,
            request.dependencies,
            metadata=request.metadata
        )
        
        if task_id:
            return TaskSubmissionResponse(
                task_id=task_id,
                status="scheduled",
                message="Task submitted successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Task submission failed")
            
    except Exception as e:
        logger.error(f"❌ Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    try:
        task_status = await task_scheduler.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a scheduled or running task"""
    try:
        success = await task_scheduler.cancel_task(task_id)
        if success:
            return {"status": "success", "message": f"Task {task_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Task cancellation failed")
            
    except Exception as e:
        logger.error(f"❌ Task cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Result Aggregation Endpoints
@app.post("/aggregation/submit")
async def submit_aggregation_request(request: AggregationRequest):
    """Submit request for result aggregation"""
    try:
        # Convert string enums
        result_type = ResultType(request.result_type)
        strategy = AggregationStrategy(request.strategy)
        
        request_id = await result_aggregator.submit_aggregation_request(
            request.task_id,
            result_type,
            strategy,
            request.weights,
            request.quality_requirements,
            request.timeout_seconds
        )
        
        return {"request_id": request_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"❌ Aggregation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/aggregation/{request_id}")
async def get_aggregation_result(request_id: str):
    """Get aggregated result"""
    try:
        result = await result_aggregator.get_aggregated_result(request_id)
        if not result:
            raise HTTPException(status_code=404, detail="Aggregation result not found")
        
        return {
            "request_id": result.request_id,
            "task_id": result.task_id,
            "result_type": result.result_type.value,
            "strategy_used": result.strategy_used.value,
            "aggregated_data": result.aggregated_data,
            "confidence": result.confidence,
            "quality_score": result.quality_score,
            "contributing_brains": result.contributing_brains,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Aggregation result retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Management Endpoints
@app.get("/config")
async def get_all_configurations():
    """Get all coordination configurations"""
    try:
        summary = await coordination_config_manager.get_configuration_summary()
        return summary
        
    except Exception as e:
        logger.error(f"❌ Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/{config_id}")
async def get_configuration(config_id: str, scope: str = "global"):
    """Get specific configuration value"""
    try:
        from .coordination_config import ConfigScope
        config_scope = ConfigScope(scope)
        
        value = await coordination_config_manager.get_config(config_id, config_scope)
        if value is None:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return {"config_id": config_id, "scope": scope, "value": value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/config/{config_id}")
async def update_configuration(config_id: str, request: ConfigUpdateRequest):
    """Update configuration value"""
    try:
        from .coordination_config import ConfigScope
        config_scope = ConfigScope(request.scope)
        
        success = await coordination_config_manager.set_config(
            config_id, request.value, config_scope, request.updated_by
        )
        
        if success:
            return {"status": "success", "message": f"Configuration {config_id} updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Configuration update failed")
            
    except Exception as e:
        logger.error(f"❌ Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics and Monitoring Endpoints
@app.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    try:
        metrics = {
            "coordination": await brain_coordinator.get_coordination_metrics(),
            "load_balancing": await load_balancer.get_load_balancing_metrics(),
            "task_scheduling": await task_scheduler.get_scheduler_metrics(),
            "health_monitoring": await brain_health_monitor.get_health_summary(),
            "result_aggregation": await result_aggregator.get_aggregation_metrics(),
            "brain_discovery": await brain_discovery.get_discovery_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get performance-specific metrics"""
    try:
        # Generate performance report
        report = await coordination_metrics.generate_coordination_report(timedelta(hours=1))
        
        return {
            "report_id": report.report_id,
            "time_period_hours": 1,
            "performance_summary": report.performance_summary,
            "recommendations": report.recommendations,
            "alerts": report.alerts,
            "generated_at": report.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Failover Management Endpoints
@app.post("/failover/{brain_id}")
async def trigger_manual_failover(brain_id: str, reason: str = "Manual trigger"):
    """Trigger manual failover for a brain"""
    try:
        from .failover_manager import FailoverTrigger
        
        success = await failover_manager.trigger_failover(
            brain_id, FailoverTrigger.MANUAL_TRIGGER, reason, manual=True
        )
        
        if success:
            return {"status": "success", "message": f"Failover triggered for brain {brain_id}"}
        else:
            raise HTTPException(status_code=400, detail="Failover trigger failed")
            
    except Exception as e:
        logger.error(f"❌ Manual failover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/failover/status")
async def get_failover_status():
    """Get current failover status"""
    try:
        status = await failover_manager.get_failover_status()
        return status
        
    except Exception as e:
        logger.error(f"❌ Failover status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Main application entry point
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Four-Brain Coordination API...")
    uvicorn.run(
        "coordination_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
