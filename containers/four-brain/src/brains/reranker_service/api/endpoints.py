"""
Brain 2 FastAPI Endpoints
API endpoints for Qwen3-Reranker-4B service
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..brain2_manager import Brain2Manager
from ..config.settings import get_brain2_settings
from .models import (
    RerankRequest, RerankResponse, HealthResponse, 
    ModelStatusResponse, TaskRequest, TaskResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/brain2", tags=["Brain 2 - Qwen3 Reranker"])

# Global Brain 2 manager instance - initialized at startup, not on-demand
brain2_manager: Brain2Manager = None


def set_brain2_manager(manager: Brain2Manager):
    """Set the global Brain 2 manager instance - called during startup"""
    global brain2_manager
    brain2_manager = manager
    logger.info("✅ Brain 2 manager instance set for API endpoints")


async def get_brain2_manager() -> Brain2Manager:
    """Dependency to get Brain 2 manager instance - FIXED: No lazy loading"""
    global brain2_manager
    if brain2_manager is None:
        # CRITICAL FIX: Do not initialize here - should be done at startup
        # This was causing lazy loading and breaking sequential GPU memory allocation
        raise HTTPException(
            status_code=503,
            detail="Brain 2 manager not initialized. Service starting up..."
        )
    return brain2_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(manager: Brain2Manager = Depends(get_brain2_manager)):
    """
    Health check endpoint for Brain 2 service
    Returns comprehensive health status including model loading and memory usage
    """
    try:
        health_data = await manager.health_check()
        
        return HealthResponse(
            status="healthy" if health_data.get("healthy", False) else "unhealthy",
            brain_id=manager.settings.brain_id,
            model_loaded=health_data.get("model_loaded", False),
            quantization_enabled=health_data.get("quantization_enabled", False),
            memory_usage=health_data.get("memory_usage", {}),
            uptime_seconds=health_data.get("uptime_seconds", 0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(manager: Brain2Manager = Depends(get_brain2_manager)):
    """
    Get detailed model status and performance metrics
    """
    try:
        status_data = manager.get_status()
        
        return ModelStatusResponse(
            model_name=status_data.get("model_name", ""),
            model_loaded=status_data.get("model_loaded", False),
            quantized=status_data.get("quantization_enabled", False),
            quantization_type=status_data.get("quantization_type"),
            optimization=status_data.get("optimization"),
            loading_time_seconds=status_data.get("loading_time_seconds"),
            memory_usage=status_data.get("memory_usage", {}),
            performance_metrics={
                "total_requests": status_data.get("total_requests", 0),
                "average_processing_time_ms": status_data.get("average_processing_time_ms", 0),
                "uptime_seconds": status_data.get("uptime_seconds", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Model status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model status check failed: {str(e)}")


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(
    request: RerankRequest,
    manager: Brain2Manager = Depends(get_brain2_manager)
):
    """
    Rerank documents based on query relevance
    
    This is the main functionality endpoint for Brain 2.
    Takes a query and list of documents, returns them ranked by relevance.
    """
    try:
        logger.info(f"🔄 Reranking request: {len(request.documents)} documents, top_k={request.top_k}")
        
        # Validate model is loaded
        if not manager.model_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Brain 2 model not loaded. Service unavailable."
            )
        
        # Convert request documents to the format expected by manager
        documents = []
        for doc in request.documents:
            documents.append({
                "text": doc.text,
                "doc_id": doc.doc_id,
                "metadata": doc.metadata or {}
            })
        
        # Perform reranking
        result = await manager.rerank_documents(
            query=request.query,
            documents=documents,
            top_k=request.top_k
        )
        
        # Convert results to response format
        rerank_results = []
        for item in result["results"]:
            rerank_results.append({
                "text": item["text"],
                "relevance_score": item["relevance_score"],
                "rank": item["rank"],
                "doc_id": item.get("doc_id"),
                "metadata": item.get("metadata", {})
            })
        
        return RerankResponse(
            results=rerank_results,
            query=result["query"],
            total_documents=result["total_documents"],
            returned_documents=result["returned_documents"],
            processing_time_ms=result["processing_time_ms"],
            model_info=result["model_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@router.post("/rerank/async", response_model=TaskResponse)
async def rerank_documents_async(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    manager: Brain2Manager = Depends(get_brain2_manager)
):
    """
    Asynchronous document reranking
    
    Submits a reranking task for background processing.
    Useful for large document sets or when immediate response is not required.
    """
    try:
        # Validate model is loaded
        if not manager.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Brain 2 model not loaded. Service unavailable."
            )
        
        # Create task response
        task_response = TaskResponse(
            task_id=request.task_id,
            status="submitted",
            created_at=datetime.now().isoformat(),
            completed_at=None,
            results=None,
            error_message=None
        )
        
        # Add background task
        background_tasks.add_task(
            _process_async_rerank_task,
            request,
            manager
        )
        
        logger.info(f"📋 Async reranking task submitted: {request.task_id}")
        return task_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Async reranking submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get status of an asynchronous reranking task
    
    Note: This is a simplified implementation.
    In production, task status would be stored in Redis or database.
    """
    # This is a placeholder implementation
    # In a real system, you would query task status from Redis or database
    return TaskResponse(
        task_id=task_id,
        status="completed",  # Placeholder
        created_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        results=None,
        error_message=None
    )


@router.get("/metrics")
async def get_metrics(manager: Brain2Manager = Depends(get_brain2_manager)):
    """
    Get Brain 2 performance metrics in Prometheus format
    """
    try:
        status = manager.get_status()
        
        # Simple metrics in Prometheus format
        metrics = f"""# HELP brain2_requests_total Total number of reranking requests
# TYPE brain2_requests_total counter
brain2_requests_total {status.get('total_requests', 0)}

# HELP brain2_processing_time_ms Average processing time in milliseconds
# TYPE brain2_processing_time_ms gauge
brain2_processing_time_ms {status.get('average_processing_time_ms', 0)}

# HELP brain2_memory_usage_mb GPU memory usage in MB
# TYPE brain2_memory_usage_mb gauge
brain2_memory_usage_mb {status.get('memory_usage', {}).get('gpu', {}).get('allocated_mb', 0)}

# HELP brain2_model_loaded Whether the model is loaded
# TYPE brain2_model_loaded gauge
brain2_model_loaded {1 if status.get('model_loaded', False) else 0}
"""
        
        return JSONResponse(
            content=metrics,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


async def _process_async_rerank_task(request: TaskRequest, manager: Brain2Manager):
    """
    Background task processor for async reranking
    
    This would typically update task status in Redis or database.
    """
    try:
        logger.info(f"🔄 Processing async task: {request.task_id}")
        
        # Convert request documents
        documents = []
        for doc in request.documents:
            documents.append({
                "text": doc.text,
                "doc_id": doc.doc_id,
                "metadata": doc.metadata or {}
            })
        
        # Perform reranking
        result = await manager.rerank_documents(
            query=request.query,
            documents=documents,
            top_k=request.top_k
        )
        
        logger.info(f"✅ Async task completed: {request.task_id}")
        
        # In production, update task status in Redis/database
        # and optionally call callback URL if provided
        
    except Exception as e:
        logger.error(f"❌ Async task failed: {request.task_id} - {e}")
        # In production, update task status with error
