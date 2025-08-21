"""
Metrics endpoint for Brain 1 - Qwen3-4B Embedding Service
Date: 2025-07-19 AEST
Purpose: Expose Prometheus metrics via HTTP endpoint
"""

from fastapi import APIRouter, Response
from ..metrics.prometheus_metrics import get_brain1_metrics
import logging

logger = logging.getLogger(__name__)

# Create metrics router
metrics_router = APIRouter(tags=["metrics"])

@metrics_router.get("/metrics", response_class=Response)
async def get_metrics():
    """
    Expose Prometheus metrics for Brain 1
    
    Returns:
        Response: Prometheus metrics in text format
    """
    try:
        metrics_instance = get_brain1_metrics()
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

@metrics_router.get("/health")
async def metrics_health():
    """
    Health check for metrics endpoint
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": "brain1_metrics",
        "timestamp": "2025-07-19T00:00:00Z"
    }
