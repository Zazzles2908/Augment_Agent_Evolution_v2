"""
Monitoring and metrics API endpoints for Brain 4
Provides REAL system performance and processing metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncpg
import asyncpg.exceptions
import psutil
import logging

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None

from ..utils.performance_monitor import PerformanceMonitor
from ..config.settings import Brain4Settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = Brain4Settings()

async def _get_real_gpu_metrics() -> Dict[str, Any]:
    """Get REAL GPU metrics using nvidia-ml-py"""
    if not NVIDIA_ML_AVAILABLE:
        logger.warning("nvidia-ml-py not available, GPU metrics unavailable")
        return {
            "available": False,
            "error": "nvidia-ml-py not available",
            "monitoring_status": "failed"
        }

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            return {
                "available": False,
                "error": "No GPU devices found",
                "monitoring_status": "no_devices"
            }

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # RTX 5070 Ti

        # Real GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Real GPU memory info
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_mb = memory_info.used // (1024 * 1024)
        memory_total_mb = memory_info.total // (1024 * 1024)
        memory_usage_percent = (memory_info.used / memory_info.total) * 100

        # Real GPU temperature
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        gpu_metrics = {
            "available": True,
            "memory_usage_percent": round(memory_usage_percent, 1),
            "temperature_c": temperature,
            "utilization_percent": utilization.gpu,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "monitoring_status": "active"
        }

        pynvml.nvmlShutdown()
        return gpu_metrics

    except Exception as e:
        logger.error(f"Error getting GPU metrics: {e}")
        # HONEST FAILURE REPORTING - Don't mask GPU monitoring failures
        return {
            "available": False,
            "error": str(e),
            "monitoring_status": "failed"
        }

async def _get_real_processing_metrics() -> Dict[str, Any]:
    """Get REAL processing metrics from database"""
    try:
        conn = await asyncpg.connect(settings.database_url, timeout=5.0)

        # Real task counts from database
        status_query = """
        SELECT processing_status, COUNT(*) as count
        FROM augment_agent.documents
        GROUP BY processing_status
        """

        results = await conn.fetch(status_query)
        await conn.close()

        processing_metrics = {
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_documents": 0,
            "database_status": "connected"
        }

        for row in results:
            status = row['processing_status']
            count = row['count']

            if status == 'processing':
                processing_metrics['active_tasks'] = count
            elif status == 'completed':
                processing_metrics['completed_tasks'] = count
            elif status in ['failed', 'error']:
                processing_metrics['failed_tasks'] += count

            processing_metrics['total_documents'] += count

        return processing_metrics

    except Exception as e:
        logger.error(f"Error getting processing metrics: {e}")
        # HONEST FAILURE REPORTING - Don't mask database failures
        raise RuntimeError(f"Database monitoring failed: {e}") from e

@router.get("/metrics")
async def get_metrics():
    """Get current REAL system metrics"""

    try:
        # REAL system metrics using psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # REAL GPU metrics
        gpu_metrics = await _get_real_gpu_metrics()

        # REAL processing metrics from database
        try:
            processing_metrics = await _get_real_processing_metrics()
        except Exception as e:
            logger.error(f"Processing metrics unavailable: {e}")
            processing_metrics = {
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_documents": 0,
                "database_status": "unavailable"
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_metrics,
            "system": {
                "memory_usage_percent": round(memory.percent, 1),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "cpu_usage_percent": round(cpu_percent, 1),
                "cpu_count": psutil.cpu_count()
            },
            "processing": processing_metrics
        }

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = Query(default=1, le=24),
    metric_type: Optional[str] = Query(default=None)
):
    """Get REAL historical metrics from database"""

    try:
        conn = await asyncpg.connect(settings.database_url, timeout=10.0)

        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Query for historical processing data
        history_query = """
        SELECT
            DATE_TRUNC('minute', processing_timestamp) as time_bucket,
            COUNT(*) as documents_processed,
            NULL::float as avg_processing_time,
            COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as successful_count,
            COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_count
        FROM augment_agent.documents
        WHERE processing_timestamp IS NOT NULL AND processing_timestamp >= $1 AND processing_timestamp <= $2
        GROUP BY time_bucket
        ORDER BY time_bucket
        """

        results = await conn.fetch(history_query, start_time, end_time)
        await conn.close()

        # Convert to time series data
        data_points = []
        for row in results:
            data_point = {
                "timestamp": row['time_bucket'].isoformat(),
                "documents_processed": row['documents_processed'],
                "avg_processing_time": float(row['avg_processing_time']) if row['avg_processing_time'] else 0.0,
                "successful_count": row['successful_count'],
                "failed_count": row['failed_count'],
                "success_rate": (row['successful_count'] / row['documents_processed'] * 100) if row['documents_processed'] > 0 else 0.0
            }

            # Filter by metric type if specified
            if metric_type:
                if metric_type == "processing_time" and row['avg_processing_time']:
                    data_points.append(data_point)
                elif metric_type == "throughput":
                    data_points.append(data_point)
                elif metric_type == "success_rate":
                    data_points.append(data_point)
            else:
                data_points.append(data_point)

        logger.info(f"Retrieved {len(data_points)} historical data points for {hours} hours")

        return {
            "time_window_hours": hours,
            "metric_type": metric_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "data_points": data_points,
            "total_points": len(data_points)
        }

    except asyncpg.exceptions.PostgresConnectionError as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=503, detail="Database error")
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics history: {str(e)}")
