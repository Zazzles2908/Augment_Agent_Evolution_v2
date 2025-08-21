"""
Prometheus-compatible metrics endpoint for Brain 4 Docling
Provides metrics in Prometheus format for Alloy collection
"""

from fastapi import APIRouter, Response
from typing import Dict, Any
import time
import psutil
import logging

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None

router = APIRouter()
logger = logging.getLogger(__name__)

# Global metrics storage
_metrics_storage = {
    "start_time": time.time(),
    "requests_total": 0,
    "processing_time_total": 0.0,
    "documents_processed_total": 0,
    "errors_total": 0,
    "processing_duration_buckets": {
        "0.1": 0, "0.5": 0, "1.0": 0, "2.0": 0, "5.0": 0, "10.0": 0, "+Inf": 0
    }
}

def update_metrics(processing_time: float = 0.0, documents_processed: int = 0, errors: int = 0):
    """Update internal metrics storage"""
    global _metrics_storage
    _metrics_storage["requests_total"] += 1
    _metrics_storage["processing_time_total"] += processing_time
    _metrics_storage["documents_processed_total"] += documents_processed
    _metrics_storage["errors_total"] += errors

    # Update histogram buckets for processing duration
    if processing_time > 0:
        buckets = _metrics_storage["processing_duration_buckets"]
        if processing_time <= 0.1:
            buckets["0.1"] += 1
        if processing_time <= 0.5:
            buckets["0.5"] += 1
        if processing_time <= 1.0:
            buckets["1.0"] += 1
        if processing_time <= 2.0:
            buckets["2.0"] += 1
        if processing_time <= 5.0:
            buckets["5.0"] += 1
        if processing_time <= 10.0:
            buckets["10.0"] += 1
        buckets["+Inf"] += 1

def get_system_metrics() -> Dict[str, Any]:
    """Get basic system metrics"""
    try:
        # CPU and Memory metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "process_memory_rss": process_memory.rss,
            "process_memory_vms": process_memory.vms,
            "process_cpu_percent": process.cpu_percent()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {}

def get_gpu_metrics() -> Dict[str, Any]:
    """Get GPU metrics if available"""
    if not NVIDIA_ML_AVAILABLE:
        return {"gpu_available": False}
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            return {"gpu_available": False}
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU metrics
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        
        return {
            "gpu_available": True,
            "gpu_memory_total": memory_info.total,
            "gpu_memory_used": memory_info.used,
            "gpu_memory_free": memory_info.free,
            "gpu_utilization": utilization.gpu,
            "gpu_memory_utilization": utilization.memory,
            "gpu_temperature": temperature,
            "gpu_power_usage": power
        }
    except Exception as e:
        logger.error(f"Error getting GPU metrics: {e}")
        return {"gpu_available": False, "error": str(e)}

@router.get("/prometheus-metrics")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    
    # Get current metrics
    system_metrics = get_system_metrics()
    gpu_metrics = get_gpu_metrics()
    current_time = time.time()
    uptime = current_time - _metrics_storage["start_time"]
    
    # Build Prometheus format response
    metrics_lines = []
    
    # Brain 4 specific metrics
    metrics_lines.extend([
        "# HELP brain4_uptime_seconds Brain 4 uptime in seconds",
        "# TYPE brain4_uptime_seconds counter",
        f"brain4_uptime_seconds {uptime:.2f}",
        "",
        "# HELP brain4_requests_total Total number of requests processed",
        "# TYPE brain4_requests_total counter", 
        f"brain4_requests_total {_metrics_storage['requests_total']}",
        "",
        "# HELP brain4_document_processing_total Total number of documents processed",
        "# TYPE brain4_document_processing_total counter",
        f"brain4_document_processing_total {_metrics_storage['documents_processed_total']}",
        "",
        "# HELP brain4_errors_total Total number of errors",
        "# TYPE brain4_errors_total counter",
        f"brain4_errors_total {_metrics_storage['errors_total']}",
        "",
        "# HELP brain4_processing_time_total Total processing time in seconds",
        "# TYPE brain4_processing_time_total counter",
        f"brain4_processing_time_total {_metrics_storage['processing_time_total']:.2f}",
        "",
        "# HELP brain4_processing_duration_seconds Processing duration histogram",
        "# TYPE brain4_processing_duration_seconds histogram"
    ])

    # Add histogram buckets
    buckets = _metrics_storage["processing_duration_buckets"]
    for bucket, count in buckets.items():
        if bucket == "+Inf":
            metrics_lines.append(f"brain4_processing_duration_seconds_bucket{{le=\"{bucket}\"}} {count}")
        else:
            metrics_lines.append(f"brain4_processing_duration_seconds_bucket{{le=\"{bucket}\"}} {count}")

    # Add histogram count and sum
    metrics_lines.extend([
        f"brain4_processing_duration_seconds_count {_metrics_storage['requests_total']}",
        f"brain4_processing_duration_seconds_sum {_metrics_storage['processing_time_total']:.2f}",
        ""
    ])

    # System metrics
    if system_metrics:
        metrics_lines.extend([
            "# HELP brain4_cpu_percent CPU usage percentage",
            "# TYPE brain4_cpu_percent gauge",
            f"brain4_cpu_percent {system_metrics.get('cpu_percent', 0):.2f}",
            "",
            "# HELP brain4_memory_percent Memory usage percentage", 
            "# TYPE brain4_memory_percent gauge",
            f"brain4_memory_percent {system_metrics.get('memory_percent', 0):.2f}",
            "",
            "# HELP brain4_process_memory_rss Process RSS memory in bytes",
            "# TYPE brain4_process_memory_rss gauge",
            f"brain4_process_memory_rss {system_metrics.get('process_memory_rss', 0)}",
            ""
        ])
    
    # GPU metrics
    if gpu_metrics.get("gpu_available", False):
        metrics_lines.extend([
            "# HELP brain4_gpu_memory_used GPU memory used in bytes",
            "# TYPE brain4_gpu_memory_used gauge",
            f"brain4_gpu_memory_used {gpu_metrics.get('gpu_memory_used', 0)}",
            "",
            "# HELP brain4_gpu_memory_total GPU memory total in bytes",
            "# TYPE brain4_gpu_memory_total gauge", 
            f"brain4_gpu_memory_total {gpu_metrics.get('gpu_memory_total', 0)}",
            "",
            "# HELP brain4_gpu_utilization GPU utilization percentage",
            "# TYPE brain4_gpu_utilization gauge",
            f"brain4_gpu_utilization {gpu_metrics.get('gpu_utilization', 0)}",
            "",
            "# HELP brain4_gpu_temperature GPU temperature in Celsius",
            "# TYPE brain4_gpu_temperature gauge",
            f"brain4_gpu_temperature {gpu_metrics.get('gpu_temperature', 0)}",
            "",
            "# HELP brain4_gpu_power_usage GPU power usage in watts",
            "# TYPE brain4_gpu_power_usage gauge",
            f"brain4_gpu_power_usage {gpu_metrics.get('gpu_power_usage', 0):.2f}",
            ""
        ])
    else:
        metrics_lines.extend([
            "# HELP brain4_gpu_available GPU availability status",
            "# TYPE brain4_gpu_available gauge",
            "brain4_gpu_available 0",
            ""
        ])
    
    # Health status
    metrics_lines.extend([
        "# HELP brain4_health_status Brain 4 health status (1=healthy, 0=unhealthy)",
        "# TYPE brain4_health_status gauge",
        "brain4_health_status 1",
        ""
    ])
    
    # Join all metrics
    prometheus_output = "\n".join(metrics_lines)
    
    return Response(
        content=prometheus_output,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@router.get("/health-metrics")
async def get_health_metrics():
    """Simple health check with basic metrics"""
    current_time = time.time()
    uptime = current_time - _metrics_storage["start_time"]
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "requests_total": _metrics_storage["requests_total"],
        "documents_processed": _metrics_storage["documents_processed_total"],
        "errors_total": _metrics_storage["errors_total"],
        "timestamp": current_time
    }

# Utility function to be called from other parts of the application
def increment_request_metrics(processing_time: float = 0.0, success: bool = True, documents_count: int = 0):
    """Increment request metrics from other parts of the application"""
    errors = 0 if success else 1
    update_metrics(processing_time, documents_count, errors)
