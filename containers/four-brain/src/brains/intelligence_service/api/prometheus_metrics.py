"""
Prometheus-compatible metrics endpoint for Brain 3 Augment
Provides metrics in Prometheus format for Alloy collection
"""

from fastapi import APIRouter, Response
from typing import Dict, Any
import time
import psutil
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Global metrics storage
_metrics_storage = {
    "start_time": time.time(),
    "requests_total": 0,
    "processing_time_total": 0.0,
    "tasks_processed_total": 0,
    "k2_requests_total": 0,
    "supabase_operations_total": 0,
    "errors_total": 0,
    "processing_duration_buckets": {
        "0.1": 0, "0.5": 0, "1.0": 0, "2.0": 0, "5.0": 0, "10.0": 0, "+Inf": 0
    }
}

def update_metrics(processing_time: float = 0.0, tasks_processed: int = 0, k2_requests: int = 0, supabase_ops: int = 0, errors: int = 0):
    """Update internal metrics storage"""
    global _metrics_storage
    _metrics_storage["requests_total"] += 1
    _metrics_storage["processing_time_total"] += processing_time
    _metrics_storage["tasks_processed_total"] += tasks_processed
    _metrics_storage["k2_requests_total"] += k2_requests
    _metrics_storage["supabase_operations_total"] += supabase_ops
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

@router.get("/prometheus-metrics")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    
    # Get current metrics
    system_metrics = get_system_metrics()
    current_time = time.time()
    uptime = current_time - _metrics_storage["start_time"]
    
    # Build Prometheus format response
    metrics_lines = []
    
    # Brain 3 specific metrics
    metrics_lines.extend([
        "# HELP brain3_uptime_seconds Brain 3 uptime in seconds",
        "# TYPE brain3_uptime_seconds counter",
        f"brain3_uptime_seconds {uptime:.2f}",
        "",
        "# HELP brain3_requests_total Total number of requests processed",
        "# TYPE brain3_requests_total counter", 
        f"brain3_requests_total {_metrics_storage['requests_total']}",
        "",
        "# HELP brain3_tasks_processed_total Total number of tasks processed",
        "# TYPE brain3_tasks_processed_total counter",
        f"brain3_tasks_processed_total {_metrics_storage['tasks_processed_total']}",
        "",
        "# HELP brain3_k2_requests_total Total number of K2 API requests",
        "# TYPE brain3_k2_requests_total counter",
        f"brain3_k2_requests_total {_metrics_storage['k2_requests_total']}",
        "",
        "# HELP brain3_supabase_operations_total Total number of Supabase operations",
        "# TYPE brain3_supabase_operations_total counter",
        f"brain3_supabase_operations_total {_metrics_storage['supabase_operations_total']}",
        "",
        "# HELP brain3_errors_total Total number of errors",
        "# TYPE brain3_errors_total counter",
        f"brain3_errors_total {_metrics_storage['errors_total']}",
        "",
        "# HELP brain3_processing_time_total Total processing time in seconds",
        "# TYPE brain3_processing_time_total counter",
        f"brain3_processing_time_total {_metrics_storage['processing_time_total']:.2f}",
        "",
        "# HELP brain3_processing_duration_seconds Processing duration histogram",
        "# TYPE brain3_processing_duration_seconds histogram"
    ])

    # Add histogram buckets
    buckets = _metrics_storage["processing_duration_buckets"]
    for bucket, count in buckets.items():
        if bucket == "+Inf":
            metrics_lines.append(f"brain3_processing_duration_seconds_bucket{{le=\"{bucket}\"}} {count}")
        else:
            metrics_lines.append(f"brain3_processing_duration_seconds_bucket{{le=\"{bucket}\"}} {count}")

    # Add histogram count and sum
    metrics_lines.extend([
        f"brain3_processing_duration_seconds_count {_metrics_storage['requests_total']}",
        f"brain3_processing_duration_seconds_sum {_metrics_storage['processing_time_total']:.2f}",
        ""
    ])

    # System metrics
    if system_metrics:
        metrics_lines.extend([
            "# HELP brain3_cpu_percent CPU usage percentage",
            "# TYPE brain3_cpu_percent gauge",
            f"brain3_cpu_percent {system_metrics.get('cpu_percent', 0):.2f}",
            "",
            "# HELP brain3_memory_percent Memory usage percentage", 
            "# TYPE brain3_memory_percent gauge",
            f"brain3_memory_percent {system_metrics.get('memory_percent', 0):.2f}",
            "",
            "# HELP brain3_process_memory_rss Process RSS memory in bytes",
            "# TYPE brain3_process_memory_rss gauge",
            f"brain3_process_memory_rss {system_metrics.get('process_memory_rss', 0)}",
            ""
        ])
    
    # Brain 3 specific operational metrics
    metrics_lines.extend([
        "# HELP brain3_k2_bridge_status K2 Vector Bridge status (1=active, 0=inactive)",
        "# TYPE brain3_k2_bridge_status gauge",
        "brain3_k2_bridge_status 1",
        "",
        "# HELP brain3_supabase_connection_status Supabase connection status (1=connected, 0=disconnected)",
        "# TYPE brain3_supabase_connection_status gauge",
        "brain3_supabase_connection_status 1",
        "",
        "# HELP brain3_redis_connection_status Redis connection status (1=connected, 0=disconnected)",
        "# TYPE brain3_redis_connection_status gauge",
        "brain3_redis_connection_status 1",
        ""
    ])
    
    # Health status
    metrics_lines.extend([
        "# HELP brain3_health_status Brain 3 health status (1=healthy, 0=unhealthy)",
        "# TYPE brain3_health_status gauge",
        "brain3_health_status 1",
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
        "tasks_processed": _metrics_storage["tasks_processed_total"],
        "k2_requests": _metrics_storage["k2_requests_total"],
        "supabase_operations": _metrics_storage["supabase_operations_total"],
        "errors_total": _metrics_storage["errors_total"],
        "timestamp": current_time
    }

# Utility function to be called from other parts of the application
def increment_request_metrics(processing_time: float = 0.0, success: bool = True, tasks_count: int = 0, k2_requests: int = 0, supabase_ops: int = 0):
    """Increment request metrics from other parts of the application"""
    errors = 0 if success else 1
    update_metrics(processing_time, tasks_count, k2_requests, supabase_ops, errors)
