"""
Metrics Collector - System Metrics Gathering
Collects comprehensive system metrics for the Four-Brain system

This module provides system metrics collection including CPU, memory, GPU,
network, and application-specific metrics for monitoring and optimization.

Created: 2025-07-29 AEST
Purpose: Comprehensive system metrics collection
Module Size: 150 lines (modular design)
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[List[float]] = None


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    brain_id: str
    active_connections: int
    requests_per_second: float
    response_time_avg: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    custom_metrics: Dict[str, Any]


class MetricsCollector:
    """
    System Metrics Collector
    
    Provides comprehensive metrics collection for system monitoring,
    performance analysis, and optimization recommendations.
    """
    
    def __init__(self, brain_id: str, collection_interval: float = 30.0):
        """Initialize metrics collector"""
        self.brain_id = brain_id
        self.collection_interval = collection_interval
        self.enabled = True
        
        # Metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.application_metrics: List[ApplicationMetrics] = []
        self.max_metrics = 1000  # Limit memory usage
        
        # Collection thread
        self._collection_thread = None
        self._stop_event = threading.Event()
        
        # Application metrics tracking
        self.app_counters = {
            "requests_total": 0,
            "errors_total": 0,
            "connections_active": 0,
            "response_times": []
        }
        
        # Custom metrics registry
        self.custom_metrics = {}
        self.metric_callbacks = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"📊 Metrics Collector initialized for {brain_id}")
    
    def start_collection(self):
        """Start automatic metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            logger.warning("⚠️ Metrics collection already running")
            return
        
        self._stop_event.clear()
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        
        logger.info(f"🚀 Metrics collection started (interval: {self.collection_interval}s)")
    
    def stop_collection(self):
        """Stop automatic metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_event.set()
            self._collection_thread.join(timeout=5)
            logger.info("🛑 Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_event.is_set():
            try:
                if self.enabled:
                    self.collect_system_metrics()
                    self.collect_application_metrics()
                
                # Wait for next collection interval
                self._stop_event.wait(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                self._stop_event.wait(5)  # Wait before retrying
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems only)
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
                
                # Maintain size limit
                if len(self.system_metrics) > self.max_metrics:
                    self.system_metrics = self.system_metrics[-self.max_metrics:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
            raise
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            with self._lock:
                # Calculate response time average
                response_times = self.app_counters["response_times"]
                response_time_avg = sum(response_times) / len(response_times) if response_times else 0.0
                
                # Calculate error rate
                total_requests = self.app_counters["requests_total"]
                total_errors = self.app_counters["errors_total"]
                error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
                
                # Get current process metrics
                current_process = psutil.Process()
                memory_usage_mb = current_process.memory_info().rss / (1024**2)
                cpu_usage_percent = current_process.cpu_percent()
                
                # Collect custom metrics
                custom_metrics = {}
                for name, callback in self.metric_callbacks.items():
                    try:
                        custom_metrics[name] = callback()
                    except Exception as e:
                        logger.warning(f"⚠️ Custom metric '{name}' collection failed: {e}")
                        custom_metrics[name] = None
                
                # Add static custom metrics
                custom_metrics.update(self.custom_metrics)
                
                metrics = ApplicationMetrics(
                    timestamp=time.time(),
                    brain_id=self.brain_id,
                    active_connections=self.app_counters["connections_active"],
                    requests_per_second=0.0,  # TODO: Calculate based on time window
                    response_time_avg=response_time_avg,
                    error_rate=error_rate,
                    memory_usage_mb=memory_usage_mb,
                    cpu_usage_percent=cpu_usage_percent,
                    custom_metrics=custom_metrics
                )
                
                self.application_metrics.append(metrics)
                
                # Maintain size limit
                if len(self.application_metrics) > self.max_metrics:
                    self.application_metrics = self.application_metrics[-self.max_metrics:]
                
                # Clear response times to prevent memory growth
                self.app_counters["response_times"] = []
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Application metrics collection failed: {e}")
            raise
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metrics"""
        with self._lock:
            self.app_counters["requests_total"] += 1
            self.app_counters["response_times"].append(response_time)
            
            if not success:
                self.app_counters["errors_total"] += 1
    
    def record_connection_change(self, delta: int):
        """Record connection count change"""
        with self._lock:
            self.app_counters["connections_active"] = max(0, self.app_counters["connections_active"] + delta)
    
    def set_custom_metric(self, name: str, value: Any):
        """Set a custom metric value"""
        with self._lock:
            self.custom_metrics[name] = value
    
    def register_metric_callback(self, name: str, callback: Callable[[], Any]):
        """Register a callback for dynamic metric collection"""
        self.metric_callbacks[name] = callback
        logger.info(f"📊 Registered metric callback: {name}")
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics"""
        with self._lock:
            return self.system_metrics[-1] if self.system_metrics else None
    
    def get_latest_application_metrics(self) -> Optional[ApplicationMetrics]:
        """Get the latest application metrics"""
        with self._lock:
            return self.application_metrics[-1] if self.application_metrics else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            system_latest = self.system_metrics[-1] if self.system_metrics else None
            app_latest = self.application_metrics[-1] if self.application_metrics else None
            
            return {
                "brain_id": self.brain_id,
                "collection_enabled": self.enabled,
                "collection_interval": self.collection_interval,
                "metrics_collected": {
                    "system_metrics_count": len(self.system_metrics),
                    "application_metrics_count": len(self.application_metrics)
                },
                "latest_system_metrics": asdict(system_latest) if system_latest else None,
                "latest_application_metrics": asdict(app_latest) if app_latest else None,
                "counters": self.app_counters.copy()
            }
    
    def get_historical_metrics(self, metric_type: str, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical metrics for specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self._lock:
            if metric_type == "system":
                metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
                return [asdict(m) for m in metrics]
            elif metric_type == "application":
                metrics = [m for m in self.application_metrics if m.timestamp >= cutoff_time]
                return [asdict(m) for m in metrics]
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics"""
        with self._lock:
            data = {
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "application_metrics": [asdict(m) for m in self.application_metrics],
                "summary": self.get_metrics_summary()
            }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self):
        """Clear stored metrics"""
        with self._lock:
            self.system_metrics.clear()
            self.application_metrics.clear()
            logger.info(f"🧹 Metrics cleared for {self.brain_id}")


# Factory function for easy creation
def create_metrics_collector(brain_id: str, collection_interval: float = 30.0) -> MetricsCollector:
    """Factory function to create metrics collector"""
    return MetricsCollector(brain_id, collection_interval)
