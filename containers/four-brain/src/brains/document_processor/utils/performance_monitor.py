"""
Performance Monitor for Brain 4
Collects and tracks system performance metrics for optimization
"""

import asyncio
import logging
import time
import psutil
import torch
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import deque

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None

class PerformanceMonitor:
    """
    Performance monitoring for Brain 4 system
    Tracks GPU, CPU, memory, and processing metrics
    """
    
    def __init__(self, history_size: int = 100):
        self.logger = logging.getLogger(__name__)
        self.history_size = history_size
        
        # Metric history storage
        self.metric_history = {
            "gpu_usage": deque(maxlen=history_size),
            "cpu_usage": deque(maxlen=history_size),
            "memory_usage": deque(maxlen=history_size),
            "processing_times": deque(maxlen=history_size),
            "throughput": deque(maxlen=history_size)
        }
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.logger.info(f"Performance monitoring initialized for GPU: {torch.cuda.get_device_name(self.gpu_device)}")
        else:
            self.logger.info("Performance monitoring initialized for CPU-only mode")
        
        # Processing statistics
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "peak_memory_usage": 0.0,
            "errors_count": 0
        }
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive system metrics
        
        Returns:
            Dictionary with current system metrics
        """
        
        try:
            timestamp = datetime.now()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            metrics = {
                "timestamp": timestamp.isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                "memory": {
                    "virtual": memory_info._asdict(),
                    "swap": psutil.swap_memory()._asdict()
                },
                "disk": {
                    "usage": psutil.disk_usage('/')._asdict()
                },
                "network": self._get_network_stats(),
                "processing": self.processing_stats.copy()
            }
            
            # GPU metrics if available
            if self.gpu_available:
                metrics["gpu"] = await self._collect_gpu_metrics()
            else:
                metrics["gpu"] = {"available": False}
            
            # Store in history
            self._store_metrics_in_history(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """
        Collect GPU-specific metrics
        
        Returns:
            Dictionary with GPU metrics
        """
        
        try:
            if not self.gpu_available:
                return {"available": False}
            
            # Get GPU memory info
            torch.cuda.synchronize()
            memory_stats = torch.cuda.memory_stats(self.gpu_device)
            device_props = torch.cuda.get_device_properties(self.gpu_device)
            
            total_memory = device_props.total_memory
            allocated_memory = memory_stats.get("allocated_bytes.all.current", 0)
            reserved_memory = memory_stats.get("reserved_bytes.all.current", 0)
            
            gpu_metrics = {
                "available": True,
                "device_name": torch.cuda.get_device_name(self.gpu_device),
                "device_capability": f"{device_props.major}.{device_props.minor}",
                "multiprocessor_count": device_props.multi_processor_count,
                "memory": {
                    "total_gb": total_memory / (1024**3),
                    "allocated_gb": allocated_memory / (1024**3),
                    "reserved_gb": reserved_memory / (1024**3),
                    "free_gb": (total_memory - allocated_memory) / (1024**3),
                    "usage_percent": (allocated_memory / total_memory) * 100
                },
                "utilization": {
                    "gpu_util": self._get_gpu_utilization(),
                    "memory_util": (allocated_memory / total_memory) * 100
                }
            }
            
            # Try to get temperature (may not be available on all systems)
            try:
                if hasattr(torch.cuda, 'temperature'):
                    gpu_metrics["temperature_c"] = torch.cuda.temperature(self.gpu_device)
            except:
                pass
            
            return gpu_metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")
            return {"available": True, "error": str(e)}
    
    def _get_gpu_utilization(self) -> float:
        """
        Get REAL GPU utilization percentage using nvidia-ml-py
        """

        if not self.gpu_available:
            return 0.0

        if not NVIDIA_ML_AVAILABLE:
            # Fallback to memory-based estimation if nvidia-ml-py unavailable
            try:
                memory_stats = torch.cuda.memory_stats(self.gpu_device)
                allocated = memory_stats.get("allocated_bytes.all.current", 0)
                total = torch.cuda.get_device_properties(self.gpu_device).total_memory
                return (allocated / total) * 100
            except:
                return 0.0

        try:
            # REAL GPU utilization using nvidia-ml-py
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device)

            # Get actual GPU utilization rates
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu

            pynvml.nvmlShutdown()
            return float(gpu_util)

        except Exception as e:
            self.logger.warning(f"Error getting real GPU utilization: {e}")
            # Fallback to memory-based estimation
            try:
                memory_stats = torch.cuda.memory_stats(self.gpu_device)
                allocated = memory_stats.get("allocated_bytes.all.current", 0)
                total = torch.cuda.get_device_properties(self.gpu_device).total_memory
                return (allocated / total) * 100
            except:
                return 0.0

    def _get_comprehensive_gpu_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive REAL GPU metrics using nvidia-ml-py
        """
        gpu_metrics = {
            "available": self.gpu_available,
            "utilization_percent": 0.0,
            "memory_usage_percent": 0.0,
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "temperature_c": 0,
            "power_usage_w": 0,
            "fan_speed_percent": 0
        }

        if not self.gpu_available:
            return gpu_metrics

        if not NVIDIA_ML_AVAILABLE:
            # Basic metrics using PyTorch
            try:
                memory_stats = torch.cuda.memory_stats(self.gpu_device)
                allocated = memory_stats.get("allocated_bytes.all.current", 0)
                total = torch.cuda.get_device_properties(self.gpu_device).total_memory

                gpu_metrics.update({
                    "utilization_percent": (allocated / total) * 100,
                    "memory_usage_percent": (allocated / total) * 100,
                    "memory_used_mb": allocated // (1024 * 1024),
                    "memory_total_mb": total // (1024 * 1024)
                })
            except Exception as e:
                self.logger.warning(f"Error getting basic GPU metrics: {e}")
            return gpu_metrics

        try:
            # COMPREHENSIVE GPU metrics using nvidia-ml-py
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device)

            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
            except:
                power_usage = 0

            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                fan_speed = 0

            gpu_metrics.update({
                "utilization_percent": float(utilization.gpu),
                "memory_usage_percent": (memory_info.used / memory_info.total) * 100,
                "memory_used_mb": memory_info.used // (1024 * 1024),
                "memory_total_mb": memory_info.total // (1024 * 1024),
                "temperature_c": temperature,
                "power_usage_w": power_usage,
                "fan_speed_percent": fan_speed
            })

            pynvml.nvmlShutdown()

        except Exception as e:
            self.logger.error(f"Error getting comprehensive GPU metrics: {e}")

        return gpu_metrics

    def _get_network_stats(self) -> Dict[str, Any]:
        """
        Get network statistics
        
        Returns:
            Dictionary with network metrics
        """
        
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            self.logger.error(f"Error getting network stats: {e}")
            return {"error": str(e)}
    
    def _store_metrics_in_history(self, metrics: Dict[str, Any]):
        """
        Store metrics in history for trend analysis
        
        Args:
            metrics: Current metrics to store
        """
        
        try:
            timestamp = datetime.now()
            
            # Store key metrics in history
            self.metric_history["cpu_usage"].append({
                "timestamp": timestamp,
                "value": metrics.get("cpu", {}).get("usage_percent", 0)
            })
            
            self.metric_history["memory_usage"].append({
                "timestamp": timestamp,
                "value": metrics.get("memory", {}).get("virtual", {}).get("percent", 0)
            })
            
            if metrics.get("gpu", {}).get("available"):
                self.metric_history["gpu_usage"].append({
                    "timestamp": timestamp,
                    "value": metrics["gpu"]["memory"]["usage_percent"]
                })
            
        except Exception as e:
            self.logger.error(f"Error storing metrics in history: {e}")
    
    def record_processing_time(self, processing_time: float):
        """
        Record document processing time
        
        Args:
            processing_time: Processing time in seconds
        """
        
        try:
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["average_processing_time"] = (
                self.processing_stats["total_processing_time"] / 
                self.processing_stats["documents_processed"]
            )
            
            # Store in history
            self.metric_history["processing_times"].append({
                "timestamp": datetime.now(),
                "value": processing_time
            })
            
        except Exception as e:
            self.logger.error(f"Error recording processing time: {e}")
    
    def record_error(self):
        """Record processing error"""
        self.processing_stats["errors_count"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary and trends
        
        Returns:
            Dictionary with performance summary
        """
        
        try:
            summary = {
                "current_stats": self.processing_stats.copy(),
                "trends": {},
                "health_indicators": {}
            }
            
            # Calculate trends from history
            for metric_name, history in self.metric_history.items():
                if len(history) > 1:
                    recent_values = [item["value"] for item in list(history)[-10:]]
                    summary["trends"][metric_name] = {
                        "current": recent_values[-1] if recent_values else 0,
                        "average_recent": sum(recent_values) / len(recent_values),
                        "min_recent": min(recent_values),
                        "max_recent": max(recent_values)
                    }
            
            # Health indicators
            if "cpu_usage" in summary["trends"]:
                cpu_avg = summary["trends"]["cpu_usage"]["average_recent"]
                summary["health_indicators"]["cpu_health"] = (
                    "good" if cpu_avg < 70 else "warning" if cpu_avg < 90 else "critical"
                )
            
            if "memory_usage" in summary["trends"]:
                mem_avg = summary["trends"]["memory_usage"]["average_recent"]
                summary["health_indicators"]["memory_health"] = (
                    "good" if mem_avg < 80 else "warning" if mem_avg < 95 else "critical"
                )
            
            if "gpu_usage" in summary["trends"]:
                gpu_avg = summary["trends"]["gpu_usage"]["average_recent"]
                summary["health_indicators"]["gpu_health"] = (
                    "good" if gpu_avg < 75 else "warning" if gpu_avg < 90 else "critical"
                )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific metric
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to return
            
        Returns:
            List of metric data points
        """
        
        try:
            if metric_name not in self.metric_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = self.metric_history[metric_name]
            
            return [
                item for item in history 
                if item["timestamp"] >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting metric history: {e}")
            return []
