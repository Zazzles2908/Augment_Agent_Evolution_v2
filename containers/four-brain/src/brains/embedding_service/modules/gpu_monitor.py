"""
GPU Monitoring Module for RTX 5070 Ti Four-Brain System
Implements comprehensive GPU metrics collection and Prometheus integration

Created: 2025-08-04 AEST
Author: AugmentAI - Four-Brain Architecture Monitoring
"""

import logging
import time
import torch
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Prometheus metrics imports with fallback
try:
    from prometheus_client import Gauge, Counter, Histogram, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus metrics available")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ Prometheus not available - metrics collection disabled")

# NVIDIA ML imports for detailed GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
    # Do NOT call nvmlInit() at import time; initialize lazily in __init__
    logger.info("ℹ️ NVIDIA ML Python bindings detected (lazy init)")
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("⚠️ NVIDIA ML not available - using basic GPU monitoring")


@dataclass
class GPUMetrics:
    """GPU metrics data structure"""
    memory_used: float
    memory_total: float
    memory_utilization: float
    gpu_utilization: float
    temperature: float
    power_usage: float
    compute_capability: tuple
    device_name: str
    timestamp: datetime


@dataclass
class BrainMetrics:
    """Brain-specific metrics data structure"""
    brain_id: str
    memory_allocated: float
    memory_reserved: float
    memory_fraction: float
    inference_count: int
    avg_inference_time: float
    model_loaded: bool
    tensorrt_enabled: bool
    quantization_type: str


class GPUMonitor:
    """
    Comprehensive GPU monitoring for RTX 5070 Ti Four-Brain System
    Collects GPU metrics, brain-specific metrics, and exports to Prometheus
    """
    
    def __init__(self, brain_id: str = "brain1", prometheus_port: int = 8001):
        """Initialize GPU Monitor"""
        self.brain_id = brain_id
        self.prometheus_port = prometheus_port
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history: List[GPUMetrics] = []
        self.brain_metrics_history: List[BrainMetrics] = []
        
        # Initialize NVIDIA ML if available
        self.nvml_handle = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("✅ NVIDIA ML GPU handle initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize NVIDIA ML handle: {e}")
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info(f"🔍 GPU Monitor initialized for {brain_id}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # GPU Hardware Metrics
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes', 
            'GPU memory used in bytes',
            ['device', 'brain_id']
        )
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_bytes', 
            'GPU total memory in bytes',
            ['device', 'brain_id']
        )
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent', 
            'GPU utilization percentage',
            ['device', 'brain_id']
        )
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius', 
            'GPU temperature in Celsius',
            ['device', 'brain_id']
        )
        self.gpu_power_usage = Gauge(
            'gpu_power_usage_watts', 
            'GPU power usage in watts',
            ['device', 'brain_id']
        )
        
        # Brain-Specific Metrics
        self.brain_memory_allocated = Gauge(
            'brain_memory_allocated_bytes', 
            'Memory allocated by brain service',
            ['brain_id']
        )
        self.brain_memory_fraction = Gauge(
            'brain_memory_fraction', 
            'Memory fraction allocated to brain',
            ['brain_id']
        )
        self.brain_inference_count = Counter(
            'brain_inference_total', 
            'Total number of inferences performed',
            ['brain_id']
        )
        self.brain_inference_time = Histogram(
            'brain_inference_duration_seconds', 
            'Time spent on inference',
            ['brain_id', 'model_type']
        )
        
        # TensorRT Metrics
        self.tensorrt_engine_build_time = Histogram(
            'tensorrt_engine_build_duration_seconds', 
            'Time to build TensorRT engine',
            ['brain_id', 'precision']
        )
        self.tensorrt_inference_speedup = Gauge(
            'tensorrt_inference_speedup_ratio', 
            'TensorRT inference speedup ratio vs PyTorch',
            ['brain_id', 'precision']
        )
        
        # System Health Metrics
        self.brain_health_status = Gauge(
            'brain_health_status', 
            'Brain service health status (1=healthy, 0=unhealthy)',
            ['brain_id']
        )
        
        logger.info("✅ Prometheus metrics initialized")
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics"""
        try:
            if not torch.cuda.is_available():
                return None
            
            # Basic PyTorch metrics
            memory_used = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            device_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            
            # Advanced metrics via NVIDIA ML
            gpu_util = 0.0
            temperature = 0.0
            power_usage = 0.0
            
            if NVML_AVAILABLE and self.nvml_handle:
                try:
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    gpu_util = util_info.gpu
                    
                    temperature = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Convert to watts
                except Exception as e:
                    logger.debug(f"NVML metrics collection failed: {e}")
            
            metrics = GPUMetrics(
                memory_used=memory_used,
                memory_total=memory_total,
                memory_utilization=(memory_used / memory_total) * 100,
                gpu_utilization=gpu_util,
                temperature=temperature,
                power_usage=power_usage,
                compute_capability=compute_cap,
                device_name=device_name,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error collecting GPU metrics: {e}")
            return None
    
    def get_brain_metrics(self) -> Optional[BrainMetrics]:
        """Get brain-specific metrics"""
        try:
            if not torch.cuda.is_available():
                return None
            
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = memory_allocated / memory_total
            
            # Get memory fraction from environment
            import os
            target_fraction = float(os.getenv("TORCH_CUDA_MEMORY_FRACTION", "0.35"))
            
            metrics = BrainMetrics(
                brain_id=self.brain_id,
                memory_allocated=memory_allocated,
                memory_reserved=memory_reserved,
                memory_fraction=memory_fraction,
                inference_count=0,  # Will be updated by inference tracking
                avg_inference_time=0.0,  # Will be updated by inference tracking
                model_loaded=memory_allocated > 1024 * 1024 * 1024,  # >1GB indicates model loaded
                tensorrt_enabled=False,  # Will be updated by TensorRT integration
                quantization_type="8bit"  # Default, will be updated
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error collecting brain metrics: {e}")
            return None
    
    def update_prometheus_metrics(self):
        """Update Prometheus metrics with current values"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Get current metrics
            gpu_metrics = self.get_gpu_metrics()
            brain_metrics = self.get_brain_metrics()
            
            if gpu_metrics:
                device_name = gpu_metrics.device_name.replace(" ", "_")
                
                # Update GPU metrics
                self.gpu_memory_used.labels(device=device_name, brain_id=self.brain_id).set(gpu_metrics.memory_used)
                self.gpu_memory_total.labels(device=device_name, brain_id=self.brain_id).set(gpu_metrics.memory_total)
                self.gpu_utilization.labels(device=device_name, brain_id=self.brain_id).set(gpu_metrics.gpu_utilization)
                self.gpu_temperature.labels(device=device_name, brain_id=self.brain_id).set(gpu_metrics.temperature)
                self.gpu_power_usage.labels(device=device_name, brain_id=self.brain_id).set(gpu_metrics.power_usage)
            
            if brain_metrics:
                # Update brain metrics
                self.brain_memory_allocated.labels(brain_id=self.brain_id).set(brain_metrics.memory_allocated)
                self.brain_memory_fraction.labels(brain_id=self.brain_id).set(brain_metrics.memory_fraction)
                self.brain_health_status.labels(brain_id=self.brain_id).set(1 if brain_metrics.model_loaded else 0)
            
        except Exception as e:
            logger.error(f"❌ Error updating Prometheus metrics: {e}")
    
    def start_monitoring(self, interval: float = 10.0):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            logger.info(f"🔍 Starting GPU monitoring loop (interval: {interval}s)")
            while self.monitoring_active:
                try:
                    self.update_prometheus_metrics()
                    
                    # Store metrics history (keep last 1000 entries)
                    gpu_metrics = self.get_gpu_metrics()
                    if gpu_metrics:
                        self.metrics_history.append(gpu_metrics)
                        if len(self.metrics_history) > 1000:
                            self.metrics_history.pop(0)
                    
                    brain_metrics = self.get_brain_metrics()
                    if brain_metrics:
                        self.brain_metrics_history.append(brain_metrics)
                        if len(self.brain_metrics_history) > 1000:
                            self.brain_metrics_history.pop(0)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("✅ GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("🛑 GPU monitoring stopped")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_memory_util = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_temperature = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
        avg_power = sum(m.power_usage for m in recent_metrics) / len(recent_metrics)
        
        return {
            "brain_id": self.brain_id,
            "device_name": recent_metrics[-1].device_name,
            "compute_capability": recent_metrics[-1].compute_capability,
            "average_memory_utilization": f"{avg_memory_util:.1f}%",
            "average_gpu_utilization": f"{avg_gpu_util:.1f}%",
            "average_temperature": f"{avg_temperature:.1f}°C",
            "average_power_usage": f"{avg_power:.1f}W",
            "total_memory": f"{recent_metrics[-1].memory_total / 1024**3:.1f}GB",
            "current_memory_used": f"{recent_metrics[-1].memory_used / 1024**3:.1f}GB",
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.metrics_history)
        }
