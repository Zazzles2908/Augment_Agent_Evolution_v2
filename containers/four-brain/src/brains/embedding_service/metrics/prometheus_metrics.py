"""
Prometheus Metrics for Brain 1 - Qwen3-4B Embedding Service
Date: 2025-07-19 AEST
Purpose: Custom metrics for thinking mode, embedding performance, and GPU utilization
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import time
import psutil
import GPUtil
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Brain1Metrics:
    """Prometheus metrics collector for Brain 1 embedding service"""
    
    def __init__(self):
        # Create custom registry for Brain 1
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.requests_total = Counter(
            'brain1_requests_total',
            'Total number of embedding requests processed',
            ['endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'brain1_request_duration_seconds',
            'Time spent processing embedding requests',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Thinking mode metrics
        self.thinking_mode_enabled = Gauge(
            'brain1_thinking_mode_enabled',
            'Whether thinking mode is currently enabled (1=enabled, 0=disabled)',
            registry=self.registry
        )
        
        self.thinking_iterations = Counter(
            'brain1_thinking_iterations_total',
            'Total number of thinking iterations performed',
            ['complexity_level'],
            registry=self.registry
        )
        
        self.thinking_duration = Histogram(
            'brain1_thinking_duration_seconds',
            'Time spent in thinking mode processing',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
        
        # Embedding performance metrics
        self.embeddings_generated = Counter(
            'brain1_embeddings_generated_total',
            'Total number of embeddings generated',
            ['model_type', 'dimension'],
            registry=self.registry
        )
        
        self.embedding_quality_score = Gauge(
            'brain1_embedding_quality_score',
            'Quality score of generated embeddings (0-1)',
            registry=self.registry
        )
        
        self.mrl_truncations = Counter(
            'brain1_mrl_truncations_total',
            'Total number of MRL truncations (2560→2000 dimensions)',
            registry=self.registry
        )
        
        # GPU utilization metrics
        self.gpu_memory_usage_bytes = Gauge(
            'brain1_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_memory_usage_percent = Gauge(
            'brain1_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_utilization_percent = Gauge(
            'brain1_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )
        
        self.gpu_temperature_celsius = Gauge(
            'brain1_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            registry=self.registry
        )
        
        # Model loading metrics
        self.model_load_duration = Histogram(
            'brain1_model_load_duration_seconds',
            'Time spent loading the Qwen3-4B model',
            buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
            registry=self.registry
        )
        
        self.model_loaded = Gauge(
            'brain1_model_loaded',
            'Whether the model is currently loaded (1=loaded, 0=not loaded)',
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_percent = Gauge(
            'brain1_cpu_usage_percent',
            'CPU usage percentage for Brain 1 process',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'brain1_memory_usage_bytes',
            'Memory usage in bytes for Brain 1 process',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'brain1_service_info',
            'Information about Brain 1 service',
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '2.1.0',
            'model': 'qwen3-4b',
            'quantization': '8-bit',
            'thinking_mode': 'enabled',
            'mrl_truncation': 'enabled',
            'gpu_acceleration': 'enabled'
        })
    
    def record_request(self, endpoint: str, status: str, duration: float):
        """Record a request with its duration and status"""
        self.requests_total.labels(endpoint=endpoint, status=status).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration)
    
    def record_thinking_iteration(self, complexity_level: str, duration: float):
        """Record a thinking mode iteration"""
        self.thinking_iterations.labels(complexity_level=complexity_level).inc()
        self.thinking_duration.observe(duration)
    
    def record_embedding_generation(self, model_type: str, dimension: int, quality_score: float):
        """Record embedding generation with quality metrics"""
        self.embeddings_generated.labels(model_type=model_type, dimension=str(dimension)).inc()
        self.embedding_quality_score.set(quality_score)
        
        # Record MRL truncation if applicable
        if dimension == 2000:
            self.mrl_truncations.inc()
    
    def update_gpu_metrics(self):
        """Update GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assume single GPU
                self.gpu_memory_usage_bytes.set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes
                self.gpu_memory_usage_percent.set(gpu.memoryUtil * 100)
                self.gpu_utilization_percent.set(gpu.load * 100)
                self.gpu_temperature_celsius.set(gpu.temperature)
        except Exception as e:
            logger.warning(f"Failed to update GPU metrics: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage_percent.set(cpu_percent)
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage_bytes.set(memory_info.rss)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def set_thinking_mode(self, enabled: bool):
        """Set thinking mode status"""
        self.thinking_mode_enabled.set(1 if enabled else 0)
    
    def set_model_loaded(self, loaded: bool):
        """Set model loaded status"""
        self.model_loaded.set(1 if loaded else 0)
    
    def record_model_load_time(self, duration: float):
        """Record model loading duration"""
        self.model_load_duration.observe(duration)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        # Update dynamic metrics before returning
        self.update_gpu_metrics()
        self.update_system_metrics()
        
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
brain1_metrics = Brain1Metrics()

def get_brain1_metrics() -> Brain1Metrics:
    """Get the global Brain 1 metrics instance"""
    return brain1_metrics
