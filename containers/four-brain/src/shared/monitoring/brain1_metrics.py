#!/usr/bin/env python3
"""
Brain 1 (Embedding) Prometheus Metrics
Exposes performance metrics for the embedded Brain 1 system
"""

import time
import threading
import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server, Info

logger = logging.getLogger(__name__)

class Brain1Metrics:
    """Prometheus metrics for Brain 1 (Qwen3-4B Embedding)"""
    
    def __init__(self):
        # Brain 1 status metrics
        self.brain1_status = Gauge(
            'brain1_status',
            'Brain 1 status (1=initialized, 0=not initialized)'
        )
        
        self.brain1_uptime_seconds = Gauge(
            'brain1_uptime_seconds',
            'Brain 1 uptime in seconds'
        )
        
        # Model metrics
        self.brain1_model_loaded = Gauge(
            'brain1_model_loaded',
            'Brain 1 model loaded status (1=loaded, 0=not loaded)'
        )
        
        self.brain1_model_memory_mb = Gauge(
            'brain1_model_memory_mb',
            'Brain 1 model memory usage in MB'
        )
        
        # Embedding generation metrics
        self.brain1_embedding_requests_total = Counter(
            'brain1_embedding_requests_total',
            'Total number of embedding requests processed',
            ['status']  # success, error
        )
        
        self.brain1_embedding_duration_seconds = Histogram(
            'brain1_embedding_duration_seconds',
            'Time spent generating embeddings',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.brain1_embedding_dimensions = Gauge(
            'brain1_embedding_dimensions',
            'Number of dimensions in generated embeddings'
        )
        
        # Cache metrics
        self.brain1_cache_hits_total = Counter(
            'brain1_cache_hits_total',
            'Total number of cache hits'
        )
        
        self.brain1_cache_misses_total = Counter(
            'brain1_cache_misses_total',
            'Total number of cache misses'
        )
        
        # GPU metrics (if available)
        self.brain1_gpu_memory_allocated_mb = Gauge(
            'brain1_gpu_memory_allocated_mb',
            'GPU memory allocated by Brain 1 in MB'
        )
        
        # Model info
        self.brain1_model_info = Info(
            'brain1_model_info',
            'Brain 1 model information'
        )
        
        # Initialize
        self.start_time = time.time()
        self.metrics_server_started = False
        
    def start_metrics_server(self, port: int = 9090):
        """Start the Prometheus metrics HTTP server"""
        try:
            if not self.metrics_server_started:
                start_http_server(port)
                self.metrics_server_started = True
                logger.info(f"✅ Brain 1 metrics server started on port {port}")
                
                # Set initial model info
                self.brain1_model_info.info({
                    'model_name': 'Qwen3-4B-Embedding',
                    'model_path': '/tmp/models/qwen3/embedding-4b',
                    'dimensions': '2560',
                    'truncated_dimensions': '2000',
                    'quantization': '8-bit'
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to start Brain 1 metrics server: {e}")
    
    def update_status(self, initialized: bool):
        """Update Brain 1 initialization status"""
        self.brain1_status.set(1 if initialized else 0)
        
    def update_uptime(self):
        """Update Brain 1 uptime"""
        uptime = time.time() - self.start_time
        self.brain1_uptime_seconds.set(uptime)
        
    def update_model_status(self, loaded: bool, memory_mb: float = 0):
        """Update model loading status and memory usage"""
        self.brain1_model_loaded.set(1 if loaded else 0)
        if memory_mb > 0:
            self.brain1_model_memory_mb.set(memory_mb)
    
    def record_embedding_request(self, duration: float, success: bool, dimensions: int = 0):
        """Record an embedding generation request"""
        status = 'success' if success else 'error'
        self.brain1_embedding_requests_total.labels(status=status).inc()
        
        if success:
            self.brain1_embedding_duration_seconds.observe(duration)
            if dimensions > 0:
                self.brain1_embedding_dimensions.set(dimensions)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.brain1_cache_hits_total.inc()
        
    def record_cache_miss(self):
        """Record a cache miss"""
        self.brain1_cache_misses_total.inc()
        
    def update_gpu_memory(self, allocated_mb: float):
        """Update GPU memory allocation"""
        self.brain1_gpu_memory_allocated_mb.set(allocated_mb)

# Global metrics instance
brain1_metrics = Brain1Metrics()

def start_brain1_metrics_server(port: int = 9090):
    """Start the Brain 1 metrics server in a separate thread"""
    def _start_server():
        brain1_metrics.start_metrics_server(port)
        
        # Update uptime every 30 seconds
        while True:
            try:
                brain1_metrics.update_uptime()
                time.sleep(30)
            except Exception as e:
                logger.error(f"❌ Error updating Brain 1 metrics: {e}")
                time.sleep(30)
    
    metrics_thread = threading.Thread(target=_start_server, daemon=True)
    metrics_thread.start()
    logger.info("🚀 Brain 1 metrics thread started")

if __name__ == "__main__":
    # Test the metrics server
    logging.basicConfig(level=logging.INFO)
    start_brain1_metrics_server()
    
    # Simulate some metrics
    brain1_metrics.update_status(True)
    brain1_metrics.update_model_status(True, 4500.0)
    brain1_metrics.record_embedding_request(0.15, True, 2000)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
