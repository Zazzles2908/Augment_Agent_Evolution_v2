"""
Prometheus Metrics for Brain 2 - Qwen3-Reranker-4B Service
Date: 2025-07-19 AEST
Purpose: Custom metrics for MoE efficiency, reranking performance, and active parameter tracking
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import time
import psutil
import GPUtil
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class Brain2Metrics:
    """Prometheus metrics collector for Brain 2 reranker service"""
    
    def __init__(self):
        # Create custom registry for Brain 2
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.rerank_requests_total = Counter(
            'brain2_rerank_requests_total',
            'Total number of reranking requests processed',
            ['query_type', 'status'],
            registry=self.registry
        )
        
        self.rerank_duration = Histogram(
            'brain2_rerank_duration_seconds',
            'Time spent processing reranking requests',
            ['query_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # MoE (Mixture of Experts) efficiency metrics
        self.moe_active_experts = Gauge(
            'brain2_moe_active_experts',
            'Number of currently active experts in MoE model',
            registry=self.registry
        )
        
        self.moe_total_experts = Gauge(
            'brain2_moe_total_experts',
            'Total number of experts available in MoE model',
            registry=self.registry
        )
        
        self.moe_active_experts_ratio = Gauge(
            'brain2_moe_active_experts_ratio',
            'Ratio of active experts to total experts (efficiency metric)',
            registry=self.registry
        )
        
        self.moe_expert_utilization = Histogram(
            'brain2_moe_expert_utilization_seconds',
            'Time each expert spends processing',
            ['expert_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.moe_parameter_efficiency = Gauge(
            'brain2_moe_parameter_efficiency_percent',
            'Percentage of parameters actively used (3B/30B target)',
            registry=self.registry
        )
        
        # Reranking performance metrics
        self.documents_reranked = Counter(
            'brain2_documents_reranked_total',
            'Total number of documents reranked',
            ['batch_size_range'],
            registry=self.registry
        )
        
        self.reranking_accuracy = Gauge(
            'brain2_reranking_accuracy_score',
            'Reranking accuracy score (0-1)',
            registry=self.registry
        )
        
        self.relevance_score_improvement = Histogram(
            'brain2_relevance_score_improvement',
            'Improvement in relevance scores after reranking',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            registry=self.registry
        )
        
        self.batch_processing_efficiency = Gauge(
            'brain2_batch_processing_efficiency',
            'Efficiency of batch processing (documents/second)',
            registry=self.registry
        )
        
        # GPU and memory metrics
        self.gpu_memory_usage_bytes = Gauge(
            'brain2_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_memory_usage_percent = Gauge(
            'brain2_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            registry=self.registry
        )
        
        self.memory_pressure_fallback = Counter(
            'brain2_memory_pressure_fallback_total',
            'Number of times 4-bit fallback was triggered due to memory pressure',
            registry=self.registry
        )
        
        self.quantization_mode = Gauge(
            'brain2_quantization_mode',
            'Current quantization mode (8=8-bit, 4=4-bit)',
            registry=self.registry
        )
        
        # Model loading and caching metrics
        self.model_load_duration = Histogram(
            'brain2_model_load_duration_seconds',
            'Time spent loading the Qwen3-Reranker-4B model',
            buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
            registry=self.registry
        )
        
        self.model_cache_hits = Counter(
            'brain2_model_cache_hits_total',
            'Number of model cache hits',
            registry=self.registry
        )
        
        self.model_cache_misses = Counter(
            'brain2_model_cache_misses_total',
            'Number of model cache misses',
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_percent = Gauge(
            'brain2_cpu_usage_percent',
            'CPU usage percentage for Brain 2 process',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'brain2_memory_usage_bytes',
            'Memory usage in bytes for Brain 2 process',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'brain2_service_info',
            'Information about Brain 2 service',
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '2.1.0',
            'model': 'qwen3-reranker-4b',
            'quantization': '8-bit-primary',
            'moe_efficiency': 'enabled',
            'fallback_mode': '4-bit',
            'target_efficiency': '3B/30B'
        })
        
        # Initialize MoE metrics with default values
        self.moe_total_experts.set(30)  # 30B total parameters
        self.moe_active_experts.set(3)   # 3B active parameters target
        self.update_moe_efficiency()
    
    def record_rerank_request(self, query_type: str, status: str, duration: float, 
                            documents_count: int, accuracy_score: float):
        """Record a reranking request with performance metrics"""
        self.rerank_requests_total.labels(query_type=query_type, status=status).inc()
        self.rerank_duration.labels(query_type=query_type).observe(duration)
        
        # Determine batch size range
        if documents_count <= 10:
            batch_range = "small"
        elif documents_count <= 50:
            batch_range = "medium"
        else:
            batch_range = "large"
        
        self.documents_reranked.labels(batch_size_range=batch_range).inc(documents_count)
        self.reranking_accuracy.set(accuracy_score)
        
        # Calculate processing efficiency
        if duration > 0:
            efficiency = documents_count / duration
            self.batch_processing_efficiency.set(efficiency)
    
    def record_moe_expert_usage(self, expert_id: str, processing_time: float):
        """Record MoE expert utilization"""
        self.moe_expert_utilization.labels(expert_id=expert_id).observe(processing_time)
    
    def update_moe_efficiency(self):
        """Update MoE efficiency metrics"""
        active = self.moe_active_experts._value._value
        total = self.moe_total_experts._value._value
        
        if total > 0:
            ratio = active / total
            self.moe_active_experts_ratio.set(ratio)
            
            # Calculate parameter efficiency percentage (3B/30B = 10%)
            parameter_efficiency = (active / total) * 100
            self.moe_parameter_efficiency.set(parameter_efficiency)
    
    def set_active_experts(self, count: int):
        """Set the number of active experts"""
        self.moe_active_experts.set(count)
        self.update_moe_efficiency()
    
    def record_memory_pressure_fallback(self):
        """Record when 4-bit fallback is triggered"""
        self.memory_pressure_fallback.inc()
        self.quantization_mode.set(4)  # Switch to 4-bit mode
    
    def set_quantization_mode(self, bits: int):
        """Set current quantization mode"""
        self.quantization_mode.set(bits)
    
    def record_relevance_improvement(self, improvement: float):
        """Record relevance score improvement"""
        self.relevance_score_improvement.observe(improvement)
    
    def record_cache_hit(self):
        """Record model cache hit"""
        self.model_cache_hits.inc()
    
    def record_cache_miss(self):
        """Record model cache miss"""
        self.model_cache_misses.inc()
    
    def record_model_load_time(self, duration: float):
        """Record model loading duration"""
        self.model_load_duration.observe(duration)
    
    def update_gpu_metrics(self):
        """Update GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assume single GPU
                self.gpu_memory_usage_bytes.set(gpu.memoryUsed * 1024 * 1024)
                self.gpu_memory_usage_percent.set(gpu.memoryUtil * 100)
        except Exception as e:
            logger.warning(f"Failed to update GPU metrics: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage_percent.set(cpu_percent)
            
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage_bytes.set(memory_info.rss)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        # Update dynamic metrics before returning
        self.update_gpu_metrics()
        self.update_system_metrics()
        
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
brain2_metrics = Brain2Metrics()

def get_brain2_metrics() -> Brain2Metrics:
    """Get the global Brain 2 metrics instance"""
    return brain2_metrics
