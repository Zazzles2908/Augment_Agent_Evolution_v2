"""
Prometheus Metrics for K2-Vector-Hub - Kimi K2 Strategy Coordinator
Date: 2025-07-19 AEST
Purpose: Custom metrics for Kimi K2 model switching, API usage tracking, and adaptive selection performance
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import time
import psutil
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class K2VectorHubMetrics:
    """Prometheus metrics collector for K2-Vector-Hub service"""
    
    def __init__(self):
        # Create custom registry for Orchestrator Hub
        self.registry = CollectorRegistry()

        # Kimi K2 API usage metrics
        self.kimi_api_requests_total = Counter(
            'orchestrator_hub_kimi_api_requests_total',
            'Total number of Kimi K2 API requests',
            ['model', 'status'],
            registry=self.registry
        )

        self.kimi_api_latency = Histogram(
            'orchestrator_hub_kimi_api_latency_seconds',
            'Latency of Kimi K2 API requests',
            ['model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
            registry=self.registry
        )

        self.kimi_api_tokens_used = Counter(
            'orchestrator_hub_kimi_api_tokens_used_total',
            'Total number of tokens used in Kimi K2 API',
            ['model', 'token_type'],
            registry=self.registry
        )
        
        self.kimi_api_cost_estimate = Counter(
            'k2_hub_kimi_api_cost_estimate_total',
            'Estimated cost of Kimi K2 API usage',
            ['model'],
            registry=self.registry
        )
        
        # Model switching metrics
        self.model_switches_total = Counter(
            'k2_hub_model_switches_total',
            'Total number of model switches',
            ['from_model', 'to_model', 'reason'],
            registry=self.registry
        )
        
        self.current_active_model = Gauge(
            'k2_hub_current_active_model',
            'Currently active model (1=kimi-k2-0711, 2=kimi-thinking)',
            registry=self.registry
        )
        
        self.model_selection_latency = Histogram(
            'k2_hub_model_selection_latency_seconds',
            'Time spent selecting optimal model',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=self.registry
        )
        
        self.adaptive_selection_accuracy = Gauge(
            'k2_hub_adaptive_selection_accuracy',
            'Accuracy of adaptive model selection (0-1)',
            registry=self.registry
        )
        
        # Task complexity analysis metrics
        self.task_complexity_scores = Histogram(
            'k2_hub_task_complexity_scores',
            'Distribution of task complexity scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.complexity_analysis_duration = Histogram(
            'k2_hub_complexity_analysis_duration_seconds',
            'Time spent analyzing task complexity',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry
        )
        
        # Performance optimization metrics
        self.response_cache_hits = Counter(
            'k2_hub_response_cache_hits_total',
            'Total number of response cache hits',
            ['model'],
            registry=self.registry
        )
        
        self.response_cache_misses = Counter(
            'k2_hub_response_cache_misses_total',
            'Total number of response cache misses',
            ['model'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'k2_hub_cache_hit_ratio',
            'Response cache hit ratio (0-1)',
            ['model'],
            registry=self.registry
        )
        
        self.load_balancing_efficiency = Gauge(
            'k2_hub_load_balancing_efficiency',
            'Load balancing efficiency across models',
            registry=self.registry
        )
        
        # Model performance tracking
        self.model_performance_scores = Gauge(
            'k2_hub_model_performance_scores',
            'Performance scores for each model',
            ['model', 'metric_type'],
            registry=self.registry
        )
        
        self.model_usage_distribution = Gauge(
            'k2_hub_model_usage_distribution',
            'Distribution of model usage (percentage)',
            ['model'],
            registry=self.registry
        )
        
        self.thinking_mode_utilization = Gauge(
            'k2_hub_thinking_mode_utilization',
            'Utilization rate of thinking mode (0-1)',
            registry=self.registry
        )
        
        # Request routing and coordination metrics
        self.brain_coordination_requests = Counter(
            'k2_hub_brain_coordination_requests_total',
            'Total requests for brain coordination',
            ['source_brain', 'coordination_type'],
            registry=self.registry
        )
        
        self.strategy_decisions = Counter(
            'k2_hub_strategy_decisions_total',
            'Total strategy decisions made',
            ['strategy_type', 'outcome'],
            registry=self.registry
        )
        
        self.coordination_latency = Histogram(
            'k2_hub_coordination_latency_seconds',
            'Latency of brain coordination operations',
            ['coordination_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Error tracking and reliability metrics
        self.api_errors = Counter(
            'k2_hub_api_errors_total',
            'Total API errors',
            ['error_type', 'model'],
            registry=self.registry
        )
        
        self.retry_attempts = Counter(
            'k2_hub_retry_attempts_total',
            'Total retry attempts',
            ['operation_type', 'model'],
            registry=self.registry
        )
        
        self.fallback_activations = Counter(
            'k2_hub_fallback_activations_total',
            'Total fallback activations',
            ['fallback_type', 'trigger_reason'],
            registry=self.registry
        )
        
        self.service_availability = Gauge(
            'k2_hub_service_availability',
            'Service availability (0-1)',
            ['service_component'],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_percent = Gauge(
            'k2_hub_cpu_usage_percent',
            'CPU usage percentage for K2-Vector-Hub process',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'k2_hub_memory_usage_bytes',
            'Memory usage in bytes for K2-Vector-Hub process',
            registry=self.registry
        )
        
        self.concurrent_requests = Gauge(
            'k2_hub_concurrent_requests',
            'Number of concurrent requests being processed',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'k2_hub_service_info',
            'Information about K2-Vector-Hub service',
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '2.1.0',
            'role': 'strategy_coordinator',
            'kimi_k2_integration': 'active',
            'adaptive_switching': 'enabled',
            'models_available': 'kimi-k2-0711-preview,kimi-thinking-preview',
            'cache_enabled': 'true',
            'load_balancing': 'enabled'
        })
        
        # Initialize model performance scores
        self.initialize_model_metrics()
    
    def initialize_model_metrics(self):
        """Initialize model-specific metrics with default values"""
        models = ['kimi-k2-0711-preview', 'kimi-thinking-preview']
        
        for model in models:
            # Set initial performance scores
            self.model_performance_scores.labels(model=model, metric_type='latency').set(0.0)
            self.model_performance_scores.labels(model=model, metric_type='quality').set(0.0)
            self.model_performance_scores.labels(model=model, metric_type='cost_efficiency').set(0.0)
            
            # Set initial usage distribution
            self.model_usage_distribution.labels(model=model).set(50.0)  # 50/50 split initially
            
            # Set initial cache hit ratios
            self.cache_hit_ratio.labels(model=model).set(0.0)
            
            # Set initial service availability
            self.service_availability.labels(service_component=f'{model}_api').set(1.0)
    
    def record_kimi_api_request(self, model: str, status: str, latency: float, 
                              input_tokens: int, output_tokens: int, cost: float):
        """Record Kimi K2 API request with comprehensive metrics"""
        self.kimi_api_requests_total.labels(model=model, status=status).inc()
        self.kimi_api_latency.labels(model=model).observe(latency)
        
        # Record token usage
        self.kimi_api_tokens_used.labels(model=model, token_type='input').inc(input_tokens)
        self.kimi_api_tokens_used.labels(model=model, token_type='output').inc(output_tokens)
        
        # Record cost
        self.kimi_api_cost_estimate.labels(model=model).inc(cost)
    
    def record_model_switch(self, from_model: str, to_model: str, reason: str, selection_time: float):
        """Record model switching event"""
        self.model_switches_total.labels(from_model=from_model, to_model=to_model, reason=reason).inc()
        self.model_selection_latency.observe(selection_time)
        
        # Update current active model
        if to_model == 'kimi-k2-0711-preview':
            self.current_active_model.set(1)
        elif to_model == 'kimi-thinking-preview':
            self.current_active_model.set(2)
    
    def record_task_complexity_analysis(self, complexity_score: float, analysis_time: float):
        """Record task complexity analysis"""
        self.task_complexity_scores.observe(complexity_score)
        self.complexity_analysis_duration.observe(analysis_time)
    
    def record_cache_operation(self, model: str, hit: bool):
        """Record cache operation"""
        if hit:
            self.response_cache_hits.labels(model=model).inc()
        else:
            self.response_cache_misses.labels(model=model).inc()
        
        self.update_cache_hit_ratio(model)
    
    def record_brain_coordination(self, source_brain: str, coordination_type: str, latency: float):
        """Record brain coordination request"""
        self.brain_coordination_requests.labels(source_brain=source_brain, coordination_type=coordination_type).inc()
        self.coordination_latency.labels(coordination_type=coordination_type).observe(latency)
    
    def record_strategy_decision(self, strategy_type: str, outcome: str):
        """Record strategy decision"""
        self.strategy_decisions.labels(strategy_type=strategy_type, outcome=outcome).inc()
    
    def record_api_error(self, error_type: str, model: str):
        """Record API error"""
        self.api_errors.labels(error_type=error_type, model=model).inc()
    
    def record_retry_attempt(self, operation_type: str, model: str):
        """Record retry attempt"""
        self.retry_attempts.labels(operation_type=operation_type, model=model).inc()
    
    def record_fallback_activation(self, fallback_type: str, trigger_reason: str):
        """Record fallback activation"""
        self.fallback_activations.labels(fallback_type=fallback_type, trigger_reason=trigger_reason).inc()
    
    def update_model_performance(self, model: str, metric_type: str, score: float):
        """Update model performance score"""
        self.model_performance_scores.labels(model=model, metric_type=metric_type).set(score)
    
    def update_model_usage_distribution(self, model: str, percentage: float):
        """Update model usage distribution"""
        self.model_usage_distribution.labels(model=model).set(percentage)
    
    def set_thinking_mode_utilization(self, utilization: float):
        """Set thinking mode utilization rate"""
        self.thinking_mode_utilization.set(utilization)
    
    def set_adaptive_selection_accuracy(self, accuracy: float):
        """Set adaptive selection accuracy"""
        self.adaptive_selection_accuracy.set(accuracy)
    
    def set_load_balancing_efficiency(self, efficiency: float):
        """Set load balancing efficiency"""
        self.load_balancing_efficiency.set(efficiency)
    
    def set_service_availability(self, component: str, availability: float):
        """Set service availability"""
        self.service_availability.labels(service_component=component).set(availability)
    
    def set_concurrent_requests(self, count: int):
        """Set number of concurrent requests"""
        self.concurrent_requests.set(count)
    
    def update_cache_hit_ratio(self, model: str):
        """Update cache hit ratio for a model"""
        # This would calculate based on hits vs total operations
        # For now, set a placeholder value
        self.cache_hit_ratio.labels(model=model).set(0.80)
    
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
        self.update_system_metrics()
        
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
orchestrator_hub_metrics = K2VectorHubMetrics()

def get_orchestrator_hub_metrics() -> K2VectorHubMetrics:
    """Get the global Orchestrator Hub metrics instance"""
    return orchestrator_hub_metrics
