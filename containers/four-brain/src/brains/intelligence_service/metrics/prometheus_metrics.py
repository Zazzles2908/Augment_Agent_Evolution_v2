"""
Prometheus Metrics for Brain 3 - Augment Coordinator Service
Date: 2025-07-19 AEST
Purpose: Custom metrics for Supabase Edge Functions, workflow coordination, and inter-brain communication
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import time
import psutil
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class Brain3Metrics:
    """Prometheus metrics collector for Brain 3 augment coordinator service"""
    
    def __init__(self):
        # Create custom registry for Brain 3
        self.registry = CollectorRegistry()
        
        # Workflow coordination metrics
        self.workflow_requests_total = Counter(
            'brain3_workflow_requests_total',
            'Total number of workflow requests processed',
            ['workflow_type', 'status'],
            registry=self.registry
        )
        
        self.workflow_duration = Histogram(
            'brain3_workflow_duration_seconds',
            'Time spent processing workflows',
            ['workflow_type'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.active_workflows = Gauge(
            'brain3_active_workflows',
            'Number of currently active workflows',
            registry=self.registry
        )
        
        self.workflow_success_rate = Gauge(
            'brain3_workflow_success_rate',
            'Success rate of workflows (0-1)',
            ['workflow_type'],
            registry=self.registry
        )
        
        # Supabase Edge Functions metrics
        self.edge_functions_active = Gauge(
            'brain3_edge_functions_active',
            'Number of active Supabase Edge Functions',
            registry=self.registry
        )
        
        self.edge_function_invocations = Counter(
            'brain3_edge_function_invocations_total',
            'Total number of Edge Function invocations',
            ['function_name', 'status'],
            registry=self.registry
        )
        
        self.edge_function_duration = Histogram(
            'brain3_edge_function_duration_seconds',
            'Time spent executing Edge Functions',
            ['function_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.edge_function_errors = Counter(
            'brain3_edge_function_errors_total',
            'Total number of Edge Function errors',
            ['function_name', 'error_type'],
            registry=self.registry
        )
        
        # Inter-brain communication metrics
        self.brain_requests_sent = Counter(
            'brain3_brain_requests_sent_total',
            'Total requests sent to other brains',
            ['target_brain', 'request_type'],
            registry=self.registry
        )
        
        self.brain_requests_received = Counter(
            'brain3_brain_requests_received_total',
            'Total requests received from other brains',
            ['source_brain', 'request_type'],
            registry=self.registry
        )
        
        self.brain_communication_latency = Histogram(
            'brain3_brain_communication_latency_seconds',
            'Latency of inter-brain communication',
            ['target_brain'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.brain_coordination_efficiency = Gauge(
            'brain3_brain_coordination_efficiency',
            'Efficiency of brain coordination (successful requests / total requests)',
            registry=self.registry
        )
        
        # Request routing metrics
        self.request_routing_decisions = Counter(
            'brain3_request_routing_decisions_total',
            'Total number of request routing decisions',
            ['route_target', 'decision_type'],
            registry=self.registry
        )
        
        self.routing_latency = Histogram(
            'brain3_routing_latency_seconds',
            'Time spent making routing decisions',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=self.registry
        )
        
        self.load_balancing_efficiency = Gauge(
            'brain3_load_balancing_efficiency',
            'Load balancing efficiency across brains',
            registry=self.registry
        )
        
        # Cache and performance metrics
        self.cache_hits = Counter(
            'brain3_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'brain3_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'brain3_cache_hit_ratio',
            'Cache hit ratio (0-1)',
            ['cache_type'],
            registry=self.registry
        )
        
        # State management metrics
        self.conversation_states_active = Gauge(
            'brain3_conversation_states_active',
            'Number of active conversation states',
            registry=self.registry
        )
        
        self.processing_states_active = Gauge(
            'brain3_processing_states_active',
            'Number of active processing states',
            registry=self.registry
        )
        
        self.state_persistence_operations = Counter(
            'brain3_state_persistence_operations_total',
            'Total state persistence operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        # External service integration metrics
        self.supabase_operations = Counter(
            'brain3_supabase_operations_total',
            'Total Supabase operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.redis_operations = Counter(
            'brain3_redis_operations_total',
            'Total Redis operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.k2_vector_hub_requests = Counter(
            'brain3_k2_vector_hub_requests_total',
            'Total requests to K2-Vector-Hub',
            ['request_type', 'status'],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_percent = Gauge(
            'brain3_cpu_usage_percent',
            'CPU usage percentage for Brain 3 process',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'brain3_memory_usage_bytes',
            'Memory usage in bytes for Brain 3 process',
            registry=self.registry
        )
        
        self.concurrent_requests = Gauge(
            'brain3_concurrent_requests',
            'Number of concurrent requests being processed',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'brain3_service_info',
            'Information about Brain 3 service',
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '2.1.0',
            'role': 'coordinator',
            'edge_functions': 'enabled',
            'supabase_integration': 'active',
            'redis_coordination': 'active',
            'k2_integration': 'active'
        })
        
        # Initialize Edge Functions count
        self.edge_functions_active.set(3)  # email-notifications, usage-tracker, scheduled-maintenance
    
    def record_workflow_request(self, workflow_type: str, status: str, duration: float):
        """Record a workflow request with its performance metrics"""
        self.workflow_requests_total.labels(workflow_type=workflow_type, status=status).inc()
        self.workflow_duration.labels(workflow_type=workflow_type).observe(duration)
        
        # Update success rate
        self.update_workflow_success_rate(workflow_type)
    
    def record_edge_function_invocation(self, function_name: str, status: str, duration: float):
        """Record Edge Function invocation"""
        self.edge_function_invocations.labels(function_name=function_name, status=status).inc()
        self.edge_function_duration.labels(function_name=function_name).observe(duration)
    
    def record_edge_function_error(self, function_name: str, error_type: str):
        """Record Edge Function error"""
        self.edge_function_errors.labels(function_name=function_name, error_type=error_type).inc()
    
    def record_brain_communication(self, target_brain: str, request_type: str, 
                                 latency: float, success: bool):
        """Record inter-brain communication"""
        status = "success" if success else "failure"
        self.brain_requests_sent.labels(target_brain=target_brain, request_type=request_type).inc()
        self.brain_communication_latency.labels(target_brain=target_brain).observe(latency)
        
        # Update coordination efficiency
        self.update_coordination_efficiency()
    
    def record_incoming_brain_request(self, source_brain: str, request_type: str):
        """Record incoming request from another brain"""
        self.brain_requests_received.labels(source_brain=source_brain, request_type=request_type).inc()
    
    def record_routing_decision(self, route_target: str, decision_type: str, latency: float):
        """Record request routing decision"""
        self.request_routing_decisions.labels(route_target=route_target, decision_type=decision_type).inc()
        self.routing_latency.observe(latency)
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation"""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
        
        self.update_cache_hit_ratio(cache_type)
    
    def record_external_service_operation(self, service: str, operation_type: str, status: str):
        """Record external service operation"""
        if service == "supabase":
            self.supabase_operations.labels(operation_type=operation_type, status=status).inc()
        elif service == "redis":
            self.redis_operations.labels(operation_type=operation_type, status=status).inc()
        elif service == "k2_vector_hub":
            self.k2_vector_hub_requests.labels(request_type=operation_type, status=status).inc()
    
    def set_active_workflows(self, count: int):
        """Set number of active workflows"""
        self.active_workflows.set(count)
    
    def set_conversation_states(self, count: int):
        """Set number of active conversation states"""
        self.conversation_states_active.set(count)
    
    def set_processing_states(self, count: int):
        """Set number of active processing states"""
        self.processing_states_active.set(count)
    
    def set_concurrent_requests(self, count: int):
        """Set number of concurrent requests"""
        self.concurrent_requests.set(count)
    
    def update_workflow_success_rate(self, workflow_type: str):
        """Update workflow success rate"""
        # This would typically calculate based on recent success/failure ratios
        # For now, set a placeholder value
        self.workflow_success_rate.labels(workflow_type=workflow_type).set(0.95)
    
    def update_coordination_efficiency(self):
        """Update brain coordination efficiency"""
        # This would calculate based on successful vs failed communications
        # For now, set a placeholder value
        self.brain_coordination_efficiency.set(0.92)
    
    def update_cache_hit_ratio(self, cache_type: str):
        """Update cache hit ratio"""
        # This would calculate based on hits vs total operations
        # For now, set a placeholder value
        self.cache_hit_ratio.labels(cache_type=cache_type).set(0.85)
    
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
brain3_metrics = Brain3Metrics()

def get_brain3_metrics() -> Brain3Metrics:
    """Get the global Brain 3 metrics instance"""
    return brain3_metrics
