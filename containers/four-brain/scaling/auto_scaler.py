#!/usr/bin/env python3
"""
Four-Brain System Auto-Scaling Manager
Production-grade horizontal scaling with load-based auto-scaling
Version: Production v1.0
"""

import os
import sys
import time
import json
import logging
import asyncio
import docker
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    NONE = "none"

class BrainService(Enum):
    """Brain service types"""
    BRAIN1_EMBEDDING = "brain1-embedding"
    BRAIN2_RERANKER = "brain2-reranker"
    BRAIN3_AGENTIC = "brain3-agentic"
    BRAIN4_DOCLING = "brain4-docling"

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: datetime
    service: str
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    queue_depth: int
    request_rate: float
    response_time_p95: float
    active_connections: int
    error_rate: float

@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    service: BrainService
    metric_type: str  # cpu, memory, gpu, queue_depth, request_rate
    threshold_up: float
    threshold_down: float
    min_instances: int
    max_instances: int
    scale_up_cooldown: int  # seconds
    scale_down_cooldown: int  # seconds
    evaluation_periods: int
    enabled: bool = True

@dataclass
class ServiceInstance:
    """Service instance information"""
    container_id: str
    service_name: str
    instance_id: str
    status: str
    created_at: datetime
    cpu_limit: str
    memory_limit: str
    port: int
    health_status: str

class AutoScaler:
    """Auto-scaling manager for Four-Brain services"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.scaling_config_file = '/app/scaling/scaling_config.json'
        self.metrics_history = {}
        self.last_scaling_action = {}
        
        # Default scaling rules
        self.scaling_rules = self._create_default_scaling_rules()
        
        # Scaling thresholds
        self.default_thresholds = {
            'cpu_up': 80.0,
            'cpu_down': 30.0,
            'memory_up': 85.0,
            'memory_down': 40.0,
            'gpu_up': 90.0,
            'gpu_down': 50.0,
            'queue_depth_up': 100,
            'queue_depth_down': 10,
            'request_rate_up': 1000,
            'request_rate_down': 100,
            'response_time_up': 5.0,
            'response_time_down': 1.0
        }
        
        # Load balancer configuration
        self.load_balancer_config = {
            'algorithm': 'least_conn',  # round_robin, least_conn, ip_hash
            'health_check_interval': 30,
            'health_check_timeout': 10,
            'max_fails': 3,
            'fail_timeout': 30
        }
        
        logger.info("Auto-scaler initialized")
    
    def _create_default_scaling_rules(self) -> List[ScalingRule]:
        """Create default scaling rules for all brain services"""
        rules = []
        
        for brain_service in BrainService:
            # CPU-based scaling
            rules.append(ScalingRule(
                name=f"{brain_service.value}_cpu_scaling",
                service=brain_service,
                metric_type="cpu",
                threshold_up=80.0,
                threshold_down=30.0,
                min_instances=1,
                max_instances=5,
                scale_up_cooldown=300,  # 5 minutes
                scale_down_cooldown=600,  # 10 minutes
                evaluation_periods=3
            ))
            
            # Memory-based scaling
            rules.append(ScalingRule(
                name=f"{brain_service.value}_memory_scaling",
                service=brain_service,
                metric_type="memory",
                threshold_up=85.0,
                threshold_down=40.0,
                min_instances=1,
                max_instances=5,
                scale_up_cooldown=300,
                scale_down_cooldown=600,
                evaluation_periods=3
            ))
            
            # Queue depth scaling (for high-throughput scenarios)
            rules.append(ScalingRule(
                name=f"{brain_service.value}_queue_scaling",
                service=brain_service,
                metric_type="queue_depth",
                threshold_up=100,
                threshold_down=10,
                min_instances=1,
                max_instances=10,
                scale_up_cooldown=180,  # 3 minutes
                scale_down_cooldown=600,  # 10 minutes
                evaluation_periods=2
            ))
        
        return rules
    
    def collect_metrics(self, service: str) -> Optional[ScalingMetrics]:
        """Collect metrics for a service"""
        try:
            # Get service containers
            containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.compose.service={service}'}
            )
            
            if not containers:
                logger.warning(f"No containers found for service: {service}")
                return None
            
            # Aggregate metrics from all containers
            total_cpu = 0.0
            total_memory = 0.0
            total_connections = 0
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * 100.0
                    else:
                        cpu_percent = 0.0
                    
                    # Calculate memory percentage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0
                    
                    total_cpu += cpu_percent
                    total_memory += memory_percent
                    
                except Exception as e:
                    logger.warning(f"Error collecting stats for container {container.id}: {e}")
                    continue
            
            # Average metrics across containers
            num_containers = len(containers)
            avg_cpu = total_cpu / num_containers if num_containers > 0 else 0.0
            avg_memory = total_memory / num_containers if num_containers > 0 else 0.0
            
            # TODO: Collect additional metrics from Prometheus/monitoring system
            # For now, using placeholder values
            gpu_utilization = 0.0  # Would come from nvidia-ml-py
            queue_depth = 0  # Would come from Redis streams
            request_rate = 0.0  # Would come from Prometheus
            response_time_p95 = 0.0  # Would come from Prometheus
            error_rate = 0.0  # Would come from Prometheus
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                service=service,
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                gpu_utilization=gpu_utilization,
                queue_depth=queue_depth,
                request_rate=request_rate,
                response_time_p95=response_time_p95,
                active_connections=total_connections,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {service}: {e}")
            return None
    
    def evaluate_scaling_decision(self, service: str, metrics: ScalingMetrics) -> ScalingDirection:
        """Evaluate if scaling is needed based on metrics and rules"""
        service_rules = [rule for rule in self.scaling_rules 
                        if rule.service.value == service and rule.enabled]
        
        if not service_rules:
            return ScalingDirection.NONE
        
        # Store metrics in history
        if service not in self.metrics_history:
            self.metrics_history[service] = []
        
        self.metrics_history[service].append(metrics)
        
        # Keep only recent metrics (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metrics_history[service] = [
            m for m in self.metrics_history[service] 
            if m.timestamp > cutoff_time
        ]
        
        # Evaluate each rule
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in service_rules:
            decision = self._evaluate_rule(rule, service)
            
            if decision == ScalingDirection.UP:
                scale_up_votes += 1
            elif decision == ScalingDirection.DOWN:
                scale_down_votes += 1
        
        # Make final decision
        if scale_up_votes > scale_down_votes:
            return ScalingDirection.UP
        elif scale_down_votes > scale_up_votes:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE
    
    def _evaluate_rule(self, rule: ScalingRule, service: str) -> ScalingDirection:
        """Evaluate a single scaling rule"""
        if service not in self.metrics_history:
            return ScalingDirection.NONE
        
        recent_metrics = self.metrics_history[service][-rule.evaluation_periods:]
        
        if len(recent_metrics) < rule.evaluation_periods:
            return ScalingDirection.NONE
        
        # Get metric values based on rule type
        metric_values = []
        for metrics in recent_metrics:
            if rule.metric_type == "cpu":
                metric_values.append(metrics.cpu_percent)
            elif rule.metric_type == "memory":
                metric_values.append(metrics.memory_percent)
            elif rule.metric_type == "gpu":
                metric_values.append(metrics.gpu_utilization)
            elif rule.metric_type == "queue_depth":
                metric_values.append(metrics.queue_depth)
            elif rule.metric_type == "request_rate":
                metric_values.append(metrics.request_rate)
        
        if not metric_values:
            return ScalingDirection.NONE
        
        # Calculate average metric value
        avg_metric = sum(metric_values) / len(metric_values)
        
        # Check cooldown periods
        last_action_time = self.last_scaling_action.get(service)
        if last_action_time:
            time_since_last = (datetime.now() - last_action_time).total_seconds()
            
            if time_since_last < rule.scale_up_cooldown:
                return ScalingDirection.NONE
        
        # Evaluate thresholds
        if avg_metric > rule.threshold_up:
            return ScalingDirection.UP
        elif avg_metric < rule.threshold_down:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE
    
    def scale_service(self, service: str, direction: ScalingDirection) -> bool:
        """Scale a service up or down"""
        try:
            current_instances = self.get_service_instances(service)
            current_count = len(current_instances)
            
            # Get scaling rule for limits
            service_rule = next(
                (rule for rule in self.scaling_rules 
                 if rule.service.value == service and rule.metric_type == "cpu"),
                None
            )
            
            if not service_rule:
                logger.error(f"No scaling rule found for service: {service}")
                return False
            
            if direction == ScalingDirection.UP:
                if current_count >= service_rule.max_instances:
                    logger.info(f"Service {service} already at max instances ({service_rule.max_instances})")
                    return False
                
                # Scale up
                new_instance_id = f"{service}-{current_count + 1}"
                success = self._create_service_instance(service, new_instance_id)
                
                if success:
                    logger.info(f"Scaled up {service}: {current_count} -> {current_count + 1}")
                    self.last_scaling_action[service] = datetime.now()
                    return True
                
            elif direction == ScalingDirection.DOWN:
                if current_count <= service_rule.min_instances:
                    logger.info(f"Service {service} already at min instances ({service_rule.min_instances})")
                    return False
                
                # Scale down (remove oldest instance)
                if current_instances:
                    oldest_instance = min(current_instances, key=lambda x: x.created_at)
                    success = self._remove_service_instance(oldest_instance.container_id)
                    
                    if success:
                        logger.info(f"Scaled down {service}: {current_count} -> {current_count - 1}")
                        self.last_scaling_action[service] = datetime.now()
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error scaling service {service}: {e}")
            return False
    
    def get_service_instances(self, service: str) -> List[ServiceInstance]:
        """Get all instances of a service"""
        try:
            containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.compose.service={service}'}
            )
            
            instances = []
            for container in containers:
                # Get container port
                port_bindings = container.attrs['NetworkSettings']['Ports']
                port = None
                for container_port, host_bindings in port_bindings.items():
                    if host_bindings:
                        port = int(host_bindings[0]['HostPort'])
                        break
                
                instance = ServiceInstance(
                    container_id=container.id,
                    service_name=service,
                    instance_id=container.name,
                    status=container.status,
                    created_at=datetime.fromisoformat(
                        container.attrs['Created'].replace('Z', '+00:00')
                    ),
                    cpu_limit=container.attrs['HostConfig'].get('CpuQuota', 'unlimited'),
                    memory_limit=container.attrs['HostConfig'].get('Memory', 'unlimited'),
                    port=port or 0,
                    health_status='healthy' if container.status == 'running' else 'unhealthy'
                )
                instances.append(instance)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error getting service instances for {service}: {e}")
            return []
    
    def _create_service_instance(self, service: str, instance_id: str) -> bool:
        """Create a new service instance"""
        try:
            # This would typically use Docker Compose or Kubernetes
            # For now, we'll use Docker API directly
            
            # Get base configuration from existing container
            existing_containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.compose.service={service}'},
                limit=1
            )
            
            if not existing_containers:
                logger.error(f"No existing containers found for service: {service}")
                return False
            
            base_container = existing_containers[0]
            base_config = base_container.attrs['Config']
            base_host_config = base_container.attrs['HostConfig']
            
            # Create new container with similar configuration
            new_container = self.docker_client.containers.run(
                image=base_config['Image'],
                environment=base_config['Env'],
                volumes=base_host_config.get('Binds', []),
                network_mode=base_host_config.get('NetworkMode'),
                mem_limit=base_host_config.get('Memory'),
                cpu_quota=base_host_config.get('CpuQuota'),
                labels={
                    'com.docker.compose.service': service,
                    'scaling.instance_id': instance_id
                },
                detach=True,
                name=instance_id
            )
            
            logger.info(f"Created new instance {instance_id} for service {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating service instance: {e}")
            return False
    
    def _remove_service_instance(self, container_id: str) -> bool:
        """Remove a service instance"""
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=30)
            container.remove()
            
            logger.info(f"Removed service instance: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing service instance {container_id}: {e}")
            return False
    
    async def auto_scaling_loop(self):
        """Main auto-scaling loop"""
        logger.info("Starting auto-scaling loop")
        
        while True:
            try:
                for brain_service in BrainService:
                    service_name = brain_service.value
                    
                    # Collect metrics
                    metrics = self.collect_metrics(service_name)
                    if not metrics:
                        continue
                    
                    # Evaluate scaling decision
                    scaling_decision = self.evaluate_scaling_decision(service_name, metrics)
                    
                    # Execute scaling if needed
                    if scaling_decision != ScalingDirection.NONE:
                        logger.info(f"Scaling decision for {service_name}: {scaling_decision.value}")
                        self.scale_service(service_name, scaling_decision)
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(30)  # Short sleep on error

def main():
    """Main function"""
    try:
        auto_scaler = AutoScaler()
        
        # Test metrics collection
        for brain_service in BrainService:
            metrics = auto_scaler.collect_metrics(brain_service.value)
            if metrics:
                logger.info(f"Metrics for {brain_service.value}: "
                          f"CPU {metrics.cpu_percent:.1f}%, "
                          f"Memory {metrics.memory_percent:.1f}%")
        
        # Start auto-scaling loop
        asyncio.run(auto_scaler.auto_scaling_loop())
        
    except KeyboardInterrupt:
        logger.info("Auto-scaler stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
