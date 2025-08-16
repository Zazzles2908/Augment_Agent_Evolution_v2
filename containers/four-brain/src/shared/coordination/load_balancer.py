"""
Load Balancing System for Four-Brain System v2
Intelligent load distribution across multiple AI brain instances

Created: 2025-07-30 AEST
Purpose: Optimize resource utilization and performance across Brain1, Brain2, Brain3, and Brain4
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"

class HealthStatus(Enum):
    """Health status for load balancing decisions"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class LoadMetrics:
    """Load metrics for a brain instance"""
    brain_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    active_connections: int
    requests_per_second: float
    average_response_time: float
    error_rate: float
    queue_length: int
    last_updated: datetime

@dataclass
class LoadBalancingRule:
    """Load balancing rule configuration"""
    rule_id: str
    strategy: LoadBalancingStrategy
    weight: float
    conditions: Dict[str, Any]
    priority: int
    enabled: bool

@dataclass
class BrainWeight:
    """Brain weight for load balancing"""
    brain_id: str
    base_weight: float
    dynamic_weight: float
    performance_factor: float
    health_factor: float
    final_weight: float

class LoadBalancer:
    """
    Intelligent load balancing system for Four-Brain coordination
    
    Features:
    - Multiple load balancing strategies
    - Real-time performance monitoring
    - Adaptive weight adjustment
    - Health-aware routing
    - Circuit breaker pattern
    - Performance optimization
    - Comprehensive metrics collection
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/12"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Load balancing state
        self.brain_metrics: Dict[str, LoadMetrics] = {}
        self.brain_weights: Dict[str, BrainWeight] = {}
        self.round_robin_index = 0
        
        # Load balancing configuration
        self.config = {
            'default_strategy': LoadBalancingStrategy.ADAPTIVE,
            'health_check_interval': 30,
            'metrics_collection_interval': 10,
            'weight_adjustment_interval': 60,
            'circuit_breaker_threshold': 0.5,
            'circuit_breaker_timeout': 300,
            'performance_window_minutes': 10,
            'adaptive_learning_rate': 0.1
        }
        
        # Load balancing rules
        self.balancing_rules: Dict[str, LoadBalancingRule] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Performance history
        self.performance_history: Dict[str, List[float]] = {}
        
        # Load balancing metrics
        self.lb_metrics = {
            'total_requests': 0,
            'requests_by_brain': {},
            'average_response_times': {},
            'error_rates': {},
            'load_distribution': {},
            'strategy_effectiveness': {}
        }
        
        logger.info("⚖️ Load Balancer initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start load balancing services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize default rules
            await self._initialize_default_rules()
            
            # Load existing metrics
            await self._load_brain_metrics()
            
            # Start background services
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._weight_adjuster())
            asyncio.create_task(self._circuit_breaker_monitor())
            
            logger.info("✅ Load Balancer Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Load Balancer: {e}")
            raise
    
    async def select_brain(self, task_type: str, available_brains: List[str],
                          strategy: Optional[LoadBalancingStrategy] = None) -> Optional[str]:
        """Select the best brain for a task using load balancing"""
        try:
            if not available_brains:
                return None
            
            # Use specified strategy or default
            strategy = strategy or self.config['default_strategy']
            
            # Filter out unhealthy brains
            healthy_brains = await self._filter_healthy_brains(available_brains)
            if not healthy_brains:
                logger.warning("No healthy brains available for load balancing")
                return None
            
            # Apply load balancing strategy
            selected_brain = await self._apply_strategy(strategy, healthy_brains, task_type)
            
            if selected_brain:
                # Update metrics
                self.lb_metrics['total_requests'] += 1
                self.lb_metrics['requests_by_brain'][selected_brain] = (
                    self.lb_metrics['requests_by_brain'].get(selected_brain, 0) + 1
                )
                
                # Update load distribution
                await self._update_load_distribution()
                
                logger.debug(f"🎯 Brain selected: {selected_brain} using {strategy.value}")
            
            return selected_brain
            
        except Exception as e:
            logger.error(f"❌ Brain selection failed: {e}")
            return available_brains[0] if available_brains else None
    
    async def _apply_strategy(self, strategy: LoadBalancingStrategy, 
                            brains: List[str], task_type: str) -> Optional[str]:
        """Apply specific load balancing strategy"""
        try:
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return await self._round_robin_selection(brains)
            
            elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return await self._least_connections_selection(brains)
            
            elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._weighted_round_robin_selection(brains)
            
            elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return await self._least_response_time_selection(brains)
            
            elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return await self._resource_based_selection(brains)
            
            elif strategy == LoadBalancingStrategy.ADAPTIVE:
                return await self._adaptive_selection(brains, task_type)
            
            else:
                return brains[0]  # Fallback
                
        except Exception as e:
            logger.error(f"❌ Strategy application failed: {e}")
            return brains[0] if brains else None
    
    async def _round_robin_selection(self, brains: List[str]) -> str:
        """Round robin load balancing"""
        if not brains:
            return None
        
        selected_brain = brains[self.round_robin_index % len(brains)]
        self.round_robin_index += 1
        return selected_brain
    
    async def _least_connections_selection(self, brains: List[str]) -> str:
        """Select brain with least active connections"""
        min_connections = float('inf')
        selected_brain = brains[0]
        
        for brain_id in brains:
            metrics = self.brain_metrics.get(brain_id)
            if metrics:
                if metrics.active_connections < min_connections:
                    min_connections = metrics.active_connections
                    selected_brain = brain_id
        
        return selected_brain
    
    async def _weighted_round_robin_selection(self, brains: List[str]) -> str:
        """Weighted round robin based on brain weights"""
        if not brains:
            return None
        
        # Calculate cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for brain_id in brains:
            weight = self.brain_weights.get(brain_id, BrainWeight(brain_id, 1.0, 1.0, 1.0, 1.0, 1.0))
            total_weight += weight.final_weight
            cumulative_weights.append(total_weight)
        
        # Select based on weight
        import random
        random_value = random.uniform(0, total_weight)
        
        for i, cumulative_weight in enumerate(cumulative_weights):
            if random_value <= cumulative_weight:
                return brains[i]
        
        return brains[0]  # Fallback
    
    async def _least_response_time_selection(self, brains: List[str]) -> str:
        """Select brain with lowest average response time"""
        min_response_time = float('inf')
        selected_brain = brains[0]
        
        for brain_id in brains:
            metrics = self.brain_metrics.get(brain_id)
            if metrics:
                if metrics.average_response_time < min_response_time:
                    min_response_time = metrics.average_response_time
                    selected_brain = brain_id
        
        return selected_brain
    
    async def _resource_based_selection(self, brains: List[str]) -> str:
        """Select brain based on resource utilization"""
        min_utilization = float('inf')
        selected_brain = brains[0]
        
        for brain_id in brains:
            metrics = self.brain_metrics.get(brain_id)
            if metrics:
                # Calculate combined utilization score
                utilization_score = (
                    metrics.cpu_usage * 0.3 +
                    metrics.memory_usage * 0.3 +
                    metrics.gpu_usage * 0.4
                )
                
                if utilization_score < min_utilization:
                    min_utilization = utilization_score
                    selected_brain = brain_id
        
        return selected_brain
    
    async def _adaptive_selection(self, brains: List[str], task_type: str) -> str:
        """Adaptive selection based on performance history and current metrics"""
        best_score = float('-inf')
        selected_brain = brains[0]
        
        for brain_id in brains:
            score = await self._calculate_adaptive_score(brain_id, task_type)
            if score > best_score:
                best_score = score
                selected_brain = brain_id
        
        return selected_brain
    
    async def _calculate_adaptive_score(self, brain_id: str, task_type: str) -> float:
        """Calculate adaptive score for brain selection"""
        try:
            metrics = self.brain_metrics.get(brain_id)
            weight = self.brain_weights.get(brain_id)
            
            if not metrics or not weight:
                return 0.0
            
            # Base score from weights
            score = weight.final_weight
            
            # Adjust for current load
            load_factor = 1.0 - (metrics.cpu_usage + metrics.memory_usage + metrics.gpu_usage) / 3.0
            score *= load_factor
            
            # Adjust for response time
            if metrics.average_response_time > 0:
                response_factor = 1.0 / (1.0 + metrics.average_response_time)
                score *= response_factor
            
            # Adjust for error rate
            error_factor = 1.0 - metrics.error_rate
            score *= error_factor
            
            # Adjust for queue length
            queue_factor = 1.0 / (1.0 + metrics.queue_length * 0.1)
            score *= queue_factor
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Adaptive score calculation failed: {e}")
            return 0.0
    
    async def _filter_healthy_brains(self, brains: List[str]) -> List[str]:
        """Filter out unhealthy brains"""
        healthy_brains = []
        
        for brain_id in brains:
            # Check circuit breaker
            if self._is_circuit_breaker_open(brain_id):
                continue
            
            # Check health metrics
            metrics = self.brain_metrics.get(brain_id)
            if metrics and self._is_brain_healthy(metrics):
                healthy_brains.append(brain_id)
        
        return healthy_brains
    
    def _is_brain_healthy(self, metrics: LoadMetrics) -> bool:
        """Check if brain is healthy based on metrics"""
        # Check error rate
        if metrics.error_rate > 0.1:  # 10% error rate threshold
            return False
        
        # Check resource utilization
        if (metrics.cpu_usage > 0.9 or 
            metrics.memory_usage > 0.9 or 
            metrics.gpu_usage > 0.9):
            return False
        
        # Check response time
        if metrics.average_response_time > 10.0:  # 10 second threshold
            return False
        
        return True
    
    def _is_circuit_breaker_open(self, brain_id: str) -> bool:
        """Check if circuit breaker is open for brain"""
        breaker = self.circuit_breakers.get(brain_id)
        if not breaker:
            return False
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if datetime.now() > breaker['timeout']:
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    async def update_brain_metrics(self, brain_id: str, cpu_usage: float, memory_usage: float,
                                 gpu_usage: float, active_connections: int, 
                                 requests_per_second: float, average_response_time: float,
                                 error_rate: float, queue_length: int):
        """Update metrics for a brain instance"""
        try:
            metrics = LoadMetrics(
                brain_id=brain_id,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                active_connections=active_connections,
                requests_per_second=requests_per_second,
                average_response_time=average_response_time,
                error_rate=error_rate,
                queue_length=queue_length,
                last_updated=datetime.now()
            )
            
            self.brain_metrics[brain_id] = metrics
            
            # Update performance history
            if brain_id not in self.performance_history:
                self.performance_history[brain_id] = []
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(metrics)
            self.performance_history[brain_id].append(performance_score)
            
            # Keep only recent history
            max_history = self.config['performance_window_minutes']
            if len(self.performance_history[brain_id]) > max_history:
                self.performance_history[brain_id] = self.performance_history[brain_id][-max_history:]
            
            # Update circuit breaker
            await self._update_circuit_breaker(brain_id, metrics)
            
            # Store in Redis
            await self._store_brain_metrics(metrics)
            
        except Exception as e:
            logger.error(f"❌ Failed to update brain metrics: {e}")
    
    def _calculate_performance_score(self, metrics: LoadMetrics) -> float:
        """Calculate performance score for a brain"""
        try:
            # Normalize metrics to 0-1 scale
            cpu_score = 1.0 - metrics.cpu_usage
            memory_score = 1.0 - metrics.memory_usage
            gpu_score = 1.0 - metrics.gpu_usage
            
            # Response time score (lower is better)
            response_score = 1.0 / (1.0 + metrics.average_response_time)
            
            # Error rate score (lower is better)
            error_score = 1.0 - metrics.error_rate
            
            # Queue score (lower is better)
            queue_score = 1.0 / (1.0 + metrics.queue_length * 0.1)
            
            # Weighted average
            performance_score = (
                cpu_score * 0.2 +
                memory_score * 0.2 +
                gpu_score * 0.2 +
                response_score * 0.2 +
                error_score * 0.1 +
                queue_score * 0.1
            )
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            logger.error(f"❌ Performance score calculation failed: {e}")
            return 0.5  # Default neutral score
    
    async def _update_circuit_breaker(self, brain_id: str, metrics: LoadMetrics):
        """Update circuit breaker state based on metrics"""
        try:
            if brain_id not in self.circuit_breakers:
                self.circuit_breakers[brain_id] = {
                    'state': 'closed',
                    'failure_count': 0,
                    'timeout': None
                }
            
            breaker = self.circuit_breakers[brain_id]
            
            # Check for failures
            if metrics.error_rate > self.config['circuit_breaker_threshold']:
                breaker['failure_count'] += 1
                
                if breaker['failure_count'] >= 5:  # Threshold for opening
                    breaker['state'] = 'open'
                    breaker['timeout'] = datetime.now() + timedelta(
                        seconds=self.config['circuit_breaker_timeout']
                    )
                    logger.warning(f"🔴 Circuit breaker opened for brain {brain_id}")
            else:
                # Reset failure count on success
                if breaker['state'] == 'half_open':
                    breaker['state'] = 'closed'
                    logger.info(f"🟢 Circuit breaker closed for brain {brain_id}")
                
                breaker['failure_count'] = max(0, breaker['failure_count'] - 1)
            
        except Exception as e:
            logger.error(f"❌ Circuit breaker update failed: {e}")
    
    async def _metrics_collector(self):
        """Background metrics collection"""
        while True:
            try:
                await asyncio.sleep(self.config['metrics_collection_interval'])
                
                # Update load balancing metrics
                await self._update_load_distribution()
                await self._calculate_strategy_effectiveness()
                
                # Store metrics
                await self._store_lb_metrics()
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
    
    async def _weight_adjuster(self):
        """Background weight adjustment based on performance"""
        while True:
            try:
                await asyncio.sleep(self.config['weight_adjustment_interval'])
                
                for brain_id in self.brain_metrics.keys():
                    await self._adjust_brain_weight(brain_id)
                
            except Exception as e:
                logger.error(f"❌ Weight adjustment error: {e}")
    
    async def _adjust_brain_weight(self, brain_id: str):
        """Adjust brain weight based on performance history"""
        try:
            if brain_id not in self.performance_history:
                return
            
            history = self.performance_history[brain_id]
            if len(history) < 3:  # Need minimum history
                return
            
            # Calculate average performance
            avg_performance = statistics.mean(history)
            
            # Get or create weight
            if brain_id not in self.brain_weights:
                self.brain_weights[brain_id] = BrainWeight(
                    brain_id=brain_id,
                    base_weight=1.0,
                    dynamic_weight=1.0,
                    performance_factor=1.0,
                    health_factor=1.0,
                    final_weight=1.0
                )
            
            weight = self.brain_weights[brain_id]
            
            # Adjust performance factor
            learning_rate = self.config['adaptive_learning_rate']
            weight.performance_factor = (
                weight.performance_factor * (1 - learning_rate) +
                avg_performance * learning_rate
            )
            
            # Adjust health factor based on current metrics
            metrics = self.brain_metrics.get(brain_id)
            if metrics:
                if self._is_brain_healthy(metrics):
                    weight.health_factor = min(1.0, weight.health_factor + 0.1)
                else:
                    weight.health_factor = max(0.1, weight.health_factor - 0.1)
            
            # Calculate final weight
            weight.final_weight = (
                weight.base_weight *
                weight.dynamic_weight *
                weight.performance_factor *
                weight.health_factor
            )
            
        except Exception as e:
            logger.error(f"❌ Weight adjustment failed for {brain_id}: {e}")
    
    async def _circuit_breaker_monitor(self):
        """Monitor and manage circuit breakers"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                
                for brain_id, breaker in self.circuit_breakers.items():
                    if (breaker['state'] == 'open' and 
                        breaker['timeout'] and 
                        current_time > breaker['timeout']):
                        
                        breaker['state'] = 'half_open'
                        logger.info(f"🟡 Circuit breaker half-open for brain {brain_id}")
                
            except Exception as e:
                logger.error(f"❌ Circuit breaker monitor error: {e}")
    
    async def _initialize_default_rules(self):
        """Initialize default load balancing rules"""
        self.balancing_rules = {
            'default_adaptive': LoadBalancingRule(
                rule_id='default_adaptive',
                strategy=LoadBalancingStrategy.ADAPTIVE,
                weight=1.0,
                conditions={},
                priority=1,
                enabled=True
            ),
            'high_load_resource_based': LoadBalancingRule(
                rule_id='high_load_resource_based',
                strategy=LoadBalancingStrategy.RESOURCE_BASED,
                weight=1.5,
                conditions={'system_load': '>0.8'},
                priority=2,
                enabled=True
            )
        }
    
    async def _update_load_distribution(self):
        """Update load distribution metrics"""
        total_requests = sum(self.lb_metrics['requests_by_brain'].values())
        if total_requests > 0:
            for brain_id, requests in self.lb_metrics['requests_by_brain'].items():
                self.lb_metrics['load_distribution'][brain_id] = requests / total_requests
    
    async def _calculate_strategy_effectiveness(self):
        """Calculate effectiveness of different strategies"""
        # This would analyze performance data to determine strategy effectiveness
        # For now, placeholder implementation
        pass
    
    async def _store_brain_metrics(self, metrics: LoadMetrics):
        """Store brain metrics in Redis"""
        if self.redis_client:
            try:
                key = f"brain_metrics:{metrics.brain_id}"
                data = json.dumps(asdict(metrics), default=str)
                await self.redis_client.setex(key, 300, data)  # 5 minute TTL
            except Exception as e:
                logger.error(f"Failed to store brain metrics: {e}")
    
    async def _load_brain_metrics(self):
        """Load brain metrics from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("brain_metrics:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        metrics_data = json.loads(data)
                        # Convert back to LoadMetrics object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load brain metrics: {e}")
    
    async def _store_lb_metrics(self):
        """Store load balancing metrics in Redis"""
        if self.redis_client:
            try:
                key = "load_balancer_metrics"
                data = json.dumps(self.lb_metrics, default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store LB metrics: {e}")
    
    async def get_load_balancing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing metrics"""
        return {
            'metrics': self.lb_metrics.copy(),
            'brain_weights': {k: asdict(v) for k, v in self.brain_weights.items()},
            'circuit_breakers': self.circuit_breakers.copy(),
            'configuration': self.config,
            'active_brains': len(self.brain_metrics),
            'timestamp': datetime.now().isoformat()
        }

# Global load balancer instance
load_balancer = LoadBalancer()

async def initialize_load_balancer():
    """Initialize the global load balancer"""
    await load_balancer.initialize()

if __name__ == "__main__":
    # Test the load balancer
    async def test_load_balancer():
        await initialize_load_balancer()
        
        # Update test metrics
        await load_balancer.update_brain_metrics(
            "brain1", 0.3, 0.4, 0.5, 2, 10.0, 0.1, 0.02, 1
        )
        await load_balancer.update_brain_metrics(
            "brain2", 0.6, 0.7, 0.8, 5, 8.0, 0.2, 0.05, 3
        )
        
        # Test brain selection
        selected = await load_balancer.select_brain(
            "embedding", ["brain1", "brain2"], LoadBalancingStrategy.ADAPTIVE
        )
        print(f"Selected brain: {selected}")
        
        # Get metrics
        metrics = await load_balancer.get_load_balancing_metrics()
        print(f"Load balancing metrics: {metrics}")
    
    asyncio.run(test_load_balancer())
