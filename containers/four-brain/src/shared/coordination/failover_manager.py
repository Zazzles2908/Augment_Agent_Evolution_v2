"""
Failover Management System for Four-Brain System v2
Automatic failover and recovery management for high availability

Created: 2025-07-30 AEST
Purpose: Ensure system resilience and automatic recovery from brain failures
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailoverTrigger(Enum):
    """Failover trigger conditions"""
    HEALTH_CHECK_FAILURE = "health_check_failure"
    RESPONSE_TIMEOUT = "response_timeout"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MANUAL_TRIGGER = "manual_trigger"
    CIRCUIT_BREAKER = "circuit_breaker"

class FailoverStatus(Enum):
    """Failover operation status"""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED_OVER = "failed_over"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    MANUAL = "manual"
    SCHEDULED = "scheduled"

@dataclass
class FailoverEvent:
    """Failover event record"""
    event_id: str
    trigger: FailoverTrigger
    failed_brain: str
    backup_brain: Optional[str]
    timestamp: datetime
    duration: Optional[float]
    success: bool
    error_message: Optional[str]
    recovery_time: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class BrainFailoverConfig:
    """Failover configuration for a brain"""
    brain_id: str
    backup_brains: List[str]
    health_check_interval: int
    failure_threshold: int
    recovery_strategy: RecoveryStrategy
    auto_recovery_enabled: bool
    priority: int
    metadata: Dict[str, Any]

@dataclass
class FailoverMetrics:
    """Failover system metrics"""
    total_failovers: int
    successful_failovers: int
    failed_failovers: int
    average_failover_time: float
    average_recovery_time: float
    failovers_by_trigger: Dict[str, int]
    failovers_by_brain: Dict[str, int]
    current_failed_brains: int

class FailoverManager:
    """
    Comprehensive failover management system
    
    Features:
    - Automatic failure detection and failover
    - Multiple backup brain strategies
    - Health monitoring and recovery
    - Graceful degradation
    - Recovery automation
    - Failover metrics and reporting
    - Manual failover controls
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/13"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Failover state
        self.brain_configs: Dict[str, BrainFailoverConfig] = {}
        self.brain_status: Dict[str, FailoverStatus] = {}
        self.active_failovers: Dict[str, FailoverEvent] = {}
        self.failover_history: List[FailoverEvent] = []
        
        # Health monitoring
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Configuration
        self.config = {
            'health_check_interval': 30,
            'failure_threshold': 3,
            'failover_timeout': 60,
            'recovery_check_interval': 120,
            'max_concurrent_failovers': 2,
            'graceful_shutdown_timeout': 30,
            'backup_brain_selection_strategy': 'priority',
            'auto_recovery_enabled': True
        }
        
        # Failover metrics
        self.metrics = FailoverMetrics(
            total_failovers=0,
            successful_failovers=0,
            failed_failovers=0,
            average_failover_time=0.0,
            average_recovery_time=0.0,
            failovers_by_trigger={trigger.value: 0 for trigger in FailoverTrigger},
            failovers_by_brain={},
            current_failed_brains=0
        )
        
        # Event callbacks
        self.failover_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        logger.info("🔄 Failover Manager initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start failover monitoring"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load existing configurations
            await self._load_failover_configs()
            
            # Start monitoring services
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._recovery_monitor())
            asyncio.create_task(self._metrics_collector())
            
            logger.info("✅ Failover Manager Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Failover Manager: {e}")
            raise
    
    async def register_brain_failover(self, brain_id: str, backup_brains: List[str],
                                    health_check_interval: int = 30,
                                    failure_threshold: int = 3,
                                    recovery_strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL,
                                    auto_recovery_enabled: bool = True,
                                    priority: int = 1) -> bool:
        """Register failover configuration for a brain"""
        try:
            config = BrainFailoverConfig(
                brain_id=brain_id,
                backup_brains=backup_brains,
                health_check_interval=health_check_interval,
                failure_threshold=failure_threshold,
                recovery_strategy=recovery_strategy,
                auto_recovery_enabled=auto_recovery_enabled,
                priority=priority,
                metadata={}
            )
            
            self.brain_configs[brain_id] = config
            self.brain_status[brain_id] = FailoverStatus.ACTIVE
            self.failure_counts[brain_id] = 0
            
            # Initialize health check
            self.health_checks[brain_id] = {
                'last_check': datetime.now(),
                'consecutive_failures': 0,
                'last_success': datetime.now(),
                'status': 'healthy'
            }
            
            # Store configuration
            await self._store_failover_config(config)
            
            logger.info(f"✅ Failover registered for brain {brain_id} with {len(backup_brains)} backups")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register failover for {brain_id}: {e}")
            return False
    
    async def trigger_failover(self, brain_id: str, trigger: FailoverTrigger,
                             error_message: str = None, manual: bool = False) -> bool:
        """Trigger failover for a specific brain"""
        try:
            # Check if brain is configured for failover
            if brain_id not in self.brain_configs:
                logger.error(f"Brain {brain_id} not configured for failover")
                return False
            
            # Check if already failed over
            if self.brain_status[brain_id] == FailoverStatus.FAILED_OVER:
                logger.warning(f"Brain {brain_id} already failed over")
                return True
            
            # Check concurrent failover limit
            active_failovers = len([f for f in self.active_failovers.values() if not f.success])
            if active_failovers >= self.config['max_concurrent_failovers'] and not manual:
                logger.warning(f"Maximum concurrent failovers reached ({active_failovers})")
                return False
            
            # Create failover event
            event_id = f"failover_{brain_id}_{int(time.time())}"
            failover_event = FailoverEvent(
                event_id=event_id,
                trigger=trigger,
                failed_brain=brain_id,
                backup_brain=None,
                timestamp=datetime.now(),
                duration=None,
                success=False,
                error_message=error_message,
                recovery_time=None,
                metadata={'manual': manual}
            )
            
            # Execute failover
            success = await self._execute_failover(failover_event)
            
            if success:
                # Update status
                self.brain_status[brain_id] = FailoverStatus.FAILED_OVER
                self.metrics.successful_failovers += 1
                
                # Notify callbacks
                for callback in self.failover_callbacks:
                    try:
                        await callback(failover_event)
                    except Exception as e:
                        logger.error(f"Failover callback error: {e}")
                
                logger.warning(f"🔄 Failover completed for brain {brain_id}")
            else:
                self.metrics.failed_failovers += 1
                logger.error(f"❌ Failover failed for brain {brain_id}")
            
            # Update metrics
            self.metrics.total_failovers += 1
            self.metrics.failovers_by_trigger[trigger.value] += 1
            self.metrics.failovers_by_brain[brain_id] = (
                self.metrics.failovers_by_brain.get(brain_id, 0) + 1
            )
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failover trigger failed: {e}")
            return False
    
    async def _execute_failover(self, failover_event: FailoverEvent) -> bool:
        """Execute the actual failover process"""
        try:
            start_time = time.time()
            brain_id = failover_event.failed_brain
            config = self.brain_configs[brain_id]
            
            # Find suitable backup brain
            backup_brain = await self._select_backup_brain(brain_id, config.backup_brains)
            if not backup_brain:
                failover_event.error_message = "No suitable backup brain available"
                return False
            
            failover_event.backup_brain = backup_brain
            
            # Perform graceful shutdown of failed brain
            await self._graceful_shutdown(brain_id)
            
            # Activate backup brain
            success = await self._activate_backup_brain(backup_brain, brain_id)
            if not success:
                failover_event.error_message = "Failed to activate backup brain"
                return False
            
            # Update routing to use backup brain
            await self._update_routing(brain_id, backup_brain)
            
            # Calculate failover duration
            failover_event.duration = time.time() - start_time
            failover_event.success = True
            
            # Store failover event
            self.active_failovers[failover_event.event_id] = failover_event
            self.failover_history.append(failover_event)
            await self._store_failover_event(failover_event)
            
            # Update average failover time
            self._update_average_failover_time(failover_event.duration)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failover execution failed: {e}")
            failover_event.error_message = str(e)
            return False
    
    async def _select_backup_brain(self, failed_brain: str, backup_brains: List[str]) -> Optional[str]:
        """Select the best available backup brain"""
        try:
            available_backups = []
            
            for backup_id in backup_brains:
                # Check if backup is healthy and available
                if (backup_id in self.brain_status and
                    self.brain_status[backup_id] == FailoverStatus.STANDBY and
                    await self._is_brain_healthy(backup_id)):
                    available_backups.append(backup_id)
            
            if not available_backups:
                return None
            
            # Select based on strategy
            strategy = self.config['backup_brain_selection_strategy']
            
            if strategy == 'priority':
                # Select highest priority backup
                return min(available_backups, 
                          key=lambda b: self.brain_configs.get(b, BrainFailoverConfig(b, [], 30, 3, RecoveryStrategy.GRADUAL, True, 999, {})).priority)
            elif strategy == 'random':
                import random
                return random.choice(available_backups)
            else:
                return available_backups[0]
            
        except Exception as e:
            logger.error(f"❌ Backup brain selection failed: {e}")
            return None
    
    async def _is_brain_healthy(self, brain_id: str) -> bool:
        """Check if brain is healthy for failover"""
        try:
            health_check = self.health_checks.get(brain_id)
            if not health_check:
                return False
            
            # Check recent health status
            if health_check['status'] != 'healthy':
                return False
            
            # Check if last success was recent
            time_since_success = datetime.now() - health_check['last_success']
            if time_since_success.total_seconds() > 300:  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {brain_id}: {e}")
            return False
    
    async def _graceful_shutdown(self, brain_id: str):
        """Perform graceful shutdown of failed brain"""
        try:
            # This would implement graceful shutdown logic
            # For now, just mark as shutting down
            logger.info(f"🔄 Gracefully shutting down brain {brain_id}")
            
            # Wait for current tasks to complete
            await asyncio.sleep(self.config['graceful_shutdown_timeout'])
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown failed for {brain_id}: {e}")
    
    async def _activate_backup_brain(self, backup_brain: str, failed_brain: str) -> bool:
        """Activate backup brain to replace failed brain"""
        try:
            logger.info(f"🔄 Activating backup brain {backup_brain} for {failed_brain}")
            
            # Update status
            self.brain_status[backup_brain] = FailoverStatus.ACTIVE
            
            # This would implement actual brain activation logic
            # For now, simulate activation
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup brain activation failed: {e}")
            return False
    
    async def _update_routing(self, failed_brain: str, backup_brain: str):
        """Update routing to use backup brain"""
        try:
            # This would update load balancer and routing tables
            # For now, just log the change
            logger.info(f"🔄 Updating routing: {failed_brain} -> {backup_brain}")
            
        except Exception as e:
            logger.error(f"❌ Routing update failed: {e}")
    
    async def _health_monitor(self):
        """Monitor brain health and trigger failovers"""
        while True:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                for brain_id, config in self.brain_configs.items():
                    if self.brain_status[brain_id] == FailoverStatus.ACTIVE:
                        await self._check_brain_health(brain_id, config)
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
    
    async def _check_brain_health(self, brain_id: str, config: BrainFailoverConfig):
        """Check health of a specific brain"""
        try:
            health_check = self.health_checks[brain_id]
            
            # Simulate health check (would be actual API call)
            is_healthy = await self._perform_health_check(brain_id)
            
            if is_healthy:
                # Reset failure count on success
                self.failure_counts[brain_id] = 0
                health_check['consecutive_failures'] = 0
                health_check['last_success'] = datetime.now()
                health_check['status'] = 'healthy'
            else:
                # Increment failure count
                self.failure_counts[brain_id] += 1
                health_check['consecutive_failures'] += 1
                health_check['status'] = 'unhealthy'
                
                # Check if failure threshold reached
                if self.failure_counts[brain_id] >= config.failure_threshold:
                    await self.trigger_failover(
                        brain_id, 
                        FailoverTrigger.HEALTH_CHECK_FAILURE,
                        f"Health check failed {self.failure_counts[brain_id]} times"
                    )
            
            health_check['last_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Health check failed for {brain_id}: {e}")
    
    async def _perform_health_check(self, brain_id: str) -> bool:
        """Perform actual health check on brain"""
        try:
            # This would make actual HTTP health check
            # For now, simulate with random success/failure
            import random
            return random.random() > 0.1  # 90% success rate
            
        except Exception as e:
            logger.error(f"❌ Health check API call failed for {brain_id}: {e}")
            return False
    
    async def _recovery_monitor(self):
        """Monitor for brain recovery and automatic failback"""
        while True:
            try:
                await asyncio.sleep(self.config['recovery_check_interval'])
                
                # Check for brains that can be recovered
                for brain_id, status in self.brain_status.items():
                    if status == FailoverStatus.FAILED_OVER:
                        config = self.brain_configs.get(brain_id)
                        if config and config.auto_recovery_enabled:
                            await self._check_recovery(brain_id, config)
                
            except Exception as e:
                logger.error(f"❌ Recovery monitor error: {e}")
    
    async def _check_recovery(self, brain_id: str, config: BrainFailoverConfig):
        """Check if brain can be recovered"""
        try:
            # Check if original brain is healthy again
            if await self._is_brain_healthy(brain_id):
                await self._initiate_recovery(brain_id, config)
            
        except Exception as e:
            logger.error(f"❌ Recovery check failed for {brain_id}: {e}")
    
    async def _initiate_recovery(self, brain_id: str, config: BrainFailoverConfig):
        """Initiate recovery process for a brain"""
        try:
            logger.info(f"🔄 Initiating recovery for brain {brain_id}")
            
            self.brain_status[brain_id] = FailoverStatus.RECOVERING
            
            if config.recovery_strategy == RecoveryStrategy.IMMEDIATE:
                await self._immediate_recovery(brain_id)
            elif config.recovery_strategy == RecoveryStrategy.GRADUAL:
                await self._gradual_recovery(brain_id)
            else:
                logger.info(f"Manual recovery required for brain {brain_id}")
                return
            
            # Update status
            self.brain_status[brain_id] = FailoverStatus.ACTIVE
            
            # Find and update failover event
            for event in self.active_failovers.values():
                if event.failed_brain == brain_id and not event.recovery_time:
                    event.recovery_time = datetime.now()
                    recovery_duration = (event.recovery_time - event.timestamp).total_seconds()
                    self._update_average_recovery_time(recovery_duration)
                    break
            
            # Notify callbacks
            for callback in self.recovery_callbacks:
                try:
                    await callback(brain_id)
                except Exception as e:
                    logger.error(f"Recovery callback error: {e}")
            
            logger.info(f"✅ Recovery completed for brain {brain_id}")
            
        except Exception as e:
            logger.error(f"❌ Recovery initiation failed for {brain_id}: {e}")
    
    async def _immediate_recovery(self, brain_id: str):
        """Perform immediate recovery"""
        # Switch traffic back immediately
        await self._switch_traffic_back(brain_id)
    
    async def _gradual_recovery(self, brain_id: str):
        """Perform gradual recovery"""
        # Gradually shift traffic back
        await self._gradual_traffic_shift(brain_id)
    
    async def _switch_traffic_back(self, brain_id: str):
        """Switch traffic back to recovered brain"""
        logger.info(f"🔄 Switching traffic back to {brain_id}")
        # Implementation would update routing tables
    
    async def _gradual_traffic_shift(self, brain_id: str):
        """Gradually shift traffic back to recovered brain"""
        logger.info(f"🔄 Gradually shifting traffic back to {brain_id}")
        # Implementation would gradually increase traffic percentage
    
    async def _metrics_collector(self):
        """Collect failover metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update current failed brains count
                self.metrics.current_failed_brains = len([
                    s for s in self.brain_status.values() 
                    if s == FailoverStatus.FAILED_OVER
                ])
                
                # Store metrics
                await self._store_metrics()
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
    
    def _update_average_failover_time(self, duration: float):
        """Update average failover time"""
        if self.metrics.successful_failovers == 1:
            self.metrics.average_failover_time = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_failover_time = (
                alpha * duration + 
                (1 - alpha) * self.metrics.average_failover_time
            )
    
    def _update_average_recovery_time(self, duration: float):
        """Update average recovery time"""
        if self.metrics.average_recovery_time == 0:
            self.metrics.average_recovery_time = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_recovery_time = (
                alpha * duration + 
                (1 - alpha) * self.metrics.average_recovery_time
            )
    
    async def _store_failover_config(self, config: BrainFailoverConfig):
        """Store failover configuration in Redis"""
        if self.redis_client:
            try:
                key = f"failover_config:{config.brain_id}"
                data = json.dumps(asdict(config), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store failover config: {e}")
    
    async def _store_failover_event(self, event: FailoverEvent):
        """Store failover event in Redis"""
        if self.redis_client:
            try:
                key = f"failover_event:{event.event_id}"
                data = json.dumps(asdict(event), default=str)
                await self.redis_client.setex(key, 86400 * 30, data)  # 30 days retention
            except Exception as e:
                logger.error(f"Failed to store failover event: {e}")
    
    async def _load_failover_configs(self):
        """Load failover configurations from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("failover_config:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        config_data = json.loads(data)
                        # Convert back to BrainFailoverConfig object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load failover configs: {e}")
    
    async def _store_metrics(self):
        """Store failover metrics in Redis"""
        if self.redis_client:
            try:
                key = "failover_metrics"
                data = json.dumps(asdict(self.metrics), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store failover metrics: {e}")
    
    def register_failover_callback(self, callback: Callable):
        """Register callback for failover events"""
        self.failover_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """Register callback for recovery events"""
        self.recovery_callbacks.append(callback)
    
    async def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status"""
        return {
            'brain_status': {k: v.value for k, v in self.brain_status.items()},
            'active_failovers': len(self.active_failovers),
            'health_checks': self.health_checks.copy(),
            'failure_counts': self.failure_counts.copy(),
            'metrics': asdict(self.metrics),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global failover manager instance
failover_manager = FailoverManager()

async def initialize_failover_manager():
    """Initialize the global failover manager"""
    await failover_manager.initialize()

if __name__ == "__main__":
    # Test the failover manager
    async def test_failover_manager():
        await initialize_failover_manager()
        
        # Register test brain
        await failover_manager.register_brain_failover(
            "brain1", ["brain1_backup"], 30, 3, RecoveryStrategy.GRADUAL, True, 1
        )
        
        # Trigger test failover
        success = await failover_manager.trigger_failover(
            "brain1", FailoverTrigger.MANUAL_TRIGGER, "Test failover", True
        )
        
        print(f"Failover success: {success}")
        
        # Get status
        status = await failover_manager.get_failover_status()
        print(f"Failover status: {status}")
    
    asyncio.run(test_failover_manager())
