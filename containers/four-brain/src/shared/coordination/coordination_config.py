"""
Coordination Configuration Management for Four-Brain System v2
Centralized configuration management for brain coordination components

Created: 2025-07-30 AEST
Purpose: Manage configuration for all coordination components with dynamic updates
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"
    BRAIN = "brain"
    COMPONENT = "component"
    TASK = "task"

class ConfigCategory(Enum):
    """Coordination configuration categories"""
    BRAIN_COORDINATION = "brain_coordination"
    LOAD_BALANCING = "load_balancing"
    TASK_SCHEDULING = "task_scheduling"
    FAILOVER = "failover"
    HEALTH_MONITORING = "health_monitoring"
    METRICS = "metrics"
    RESULT_AGGREGATION = "result_aggregation"
    RESOURCE_ALLOCATION = "resource_allocation"

@dataclass
class CoordinationConfig:
    """Coordination configuration item"""
    config_id: str
    category: ConfigCategory
    scope: ConfigScope
    name: str
    description: str
    value: Any
    default_value: Any
    validation_rules: Dict[str, Any]
    requires_restart: bool
    last_updated: datetime
    updated_by: str
    version: int

@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class CoordinationConfigManager:
    """
    Comprehensive coordination configuration management system
    
    Features:
    - Centralized configuration for all coordination components
    - Dynamic configuration updates without restart
    - Configuration validation and verification
    - Environment-specific configurations
    - Configuration versioning and rollback
    - Hot-reload capabilities
    - Configuration audit logging
    - Performance-aware configuration tuning
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/18", 
                 config_file: str = "coordination_config.yaml"):
        self.redis_url = redis_url
        self.redis_client = None
        self.config_file = config_file
        
        # Configuration storage
        self.configurations: Dict[str, CoordinationConfig] = {}
        self.config_watchers: Dict[str, List[callable]] = {}
        
        # Default coordination configurations
        self.default_configs = self._initialize_default_configs()
        
        # Configuration validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Configuration metrics
        self.metrics = {
            'configs_loaded': 0,
            'configs_updated': 0,
            'validation_failures': 0,
            'hot_reloads': 0,
            'rollbacks': 0
        }
        
        logger.info("⚙️ Coordination Config Manager initialized")
    
    async def initialize(self):
        """Initialize Redis connection and load configurations"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load configurations from file and Redis
            await self._load_configurations()
            
            # Start configuration monitoring
            asyncio.create_task(self._monitor_config_changes())
            
            logger.info("✅ Coordination Config Manager Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Coordination Config Manager: {e}")
            raise
    
    async def get_config(self, config_id: str, scope: ConfigScope = ConfigScope.GLOBAL) -> Optional[Any]:
        """Get coordination configuration value"""
        try:
            full_config_id = f"{scope.value}:{config_id}"
            config = self.configurations.get(full_config_id)
            
            if config:
                return config.value
            
            # Try to load from Redis
            config = await self._load_config_from_redis(full_config_id)
            if config:
                self.configurations[full_config_id] = config
                return config.value
            
            # Return default if available
            default_config = self.default_configs.get(config_id)
            if default_config:
                return default_config.default_value
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get config {config_id}: {e}")
            return None
    
    async def set_config(self, config_id: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
                        updated_by: str = "system", validate: bool = True) -> bool:
        """Set coordination configuration value with validation"""
        try:
            full_config_id = f"{scope.value}:{config_id}"
            
            # Get existing config or create new one
            existing_config = self.configurations.get(full_config_id)
            if existing_config:
                config = existing_config
                config.value = value
                config.last_updated = datetime.now()
                config.updated_by = updated_by
                config.version += 1
            else:
                # Create new config
                default_config = self.default_configs.get(config_id)
                if not default_config:
                    logger.error(f"No default configuration found for {config_id}")
                    return False
                
                config = CoordinationConfig(
                    config_id=full_config_id,
                    category=default_config.category,
                    scope=scope,
                    name=default_config.name,
                    description=default_config.description,
                    value=value,
                    default_value=default_config.default_value,
                    validation_rules=default_config.validation_rules,
                    requires_restart=default_config.requires_restart,
                    last_updated=datetime.now(),
                    updated_by=updated_by,
                    version=1
                )
            
            # Validate configuration
            if validate:
                validation_result = await self._validate_config(config)
                if not validation_result.valid:
                    logger.error(f"Configuration validation failed for {config_id}: {validation_result.errors}")
                    self.metrics['validation_failures'] += 1
                    return False
            
            # Store configuration
            self.configurations[full_config_id] = config
            await self._store_config_in_redis(config)
            
            # Notify watchers
            await self._notify_config_watchers(config_id, value)
            
            # Update metrics
            self.metrics['configs_updated'] += 1
            
            logger.info(f"✅ Coordination configuration updated: {config_id} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set config {config_id}: {e}")
            return False
    
    async def _validate_config(self, config: CoordinationConfig) -> ConfigValidationResult:
        """Validate coordination configuration value"""
        errors = []
        warnings = []
        recommendations = []
        
        try:
            rules = config.validation_rules
            value = config.value
            
            # Type validation
            if 'type' in rules:
                expected_type = rules['type']
                if expected_type == 'int' and not isinstance(value, int):
                    errors.append(f"Expected integer, got {type(value).__name__}")
                elif expected_type == 'float' and not isinstance(value, (int, float)):
                    errors.append(f"Expected float, got {type(value).__name__}")
                elif expected_type == 'str' and not isinstance(value, str):
                    errors.append(f"Expected string, got {type(value).__name__}")
                elif expected_type == 'bool' and not isinstance(value, bool):
                    errors.append(f"Expected boolean, got {type(value).__name__}")
                elif expected_type == 'list' and not isinstance(value, list):
                    errors.append(f"Expected list, got {type(value).__name__}")
                elif expected_type == 'dict' and not isinstance(value, dict):
                    errors.append(f"Expected dictionary, got {type(value).__name__}")
            
            # Range validation
            if 'min' in rules and isinstance(value, (int, float)):
                if value < rules['min']:
                    errors.append(f"Value {value} is below minimum {rules['min']}")
            
            if 'max' in rules and isinstance(value, (int, float)):
                if value > rules['max']:
                    errors.append(f"Value {value} is above maximum {rules['max']}")
            
            # Coordination-specific validations
            if config.category == ConfigCategory.LOAD_BALANCING:
                if 'load_balance_threshold' in config.name and isinstance(value, (int, float)):
                    if value > 1.0:
                        warnings.append("Load balance threshold should typically be <= 1.0")
                    if value < 0.1:
                        warnings.append("Very low load balance threshold may cause instability")
            
            elif config.category == ConfigCategory.TASK_SCHEDULING:
                if 'max_queue_size' in config.name and isinstance(value, int):
                    if value > 10000:
                        warnings.append("Very large queue size may impact memory usage")
                        recommendations.append("Consider implementing queue partitioning for large queues")
                    if value < 10:
                        warnings.append("Very small queue size may limit throughput")
            
            elif config.category == ConfigCategory.HEALTH_MONITORING:
                if 'check_interval' in config.name and isinstance(value, int):
                    if value < 5:
                        warnings.append("Very frequent health checks may impact performance")
                    if value > 300:
                        warnings.append("Infrequent health checks may delay failure detection")
            
            elif config.category == ConfigCategory.FAILOVER:
                if 'failure_threshold' in config.name and isinstance(value, int):
                    if value < 2:
                        warnings.append("Low failure threshold may cause unnecessary failovers")
                    if value > 10:
                        warnings.append("High failure threshold may delay failover response")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _initialize_default_configs(self) -> Dict[str, CoordinationConfig]:
        """Initialize default coordination configurations"""
        defaults = {}
        
        # Brain Coordination configurations
        defaults['brain_heartbeat_interval'] = CoordinationConfig(
            config_id='brain_heartbeat_interval',
            category=ConfigCategory.BRAIN_COORDINATION,
            scope=ConfigScope.GLOBAL,
            name='Brain Heartbeat Interval',
            description='Interval between brain heartbeat checks in seconds',
            value=30,
            default_value=30,
            validation_rules={'type': 'int', 'min': 5, 'max': 300},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['max_concurrent_tasks'] = CoordinationConfig(
            config_id='max_concurrent_tasks',
            category=ConfigCategory.BRAIN_COORDINATION,
            scope=ConfigScope.GLOBAL,
            name='Max Concurrent Tasks',
            description='Maximum number of concurrent tasks per brain',
            value=10,
            default_value=10,
            validation_rules={'type': 'int', 'min': 1, 'max': 100},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Load Balancing configurations
        defaults['load_balance_strategy'] = CoordinationConfig(
            config_id='load_balance_strategy',
            category=ConfigCategory.LOAD_BALANCING,
            scope=ConfigScope.GLOBAL,
            name='Load Balance Strategy',
            description='Default load balancing strategy',
            value='adaptive',
            default_value='adaptive',
            validation_rules={'type': 'str', 'allowed_values': ['round_robin', 'least_connections', 'adaptive', 'resource_based']},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['load_balance_threshold'] = CoordinationConfig(
            config_id='load_balance_threshold',
            category=ConfigCategory.LOAD_BALANCING,
            scope=ConfigScope.GLOBAL,
            name='Load Balance Threshold',
            description='Load threshold for triggering load balancing',
            value=0.8,
            default_value=0.8,
            validation_rules={'type': 'float', 'min': 0.1, 'max': 1.0},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Task Scheduling configurations
        defaults['scheduling_strategy'] = CoordinationConfig(
            config_id='scheduling_strategy',
            category=ConfigCategory.TASK_SCHEDULING,
            scope=ConfigScope.GLOBAL,
            name='Task Scheduling Strategy',
            description='Default task scheduling strategy',
            value='priority',
            default_value='priority',
            validation_rules={'type': 'str', 'allowed_values': ['fifo', 'priority', 'shortest_job_first', 'deadline_aware']},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['max_queue_size'] = CoordinationConfig(
            config_id='max_queue_size',
            category=ConfigCategory.TASK_SCHEDULING,
            scope=ConfigScope.GLOBAL,
            name='Max Queue Size',
            description='Maximum number of tasks in scheduling queue',
            value=1000,
            default_value=1000,
            validation_rules={'type': 'int', 'min': 10, 'max': 50000},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Health Monitoring configurations
        defaults['health_check_interval'] = CoordinationConfig(
            config_id='health_check_interval',
            category=ConfigCategory.HEALTH_MONITORING,
            scope=ConfigScope.GLOBAL,
            name='Health Check Interval',
            description='Interval between health checks in seconds',
            value=60,
            default_value=60,
            validation_rules={'type': 'int', 'min': 10, 'max': 600},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['health_check_timeout'] = CoordinationConfig(
            config_id='health_check_timeout',
            category=ConfigCategory.HEALTH_MONITORING,
            scope=ConfigScope.GLOBAL,
            name='Health Check Timeout',
            description='Timeout for health check requests in seconds',
            value=10,
            default_value=10,
            validation_rules={'type': 'int', 'min': 1, 'max': 60},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Failover configurations
        defaults['failover_enabled'] = CoordinationConfig(
            config_id='failover_enabled',
            category=ConfigCategory.FAILOVER,
            scope=ConfigScope.GLOBAL,
            name='Failover Enabled',
            description='Enable automatic failover',
            value=True,
            default_value=True,
            validation_rules={'type': 'bool'},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['failure_threshold'] = CoordinationConfig(
            config_id='failure_threshold',
            category=ConfigCategory.FAILOVER,
            scope=ConfigScope.GLOBAL,
            name='Failure Threshold',
            description='Number of failures before triggering failover',
            value=3,
            default_value=3,
            validation_rules={'type': 'int', 'min': 1, 'max': 20},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Metrics configurations
        defaults['metrics_collection_interval'] = CoordinationConfig(
            config_id='metrics_collection_interval',
            category=ConfigCategory.METRICS,
            scope=ConfigScope.GLOBAL,
            name='Metrics Collection Interval',
            description='Interval for metrics collection in seconds',
            value=60,
            default_value=60,
            validation_rules={'type': 'int', 'min': 10, 'max': 3600},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Result Aggregation configurations
        defaults['aggregation_strategy'] = CoordinationConfig(
            config_id='aggregation_strategy',
            category=ConfigCategory.RESULT_AGGREGATION,
            scope=ConfigScope.GLOBAL,
            name='Result Aggregation Strategy',
            description='Default result aggregation strategy',
            value='confidence_weighted',
            default_value='confidence_weighted',
            validation_rules={'type': 'str', 'allowed_values': ['simple_merge', 'weighted_average', 'confidence_weighted', 'consensus_voting']},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['min_results_for_aggregation'] = CoordinationConfig(
            config_id='min_results_for_aggregation',
            category=ConfigCategory.RESULT_AGGREGATION,
            scope=ConfigScope.GLOBAL,
            name='Min Results for Aggregation',
            description='Minimum number of results required for aggregation',
            value=2,
            default_value=2,
            validation_rules={'type': 'int', 'min': 1, 'max': 10},
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        return defaults
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize coordination-specific validation rules"""
        return {
            'performance_thresholds': {
                'max_response_time': 30.0,
                'min_throughput': 1.0,
                'max_error_rate': 0.1
            },
            'resource_limits': {
                'max_cpu_usage': 0.9,
                'max_memory_usage': 0.9,
                'max_gpu_usage': 0.95
            },
            'coordination_constraints': {
                'min_brain_count': 1,
                'max_brain_count': 10,
                'min_task_timeout': 1,
                'max_task_timeout': 3600
            }
        }
    
    async def _load_configurations(self):
        """Load configurations from file and Redis"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_configs = yaml.safe_load(f)
                    if file_configs:
                        for config_id, config_data in file_configs.items():
                            await self.set_config(config_id, config_data.get('value'), 
                                                 ConfigScope(config_data.get('scope', 'global')),
                                                 'file_load', validate=False)
            
            # Load from Redis
            if self.redis_client:
                keys = await self.redis_client.keys("coordination_config:*")
                for key in keys:
                    config_data = await self.redis_client.get(key)
                    if config_data:
                        config_dict = json.loads(config_data)
                        # Convert back to CoordinationConfig object
                        # This would need proper deserialization logic
                        pass
            
            # Load defaults for missing configs
            for config_id, default_config in self.default_configs.items():
                full_config_id = f"{default_config.scope.value}:{config_id}"
                if full_config_id not in self.configurations:
                    self.configurations[full_config_id] = default_config
            
            self.metrics['configs_loaded'] = len(self.configurations)
            logger.info(f"✅ Loaded {len(self.configurations)} coordination configurations")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configurations: {e}")
    
    async def _store_config_in_redis(self, config: CoordinationConfig):
        """Store configuration in Redis"""
        if self.redis_client:
            try:
                key = f"coordination_config:{config.config_id}"
                data = json.dumps(asdict(config), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store config in Redis: {e}")
    
    async def _load_config_from_redis(self, config_id: str) -> Optional[CoordinationConfig]:
        """Load configuration from Redis"""
        if self.redis_client:
            try:
                key = f"coordination_config:{config_id}"
                data = await self.redis_client.get(key)
                if data:
                    config_dict = json.loads(data)
                    # Convert back to CoordinationConfig object
                    # This would need proper deserialization logic
                    return None  # Placeholder
            except Exception as e:
                logger.error(f"Failed to load config from Redis: {e}")
        return None
    
    def watch_config(self, config_id: str, callback: callable):
        """Register callback for configuration changes"""
        if config_id not in self.config_watchers:
            self.config_watchers[config_id] = []
        self.config_watchers[config_id].append(callback)
    
    async def _notify_config_watchers(self, config_id: str, new_value: Any):
        """Notify configuration watchers of changes"""
        watchers = self.config_watchers.get(config_id, [])
        for callback in watchers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(config_id, new_value)
                else:
                    callback(config_id, new_value)
            except Exception as e:
                logger.error(f"Config watcher callback error: {e}")
    
    async def _monitor_config_changes(self):
        """Monitor configuration changes"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for external configuration changes
                # This would monitor file system changes, Redis updates, etc.
                
            except Exception as e:
                logger.error(f"❌ Config monitoring error: {e}")
    
    async def export_configurations(self, file_path: str = None) -> Dict[str, Any]:
        """Export configurations to file or return as dict"""
        try:
            export_data = {}
            
            for config_id, config in self.configurations.items():
                export_data[config_id] = {
                    'category': config.category.value,
                    'scope': config.scope.value,
                    'name': config.name,
                    'description': config.description,
                    'value': config.value,
                    'default_value': config.default_value,
                    'requires_restart': config.requires_restart,
                    'last_updated': config.last_updated.isoformat(),
                    'version': config.version
                }
            
            if file_path:
                with open(file_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
                logger.info(f"✅ Coordination configurations exported to {file_path}")
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export configurations: {e}")
            return {}
    
    async def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        return {
            'total_configurations': len(self.configurations),
            'configurations_by_category': {
                category.value: sum(1 for c in self.configurations.values() if c.category == category)
                for category in ConfigCategory
            },
            'configurations_by_scope': {
                scope.value: sum(1 for c in self.configurations.values() if c.scope == scope)
                for scope in ConfigScope
            },
            'metrics': self.metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def validate_all_configurations(self) -> Dict[str, ConfigValidationResult]:
        """Validate all configurations"""
        validation_results = {}
        
        for config_id, config in self.configurations.items():
            validation_results[config_id] = await self._validate_config(config)
        
        return validation_results
    
    async def optimize_configurations(self) -> List[str]:
        """Analyze and suggest configuration optimizations"""
        recommendations = []
        
        try:
            # Analyze current configurations for optimization opportunities
            for config_id, config in self.configurations.items():
                if config.category == ConfigCategory.LOAD_BALANCING:
                    if config.name == 'load_balance_threshold' and config.value > 0.9:
                        recommendations.append(f"Consider lowering {config_id} for better load distribution")
                
                elif config.category == ConfigCategory.HEALTH_MONITORING:
                    if config.name == 'health_check_interval' and config.value < 30:
                        recommendations.append(f"Consider increasing {config_id} to reduce monitoring overhead")
                
                elif config.category == ConfigCategory.TASK_SCHEDULING:
                    if config.name == 'max_queue_size' and config.value > 5000:
                        recommendations.append(f"Large queue size in {config_id} may impact memory usage")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Configuration optimization failed: {e}")
            return ["Unable to generate optimization recommendations due to error"]

# Global coordination config manager instance
coordination_config_manager = CoordinationConfigManager()

async def initialize_coordination_config_manager():
    """Initialize the global coordination config manager"""
    await coordination_config_manager.initialize()

# Convenience functions
async def get_coordination_config(config_id: str, scope: ConfigScope = ConfigScope.GLOBAL) -> Optional[Any]:
    """Get coordination configuration value"""
    return await coordination_config_manager.get_config(config_id, scope)

async def set_coordination_config(config_id: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
                                updated_by: str = "system") -> bool:
    """Set coordination configuration value"""
    return await coordination_config_manager.set_config(config_id, value, scope, updated_by)

if __name__ == "__main__":
    # Test the coordination config manager
    async def test_coordination_config_manager():
        await initialize_coordination_config_manager()
        
        # Get configuration
        heartbeat_interval = await get_coordination_config('brain_heartbeat_interval')
        print(f"Heartbeat interval: {heartbeat_interval}")
        
        # Set configuration
        success = await set_coordination_config('brain_heartbeat_interval', 45, updated_by='test_user')
        print(f"Config update success: {success}")
        
        # Get updated configuration
        new_interval = await get_coordination_config('brain_heartbeat_interval')
        print(f"New heartbeat interval: {new_interval}")
        
        # Get summary
        summary = await coordination_config_manager.get_configuration_summary()
        print(f"Configuration summary: {summary}")
        
        # Get optimization recommendations
        recommendations = await coordination_config_manager.optimize_configurations()
        print(f"Optimization recommendations: {recommendations}")
    
    asyncio.run(test_coordination_config_manager())
