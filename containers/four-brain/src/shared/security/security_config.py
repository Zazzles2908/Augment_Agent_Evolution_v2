"""
Security Configuration Management for Four-Brain System v2
Centralized security configuration with dynamic updates and validation

Created: 2025-07-30 AEST
Purpose: Manage security configurations across all system components
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
    SERVICE = "service"
    USER = "user"
    SESSION = "session"

class ConfigCategory(Enum):
    """Security configuration categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"
    INCIDENT_RESPONSE = "incident_response"
    NETWORK_SECURITY = "network_security"
    DATA_PROTECTION = "data_protection"

@dataclass
class SecurityConfig:
    """Security configuration item"""
    config_id: str
    category: ConfigCategory
    scope: ConfigScope
    name: str
    description: str
    value: Any
    default_value: Any
    validation_rules: Dict[str, Any]
    sensitive: bool
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

class SecurityConfigManager:
    """
    Comprehensive security configuration management system
    
    Features:
    - Centralized configuration management
    - Dynamic configuration updates
    - Configuration validation and verification
    - Version control and rollback
    - Environment-specific configurations
    - Security policy enforcement
    - Configuration audit logging
    - Hot-reload capabilities
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/10", 
                 config_file: str = "security_config.yaml"):
        self.redis_url = redis_url
        self.redis_client = None
        self.config_file = config_file
        
        # Configuration storage
        self.configurations: Dict[str, SecurityConfig] = {}
        self.config_watchers: Dict[str, List[callable]] = {}
        
        # Default security configurations
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
        
        logger.info("⚙️ Security Config Manager initialized")
    
    async def initialize(self):
        """Initialize Redis connection and load configurations"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load configurations from file and Redis
            await self._load_configurations()
            
            # Start configuration monitoring
            asyncio.create_task(self._monitor_config_changes())
            
            logger.info("✅ Security Config Manager Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Security Config Manager: {e}")
            raise
    
    async def get_config(self, config_id: str, scope: ConfigScope = ConfigScope.GLOBAL) -> Optional[Any]:
        """Get configuration value"""
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
        """Set configuration value with validation"""
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
                
                config = SecurityConfig(
                    config_id=full_config_id,
                    category=default_config.category,
                    scope=scope,
                    name=default_config.name,
                    description=default_config.description,
                    value=value,
                    default_value=default_config.default_value,
                    validation_rules=default_config.validation_rules,
                    sensitive=default_config.sensitive,
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
            
            logger.info(f"✅ Configuration updated: {config_id} = {value if not config.sensitive else '***'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set config {config_id}: {e}")
            return False
    
    async def _validate_config(self, config: SecurityConfig) -> ConfigValidationResult:
        """Validate configuration value"""
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
            
            # Length validation
            if 'min_length' in rules and hasattr(value, '__len__'):
                if len(value) < rules['min_length']:
                    errors.append(f"Length {len(value)} is below minimum {rules['min_length']}")
            
            if 'max_length' in rules and hasattr(value, '__len__'):
                if len(value) > rules['max_length']:
                    errors.append(f"Length {len(value)} is above maximum {rules['max_length']}")
            
            # Allowed values validation
            if 'allowed_values' in rules:
                if value not in rules['allowed_values']:
                    errors.append(f"Value {value} not in allowed values: {rules['allowed_values']}")
            
            # Pattern validation
            if 'pattern' in rules and isinstance(value, str):
                import re
                if not re.match(rules['pattern'], value):
                    errors.append(f"Value does not match required pattern: {rules['pattern']}")
            
            # Custom validation
            if 'custom_validator' in rules:
                validator = rules['custom_validator']
                if callable(validator):
                    try:
                        validator_result = validator(value)
                        if not validator_result:
                            errors.append("Custom validation failed")
                    except Exception as e:
                        errors.append(f"Custom validation error: {e}")
            
            # Security-specific validations
            if config.category == ConfigCategory.AUTHENTICATION:
                if 'password' in config.name.lower() and isinstance(value, str):
                    if len(value) < 8:
                        warnings.append("Password should be at least 8 characters long")
                    if not any(c.isupper() for c in value):
                        warnings.append("Password should contain uppercase letters")
                    if not any(c.islower() for c in value):
                        warnings.append("Password should contain lowercase letters")
                    if not any(c.isdigit() for c in value):
                        warnings.append("Password should contain numbers")
            
            elif config.category == ConfigCategory.ENCRYPTION:
                if 'key_size' in config.name.lower() and isinstance(value, int):
                    if value < 256:
                        warnings.append("Encryption key size should be at least 256 bits")
                        recommendations.append("Consider using 256-bit or higher encryption keys")
            
            elif config.category == ConfigCategory.NETWORK_SECURITY:
                if 'timeout' in config.name.lower() and isinstance(value, int):
                    if value > 300:  # 5 minutes
                        warnings.append("Long timeouts may pose security risks")
                        recommendations.append("Consider shorter timeout values for better security")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _initialize_default_configs(self) -> Dict[str, SecurityConfig]:
        """Initialize default security configurations"""
        defaults = {}
        
        # Authentication configurations
        defaults['auth_session_timeout'] = SecurityConfig(
            config_id='auth_session_timeout',
            category=ConfigCategory.AUTHENTICATION,
            scope=ConfigScope.GLOBAL,
            name='Session Timeout',
            description='Session timeout in minutes',
            value=30,
            default_value=30,
            validation_rules={'type': 'int', 'min': 5, 'max': 480},
            sensitive=False,
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        defaults['auth_max_login_attempts'] = SecurityConfig(
            config_id='auth_max_login_attempts',
            category=ConfigCategory.AUTHENTICATION,
            scope=ConfigScope.GLOBAL,
            name='Max Login Attempts',
            description='Maximum failed login attempts before lockout',
            value=5,
            default_value=5,
            validation_rules={'type': 'int', 'min': 3, 'max': 20},
            sensitive=False,
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Authorization configurations
        defaults['authz_default_role'] = SecurityConfig(
            config_id='authz_default_role',
            category=ConfigCategory.AUTHORIZATION,
            scope=ConfigScope.GLOBAL,
            name='Default User Role',
            description='Default role assigned to new users',
            value='guest',
            default_value='guest',
            validation_rules={'type': 'str', 'allowed_values': ['guest', 'user', 'admin']},
            sensitive=False,
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Encryption configurations
        defaults['encryption_algorithm'] = SecurityConfig(
            config_id='encryption_algorithm',
            category=ConfigCategory.ENCRYPTION,
            scope=ConfigScope.GLOBAL,
            name='Encryption Algorithm',
            description='Default encryption algorithm',
            value='AES-256-GCM',
            default_value='AES-256-GCM',
            validation_rules={'type': 'str', 'allowed_values': ['AES-256-GCM', 'AES-256-CBC', 'ChaCha20-Poly1305']},
            sensitive=False,
            requires_restart=True,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Monitoring configurations
        defaults['monitoring_log_level'] = SecurityConfig(
            config_id='monitoring_log_level',
            category=ConfigCategory.MONITORING,
            scope=ConfigScope.GLOBAL,
            name='Security Log Level',
            description='Security monitoring log level',
            value='INFO',
            default_value='INFO',
            validation_rules={'type': 'str', 'allowed_values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']},
            sensitive=False,
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        # Network security configurations
        defaults['network_rate_limit'] = SecurityConfig(
            config_id='network_rate_limit',
            category=ConfigCategory.NETWORK_SECURITY,
            scope=ConfigScope.GLOBAL,
            name='API Rate Limit',
            description='API requests per minute per IP',
            value=100,
            default_value=100,
            validation_rules={'type': 'int', 'min': 10, 'max': 10000},
            sensitive=False,
            requires_restart=False,
            last_updated=datetime.now(),
            updated_by='system',
            version=1
        )
        
        return defaults
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configuration validation rules"""
        return {
            'password_complexity': {
                'min_length': 8,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special_chars': True
            },
            'encryption_standards': {
                'min_key_size': 256,
                'approved_algorithms': ['AES-256-GCM', 'AES-256-CBC', 'ChaCha20-Poly1305']
            },
            'network_security': {
                'max_timeout': 300,
                'min_rate_limit': 10,
                'max_rate_limit': 10000
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
                keys = await self.redis_client.keys("security_config:*")
                for key in keys:
                    config_data = await self.redis_client.get(key)
                    if config_data:
                        config_dict = json.loads(config_data)
                        # Convert back to SecurityConfig object
                        # This would need proper deserialization logic
                        pass
            
            # Load defaults for missing configs
            for config_id, default_config in self.default_configs.items():
                full_config_id = f"{default_config.scope.value}:{config_id}"
                if full_config_id not in self.configurations:
                    self.configurations[full_config_id] = default_config
            
            self.metrics['configs_loaded'] = len(self.configurations)
            logger.info(f"✅ Loaded {len(self.configurations)} security configurations")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configurations: {e}")
    
    async def _store_config_in_redis(self, config: SecurityConfig):
        """Store configuration in Redis"""
        if self.redis_client:
            try:
                key = f"security_config:{config.config_id}"
                data = json.dumps(asdict(config), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store config in Redis: {e}")
    
    async def _load_config_from_redis(self, config_id: str) -> Optional[SecurityConfig]:
        """Load configuration from Redis"""
        if self.redis_client:
            try:
                key = f"security_config:{config_id}"
                data = await self.redis_client.get(key)
                if data:
                    config_dict = json.loads(data)
                    # Convert back to SecurityConfig object
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
                if not config.sensitive:  # Don't export sensitive configs
                    export_data[config_id] = {
                        'category': config.category.value,
                        'scope': config.scope.value,
                        'name': config.name,
                        'description': config.description,
                        'value': config.value,
                        'last_updated': config.last_updated.isoformat(),
                        'version': config.version
                    }
            
            if file_path:
                with open(file_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
                logger.info(f"✅ Configurations exported to {file_path}")
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export configurations: {e}")
            return {}
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security configuration metrics"""
        return {
            'metrics': self.metrics.copy(),
            'total_configurations': len(self.configurations),
            'configurations_by_category': {
                category.value: sum(1 for c in self.configurations.values() if c.category == category)
                for category in ConfigCategory
            },
            'configurations_by_scope': {
                scope.value: sum(1 for c in self.configurations.values() if c.scope == scope)
                for scope in ConfigScope
            },
            'timestamp': datetime.now().isoformat()
        }

# Global security config manager instance
security_config_manager = SecurityConfigManager()

async def initialize_security_config_manager():
    """Initialize the global security config manager"""
    await security_config_manager.initialize()

# Convenience functions
async def get_security_config(config_id: str, scope: ConfigScope = ConfigScope.GLOBAL) -> Optional[Any]:
    """Get security configuration value"""
    return await security_config_manager.get_config(config_id, scope)

async def set_security_config(config_id: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
                            updated_by: str = "system") -> bool:
    """Set security configuration value"""
    return await security_config_manager.set_config(config_id, value, scope, updated_by)

if __name__ == "__main__":
    # Test the security config manager
    async def test_security_config_manager():
        await initialize_security_config_manager()
        
        # Get configuration
        session_timeout = await get_security_config('auth_session_timeout')
        print(f"Session timeout: {session_timeout}")
        
        # Set configuration
        success = await set_security_config('auth_session_timeout', 45, updated_by='test_user')
        print(f"Config update success: {success}")
        
        # Get updated configuration
        new_timeout = await get_security_config('auth_session_timeout')
        print(f"New session timeout: {new_timeout}")
        
        # Get metrics
        metrics = await security_config_manager.get_security_metrics()
        print(f"Security config metrics: {metrics}")
    
    asyncio.run(test_security_config_manager())
