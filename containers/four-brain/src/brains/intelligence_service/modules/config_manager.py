"""
Config Manager Module - API Configuration Management
Manages Augment Agent API configuration and settings

This module handles all configuration management for the AI interface,
including API keys, endpoints, and performance settings.

Created: 2025-07-29 AEST
Purpose: Centralized configuration management for AI integration
Module Size: 150 lines (modular design)
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration Manager for Brain-3 AI Integration
    
    Manages all configuration settings for the Augment Agent API
    and Brain-3 operational parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.environment_overrides = {}
        
        # Default configuration
        self.default_config = {
            'augment_api': {
                'api_url': 'https://api.augmentcode.com/v1',
                'model': 'claude-sonnet-4',
                'timeout_seconds': 30,
                'max_tokens': 4096,
                'temperature': 0.7,
                'retry_attempts': 3,
                'retry_delay': 5
            },
            'brain3': {
                'max_context_length': 10,
                'response_cache_enabled': True,
                'cache_ttl_seconds': 300,
                'fallback_enabled': True,
                'health_check_interval': 60
            },
            'performance': {
                'max_concurrent_requests': 5,
                'request_timeout': 30,
                'connection_pool_size': 10,
                'enable_metrics': True
            },
            'security': {
                'api_key_env_var': 'AUGMENT_API_KEY',
                'validate_ssl': True,
                'log_requests': False,  # Security: don't log API requests by default
                'mask_api_key_in_logs': True
            }
        }
        
        logger.info("⚙️ Config Manager initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment"""
        try:
            # Start with default configuration
            self.config = self.default_config.copy()
            
            # Load from file if exists
            if os.path.exists(self.config_path):
                file_config = self._load_config_file()
                self.config = self._merge_configs(self.config, file_config)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
            else:
                logger.info("📝 Using default configuration (no config file found)")
            
            # Apply environment overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            logger.info("🔄 Falling back to default configuration")
            return self.default_config
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Look for config in multiple locations
        possible_paths = [
            'config/brain3_config.yaml',
            'config/brain3_config.json',
            '../config/brain3_config.yaml',
            os.path.expanduser('~/.four_brain/brain3_config.yaml')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default to yaml in config directory
        return 'config/brain3_config.yaml'
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    return json.load(f)
                else:
                    logger.warning(f"⚠️ Unknown config file format: {self.config_path}")
                    return {}
        except Exception as e:
            logger.error(f"❌ Failed to load config file {self.config_path}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'AUGMENT_API_URL': ['augment_api', 'api_url'],
            'AUGMENT_API_KEY': ['augment_api', 'api_key'],
            'AUGMENT_MODEL': ['augment_api', 'model'],
            'AUGMENT_TIMEOUT': ['augment_api', 'timeout_seconds'],
            'AUGMENT_MAX_TOKENS': ['augment_api', 'max_tokens'],
            'AUGMENT_TEMPERATURE': ['augment_api', 'temperature'],
            'BRAIN3_FALLBACK_ENABLED': ['brain3', 'fallback_enabled'],
            'BRAIN3_CACHE_ENABLED': ['brain3', 'response_cache_enabled'],
            'BRAIN3_MAX_CONTEXT': ['brain3', 'max_context_length']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                
                # Set in config
                current = self.config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = converted_value
                
                # Track override for logging
                self.environment_overrides[env_var] = converted_value
        
        if self.environment_overrides:
            logger.info(f"🔧 Applied {len(self.environment_overrides)} environment overrides")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate API configuration
        api_config = self.config.get('augment_api', {})
        
        if not api_config.get('api_url'):
            errors.append("Missing API URL")
        
        if not api_config.get('model'):
            errors.append("Missing AI model specification")
        
        # Check for API key
        api_key = self.get_api_key()
        if not api_key:
            errors.append("Missing API key (set AUGMENT_API_KEY environment variable)")
        
        # Validate numeric ranges
        timeout = api_config.get('timeout_seconds', 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append("Invalid timeout_seconds value")
        
        max_tokens = api_config.get('max_tokens', 4096)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            errors.append("Invalid max_tokens value")
        
        temperature = api_config.get('temperature', 0.7)
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            errors.append("Invalid temperature value (must be 0-2)")
        
        if errors:
            logger.warning(f"⚠️ Configuration validation warnings: {', '.join(errors)}")
        else:
            logger.info("✅ Configuration validation passed")
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment or config"""
        # First check environment variable
        env_var = self.config.get('security', {}).get('api_key_env_var', 'AUGMENT_API_KEY')
        api_key = os.getenv(env_var)
        
        if api_key:
            return api_key
        
        # Fallback to config file (not recommended for security)
        api_key = self.config.get('augment_api', {}).get('api_key')
        if api_key:
            logger.warning("⚠️ API key found in config file - consider using environment variable")
            return api_key
        
        return None
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get specific configuration section"""
        return self.config.get(section, {})
    
    def get_config_value(self, *path: str, default: Any = None) -> Any:
        """Get specific configuration value by path"""
        current = self.config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            save_path = config_path or self.config_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Remove sensitive information before saving
            safe_config = self._sanitize_config_for_save()
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(safe_config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(safe_config, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def _sanitize_config_for_save(self) -> Dict[str, Any]:
        """Remove sensitive information from config before saving"""
        safe_config = self.config.copy()
        
        # Remove API key if present
        if 'augment_api' in safe_config and 'api_key' in safe_config['augment_api']:
            del safe_config['augment_api']['api_key']
        
        return safe_config
    
    def get_masked_config(self) -> Dict[str, Any]:
        """Get configuration with sensitive values masked for logging"""
        masked_config = self.config.copy()
        
        # Mask API key
        if 'augment_api' in masked_config and 'api_key' in masked_config['augment_api']:
            api_key = masked_config['augment_api']['api_key']
            masked_config['augment_api']['api_key'] = f"{api_key[:8]}..." if len(api_key) > 8 else "***"
        
        return masked_config
