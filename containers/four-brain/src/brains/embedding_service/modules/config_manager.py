"""
Configuration Manager Module for Brain-1
Handles configuration loading, validation, and environment management

Extracted from brain1_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration Manager for Brain-1
    Handles all configuration aspects for modular architecture
    """
    
    def __init__(self, initial_settings: Optional[Dict[str, Any]] = None):
        """Initialize Configuration Manager"""
        self.config = {}
        self.config_path = Path("/workspace/config/brain1_config.json")
        # Keep a reference to the original settings object if provided
        self.settings = initial_settings
        # Normalize initial settings to a dict for merging
        if isinstance(initial_settings, dict):
            self.initial_settings = initial_settings
        elif hasattr(initial_settings, "model_dump"):
            try:
                self.initial_settings = initial_settings.model_dump()
            except Exception:
                self.initial_settings = {}
        elif initial_settings is not None:
            # Best-effort attribute extraction
            try:
                self.initial_settings = {k: getattr(initial_settings, k) for k in dir(initial_settings) if not k.startswith("_")}
            except Exception:
                self.initial_settings = {}
        else:
            self.initial_settings = {}
        
        # Default configuration
        self.default_config = {
            "brain_id": "brain1",
            "model_path": "/workspace/models/qwen3/embedding-4b",
            "cache_dir": "/workspace/models/cache",
            "use_blackwell_quantization": True,
            "mrl_truncation_enabled": True,
            "target_dimensions": 2000,
            "native_dimensions": 2560,
            "thinking_mode_enabled": True,
            "thinking_iterations": 3,
            "thinking_temperature": 0.7,
            "max_vram_usage": 0.25,  # 25% of GPU memory
            "redis_url": "redis://redis:6379/0",
            "performance_monitoring_enabled": True,
            "health_check_interval": 30,
            # Triton configuration defaults (aligned with Brain1Settings)
            "use_triton": True,
            "triton_url": "http://triton:8000",
            "triton_model_name": "qwen3_embedding",
            "triton_timeout_s": 30
        }
        
        logger.info("🔧 Configuration Manager initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from multiple sources"""
        try:
            # Start with default configuration
            self.config = self.default_config.copy()
            
            # Apply initial settings
            if self.initial_settings:
                self.config.update(self.initial_settings)
                logger.info("✅ Initial settings applied")
            
            # Load from file if exists
            if self.config_path.exists():
                file_config = self._load_config_file()
                self.config.update(file_config)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
            else:
                logger.info("📝 Using default configuration (no config file found)")
            
            # Apply environment overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("✅ Configuration loaded and validated")
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            # Return default config on error
            return self.default_config.copy()
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config file: {e}")
            return {}
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            "BRAIN1_MODEL_PATH": "model_path",
            "BRAIN1_CACHE_DIR": "cache_dir",
            "ENABLE_BLACKWELL_QUANTIZATION": "use_blackwell_quantization",
            "BRAIN1_ENABLE_THINKING": "thinking_mode_enabled",
            "BRAIN1_THINKING_ITERATIONS": "thinking_iterations",
            "BRAIN1_THINKING_TEMPERATURE": "thinking_temperature",
            "BRAIN1_MAX_VRAM_USAGE": "max_vram_usage",
            "REDIS_URL": "redis_url",
            "BRAIN1_TARGET_DIMENSIONS": "target_dimensions",
            # Triton overrides
            "USE_TRITON": "use_triton",
            "TRITON_URL": "triton_url",
            "TRITON_MODEL_NAME": "triton_model_name",
            "TRITON_TIMEOUT_S": "triton_timeout_s"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ["use_blackwell_quantization", "thinking_mode_enabled", "mrl_truncation_enabled"]:
                    self.config[config_key] = env_value.lower() == "true"
                elif config_key in ["thinking_iterations", "target_dimensions", "native_dimensions"]:
                    self.config[config_key] = int(env_value)
                elif config_key in ["thinking_temperature", "max_vram_usage"]:
                    self.config[config_key] = float(env_value)
                else:
                    self.config[config_key] = env_value
                
                logger.debug(f"🔧 Environment override: {config_key} = {self.config[config_key]}")
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate dimensions
        if self.config["target_dimensions"] > self.config["native_dimensions"]:
            logger.warning("⚠️ Target dimensions cannot exceed native dimensions")
            self.config["target_dimensions"] = self.config["native_dimensions"]
        
        # Validate VRAM usage
        if not 0.1 <= self.config["max_vram_usage"] <= 1.0:
            logger.warning("⚠️ Invalid VRAM usage, setting to 0.25")
            self.config["max_vram_usage"] = 0.25
        
        # Validate thinking parameters
        if self.config["thinking_iterations"] < 1:
            self.config["thinking_iterations"] = 1
        elif self.config["thinking_iterations"] > 10:
            self.config["thinking_iterations"] = 10
        
        if not 0.1 <= self.config["thinking_temperature"] <= 2.0:
            self.config["thinking_temperature"] = 0.7
        
        # Validate paths
        model_path = Path(self.config["model_path"])
        if not model_path.exists():
            logger.warning(f"⚠️ Model path does not exist: {model_path}")
        
        logger.debug("✅ Configuration validation completed")
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration value(s)"""
        if key is None:
            return self.config.copy()
        return self.config.get(key)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values"""
        self.config.update(updates)
        logger.info(f"🔧 Configuration updated: {list(updates.keys())}")
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "model_path": self.config["model_path"],
            "cache_dir": self.config["cache_dir"],
            "use_blackwall_quantization": self.config["use_blackwall_quantization"],
            "max_vram_usage": self.config["max_vram_usage"]
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding-specific configuration"""
        return {
            "native_dimensions": self.config["native_dimensions"],
            "target_dimensions": self.config["target_dimensions"],
            "mrl_truncation_enabled": self.config["mrl_truncation_enabled"],
            "thinking_mode_enabled": self.config["thinking_mode_enabled"],
            "thinking_iterations": self.config["thinking_iterations"],
            "thinking_temperature": self.config["thinking_temperature"]
        }
