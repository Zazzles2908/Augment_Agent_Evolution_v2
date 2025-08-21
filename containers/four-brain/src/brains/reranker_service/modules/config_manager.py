"""
Configuration Manager Module for Brain-2
Handles configuration loading, validation, and environment management

Extracted from brain2_manager.py for modular architecture.
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
    Configuration Manager for Brain-2
    Handles all configuration aspects for modular architecture
    """
    
    def __init__(self, initial_settings: Optional[Dict[str, Any]] = None):
        """Initialize Configuration Manager"""
        self.config = {}
        self.config_path = Path("/workspace/config/brain2_config.json")
        self.initial_settings = initial_settings or {}
        
        # Default configuration for Brain-2 (Reranker)
        self.default_config = {
            "brain_id": "brain2",
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "model_path": "/workspace/models/qwen3/reranker-4b",
            "cache_dir": "/workspace/models/cache",
            "use_blackwell_quantization": True,
            "enable_4bit_quantization": True,
            "enable_8bit_quantization": True,
            "max_vram_usage": 0.20,  # 20% of GPU memory for Brain-2
            "target_vram_usage": 0.18,  # Target 18% usage
            "batch_size": 16,  # Optimized for RTX 5070 Ti
            "max_length": 512,  # Token limit for query-document pairs
            "top_k_default": 10,  # Default number of results to return
            "redis_url": "redis://redis:6379/0",
            "performance_monitoring_enabled": True,
            "health_check_interval": 30,
            # MoE Efficiency Configuration
            "enable_moe_efficiency": True,
            "active_experts": 3,  # 3B active out of 30B total
            "expert_selection_threshold": 0.1
        }
        
        logger.info("🔧 Configuration Manager (Brain-2) initialized")
    
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
            "BRAIN2_MODEL_PATH": "model_path",
            "BRAIN2_CACHE_DIR": "cache_dir",
            "ENABLE_BLACKWELL_QUANTIZATION": "use_blackwell_quantization",
            "BRAIN2_ENABLE_4BIT": "enable_4bit_quantization",
            "BRAIN2_ENABLE_8BIT": "enable_8bit_quantization",
            "BRAIN2_MAX_VRAM_USAGE": "max_vram_usage",
            "BRAIN2_TARGET_VRAM_USAGE": "target_vram_usage",
            "BRAIN2_BATCH_SIZE": "batch_size",
            "BRAIN2_MAX_LENGTH": "max_length",
            "BRAIN2_TOP_K": "top_k_default",
            "REDIS_URL": "redis_url",
            "BRAIN2_ENABLE_MOE": "enable_moe_efficiency",
            "BRAIN2_ACTIVE_EXPERTS": "active_experts"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ["use_blackwell_quantization", "enable_4bit_quantization",
                                "enable_8bit_quantization", "enable_moe_efficiency"]:
                    self.config[config_key] = env_value.lower() == "true"
                elif config_key in ["batch_size", "max_length", "top_k_default", "active_experts"]:
                    self.config[config_key] = int(env_value)
                elif config_key in ["max_vram_usage", "target_vram_usage", "expert_selection_threshold"]:
                    self.config[config_key] = float(env_value)
                else:
                    self.config[config_key] = env_value
                
                logger.debug(f"🔧 Environment override: {config_key} = {self.config[config_key]}")
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate VRAM usage
        if not 0.1 <= self.config["max_vram_usage"] <= 0.5:
            logger.warning("⚠️ Invalid max VRAM usage for Brain-2, setting to 0.20")
            self.config["max_vram_usage"] = 0.20
        
        if self.config["target_vram_usage"] >= self.config["max_vram_usage"]:
            self.config["target_vram_usage"] = self.config["max_vram_usage"] * 0.9
        
        # Validate batch size
        if self.config["batch_size"] < 1:
            self.config["batch_size"] = 1
        elif self.config["batch_size"] > 64:
            logger.warning("⚠️ Large batch size may cause memory issues")
        
        # Validate max length
        if self.config["max_length"] < 128:
            self.config["max_length"] = 128
        elif self.config["max_length"] > 2048:
            logger.warning("⚠️ Very large max_length may cause memory issues")
            self.config["max_length"] = 2048
        
        # Validate top_k
        if self.config["top_k_default"] < 1:
            self.config["top_k_default"] = 1
        elif self.config["top_k_default"] > 100:
            self.config["top_k_default"] = 100
        
        # Validate MoE settings
        if self.config["active_experts"] < 1:
            self.config["active_experts"] = 1
        elif self.config["active_experts"] > 10:
            self.config["active_experts"] = 10
        
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
            "model_name": self.config["model_name"],
            "model_path": self.config["model_path"],
            "cache_dir": self.config["cache_dir"],
            "use_blackwall_quantization": self.config["use_blackwall_quantization"],
            "enable_4bit_quantization": self.config["enable_4bit_quantization"],
            "enable_8bit_quantization": self.config["enable_8bit_quantization"],
            "max_vram_usage": self.config["max_vram_usage"]
        }
    
    def get_reranking_config(self) -> Dict[str, Any]:
        """Get reranking-specific configuration"""
        return {
            "batch_size": self.config["batch_size"],
            "max_length": self.config["max_length"],
            "top_k_default": self.config["top_k_default"],
            "enable_moe_efficiency": self.config["enable_moe_efficiency"],
            "active_experts": self.config["active_experts"],
            "expert_selection_threshold": self.config["expert_selection_threshold"]
        }
