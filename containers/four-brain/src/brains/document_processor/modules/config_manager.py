"""
Configuration Manager Module for Brain-4
Handles configuration loading, validation, and environment management

Extracted from brain4_manager.py for modular architecture.
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
    Configuration Manager for Brain-4
    Handles all configuration aspects for modular architecture
    """
    
    def __init__(self, initial_settings: Optional[Dict[str, Any]] = None):
        """Initialize Configuration Manager"""
        self.config = {}
        self.config_path = Path("/workspace/config/brain4_config.json")
        self.initial_settings = initial_settings or {}
        
        # Default configuration for Brain-4 (Document Processing)
        self.default_config = {
            "brain_id": "brain4",
            "service_name": "Docling Document Processor",
            "model_cache_dir": "/workspace/models/cache",
            "temp_dir": "/workspace/temp",
            "max_vram_usage": 0.40,  # 40% of GPU memory for Brain-4 (largest allocation)
            "target_vram_usage": 0.35,  # Target 35% usage
            "max_concurrent_tasks": 4,  # Document processing tasks
            "batch_size_documents": 2,  # Documents per batch
            "max_file_size_mb": 100,  # Maximum file size
            "chunk_size": 1000,  # Characters per chunk
            "chunk_overlap": 200,  # Overlap between chunks
            "redis_url": "redis://redis:6379/0",
            "database_url": "postgresql://user:pass@localhost/brain4",
            "performance_monitoring_enabled": True,
            "health_check_interval": 30,
            # Docling Configuration
            "enable_ocr": True,
            "enable_table_extraction": True,
            "enable_image_extraction": True,
            "enable_semantic_chunking": True,
            # Supported formats
            "supported_formats": ["pdf", "docx", "pptx", "html", "md", "txt"],
            # Integration settings
            "enable_brain1_integration": True,  # For embeddings
            "enable_brain2_integration": True,  # For reranking
            "enable_brain3_integration": True   # For enhancement
        }
        
        logger.info("🔧 Configuration Manager (Brain-4) initialized")
    
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
            "BRAIN4_MODEL_CACHE_DIR": "model_cache_dir",
            "BRAIN4_TEMP_DIR": "temp_dir",
            "BRAIN4_MAX_VRAM_USAGE": "max_vram_usage",
            "BRAIN4_TARGET_VRAM_USAGE": "target_vram_usage",
            "BRAIN4_MAX_CONCURRENT_TASKS": "max_concurrent_tasks",
            "BRAIN4_BATCH_SIZE": "batch_size_documents",
            "BRAIN4_MAX_FILE_SIZE_MB": "max_file_size_mb",
            "BRAIN4_CHUNK_SIZE": "chunk_size",
            "BRAIN4_CHUNK_OVERLAP": "chunk_overlap",
            "REDIS_URL": "redis_url",
            "DATABASE_URL": "database_url",
            "BRAIN4_ENABLE_OCR": "enable_ocr",
            "BRAIN4_ENABLE_TABLES": "enable_table_extraction",
            "BRAIN4_ENABLE_IMAGES": "enable_image_extraction",
            "BRAIN4_ENABLE_CHUNKING": "enable_semantic_chunking"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ["enable_ocr", "enable_table_extraction", "enable_image_extraction", 
                                "enable_semantic_chunking", "enable_brain1_integration", 
                                "enable_brain2_integration", "enable_brain3_integration"]:
                    self.config[config_key] = env_value.lower() == "true"
                elif config_key in ["max_concurrent_tasks", "batch_size_documents", "max_file_size_mb", 
                                  "chunk_size", "chunk_overlap"]:
                    self.config[config_key] = int(env_value)
                elif config_key in ["max_vram_usage", "target_vram_usage"]:
                    self.config[config_key] = float(env_value)
                else:
                    self.config[config_key] = env_value
                
                logger.debug(f"🔧 Environment override: {config_key} = {self.config[config_key]}")
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate VRAM usage (Brain-4 gets largest allocation)
        if not 0.2 <= self.config["max_vram_usage"] <= 0.8:
            logger.warning("⚠️ Invalid max VRAM usage for Brain-4, setting to 0.40")
            self.config["max_vram_usage"] = 0.40
        
        if self.config["target_vram_usage"] >= self.config["max_vram_usage"]:
            self.config["target_vram_usage"] = self.config["max_vram_usage"] * 0.9
        
        # Validate concurrent tasks
        if self.config["max_concurrent_tasks"] < 1:
            self.config["max_concurrent_tasks"] = 1
        elif self.config["max_concurrent_tasks"] > 10:
            logger.warning("⚠️ High concurrent tasks may cause resource issues")
        
        # Validate batch size
        if self.config["batch_size_documents"] < 1:
            self.config["batch_size_documents"] = 1
        elif self.config["batch_size_documents"] > 5:
            logger.warning("⚠️ Large batch size may cause memory issues")
        
        # Validate file size
        if self.config["max_file_size_mb"] < 1:
            self.config["max_file_size_mb"] = 1
        elif self.config["max_file_size_mb"] > 500:
            logger.warning("⚠️ Very large file size limit may cause memory issues")
        
        # Validate chunk settings
        if self.config["chunk_size"] < 100:
            self.config["chunk_size"] = 100
        elif self.config["chunk_size"] > 5000:
            self.config["chunk_size"] = 5000
        
        if self.config["chunk_overlap"] >= self.config["chunk_size"]:
            self.config["chunk_overlap"] = self.config["chunk_size"] // 4
        
        # Validate paths
        for path_key in ["model_cache_dir", "temp_dir"]:
            path = Path(self.config[path_key])
            if not path.exists():
                logger.warning(f"⚠️ Path does not exist: {path}")
        
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
    
    def get_docling_config(self) -> Dict[str, Any]:
        """Get Docling-specific configuration"""
        return {
            "model_cache_dir": self.config["model_cache_dir"],
            "enable_ocr": self.config["enable_ocr"],
            "enable_table_extraction": self.config["enable_table_extraction"],
            "enable_image_extraction": self.config["enable_image_extraction"],
            "supported_formats": self.config["supported_formats"]
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return {
            "max_concurrent_tasks": self.config["max_concurrent_tasks"],
            "batch_size_documents": self.config["batch_size_documents"],
            "max_file_size_mb": self.config["max_file_size_mb"],
            "chunk_size": self.config["chunk_size"],
            "chunk_overlap": self.config["chunk_overlap"],
            "enable_semantic_chunking": self.config["enable_semantic_chunking"],
            "temp_dir": self.config["temp_dir"]
        }
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get brain integration configuration"""
        return {
            "enable_brain1_integration": self.config["enable_brain1_integration"],
            "enable_brain2_integration": self.config["enable_brain2_integration"],
            "enable_brain3_integration": self.config["enable_brain3_integration"],
            "redis_url": self.config["redis_url"],
            "database_url": self.config["database_url"]
        }
