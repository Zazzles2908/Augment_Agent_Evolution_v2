"""
Configuration management for Brain 4 (Docling) integration
Simplified configuration for Docker container deployment
AUTHENTIC IMPLEMENTATION - Zero fabrication policy
"""

from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import logging

def configure_logging(log_level: str = "INFO"):
    """Configure logging for Brain 4 system"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

class Brain4Settings(BaseSettings):
    """
    Brain 4 configuration settings with validation
    Optimized for RTX 5070 Ti and 32GB DDR5 system
    """
    
    # Environment
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Hardware optimization settings
    max_vram_usage: float = Field(default=0.75, env="MAX_VRAM_USAGE")
    target_vram_usage: float = Field(default=0.65, env="TARGET_VRAM_USAGE")
    max_concurrent_tasks: int = Field(default=4, env="MAX_CONCURRENT_TASKS")
    batch_size_documents: int = Field(default=2, env="BATCH_SIZE_DOCUMENTS")
    
    # Model and cache settings - Docker Performance Optimization
    model_cache_dir: Path = Field(default=Path("/workspace/models"), env="MODEL_CACHE_DIR")
    qwen3_model_path: str = Field(default="/workspace/models/qwen3/embedding-4b", env="BRAIN1_MODEL_PATH")
    test_data_dir: Path = Field(default=Path("/workspace/test_data"), env="TEST_DATA_DIR")
    temp_dir: Path = Field(default=Path("/workspace/temp"), env="TEMP_DIR")
    data_dir: Path = Field(default=Path("/workspace/data"), env="DATA_DIR")
    
    # Database connections
    database_url: str = Field(default="postgresql://postgres:ai_secure_2024@postgres:5432/ai_system", env="DATABASE_URL")
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")

    # Supabase integration
    supabase_url: str = Field(default="https://placeholder.supabase.co", env="SUPABASE_URL")
    supabase_anon_key: str = Field(default="placeholder_anon_key", env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="placeholder_service_key", env="SUPABASE_SERVICE_ROLE_KEY")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="BRAIN4_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Document processing settings
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "pptx", "html", "md", "txt"],
        env="SUPPORTED_FORMATS"
    )
    ocr_enabled: bool = Field(default=True, env="OCR_ENABLED")
    table_extraction_enabled: bool = Field(default=True, env="TABLE_EXTRACTION_ENABLED")
    
    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    metrics_collection_interval: int = Field(default=30, env="METRICS_COLLECTION_INTERVAL")
    
    # Security
    jwt_secret_key: str = Field(default="placeholder_jwt_secret_key_change_in_production", env="JWT_SECRET_KEY")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")
    
    @field_validator("max_vram_usage", "target_vram_usage")
    @classmethod
    def validate_vram_usage(cls, v):
        """Validate VRAM usage percentages for RTX 5070 Ti"""
        if not 0.1 <= v <= 0.9:
            raise ValueError("VRAM usage must be between 0.1 and 0.9")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

# Global settings instance - simplified for container deployment
settings = Brain4Settings()

# Simple logging configuration for container deployment
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Reduce noise from verbose libraries
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
