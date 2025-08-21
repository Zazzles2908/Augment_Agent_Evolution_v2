"""
Brain1 Configuration Settings
Configuration for Qwen3-4B Embedding deployment with RTX 5070 Ti optimization

Created: 2025-07-13 AEST
Author: Augment Agent Evolution - Brain Architecture Standardization
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings



class Brain1Settings(BaseSettings):
    model_config = {
        'protected_namespaces': ('settings_',)
    }
    """Configuration settings for Brain1 (Qwen3-4B Embedding)"""

    # Model Configuration - Docker Performance Optimization
    model_name: str = Field(
        default="/workspace/models/qwen3/embedding-4b",
        description="Path to Qwen3-4B embedding model",
        env="BRAIN1_MODEL_PATH"
    )
    model_cache_dir: str = Field(
        default="/workspace/models",
        description="Model cache directory (container-native for 11.6x performance)",
        env="MODEL_CACHE_DIR"
    )
    shared_model_cache_dir: str = Field(
        default="/workspace/models/shared_cache",
        description="Shared model cache directory for inter-brain model sharing (Phase 3 optimization)",
        env="SHARED_MODEL_CACHE_DIR"
    )
    enable_shared_cache: bool = Field(
        default=True,
        description="Enable shared model cache to reduce disk usage by 30%",
        env="ENABLE_SHARED_CACHE"
    )

    # Embedding Configuration
    native_embedding_dimension: int = Field(
        default=2560,
        description="Native Qwen3-4B embedding dimensions",
        env="NATIVE_EMBEDDING_DIMENSION"
    )
    target_embedding_dimension: int = Field(
        default=2000,
        description="Target dimensions for Supabase compatibility",
        env="TARGET_EMBEDDING_DIMENSION"
    )
    use_mrl_truncation: bool = Field(
        default=True,
        description="Use MRL truncation for dimension reduction",
        env="USE_MRL_TRUNCATION"
    )

    # Quantization Settings (Phase 7 Strategy: 8-bit primary)
    enable_4bit_quantization: bool = Field(
        default=True,
        description="Enable 4-bit quantization fallback",
        env="ENABLE_4BIT_QUANTIZATION"
    )
    enable_8bit_quantization: bool = Field(
        default=True,
        description="Enable 8-bit quantization primary",
        env="ENABLE_8BIT_QUANTIZATION"
    )
    enable_unsloth_optimization: bool = Field(
        default=True,
        description="Enable Unsloth optimization",
        env="ENABLE_UNSLOTH_OPTIMIZATION"
    )
    enable_flash_attention: bool = Field(
        default=True,
        description="Enable Flash Attention optimization",
        env="ENABLE_FLASH_ATTENTION"
    )

    # RTX 5070 Ti Memory Management (Brain1 gets 60% allocation for Qwen3-4B FP16)
    max_vram_usage: float = Field(
        default=0.60,
        description="Maximum VRAM usage for Brain1 (60% of 16GB = 9.6GB for Qwen3-4B)",
        env="MAX_VRAM_USAGE"
    )
    target_vram_usage: float = Field(
        default=0.55,
        description="Target VRAM usage for Brain1 (55% of 16GB = 8.8GB)",
        env="TARGET_VRAM_USAGE"
    )
    gpu_memory_fraction: float = Field(
        default=0.60,
        description="GPU memory fraction for PyTorch (60% for Qwen3-4B)",
        env="TORCH_CUDA_MEMORY_FRACTION"
    )
    cuda_memory_fraction: float = Field(
        default=0.60,
        description="CUDA memory fraction (60% for Qwen3-4B)",
        env="CUDA_MEMORY_FRACTION"
    )

    # Dynamic Memory Limits (Phase 3 optimization)
    enable_dynamic_memory: bool = Field(
        default=True,
        description="Enable dynamic memory limits with extra headroom",
        env="ENABLE_DYNAMIC_MEMORY"
    )
    memory_headroom_gb: float = Field(
        default=1.0,
        description="Extra memory headroom in GB (500MB-1GB per fix_containers.md)",
        env="MEMORY_HEADROOM_GB"
    )
    soft_memory_limit_gb: float = Field(
        default=4.0,
        description="Soft memory limit in GB before triggering optimizations",
        env="SOFT_MEMORY_LIMIT_GB"
    )

    # Performance Settings
    batch_size: int = Field(
        default=32,
        description="Default batch size for embedding generation",
        env="BATCH_SIZE_DOCUMENTS"
    )
    max_sequence_length: int = Field(
        default=8000,
        description="Maximum sequence length for input text",
        env="VECTOR_MAX_CONTEXT_LENGTH"
    )
    enable_sequential_loading: bool = Field(
        default=True,
        description="Enable sequential model loading",
        env="ENABLE_SEQUENTIAL_LOADING"
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address",
        env="API_HOST"
    )
    api_port: int = Field(
        default=8001,
        description="API port for Brain1 (8001 to avoid conflict with Brain4 on 8000)",
        env="API_PORT"
    )
    api_workers: int = Field(
        default=1,
        description="Number of API workers",
        env="API_WORKERS"
    )

    # Triton Inference Server configuration
    use_triton: bool = Field(
        default=True,
        description="Use Triton Inference Server instead of local model",
        env="USE_TRITON"
    )
    triton_url: str = Field(
        default="http://triton:8000",
        description="Triton server HTTP URL",
        env="TRITON_URL"
    )
    triton_model_name: str = Field(
        default="qwen3_embedding_trt",
        description="Model name deployed in Triton model repository (default uses *_trt engine)",
        env="TRITON_MODEL_NAME"
    )
    triton_timeout_s: int = Field(
        default=30,
        description="HTTP/gRPC timeout for Triton requests (seconds)",
        env="TRITON_TIMEOUT_S"
    )


    # Database Configuration
    database_url: str = Field(
        default="",
        description="PostgreSQL database URL (must be provided via env; no default secrets)",
        env="DATABASE_URL"
    )
    supabase_url: str = Field(
        default="https://ustcfwmonegxeoqeixgg.supabase.co",
        description="Supabase URL for vector storage",
        env="SUPABASE_URL"
    )
    supabase_service_role_key: str = Field(
        default="",
        description="Supabase service role key",
        env="SUPABASE_SERVICE_ROLE_KEY"
    )
    supabase_anon_key: str = Field(
        default="",
        description="Supabase anonymous key",
        env="SUPABASE_ANON_KEY"
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis URL for inter-brain communication",
        env="REDIS_URL"
    )
    redis_host: str = Field(
        default="redis",
        description="Redis host",
        env="REDIS_HOST"
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port",
        env="REDIS_PORT"
    )
    redis_db: int = Field(
        default=0,
        description="Redis database number",
        env="REDIS_DB"
    )

    # Monitoring and Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        env="LOG_LEVEL"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
        env="LOG_FORMAT"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
        env="ENABLE_METRICS"
    )
    metrics_port: int = Field(
        default=9090,
        description="Metrics port",
        env="METRICS_PORT"
    )

    # Development and Testing
    development_mode: bool = Field(
        default=False,
        description="Enable development mode",
        env="DEVELOPMENT_MODE"
    )
    test_mode: bool = Field(
        default=False,
        description="Enable test mode",
        env="TEST_MODE"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
        env="DEBUG"
    )

    # Four-Brain Architecture
    four_brain_mode: bool = Field(
        default=True,
        description="Enable Four-Brain Architecture mode",
        env="FOUR_BRAIN_MODE"
    )
    brain1_enabled: bool = Field(
        default=True,
        description="Enable Brain1 (Embedding)",
        env="BRAIN1_ENABLED"
    )



# Global settings instance
brain1_settings = Brain1Settings()
