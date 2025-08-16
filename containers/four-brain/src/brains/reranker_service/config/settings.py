"""
Brain 2 Configuration Settings
Configuration for Qwen3-Reranker-4B deployment with RTX 5070 Ti optimization
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Config:
    protected_namespaces = ('settings_',)

class Brain2Settings(BaseSettings):
    """Configuration settings for Brain 2 (Qwen3-Reranker-4B)"""

    # Model Configuration - Local Model Path
    model_name: str = Field(
        default="/workspace/models/qwen3/reranker-4b",
        description="Path to Qwen3-Reranker-4B model",
        env="BRAIN2_MODEL_PATH"
    )
    model_cache_dir: str = Field(
        default="/workspace/models/cache",
        description="Model cache directory",
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

    # Quantization Settings (Phase 7 Strategy: 8-bit primary)
    enable_4bit_quantization: bool = Field(
        default=False,
        description="Enable 4-bit quantization fallback"
    )
    enable_8bit_quantization: bool = Field(
        default=True,
        description="Enable 8-bit quantization primary"
    )
    enable_unsloth_optimization: bool = Field(
        default=True,
        description="Enable Unsloth optimization"
    )
    enable_flash_attention: bool = Field(
        default=True,
        description="Enable Flash Attention optimization"
    )

    # RTX 5070 Ti Memory Management (Brain-2 gets 20% allocation as per Four-Brain design)
    max_vram_usage: float = Field(
        default=0.20,
        description="Maximum VRAM usage for Brain 2 (20% of 16GB = 3.2GB)",
        env="MAX_VRAM_USAGE"
    )
    target_vram_usage: float = Field(
        default=0.18,
        description="Target VRAM usage for Brain 2 (18% of 16GB = 2.9GB)",
        env="TARGET_VRAM_USAGE"
    )
    gpu_memory_fraction: float = Field(
        default=0.20,
        description="GPU memory fraction for PyTorch (20% as per Four-Brain architecture)",
        env="TORCH_CUDA_MEMORY_FRACTION"
    )
    cuda_memory_fraction: float = Field(
        default=0.20,
        description="CUDA memory fraction (20% as per Four-Brain architecture)",
        env="CUDA_MEMORY_FRACTION"
    )

    # Dynamic Memory Limits (Phase 3 optimization)
    enable_dynamic_memory: bool = Field(
        default=True,
        description="Enable dynamic memory limits with extra headroom",
        env="ENABLE_DYNAMIC_MEMORY"
    )
    memory_headroom_gb: float = Field(
        default=0.5,
        description="Extra memory headroom in GB for Brain 2",
        env="MEMORY_HEADROOM_GB"
    )
    soft_memory_limit_gb: float = Field(
        default=3.0,
        description="Soft memory limit in GB before triggering optimizations",
        env="SOFT_MEMORY_LIMIT_GB"
    )

    # Service Configuration
    brain_id: str = Field(
        default="brain2",
        description="Brain identifier for inter-brain communication"
    )
    service_host: str = Field(
        default="0.0.0.0",
        description="FastAPI service host"
    )
    service_port: int = Field(
        default=8002,
        description="FastAPI service port"
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis URL for inter-brain communication"
    )
    redis_timeout: int = Field(
        default=30,
        description="Redis operation timeout in seconds"
    )

    # Performance Settings
    max_concurrent_tasks: int = Field(
        default=2,
        description="Maximum concurrent reranking tasks"
    )
    batch_size_documents: int = Field(
        default=10,
        description="Document batch size for reranking"
    )
    max_sequence_length: int = Field(
        default=2048,
        description="Maximum sequence length for model"
    )

    # Reranking Configuration
    default_top_k: int = Field(
        default=10,
        description="Default number of top documents to return"
    )
    max_top_k: int = Field(
        default=100,
        description="Maximum number of top documents allowed"
    )
    rerank_timeout: int = Field(
        default=30,
        description="Reranking operation timeout in seconds"
    )

    # Monitoring and Logging
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    metrics_port: int = Field(
        default=9091,
        description="Metrics endpoint port"
    )

    # Health Check Configuration
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(
        default=10,
        description="Health check timeout in seconds"
    )

    class Config:
        env_prefix = "BRAIN2_"
        case_sensitive = False


def get_brain2_settings() -> Brain2Settings:
    """Get Brain 2 settings with environment variable override"""
    return Brain2Settings()
