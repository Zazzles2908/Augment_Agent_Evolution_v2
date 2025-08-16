"""
Brain1 API Models
Pydantic models for API request/response validation.

Created: 2025-07-13 AEST
Author: Augment Agent Evolution - Brain Architecture Standardization
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    batch_size: int = Field(default=32, description="Batch size for processing", ge=1, le=128)
    truncate_dimension: Optional[int] = Field(default=2000, description="Truncate to dimension for Supabase compatibility")

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used for embedding generation")
    dimensions: int = Field(..., description="Embedding dimensions")
    usage: Dict[str, int] = Field(..., description="Usage statistics")

class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: HealthStatus = Field(..., description="Health status")
    brain: str = Field(..., description="Brain identifier")
    model: str = Field(..., description="Model name")
    version: str = Field(..., description="Brain version")
    timestamp: Optional[str] = Field(default=None, description="Health check timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional health details")

class ModelInfo(BaseModel):
    """Model information response."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    parameters: int = Field(..., description="Number of parameters")
    dimensions: int = Field(..., description="Embedding dimensions")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    quantization: str = Field(..., description="Quantization type")
    device: str = Field(..., description="Device (CPU/GPU)")

class MetricsResponse(BaseModel):
    """Metrics response model."""
    model_config = ConfigDict(protected_namespaces=())

    total_requests: int = Field(..., description="Total number of requests processed")
    average_latency_ms: float = Field(..., description="Average latency in milliseconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    gpu_utilization_percent: float = Field(..., description="GPU utilization percentage")
    model_load_time_ms: float = Field(..., description="Model load time in milliseconds")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
