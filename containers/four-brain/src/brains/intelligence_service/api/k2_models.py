"""
K2 Vector Bridge API Models
Pydantic models for K2-Instruct Vector Bridge endpoints
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class K2StrategyRequest(BaseModel):
    """Request model for K2 strategy selection"""
    query: str = Field(..., description="Query text for strategy analysis")
    context: str = Field(default="general", description="Context type (engineering, creative, safety)")
    document_type: Optional[str] = Field(default=None, description="Type of document being processed")
    brains: Optional[List[str]] = Field(default=["brain1", "brain2", "brain4"], description="Available brains")
    current_load: Optional[Dict[str, float]] = Field(default={}, description="Current system load")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class K2StrategyResponse(BaseModel):
    """Response model for K2 strategy selection"""
    status: str = Field(..., description="Response status")
    strategy: str = Field(..., description="Selected strategy (analytical, creative, safety)")
    reasoning: str = Field(..., description="Reasoning for strategy selection")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    brain_allocation: Dict[str, float] = Field(..., description="Brain allocation percentages")
    expected_latency: int = Field(..., description="Expected processing latency in ms")
    parallel_processing: bool = Field(..., description="Whether parallel processing is recommended")
    cost_estimate: str = Field(..., description="Cost estimate for this request")
    processing_time_ms: int = Field(..., description="Actual processing time in ms")
    timestamp: float = Field(..., description="Response timestamp")
    vector_dimension: int = Field(..., description="Embedding vector dimension")
    embedding_source: str = Field(..., description="Source of embedding (local_qwen3_4b)")
    strategy_source: str = Field(..., description="Source of strategy (kimi_k2_instruct)")


class K2HealthResponse(BaseModel):
    """Health check response for K2 Vector Bridge"""
    service: str = Field(default="k2_vector_bridge", description="Service name")
    status: str = Field(..., description="Service status")
    k2_api_configured: bool = Field(..., description="Whether K2 API is configured")
    local_embedding_endpoint: str = Field(..., description="Local embedding endpoint")
    cost_per_month: str = Field(..., description="Estimated monthly cost")
    vram_usage: str = Field(..., description="VRAM usage (should be 0 GB)")
    timestamp: float = Field(..., description="Health check timestamp")


class K2VectorRequest(BaseModel):
    """Request model for vector coordination"""
    text: str = Field(..., description="Text to process")
    brains: Optional[List[str]] = Field(default=["brain1", "brain2", "brain4"], description="Available brains")
    load: Optional[Dict[str, float]] = Field(default={}, description="Current system load")
    strategy_hint: Optional[str] = Field(default=None, description="Strategy hint (analytical, creative, safety)")


class K2VectorResponse(BaseModel):
    """Response model for vector coordination"""
    status: str = Field(..., description="Response status")
    timestamp: float = Field(..., description="Response timestamp")
    vector_dimension: int = Field(..., description="Embedding vector dimension")
    embedding_source: str = Field(..., description="Source of embedding")
    strategy_source: str = Field(..., description="Source of strategy selection")
    strategy: K2StrategyResponse = Field(..., description="Strategy selection result")
    cost_estimate: str = Field(..., description="Cost estimate")
    processing_time_ms: int = Field(..., description="Processing time in ms")


class K2MetricsResponse(BaseModel):
    """Metrics response for K2 Vector Bridge"""
    total_requests: int = Field(..., description="Total K2 requests processed")
    strategy_selections: Dict[str, int] = Field(..., description="Strategy selections by type")
    average_confidence: float = Field(..., description="Average confidence score")
    average_latency_ms: float = Field(..., description="Average processing latency")
    cost_total: float = Field(..., description="Total cost incurred")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")


class K2ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error information")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
