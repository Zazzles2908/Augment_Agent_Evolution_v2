"""
Brain 2 API Models
Pydantic models for Brain 2 (Qwen3-Reranker-4B) API endpoints
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator, ConfigDict


class DocumentItem(BaseModel):
    """Individual document for reranking"""
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional document metadata"
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Optional document identifier"
    )


class RerankRequest(BaseModel):
    """Request model for document reranking"""
    query: str = Field(..., description="Query text for relevance ranking")
    documents: List[DocumentItem] = Field(
        ..., 
        description="List of documents to rerank"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top documents to return"
    )
    
    @validator('documents')
    def validate_documents(cls, v):
        if len(v) == 0:
            raise ValueError("At least one document must be provided")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 documents allowed")
        return v


class RerankResult(BaseModel):
    """Individual reranking result"""
    text: str = Field(..., description="Document text")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    rank: int = Field(..., description="Rank position (1-based)")
    doc_id: Optional[str] = Field(default=None, description="Document identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class RerankResponse(BaseModel):
    """Response model for document reranking"""
    model_config = ConfigDict(protected_namespaces=())

    results: List[RerankResult] = Field(..., description="Reranked documents")
    query: str = Field(..., description="Original query")
    total_documents: int = Field(..., description="Total number of input documents")
    returned_documents: int = Field(..., description="Number of returned documents")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status")
    brain_id: str = Field(..., description="Brain identifier")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    quantization_enabled: bool = Field(..., description="Whether quantization is active")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage information")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Model name/path")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    quantized: bool = Field(..., description="Whether model is quantized")
    quantization_type: Optional[str] = Field(default=None, description="Quantization type")
    optimization: Optional[str] = Field(default=None, description="Optimization method")
    loading_time_seconds: Optional[float] = Field(default=None, description="Model loading time")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage details")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")


class TaskRequest(BaseModel):
    """Background task request"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Task type")
    query: str = Field(..., description="Query for reranking")
    documents: List[DocumentItem] = Field(..., description="Documents to rerank")
    top_k: int = Field(default=10, description="Number of top results")
    callback_url: Optional[str] = Field(default=None, description="Callback URL for results")


class TaskResponse(BaseModel):
    """Background task response"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    created_at: str = Field(..., description="Task creation timestamp")
    completed_at: Optional[str] = Field(default=None, description="Task completion timestamp")
    results: Optional[RerankResponse] = Field(default=None, description="Task results")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
