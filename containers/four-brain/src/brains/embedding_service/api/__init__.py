"""
Brain1 API Module
Provides REST API endpoints for embedding generation and management.
"""

from .endpoints import router
from .models import EmbeddingRequest, EmbeddingResponse

__all__ = ["router", "EmbeddingRequest", "EmbeddingResponse"]
