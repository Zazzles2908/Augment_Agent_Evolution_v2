"""
Brain1 API Endpoints
REST API endpoints for embedding generation and management.

Created: 2025-07-13 AEST
Author: Augment Agent Evolution - Brain Architecture Standardization
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from brains.embedding_service.api.models import EmbeddingRequest, EmbeddingResponse, HealthResponse
from brains.embedding_service.core.manager_alias import Brain1Manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/brain1", tags=["brain1-embedding"])

# Global Brain1Manager instance (set by brain1_service.py)
brain1_manager: Brain1Manager = None

# Dependency to get Brain1Manager instance
def get_brain1_manager() -> Brain1Manager:
    """Get Brain1Manager instance."""
    if brain1_manager is None:
        raise HTTPException(status_code=503, detail="Brain 1 manager not initialized")
    return brain1_manager

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Brain1."""
    try:
        return HealthResponse(
            status="healthy",
            brain="brain1-embedding",
            model="qwen3-4b",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Brain1 health check failed")

@router.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    brain1_manager: Brain1Manager = Depends(get_brain1_manager)
):
    """Generate embeddings for input text."""
    try:
        # Generate embeddings using Brain1Manager (modular)
        embeddings = await brain1_manager.generate_batch_embeddings(
            texts=request.texts,
            truncate_to_2000=True
        )
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model="qwen3-4b",
            dimensions=len(embeddings[0]) if embeddings else 0,
            usage={
                "total_tokens": sum(len(text.split()) for text in request.texts),
                "total_texts": len(request.texts)
            }
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@router.get("/model/info")
async def get_model_info(
    brain1_manager: Brain1Manager = Depends(get_brain1_manager)
):
    """Get information about the loaded model."""
    try:
        model_info = await brain1_manager.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/metrics")
async def get_metrics(
    brain1_manager: Brain1Manager = Depends(get_brain1_manager)
):
    """Get Brain1 performance metrics."""
    try:
        metrics = await brain1_manager.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")
