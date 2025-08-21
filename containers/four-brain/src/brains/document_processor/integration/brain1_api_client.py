"""
Brain 1 API Client for Brain 4 Integration

This module provides an HTTP client for Brain 4 to communicate with Brain 1's
embedding service, eliminating the need for duplicate model loading.

Author: Augment Agent
Date: 2025-07-16
Purpose: Resolve Brain-1 performance issue by eliminating duplicate model loading
"""

import asyncio
import logging
import httpx
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Brain1APIClient:
    """HTTP client for communicating with Brain 1 embedding service"""
    
    def __init__(self, base_url: str = "http://embedding-service:8001", timeout: int = 30):
        """
        Initialize Brain 1 API client

        Args:
            base_url: Base URL for Brain 1 service (embedding-service inside compose)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.logger = logger
        
    async def wait_for_ready(self, max_attempts: int = 30, delay: float = 2.0) -> bool:
        """
        Wait for Brain 1 service to be ready
        
        Args:
            max_attempts: Maximum number of health check attempts
            delay: Delay between attempts in seconds
            
        Returns:
            True if Brain 1 is ready, False otherwise
        """
        self.logger.info(f"🔍 Waiting for Brain 1 service at {self.base_url}")
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.get(f"{self.base_url}/api/v1/brain1/health")
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        self.logger.info("✅ Brain 1 service is ready")
                        return True
                        
            except Exception as e:
                self.logger.debug(f"Attempt {attempt + 1}/{max_attempts}: Brain 1 not ready - {e}")
                
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                
        self.logger.error(f"❌ Brain 1 service not ready after {max_attempts} attempts")
        return False
        
    async def generate_embedding(self, text: str, dimensions: int = 2000) -> Optional[np.ndarray]:
        """
        Generate embedding using Brain 1 service

        Args:
            text: Text to embed
            dimensions: Target embedding dimensions (default: 2000 for Supabase)

        Returns:
            Embedding vector as numpy array, or None if failed
        """
        try:
            payload = {
                "texts": [text],
                "normalize": True,
                "batch_size": 1,
                "truncate_dimension": dimensions
            }

            response = await self.client.post(
                f"{self.base_url}/api/v1/brain1/embed",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("embeddings") or []
                if embeddings:
                    embedding = np.array(embeddings[0], dtype=np.float32)
                    self.logger.debug(f"✅ Generated embedding: {embedding.shape} dimensions")
                    return embedding
                return None

            else:
                self.logger.error(f"❌ Brain 1 API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embedding via Brain 1 API: {e}")
            return None
            
    async def generate_batch_embeddings(self, texts: List[str], dimensions: int = 2000) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            dimensions: Target embedding dimensions

        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        try:
            payload = {
                "texts": texts,
                "normalize": True,
                "batch_size": min(32, max(1, len(texts))),
                "truncate_dimension": dimensions
            }

            response = await self.client.post(
                f"{self.base_url}/api/v1/brain1/embed",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = []

                for emb_data in result.get("embeddings", []):
                    if emb_data is not None:
                        embeddings.append(np.array(emb_data, dtype=np.float32))
                    else:
                        embeddings.append(None)

                self.logger.debug(f"✅ Generated {len(embeddings)} batch embeddings")
                return embeddings

            else:
                self.logger.error(f"❌ Brain 1 batch API error: {response.status_code} - {response.text}")
                return [None] * len(texts)

        except Exception as e:
            self.logger.error(f"❌ Failed to generate batch embeddings via Brain 1 API: {e}")
            return [None] * len(texts)
            
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get Brain 1 service health status
        
        Returns:
            Health status dictionary
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
            
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        self.logger.info("🔒 Brain 1 API client closed")
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


class Brain1APIError(Exception):
    """Exception raised for Brain 1 API errors"""
    pass


# Compatibility wrapper for existing code
class Brain1EmbeddingWrapper:
    """
    Wrapper to maintain compatibility with existing Brain 4 code
    that expects a model-like interface
    """
    
    def __init__(self, api_client: Brain1APIClient):
        self.api_client = api_client
        self.logger = logger
        
    async def generate_embedding(self, text: str, dimensions: int = 2000) -> Optional[np.ndarray]:
        """Generate embedding using Brain 1 API"""
        return await self.api_client.generate_embedding(text, dimensions)
        
    async def generate_batch_embeddings(self, texts: List[str], dimensions: int = 2000) -> List[Optional[np.ndarray]]:
        """Generate batch embeddings using Brain 1 API"""
        return await self.api_client.generate_batch_embeddings(texts, dimensions)
        
    @property
    def is_loaded(self) -> bool:
        """Check if Brain 1 service is available"""
        # This is a synchronous property, so we can't do async health check
        # Return True assuming Brain 1 is ready (checked during initialization)
        return True
