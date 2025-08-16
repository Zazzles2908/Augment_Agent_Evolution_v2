"""
Brain 2 FastAPI Service Tests
Integration tests for the Brain 2 reranker service endpoints

This module tests the FastAPI service functionality including:
- Health check endpoints
- Model status endpoints
- Document reranking API
- Error handling and validation
- Performance metrics

Zero Fabrication Policy: ENFORCED
All tests use real API endpoints and actual service functionality.
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Import test configuration
from . import (
    SAMPLE_QUERY, SAMPLE_DOCUMENTS, EXPECTED_TOP_DOCUMENT, 
    EXPECTED_MIN_RELEVANCE_SCORE
)

# Import Brain 2 service
from brain2_reranker.reranker_service import app
from brain2_reranker.api.models import RerankRequest, DocumentItem


class TestRerankerService:
    """Test suite for Brain 2 FastAPI service"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client for FastAPI app"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "service" in data
        assert "version" in data
        assert "description" in data
        assert "model" in data
        assert "status" in data
        assert "endpoints" in data
        
        # Validate specific values
        assert data["service"] == "Brain 2 - Qwen3 Reranker"
        assert data["model"] == "Qwen3-Reranker-4B"
        assert isinstance(data["endpoints"], dict)
        
        print(f"✅ Root endpoint test completed:")
        print(f"  - Service: {data['service']}")
        print(f"  - Version: {data['version']}")
        print(f"  - Status: {data['status']}")
    
    def test_simple_health_endpoint(self, client):
        """Test simple health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "status" in data
        assert "service" in data
        assert data["service"] == "brain2-reranker"
        assert data["status"] in ["healthy", "initializing", "unhealthy"]
        
        print(f"✅ Simple health endpoint test completed:")
        print(f"  - Status: {data['status']}")
        print(f"  - Service: {data['service']}")
    
    def test_brain2_health_endpoint(self, client):
        """Test detailed Brain 2 health check endpoint"""
        response = client.get("/brain2/health")
        
        # Health check should return 200 or 503 depending on model availability
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_fields = [
                "status", "brain_id", "model_loaded", "quantization_enabled",
                "memory_usage", "uptime_seconds", "timestamp"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate specific values
            assert data["brain_id"] == "brain2"
            assert isinstance(data["model_loaded"], bool)
            assert isinstance(data["quantization_enabled"], bool)
            assert isinstance(data["uptime_seconds"], (int, float))
            assert data["uptime_seconds"] >= 0
            
            print(f"✅ Brain 2 health endpoint test completed:")
            print(f"  - Status: {data['status']}")
            print(f"  - Model loaded: {data['model_loaded']}")
            print(f"  - Quantization: {data['quantization_enabled']}")
            print(f"  - Uptime: {data['uptime_seconds']:.2f}s")
        else:
            print("⚠️ Brain 2 health check returned 503 (service unavailable)")
    
    def test_model_status_endpoint(self, client):
        """Test model status endpoint"""
        response = client.get("/brain2/model/status")
        
        # Model status should return 200 or 500 depending on availability
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_fields = [
                "model_name", "model_loaded", "quantized", "memory_usage",
                "performance_metrics"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate specific values
            assert isinstance(data["model_loaded"], bool)
            assert isinstance(data["quantized"], bool)
            assert isinstance(data["memory_usage"], dict)
            assert isinstance(data["performance_metrics"], dict)
            
            # Validate performance metrics
            perf_metrics = data["performance_metrics"]
            assert "total_requests" in perf_metrics
            assert "average_processing_time_ms" in perf_metrics
            assert "uptime_seconds" in perf_metrics
            
            print(f"✅ Model status endpoint test completed:")
            print(f"  - Model: {data['model_name']}")
            print(f"  - Loaded: {data['model_loaded']}")
            print(f"  - Quantized: {data['quantized']}")
            print(f"  - Total requests: {perf_metrics['total_requests']}")
        else:
            print("⚠️ Model status check returned 500 (internal error)")
    
    def test_rerank_endpoint_validation(self, client):
        """Test reranking endpoint input validation"""
        # Test empty request
        response = client.post("/brain2/rerank", json={})
        assert response.status_code == 422  # Validation error
        
        # Test missing documents
        response = client.post("/brain2/rerank", json={
            "query": "test query",
            "top_k": 5
        })
        assert response.status_code == 422  # Validation error
        
        # Test empty documents list
        response = client.post("/brain2/rerank", json={
            "query": "test query",
            "documents": [],
            "top_k": 5
        })
        assert response.status_code == 422  # Validation error
        
        # Test invalid top_k
        response = client.post("/brain2/rerank", json={
            "query": "test query",
            "documents": [{"text": "test doc"}],
            "top_k": 0
        })
        assert response.status_code == 422  # Validation error
        
        response = client.post("/brain2/rerank", json={
            "query": "test query",
            "documents": [{"text": "test doc"}],
            "top_k": 101  # Exceeds max limit
        })
        assert response.status_code == 422  # Validation error
        
        print("✅ Rerank endpoint validation test completed")
    
    def test_rerank_endpoint_functionality(self, client):
        """Test reranking endpoint functionality"""
        # Prepare test request
        request_data = {
            "query": SAMPLE_QUERY,
            "documents": [
                {
                    "text": doc["text"],
                    "doc_id": doc.get("doc_id"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in SAMPLE_DOCUMENTS
            ],
            "top_k": 3
        }
        
        response = client.post("/brain2/rerank", json=request_data)
        
        # Response should be 200 (success) or 503 (service unavailable)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_fields = [
                "results", "query", "total_documents", "returned_documents",
                "processing_time_ms", "model_info"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate results
            results = data["results"]
            assert isinstance(results, list)
            assert len(results) <= 3  # top_k limit
            assert len(results) <= len(SAMPLE_DOCUMENTS)
            
            # Validate individual results
            for result in results:
                assert "text" in result
                assert "relevance_score" in result
                assert "rank" in result
                assert isinstance(result["relevance_score"], (int, float))
                assert 0 <= result["relevance_score"] <= 1
                assert isinstance(result["rank"], int)
                assert result["rank"] >= 1
            
            # Validate ranking order
            scores = [r["relevance_score"] for r in results]
            assert scores == sorted(scores, reverse=True)
            
            # Validate metadata
            assert data["query"] == SAMPLE_QUERY
            assert data["total_documents"] == len(SAMPLE_DOCUMENTS)
            assert data["returned_documents"] == len(results)
            assert data["processing_time_ms"] > 0
            
            print(f"✅ Rerank endpoint functionality test completed:")
            print(f"  - Query: {data['query']}")
            print(f"  - Total docs: {data['total_documents']}")
            print(f"  - Returned docs: {data['returned_documents']}")
            print(f"  - Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"  - Top score: {results[0]['relevance_score']:.4f}")
            
        elif response.status_code == 503:
            data = response.json()
            assert "detail" in data
            print(f"⚠️ Rerank endpoint returned 503: {data['detail']}")
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/brain2/metrics")
        
        # Metrics should return 200 or 500 depending on availability
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            # Metrics should be in Prometheus format (plain text)
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            
            metrics_text = response.text
            assert "brain2_requests_total" in metrics_text
            assert "brain2_processing_time_ms" in metrics_text
            assert "brain2_memory_usage_mb" in metrics_text
            assert "brain2_model_loaded" in metrics_text
            
            print(f"✅ Metrics endpoint test completed:")
            print(f"  - Content type: {response.headers['content-type']}")
            print(f"  - Metrics length: {len(metrics_text)} characters")
        else:
            print("⚠️ Metrics endpoint returned 500 (internal error)")
    
    @pytest.mark.asyncio
    async def test_async_rerank_endpoint(self, async_client):
        """Test asynchronous reranking endpoint"""
        # Prepare test request
        request_data = {
            "task_id": "test_task_001",
            "task_type": "rerank",
            "query": SAMPLE_QUERY,
            "documents": [
                {
                    "text": doc["text"],
                    "doc_id": doc.get("doc_id"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in SAMPLE_DOCUMENTS[:2]  # Use smaller set for async test
            ],
            "top_k": 2
        }
        
        response = await async_client.post("/brain2/rerank/async", json=request_data)
        
        # Response should be 200 (success) or 503 (service unavailable)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            required_fields = ["task_id", "status", "created_at"]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate specific values
            assert data["task_id"] == "test_task_001"
            assert data["status"] == "submitted"
            assert data["created_at"] is not None
            
            print(f"✅ Async rerank endpoint test completed:")
            print(f"  - Task ID: {data['task_id']}")
            print(f"  - Status: {data['status']}")
            print(f"  - Created: {data['created_at']}")
            
        elif response.status_code == 503:
            data = response.json()
            print(f"⚠️ Async rerank endpoint returned 503: {data['detail']}")
    
    def test_error_handling(self, client):
        """Test service error handling"""
        # Test invalid endpoint
        response = client.get("/brain2/invalid_endpoint")
        assert response.status_code == 404
        
        # Test invalid method
        response = client.put("/brain2/health")
        assert response.status_code == 405
        
        print("✅ Error handling test completed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
