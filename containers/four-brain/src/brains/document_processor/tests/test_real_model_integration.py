"""
Real Model Loading Integration Tests - NO MOCKING
Tests actual Qwen3-4B model loading and embedding generation
AUTHENTIC IMPLEMENTATION - Zero fabrication policy
"""

import pytest
import asyncio
import time
import logging
from pathlib import Path
from typing import List

# Optional dependencies with skip conditions
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from brain4_docling.models.embedding_models import Qwen3EmbeddingModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    from brain4_docling.config.settings import Brain4Settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

# Skip conditions for clean environment compatibility
skip_if_no_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
skip_if_no_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="Embedding models not available")
skip_if_no_settings = pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
skip_if_no_gpu = pytest.mark.skipif(not (TORCH_AVAILABLE and torch.cuda.is_available()), reason="GPU not available")

def check_model_files_exist(settings) -> bool:
    """Check if required model files exist"""
    try:
        model_path = Path(settings.qwen3_model_path)
        return model_path.exists() and model_path.is_dir()
    except Exception:
        return False

skip_if_no_model_files = pytest.mark.skipif(
    not (SETTINGS_AVAILABLE and check_model_files_exist(Brain4Settings() if SETTINGS_AVAILABLE else None)),
    reason="Model files not available"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealModelIntegration:
    """Integration tests using real Qwen3-4B models - NO MOCKING"""
    
    @pytest.fixture
    def settings(self):
        """Real settings for model testing"""
        return Brain4Settings()
    
    @pytest.fixture
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    @skip_if_no_model_files
    async def real_qwen3_model(self, settings):
        """Load actual Qwen3-4B model from local cache"""
        model = Qwen3EmbeddingModel(
            model_path=settings.qwen3_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_mrl_truncation=True,
            embedding_dim=2000
        )
        
        # Measure real loading time
        start_time = time.time()
        await model.load_model()
        loading_time = time.time() - start_time
        
        logger.info(f"Real model loading time: {loading_time:.2f} seconds")
        
        yield model, loading_time
        
        # Cleanup
        if hasattr(model, 'model') and model.model:
            del model.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @pytest.mark.asyncio
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    @skip_if_no_model_files
    async def test_real_model_loading_performance(self, real_qwen3_model):
        """Test actual model loading performance - NO MOCKING"""
        model, loading_time = real_qwen3_model
        
        # Verify real model is loaded
        assert model.is_loaded is True
        assert model.model is not None
        assert hasattr(model.model, 'encode')
        
        # Verify loading time is reasonable (should be <30 seconds for cached model)
        assert loading_time < 30.0, f"Model loading too slow: {loading_time:.2f}s"
        
        # Verify model dimensions
        assert model.native_embedding_dim == 2560  # Qwen3-4B native dimension
        assert model.embedding_dim == 2000  # MRL truncated dimension
        
        logger.info(f"✅ Real model loading test passed: {loading_time:.2f}s")
    
    @pytest.mark.asyncio
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    @skip_if_no_model_files
    async def test_real_embedding_generation(self, real_qwen3_model):
        """Test actual embedding generation with real Qwen3-4B model"""
        model, _ = real_qwen3_model
        
        # Test with real text content
        test_texts = [
            "This is a test document for embedding generation.",
            "Phase 6 Four-Brain AI System integration testing.",
            "Real Qwen3-4B model processing authentic content."
        ]
        
        # Measure embedding generation time
        start_time = time.time()
        embeddings = await model.encode_async(test_texts)
        generation_time = time.time() - start_time
        
        # Verify real embeddings
        assert embeddings is not None
        assert len(embeddings) == len(test_texts)
        
        for i, embedding in enumerate(embeddings):
            # Verify embedding structure
            assert isinstance(embedding, list)
            assert len(embedding) == 2000  # MRL truncated dimension
            assert all(isinstance(x, float) for x in embedding)
            
            # Verify embeddings are not zero vectors (real processing)
            assert sum(abs(x) for x in embedding) > 0.0
            
            # Verify embeddings are normalized (real SentenceTransformer output)
            magnitude = sum(x*x for x in embedding) ** 0.5
            assert 0.9 < magnitude < 1.1  # Should be approximately normalized
        
        logger.info(f"✅ Real embedding generation test passed: {generation_time:.2f}s for {len(test_texts)} texts")
    
    @pytest.mark.asyncio
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    @skip_if_no_model_files
    @skip_if_no_gpu
    async def test_real_gpu_utilization(self, real_qwen3_model):
        """Test actual GPU memory utilization during model operations"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU testing")
        
        model, _ = real_qwen3_model
        
        # Measure GPU memory before and after embedding generation
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Generate embeddings with real model
        test_text = "GPU utilization test with real Qwen3-4B model processing."
        embeddings = await model.encode_async([test_text])
        
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        # Verify real GPU usage
        memory_used = (peak_memory - initial_memory) / (1024**3)  # GB
        
        logger.info(f"GPU memory usage: {memory_used:.2f} GB")
        logger.info(f"Initial: {initial_memory/(1024**3):.2f} GB")
        logger.info(f"Peak: {peak_memory/(1024**3):.2f} GB")
        logger.info(f"Final: {final_memory/(1024**3):.2f} GB")
        
        # Verify real GPU utilization (should use >1GB for Qwen3-4B)
        assert memory_used > 1.0, f"GPU usage too low: {memory_used:.2f} GB"
        assert embeddings is not None and len(embeddings) == 1
        
        logger.info(f"✅ Real GPU utilization test passed: {memory_used:.2f} GB used")
    
    @pytest.mark.asyncio
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    @skip_if_no_model_files
    async def test_real_batch_processing(self, real_qwen3_model):
        """Test real batch processing with varying batch sizes"""
        model, _ = real_qwen3_model
        
        # Test different batch sizes with real content
        batch_sizes = [1, 4, 8]
        test_texts = [f"Batch processing test document {i}" for i in range(10)]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            embeddings = await model.encode_async(test_texts[:batch_size])
            processing_time = time.time() - start_time
            
            # Verify real batch processing
            assert len(embeddings) == batch_size
            assert all(len(emb) == 2000 for emb in embeddings)
            
            # Verify processing time scales reasonably
            time_per_doc = processing_time / batch_size
            assert time_per_doc < 5.0, f"Processing too slow: {time_per_doc:.2f}s per document"
            
            logger.info(f"Batch size {batch_size}: {processing_time:.2f}s ({time_per_doc:.2f}s per doc)")
        
        logger.info("✅ Real batch processing test passed")
    
    @pytest.mark.asyncio
    @skip_if_no_torch
    @skip_if_no_model
    @skip_if_no_settings
    async def test_real_error_handling(self, settings):
        """Test real error handling without mocking"""
        # Test with invalid model path
        invalid_model = Qwen3EmbeddingModel(
            model_path="/invalid/path/to/model",
            device="cpu"
        )
        
        # Should fail honestly, not return fake success
        with pytest.raises(Exception):
            await invalid_model.load_model()
        
        # Test with empty text
        valid_model = Qwen3EmbeddingModel(
            model_path=settings.qwen3_model_path,
            device="cpu"
        )
        await valid_model.load_model()
        
        # Should handle empty input gracefully
        empty_embeddings = await valid_model.encode_async([])
        assert empty_embeddings == []
        
        # Should handle None input gracefully
        with pytest.raises((ValueError, TypeError)):
            await valid_model.encode_async(None)
        
        logger.info("✅ Real error handling test passed")

if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v", "-s"])
