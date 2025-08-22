"""
Real PDF Processing Integration Tests - NO MOCKING
Tests actual PDF processing pipeline with real documents
AUTHENTIC IMPLEMENTATION - Zero fabrication policy
"""

import pytest
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any

# Optional dependencies with skip conditions
try:
    from brain4_docling.core.brain4_manager import Brain4Manager
    BRAIN4_MANAGER_AVAILABLE = True
except ImportError:
    BRAIN4_MANAGER_AVAILABLE = False

try:
    from brain4_docling.core.document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False

try:
    from brain4_docling.models.embedding_models import Qwen3EmbeddingModel
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False

try:
    from brain4_docling.config.settings import Brain4Settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    import docling
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Skip conditions for clean environment compatibility
skip_if_no_brain4 = pytest.mark.skipif(not BRAIN4_MANAGER_AVAILABLE, reason="Brain4Manager not available")
skip_if_no_processor = pytest.mark.skipif(not DOCUMENT_PROCESSOR_AVAILABLE, reason="DocumentProcessor not available")
skip_if_no_embedding = pytest.mark.skipif(not EMBEDDING_MODEL_AVAILABLE, reason="Embedding models not available")
skip_if_no_settings = pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
skip_if_no_docling = pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not available")

def check_test_pdf_exists(settings) -> bool:
    """Check if test PDF file exists"""
    try:
        test_pdf_path = Path(settings.test_data_dir) / "2506.16507v1.pdf"
        return test_pdf_path.exists()
    except Exception:
        return False

skip_if_no_test_pdf = pytest.mark.skipif(
    not (SETTINGS_AVAILABLE and check_test_pdf_exists(Brain4Settings() if SETTINGS_AVAILABLE else None)),
    reason="Test PDF file not available"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealPDFProcessing:
    """Integration tests using real PDF documents - NO MOCKING"""
    
    @pytest.fixture
    def settings(self):
        """Real settings for PDF processing"""
        return Brain4Settings()
    
    @pytest.fixture
    def real_test_pdf_path(self, settings):
        """Path to real test PDF document"""
        pdf_path = settings.test_data_dir / "2506.16507v1.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        return str(pdf_path)
    
    @pytest.fixture
    async def real_document_processor(self, settings):
        """Real DocumentProcessor instance"""
        processor = DocumentProcessor(settings)
        await processor.initialize()
        
        yield processor
        
        await processor.cleanup()
    
    @pytest.fixture
    async def real_qwen3_model(self, settings):
        """Real Qwen3-4B embedding model"""
        model = Qwen3EmbeddingModel(
            model_path=settings.qwen3_model_path,
            device="cuda" if settings.max_vram_usage > 0 else "cpu",
            use_mrl_truncation=True,
            embedding_dim=2000
        )
        
        await model.load_model()
        yield model
        
        # Cleanup
        if hasattr(model, 'model') and model.model:
            del model.model
    
    @pytest.mark.asyncio
    @skip_if_no_processor
    @skip_if_no_settings
    @skip_if_no_docling
    @skip_if_no_test_pdf
    async def test_real_pdf_docling_conversion(self, real_test_pdf_path, real_document_processor):
        """Test real PDF conversion using Docling - NO MOCKING"""
        processor = real_document_processor
        
        # Process real PDF document
        start_time = time.time()
        result = await processor.process_document(
            file_path=real_test_pdf_path,
            metadata={"test_type": "real_pdf_processing"}
        )
        processing_time = time.time() - start_time
        
        # Verify real processing results
        assert result is not None, "Document processing returned None"
        assert "content" in result, "Missing content in processing result"
        assert "text" in result["content"], "Missing text content"
        assert "chunks" in result, "Missing chunks in processing result"
        
        # Verify real text extraction
        extracted_text = result["content"]["text"]
        assert len(extracted_text) > 0, "No text extracted from PDF"
        assert isinstance(extracted_text, str), "Extracted text is not string"
        
        # Verify document chunks
        chunks = result["chunks"]
        assert len(chunks) > 0, "No chunks generated from PDF"
        assert all("text" in chunk for chunk in chunks), "Missing text in chunks"
        
        # Verify processing metadata
        assert result["source_path"] == real_test_pdf_path
        assert result["document_type"] == "pdf"
        assert result["processing_time"] > 0
        
        logger.info(f"✅ Real PDF processing: {len(extracted_text)} chars, {len(chunks)} chunks, {processing_time:.2f}s")
    
    @pytest.mark.asyncio
    @skip_if_no_processor
    @skip_if_no_embedding
    @skip_if_no_settings
    @skip_if_no_test_pdf
    async def test_real_pdf_embedding_generation(self, real_test_pdf_path, real_document_processor, real_qwen3_model):
        """Test real embedding generation from PDF content"""
        processor = real_document_processor
        model = real_qwen3_model
        
        # Process real PDF
        result = await processor.process_document(file_path=real_test_pdf_path)
        extracted_text = result["content"]["text"]
        
        # Generate real embeddings from extracted text
        start_time = time.time()
        
        # Process text in chunks for embedding generation
        text_chunks = [chunk["text"] for chunk in result["chunks"][:5]]  # First 5 chunks
        embeddings = await model.encode_async(text_chunks)
        
        embedding_time = time.time() - start_time
        
        # Verify real embeddings
        assert embeddings is not None, "Embedding generation returned None"
        assert len(embeddings) == len(text_chunks), "Embedding count mismatch"
        
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, list), f"Embedding {i} is not a list"
            assert len(embedding) == 2000, f"Embedding {i} wrong dimension: {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), f"Embedding {i} contains non-float values"
            
            # Verify embeddings are not zero vectors (real processing)
            magnitude = sum(x*x for x in embedding) ** 0.5
            assert magnitude > 0.1, f"Embedding {i} appears to be zero vector"
        
        logger.info(f"✅ Real embedding generation: {len(embeddings)} embeddings, {embedding_time:.2f}s")
    
    @pytest.mark.asyncio
    @skip_if_no_brain4
    @skip_if_no_settings
    @skip_if_no_test_pdf
    async def test_real_end_to_end_pdf_pipeline(self, real_test_pdf_path, settings):
        """Test complete end-to-end PDF processing pipeline"""
        # Create real Brain4Manager
        brain4_manager = Brain4Manager(settings)
        await brain4_manager.start()
        
        try:
            # Submit real document processing task
            start_time = time.time()
            task_id = await brain4_manager.submit_document_task(
                file_path=real_test_pdf_path,
                extract_tables=True,
                extract_images=True,
                generate_embeddings=True,
                priority=1
            )
            
            # Verify real task creation
            assert task_id is not None, "Task ID is None"
            assert isinstance(task_id, str), "Task ID is not string"
            assert len(task_id) > 0, "Task ID is empty"
            
            # Wait for processing to begin (real processing takes time)
            await asyncio.sleep(2.0)
            
            # Check task status
            task_status = brain4_manager.get_task_status(task_id)
            assert task_status is not None, "Task status is None"
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ End-to-end pipeline: Task {task_id}, Status: {task_status}, Time: {total_time:.2f}s")
            
        finally:
            await brain4_manager.shutdown()
    
    @pytest.mark.asyncio
    @skip_if_no_processor
    @skip_if_no_settings
    @skip_if_no_test_pdf
    async def test_real_pdf_content_validation(self, real_test_pdf_path, real_document_processor):
        """Validate real PDF content extraction quality"""
        processor = real_document_processor
        
        # Process real PDF
        result = await processor.process_document(file_path=real_test_pdf_path)
        extracted_text = result["content"]["text"]
        
        # Validate content quality (real PDF should contain meaningful text)
        assert len(extracted_text) > 100, f"Extracted text too short: {len(extracted_text)} chars"
        
        # Check for common PDF content indicators
        text_lower = extracted_text.lower()
        
        # Should contain some common academic paper elements (this is a research paper)
        content_indicators = ["abstract", "introduction", "method", "result", "conclusion", "reference"]
        found_indicators = [indicator for indicator in content_indicators if indicator in text_lower]
        
        assert len(found_indicators) > 0, f"No academic content indicators found in: {extracted_text[:200]}..."
        
        # Verify chunks contain meaningful content
        chunks = result["chunks"]
        non_empty_chunks = [chunk for chunk in chunks if len(chunk["text"].strip()) > 10]
        assert len(non_empty_chunks) > 0, "No meaningful chunks extracted"
        
        # Verify chunk text quality
        for chunk in non_empty_chunks[:3]:  # Check first 3 chunks
            chunk_text = chunk["text"].strip()
            assert len(chunk_text) > 10, f"Chunk too short: {chunk_text}"
            assert not chunk_text.isspace(), f"Chunk is only whitespace: {repr(chunk_text)}"
        
        logger.info(f"✅ PDF content validation: {len(extracted_text)} chars, {len(non_empty_chunks)} meaningful chunks")
        logger.info(f"Content indicators found: {found_indicators}")
    
    @pytest.mark.asyncio
    @skip_if_no_processor
    @skip_if_no_settings
    async def test_real_pdf_error_handling(self, settings):
        """Test real error handling with invalid PDF files"""
        processor = DocumentProcessor(settings)
        await processor.initialize()
        
        try:
            # Test with non-existent file
            with pytest.raises(Exception):
                await processor.process_document("/invalid/path/to/file.pdf")
            
            # Test with invalid file format (create temporary text file)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is not a PDF file")
                invalid_file_path = f.name
            
            try:
                # Should handle invalid format gracefully
                with pytest.raises(Exception):
                    await processor.process_document(invalid_file_path)
            finally:
                Path(invalid_file_path).unlink(missing_ok=True)
            
            logger.info("✅ Real PDF error handling test passed")
            
        finally:
            await processor.cleanup()
    
    @pytest.mark.asyncio
    @skip_if_no_processor
    @skip_if_no_settings
    @skip_if_no_test_pdf
    async def test_real_pdf_performance_benchmarks(self, real_test_pdf_path, real_document_processor):
        """Benchmark real PDF processing performance"""
        processor = real_document_processor
        
        # Measure processing performance
        times = []
        for i in range(3):  # Run 3 times for average
            start_time = time.time()
            result = await processor.process_document(file_path=real_test_pdf_path)
            processing_time = time.time() - start_time
            times.append(processing_time)
            
            # Verify consistent results
            assert result is not None
            assert len(result["content"]["text"]) > 0
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Performance assertions (reasonable processing time)
        assert avg_time < 30.0, f"Average processing too slow: {avg_time:.2f}s"
        assert max_time < 60.0, f"Max processing too slow: {max_time:.2f}s"
        
        logger.info(f"✅ Performance benchmark: Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")

if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v", "-s"])
