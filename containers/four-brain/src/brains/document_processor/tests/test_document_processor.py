"""
Comprehensive unit tests for Document Processor
Tests document processing functionality and edge cases
Based on implementation_part_6.md specifications
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

class TestDocumentProcessor:
    """Test suite for Document Processor"""
    
    @pytest.fixture
    def document_processor(self, mock_settings, mock_docling_converter):
        """Create document processor for testing"""
        
        from brain4_docling.core.document_processor import DocumentProcessor
        processor = DocumentProcessor(mock_docling_converter, mock_settings)
        
        return processor
    
    @pytest.mark.asyncio
    async def test_file_validation_success(self, document_processor):
        """Test successful file validation"""
        
        # Create valid test file
        test_file = document_processor.settings.temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Should not raise exception
        await document_processor._validate_file(test_file)
    
    @pytest.mark.asyncio
    async def test_file_validation_not_found(self, document_processor):
        """Test file validation with non-existent file"""
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            await document_processor._validate_file(Path("/non/existent/file.pdf"))
    
    @pytest.mark.asyncio
    async def test_file_validation_too_large(self, document_processor):
        """Test file validation with oversized file"""
        
        # Create oversized file (larger than 10MB limit)
        test_file = document_processor.settings.temp_dir / "large.pdf"
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        test_file.write_text(large_content)
        
        # Should raise ValueError for file too large
        with pytest.raises(ValueError, match="File too large"):
            await document_processor._validate_file(test_file)
    
    @pytest.mark.asyncio
    async def test_file_validation_unsupported_format(self, document_processor):
        """Test file validation with unsupported format"""
        
        # Create file with unsupported extension
        test_file = document_processor.settings.temp_dir / "test.xyz"
        test_file.write_text("test content")
        
        # Should raise ValueError for unsupported format
        with pytest.raises(ValueError, match="Unsupported file format"):
            await document_processor._validate_file(test_file)
    
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, document_processor):
        """Test file metadata extraction"""
        
        # Create test file
        test_file = document_processor.settings.temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        metadata = await document_processor._extract_file_metadata(test_file)
        
        assert metadata["filename"] == "test.pdf"
        assert metadata["file_size"] > 0
        assert metadata["file_hash"] is not None
        assert metadata["file_type"] == "pdf"
        assert metadata["mime_type"] is not None
        assert "created_at" in metadata
        assert "modified_at" in metadata
    
    @pytest.mark.asyncio
    async def test_file_hash_calculation(self, document_processor):
        """Test file hash calculation for deduplication"""
        
        # Create test file
        test_file = document_processor.settings.temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        hash1 = await document_processor._calculate_file_hash(test_file)
        hash2 = await document_processor._calculate_file_hash(test_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different content should produce different hash
        test_file.write_text("different content")
        hash3 = await document_processor._calculate_file_hash(test_file)
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_document_conversion(self, document_processor):
        """Test document conversion using Docling"""
        
        # Create test file
        test_file = document_processor.settings.temp_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Test conversion
        result = await document_processor._convert_document(test_file)
        
        assert result is not None
        assert hasattr(result, 'document')
        
        # Verify converter was called
        document_processor.converter.convert.assert_called_once_with(str(test_file))
    
    @pytest.mark.asyncio
    async def test_content_extraction(self, document_processor):
        """Test content extraction from conversion result"""
        
        # Create mock conversion result
        mock_result = Mock()
        mock_document = Mock()
        mock_document.export_to_markdown.return_value = "# Test Document\nContent here"
        mock_document.export_to_json.return_value = '{"title": "Test"}'
        mock_document.pages = [Mock()]
        mock_document.texts = []
        mock_document.tables = []
        mock_document.pictures = []
        mock_result.document = mock_document
        
        content = await document_processor._extract_content(mock_result)
        
        assert content["text"] == "# Test Document\nContent here"
        assert content["json"]["title"] == "Test"
        assert "structure" in content
        assert "tables" in content
        assert "images" in content
        assert "metadata_extracted" in content
    
    def test_semantic_chunking(self, document_processor):
        """Test semantic text chunking"""
        
        # Test with simple text
        test_text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = asyncio.run(document_processor._create_semantic_chunks(test_text))
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "word_count" in chunk
            assert "sentence_count" in chunk
            assert "chunk_index" in chunk
            assert chunk["word_count"] > 0
    
    def test_semantic_chunking_large_text(self, document_processor):
        """Test semantic chunking with large text"""
        
        # Create large text that should be split into multiple chunks
        test_text = "This is a test sentence. " * 200  # Large text
        
        chunks = asyncio.run(document_processor._create_semantic_chunks(test_text))
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Check chunk overlap
        if len(chunks) > 1:
            # Verify chunks have some overlap
            chunk1_end = chunks[0]["text"].split()[-5:]  # Last 5 words
            chunk2_start = chunks[1]["text"].split()[:10]  # First 10 words
            
            # Should have some overlap
            overlap = set(chunk1_end) & set(chunk2_start)
            assert len(overlap) > 0
    
    def test_sentence_splitting(self, document_processor):
        """Test sentence splitting functionality"""
        
        test_text = "This is sentence one. This is sentence two! Is this sentence three?"
        
        sentences = document_processor._split_into_sentences(test_text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "Is this sentence three" in sentences[2]
    
    def test_overlap_calculation(self, document_processor):
        """Test overlap calculation for chunking"""
        
        test_sentences = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5"]
        
        overlap = document_processor._calculate_overlap_sentences(test_sentences)
        
        # Should be 20% overlap (1 sentence) or minimum 1
        assert overlap >= 1
        assert overlap < len(test_sentences)
    
    def test_heading_level_extraction(self, document_processor):
        """Test heading level extraction from labels"""
        
        test_cases = [
            ("heading-1", 1),
            ("heading-2", 2),
            ("h3", 3),
            ("title", 1),
            ("subtitle", 2),
            ("heading", 3)
        ]
        
        for label, expected_level in test_cases:
            level = document_processor._extract_heading_level(label)
            assert level == expected_level
    
    def test_processing_statistics(self, document_processor):
        """Test processing statistics tracking"""
        
        # Initial stats should be empty
        stats = document_processor.get_processing_stats()
        assert stats["documents_processed"] == 0
        assert stats["total_processing_time"] == 0.0
        
        # Simulate processing a document
        processed_data = {
            "document_type": "pdf",
            "processing_time": 2.5,
            "content": {"structure": {"page_count": 3}}
        }
        
        document_processor._update_processing_stats(processed_data)
        
        # Check updated stats
        updated_stats = document_processor.get_processing_stats()
        assert updated_stats["documents_processed"] == 1
        assert updated_stats["total_processing_time"] == 2.5
        assert "pdf" in updated_stats["format_stats"]
        assert updated_stats["format_stats"]["pdf"]["count"] == 1
    
    @pytest.mark.asyncio
    async def test_complete_document_processing(self, document_processor):
        """Test complete document processing pipeline"""
        
        # Create test file
        test_file = document_processor.settings.temp_dir / "test.pdf"
        test_file.write_text("Test document content for complete processing")
        
        # Process document
        result = await document_processor.process_document(
            file_path=test_file,
            metadata={"test": True}
        )
        
        # Verify result structure
        assert result["source_path"] == str(test_file)
        assert result["filename"] == "test.pdf"
        assert result["document_type"] == "pdf"
        assert result["file_size"] > 0
        assert "content" in result
        assert "chunks" in result
        assert "processing_time" in result
        assert "processed_at" in result
        
        # Verify content structure
        content = result["content"]
        assert "text" in content
        assert "json" in content
        assert "structure" in content
        assert "tables" in content
        assert "images" in content
        
        # Verify chunks
        chunks = result["chunks"]
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
