"""
Comprehensive unit tests for Brain 4 Manager
Tests all core functionality and edge cases
Based on implementation_part_6.md specifications
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

class TestBrain4Manager:
    """Test suite for Brain 4 Manager"""
    
    @pytest.fixture
    async def brain4_manager(self, mock_settings):
        """Create Brain 4 manager for testing"""
        
        # Mock all external dependencies
        with patch('brain4_docling.core.brain4_manager.MemoryManager') as mock_memory, \
             patch('brain4_docling.core.brain4_manager.PerformanceMonitor') as mock_perf, \
             patch('brain4_docling.core.brain4_manager.BrainCommunicator') as mock_comm, \
             patch('brain4_docling.core.brain4_manager.DocumentStore') as mock_store, \
             patch('brain4_docling.core.brain4_manager.DocumentConverter') as mock_converter, \
             patch('brain4_docling.core.brain4_manager.DocumentProcessor') as mock_processor:
            
            # Configure memory manager mock
            mock_memory_instance = AsyncMock()
            mock_memory_instance.get_current_usage.return_value = {"gpu": {"used_gb": 4.0}}
            mock_memory_instance.allocate_memory_for_brain = AsyncMock(return_value=True)
            mock_memory_instance.deallocate_memory_for_brain = AsyncMock()
            mock_memory_instance.cleanup_unused_memory = AsyncMock()
            mock_memory.return_value = mock_memory_instance
            
            # Configure performance monitor mock
            mock_perf_instance = AsyncMock()
            mock_perf_instance.collect_metrics.return_value = {"gpu_usage": 50.0}
            mock_perf.return_value = mock_perf_instance
            
            # Configure brain communicator mock
            mock_comm_instance = AsyncMock()
            mock_comm_instance.initialize = AsyncMock()
            mock_comm_instance.register_brain = AsyncMock()
            mock_comm_instance.request_embeddings = AsyncMock(return_value=[0.1] * 384)
            mock_comm_instance.request_analysis = AsyncMock(return_value={"analysis": "test"})
            mock_comm_instance.request_summary = AsyncMock(return_value="Test summary")
            mock_comm_instance.broadcast_document_processed = AsyncMock()
            mock_comm_instance.send_heartbeat = AsyncMock()
            mock_comm_instance.close = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            
            # Configure document store mock
            mock_store_instance = AsyncMock()
            mock_store_instance.initialize = AsyncMock()
            mock_store_instance.store_document = AsyncMock(return_value=True)
            mock_store_instance.close = AsyncMock()
            mock_store.return_value = mock_store_instance
            
            # Configure document processor mock
            mock_processor_instance = AsyncMock()
            mock_processor_instance.process_document = AsyncMock(return_value={
                "source_path": "/test/file.pdf",
                "filename": "file.pdf",
                "document_type": "pdf",
                "file_size": 1024,
                "content": {"text": "Test content"},
                "chunks": [{"chunk_id": "chunk_0", "text": "Test content"}],
                "processing_time": 1.5
            })
            mock_processor.return_value = mock_processor_instance
            
            # Import and create manager
            from brain4_docling.core.brain4_manager import Brain4Manager
            manager = Brain4Manager(mock_settings)
            
            # Store mocks for access in tests
            manager._mock_memory = mock_memory_instance
            manager._mock_comm = mock_comm_instance
            manager._mock_store = mock_store_instance
            manager._mock_processor = mock_processor_instance
            
            await manager.start()
            
            yield manager
            
            # Cleanup
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, brain4_manager):
        """Test Brain 4 manager initialization"""
        
        assert brain4_manager is not None
        assert brain4_manager.settings is not None
        assert brain4_manager.memory_manager is not None
        assert brain4_manager.brain_communicator is not None
        assert brain4_manager.document_store is not None
        assert brain4_manager.document_processor is not None
        
        # Verify initialization calls
        brain4_manager._mock_comm.initialize.assert_called_once()
        brain4_manager._mock_store.initialize.assert_called_once()
        brain4_manager._mock_comm.register_brain.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_document_task(self, brain4_manager):
        """Test document task submission"""
        
        # Create test document
        test_doc_path = brain4_manager.settings.temp_dir / "test.pdf"
        test_doc_path.write_text("Test document content")
        
        # Submit task
        task_id = await brain4_manager.submit_document_task(
            file_path=str(test_doc_path),
            extract_tables=True,
            extract_images=True,
            generate_embeddings=True
        )
        
        assert task_id is not None
        assert task_id.startswith("doc_")
        assert task_id in brain4_manager.active_tasks
        
        # Check task properties
        task = brain4_manager.active_tasks[task_id]
        assert task.file_path == str(test_doc_path)
        assert task.status.value == "pending"
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, brain4_manager):
        """Test complete document processing pipeline"""
        
        # Create test document
        test_doc_path = brain4_manager.settings.temp_dir / "test.pdf"
        test_doc_path.write_text("Test document content for processing")
        
        # Submit and wait for processing
        task_id = await brain4_manager.submit_document_task(str(test_doc_path))
        
        # Wait for processing to complete
        await asyncio.sleep(0.1)
        
        # Verify processing steps were called
        brain4_manager._mock_memory.allocate_memory_for_brain.assert_called()
        brain4_manager._mock_processor.process_document.assert_called()
        brain4_manager._mock_comm.request_embeddings.assert_called()
        brain4_manager._mock_store.store_document.assert_called()
        brain4_manager._mock_comm.broadcast_document_processed.assert_called()
        brain4_manager._mock_memory.deallocate_memory_for_brain.assert_called()
    
    @pytest.mark.asyncio
    async def test_memory_allocation_failure(self, brain4_manager):
        """Test handling of memory allocation failure"""
        
        # Configure memory allocation to fail
        brain4_manager._mock_memory.allocate_memory_for_brain = AsyncMock(return_value=False)
        
        # Create test document
        test_doc_path = brain4_manager.settings.temp_dir / "test.pdf"
        test_doc_path.write_text("Test document content")
        
        # Submit task
        task_id = await brain4_manager.submit_document_task(str(test_doc_path))
        
        # Wait for processing attempt
        await asyncio.sleep(0.1)
        
        # Task should fail due to memory allocation
        if task_id in brain4_manager.completed_tasks:
            task = brain4_manager.completed_tasks[task_id]
            assert task.status.value == "failed"
            assert "memory" in task.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_cross_brain_communication(self, brain4_manager):
        """Test communication with other brains"""
        
        # Test embedding request
        embeddings = await brain4_manager._generate_embeddings("test content")
        assert embeddings is not None
        assert len(embeddings) == 384
        brain4_manager._mock_comm.request_embeddings.assert_called_with("test content")
        
        # Test analysis request
        document_data = {"content": {"text": "test"}}
        enhanced_data = await brain4_manager._enhance_with_other_brains(document_data)
        assert enhanced_data is not None
        assert "enhanced_analysis" in enhanced_data
        brain4_manager._mock_comm.request_analysis.assert_called()
        brain4_manager._mock_comm.request_summary.assert_called()
    
    @pytest.mark.asyncio
    async def test_heartbeat_system(self, brain4_manager):
        """Test heartbeat system"""
        
        # Wait for heartbeat to be sent
        await asyncio.sleep(0.1)
        
        # Verify heartbeat was sent
        brain4_manager._mock_comm.send_heartbeat.assert_called()
        
        # Check heartbeat data structure
        call_args = brain4_manager._mock_comm.send_heartbeat.call_args
        if call_args:
            heartbeat_data = call_args[0][0]
            assert "brain_id" in heartbeat_data
            assert "status" in heartbeat_data
            assert "active_tasks" in heartbeat_data
    
    @pytest.mark.asyncio
    async def test_status_summary(self, brain4_manager):
        """Test status summary generation"""
        
        status = brain4_manager.get_status_summary()
        
        assert status["brain_id"] == "brain4"
        assert status["status"] == "active"
        assert "active_tasks" in status
        assert "completed_tasks" in status
        assert "processing_stats" in status
        assert "memory_usage" in status
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, brain4_manager):
        """Test graceful shutdown"""
        
        # Shutdown should complete without errors
        await brain4_manager.shutdown()
        
        # Verify cleanup calls
        brain4_manager._mock_comm.close.assert_called_once()
        brain4_manager._mock_store.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, brain4_manager):
        """Test error handling in document processing"""
        
        # Configure processor to raise exception
        brain4_manager._mock_processor.process_document = AsyncMock(
            side_effect=Exception("Processing failed")
        )
        
        # Create test document
        test_doc_path = brain4_manager.settings.temp_dir / "test.pdf"
        test_doc_path.write_text("Test document content")
        
        # Submit task
        task_id = await brain4_manager.submit_document_task(str(test_doc_path))
        
        # Wait for processing attempt
        await asyncio.sleep(0.1)
        
        # Task should be marked as failed
        if task_id in brain4_manager.completed_tasks:
            task = brain4_manager.completed_tasks[task_id]
            assert task.status.value == "failed"
            assert task.error_message is not None
