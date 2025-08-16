"""
Integration tests for Brain 4 system
Tests end-to-end functionality and cross-component integration
Based on implementation_part_6.md specifications
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

class TestBrain4Integration:
    """Integration test suite for Brain 4 system"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_health_check_integration(self):
        """Test real health check endpoints without mocks"""

        # Import the actual health check functions
        from brain4_docling.api.health import _check_database_connection, _check_redis_connection, _check_gpu_availability

        # Test real GPU check (should work on RTX 5070 Ti)
        gpu_available = await _check_gpu_availability()
        assert isinstance(gpu_available, bool)

        # Test database connection (may fail if not running, but should not crash)
        try:
            db_connected = await _check_database_connection()
            assert isinstance(db_connected, bool)
        except Exception as e:
            # Expected if database not running - test that it fails gracefully
            assert "connection" in str(e).lower() or "timeout" in str(e).lower()

        # Test Redis connection (may fail if not running, but should not crash)
        try:
            redis_connected = await _check_redis_connection()
            assert isinstance(redis_connected, bool)
        except Exception as e:
            # Expected if Redis not running - test that it fails gracefully
            assert "connection" in str(e).lower() or "timeout" in str(e).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_memory_manager_integration(self):
        """Test real memory manager without mocks"""

        from brain4_docling.utils.memory_manager import MemoryManager

        # Create real memory manager
        memory_manager = MemoryManager()

        # Test real GPU detection
        assert isinstance(memory_manager.gpu_available, bool)
        assert isinstance(memory_manager.total_gpu_memory, float)

        if memory_manager.gpu_available:
            # Test real memory allocation (small amount)
            try:
                result = await memory_manager.allocate_memory_for_brain("test_brain", 0.1)  # 100MB
                assert isinstance(result, bool)

                if result:
                    # Test real deallocation
                    await memory_manager.deallocate_memory_for_brain("test_brain", 0.1)

            except Exception as e:
                # May fail due to insufficient memory - test that it fails gracefully
                assert "memory" in str(e).lower() or "allocation" in str(e).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_generation_failure_handling(self):
        """Test that embedding generation fails honestly instead of fabricating data"""

        # This test verifies our fix for MD5 hash fallback embeddings
        from brain4_docling.core.brain4_manager import Brain4Manager
        from brain4_docling.config.settings import Brain4Settings

        settings = Brain4Settings()
        manager = Brain4Manager(settings)

        # Test that embedding generation raises exception instead of returning fake data
        with pytest.raises(Exception) as exc_info:
            await manager._generate_embeddings("test text")

        # Verify the exception message indicates honest failure
        assert "embedding generation failed" in str(exc_info.value).lower() or "brain 1 unavailable" in str(exc_info.value).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_documents_api_integration(self):
        """Test documents API with real database integration"""

        from brain4_docling.api.documents import list_documents, get_document
        from brain4_docling.config.settings import Brain4Settings
        import asyncpg

        settings = Brain4Settings()

        # Test list_documents with real database
        try:
            result = await list_documents(limit=5, offset=0)

            # Verify response structure
            assert "documents" in result
            assert "total" in result
            assert "limit" in result
            assert "offset" in result
            assert isinstance(result["documents"], list)
            assert isinstance(result["total"], int)

        except Exception as e:
            # Expected if database not available - test that it fails honestly
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_monitoring_api_integration(self):
        """Test monitoring API with real system metrics"""

        from brain4_docling.api.monitoring import get_metrics

        # Test get_metrics with real system monitoring
        try:
            result = await get_metrics()

            # Verify response structure
            assert "timestamp" in result
            assert "gpu" in result
            assert "system" in result
            assert "processing" in result

            # Verify GPU metrics structure
            gpu_metrics = result["gpu"]
            assert "available" in gpu_metrics
            assert "memory_usage_percent" in gpu_metrics
            assert "temperature_c" in gpu_metrics
            assert isinstance(gpu_metrics["available"], bool)

            # Verify system metrics structure
            system_metrics = result["system"]
            assert "memory_usage_percent" in system_metrics
            assert "cpu_usage_percent" in system_metrics
            assert isinstance(system_metrics["memory_usage_percent"], (int, float))
            assert isinstance(system_metrics["cpu_usage_percent"], (int, float))

        except Exception as e:
            # Should not fail for system metrics
            pytest.fail(f"System metrics should always be available: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_health_check_model_verification(self):
        """Test health check with real model verification"""

        from brain4_docling.api.health import _check_docling_models

        # Test real model verification
        try:
            models_ready = await _check_docling_models()
            assert isinstance(models_ready, bool)

            # If models are ready, they should actually be functional
            if models_ready:
                # Additional verification that models are actually loaded
                from docling import DocumentConverter
                converter = DocumentConverter()
                assert hasattr(converter, 'convert')
                assert hasattr(converter, 'allowed_formats')

        except ImportError:
            # Expected if Docling not available - test that it fails honestly
            models_ready = await _check_docling_models()
            assert models_ready is False
        except Exception as e:
            # Should fail gracefully with clear error
            assert "docling" in str(e).lower() or "model" in str(e).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_api_workflow(self):
        """Test complete API workflow without mocks"""

        from brain4_docling.api.health import health_check
        from brain4_docling.api.monitoring import get_metrics
        from brain4_docling.api.documents import list_documents

        # Test health check
        try:
            health_result = await health_check()
            assert "status" in health_result
            assert "services" in health_result
            assert health_result["status"] in ["healthy", "degraded"]
        except Exception as e:
            pytest.fail(f"Health check should not raise exceptions: {e}")

        # Test monitoring
        try:
            metrics_result = await get_metrics()
            assert "timestamp" in metrics_result
            assert "gpu" in metrics_result
            assert "system" in metrics_result
        except Exception as e:
            pytest.fail(f"Monitoring should not raise exceptions: {e}")

        # Test documents (may fail if database unavailable)
        try:
            docs_result = await list_documents(limit=1)
            assert "documents" in docs_result
            assert "total" in docs_result
        except Exception as e:
            # Expected if database not available
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self, mock_settings, sample_documents):
        """Test complete end-to-end document processing"""
        
        # Mock all external dependencies
        with patch('brain4_docling.core.brain4_manager.MemoryManager') as mock_memory, \
             patch('brain4_docling.core.brain4_manager.PerformanceMonitor') as mock_perf, \
             patch('brain4_docling.core.brain4_manager.BrainCommunicator') as mock_comm, \
             patch('brain4_docling.core.brain4_manager.DocumentStore') as mock_store, \
             patch('brain4_docling.core.brain4_manager.DocumentConverter') as mock_converter:
            
            # Configure mocks for successful processing
            mock_memory.return_value.allocate_memory_for_brain = AsyncMock(return_value=True)
            mock_memory.return_value.deallocate_memory_for_brain = AsyncMock()
            mock_memory.return_value.cleanup_unused_memory = AsyncMock()
            
            mock_comm_instance = AsyncMock()
            mock_comm_instance.initialize = AsyncMock()
            mock_comm_instance.register_brain = AsyncMock()
            mock_comm_instance.request_embeddings = AsyncMock(return_value=[0.1] * 384)
            mock_comm_instance.request_analysis = AsyncMock(return_value={"classification": "document"})
            mock_comm_instance.request_summary = AsyncMock(return_value="Test summary")
            mock_comm_instance.broadcast_document_processed = AsyncMock()
            mock_comm_instance.send_heartbeat = AsyncMock()
            mock_comm_instance.close = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            
            mock_store.return_value.initialize = AsyncMock()
            mock_store.return_value.store_document = AsyncMock(return_value=True)
            mock_store.return_value.close = AsyncMock()
            
            # Mock Docling converter
            mock_result = Mock()
            mock_document = Mock()
            mock_document.export_to_markdown.return_value = "# Test Document\nContent"
            mock_document.export_to_json.return_value = '{"title": "Test"}'
            mock_document.pages = [Mock()]
            mock_document.texts = []
            mock_document.tables = []
            mock_document.pictures = []
            mock_result.document = mock_document
            mock_converter.return_value.convert.return_value = mock_result
            
            # Create Brain4Manager
            from brain4_docling.core.brain4_manager import Brain4Manager
            manager = Brain4Manager(mock_settings)
            await manager.start()
            
            try:
                # Process multiple documents
                task_ids = []
                for doc_path in sample_documents[:2]:  # Process first 2 documents
                    task_id = await manager.submit_document_task(doc_path)
                    task_ids.append(task_id)
                
                # Wait for processing to complete
                await asyncio.sleep(0.2)
                
                # Verify all tasks were processed
                for task_id in task_ids:
                    assert task_id in manager.active_tasks or task_id in manager.completed_tasks
                
                # Verify integration points were called
                mock_comm_instance.request_embeddings.assert_called()
                mock_comm_instance.request_analysis.assert_called()
                mock_comm_instance.request_summary.assert_called()
                mock_store.return_value.store_document.assert_called()
                mock_comm_instance.broadcast_document_processed.assert_called()
                
            finally:
                await manager.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_brain_communication_flow(self, mock_settings):
        """Test communication flow between Brain 4 and other brains"""
        
        with patch('brain4_docling.integration.brain_communicator.aioredis') as mock_redis:
            # Configure Redis mock
            mock_client = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            
            from brain4_docling.integration.brain_communicator import BrainCommunicator
            communicator = BrainCommunicator("redis://localhost:6379")
            
            # Mock message sending and receiving
            response_data = {"embeddings": [0.1] * 384}
            
            async def mock_send_and_wait(message, timeout=None):
                return response_data
            
            communicator._send_message_and_wait = mock_send_and_wait
            communicator.is_connected = True
            
            # Test embedding request
            embeddings = await communicator.request_embeddings("test content")
            assert embeddings == response_data["embeddings"]
            
            # Test analysis request
            analysis = await communicator.request_analysis({"content": {"text": "test"}})
            assert analysis == response_data
            
            # Test summary request
            summary = await communicator.request_summary("test content")
            assert summary == response_data
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_management_integration(self, mock_settings):
        """Test memory management integration across components"""
        
        from brain4_docling.utils.memory_manager import MemoryManager
        
        # Create memory manager
        memory_manager = MemoryManager(max_vram_usage=0.75, target_vram_usage=0.65)
        
        # Test memory allocation for multiple brains
        brain_ids = ["brain1", "brain2", "brain3", "brain4"]
        
        for brain_id in brain_ids:
            allocated = await memory_manager.allocate_memory_for_brain(brain_id, 2.0)
            # Should succeed for first few allocations
            assert isinstance(allocated, bool)
        
        # Test memory usage tracking
        usage = await memory_manager.get_current_usage()
        assert "system" in usage
        assert "allocated" in usage
        assert "total_allocated_gb" in usage
        
        # Test memory cleanup
        await memory_manager.cleanup_unused_memory()
        
        # Test memory health check
        health = await memory_manager.check_memory_health()
        assert "status" in health
        assert health["status"] in ["healthy", "warning", "critical", "error"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_integration(self, mock_database):
        """Test database integration with document storage"""
        
        from brain4_docling.integration.document_store import DocumentStore
        
        # Create document store with mock database
        with patch('brain4_docling.integration.document_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value = mock_database
            
            store = DocumentStore("postgresql://test:test@localhost:5432/test")
            await store.initialize()
            
            # Test document storage
            document_data = {
                "task_id": "test_123",
                "source_path": "/test/file.pdf",
                "document_type": "pdf",
                "file_size": 1024,
                "content": {
                    "text": "Test content",
                    "json": {"title": "Test"},
                    "structure": {"page_count": 1},
                    "tables": [],
                    "images": []
                },
                "embeddings": [0.1] * 384,
                "metadata": {"test": True}
            }
            
            result = await store.store_document(document_data)
            assert isinstance(result, bool)
            
            # Test document retrieval
            retrieved = await store.get_document("test_123")
            # Mock returns predefined data
            assert retrieved is not None
            
            await store.close()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        
        from brain4_docling.utils.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test metrics collection
        metrics = await monitor.collect_metrics()
        assert "timestamp" in metrics
        assert "cpu" in metrics
        assert "memory" in metrics
        
        # Test processing time recording
        monitor.record_processing_time(2.5)
        
        # Test error recording
        monitor.record_error()
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        assert "current_stats" in summary
        assert "trends" in summary
        assert "health_indicators" in summary
        
        # Test metric history
        history = monitor.get_metric_history("cpu_usage", hours=1)
        assert isinstance(history, list)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, mock_settings):
        """Test error recovery and resilience across components"""
        
        # Test with failing components
        with patch('brain4_docling.core.brain4_manager.MemoryManager') as mock_memory, \
             patch('brain4_docling.core.brain4_manager.BrainCommunicator') as mock_comm, \
             patch('brain4_docling.core.brain4_manager.DocumentStore') as mock_store:
            
            # Configure memory manager to fail allocation
            mock_memory.return_value.allocate_memory_for_brain = AsyncMock(return_value=False)
            mock_memory.return_value.deallocate_memory_for_brain = AsyncMock()
            
            # Configure communication to fail
            mock_comm_instance = AsyncMock()
            mock_comm_instance.initialize = AsyncMock()
            mock_comm_instance.register_brain = AsyncMock()
            mock_comm_instance.request_embeddings = AsyncMock(return_value=None)  # Fail
            mock_comm_instance.close = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            
            # Configure store to fail
            mock_store.return_value.initialize = AsyncMock()
            mock_store.return_value.store_document = AsyncMock(return_value=False)  # Fail
            mock_store.return_value.close = AsyncMock()
            
            from brain4_docling.core.brain4_manager import Brain4Manager
            manager = Brain4Manager(mock_settings)
            
            # Should handle initialization failures gracefully
            try:
                await manager.start()
                
                # Create test document
                test_doc = mock_settings.temp_dir / "test.pdf"
                test_doc.write_text("Test content")
                
                # Submit task - should handle failures gracefully
                task_id = await manager.submit_document_task(str(test_doc))
                
                # Wait for processing attempt
                await asyncio.sleep(0.1)
                
                # Task should be handled (either failed or processed with fallbacks)
                assert task_id in manager.active_tasks or task_id in manager.completed_tasks
                
            finally:
                await manager.shutdown()
    
    @pytest.mark.integration
    def test_configuration_validation(self, mock_settings):
        """Test configuration validation and settings"""
        
        # Test required settings
        assert hasattr(mock_settings, 'database_url')
        assert hasattr(mock_settings, 'redis_url')
        assert hasattr(mock_settings, 'max_concurrent_tasks')
        assert hasattr(mock_settings, 'max_file_size_mb')
        assert hasattr(mock_settings, 'supported_formats')
        
        # Test directory creation
        assert mock_settings.temp_dir.exists()
        assert mock_settings.model_cache_dir.exists()
        assert mock_settings.data_dir.exists()
        
        # Test format validation
        assert "pdf" in mock_settings.supported_formats
        assert "txt" in mock_settings.supported_formats
        
        # Test resource limits
        assert mock_settings.max_concurrent_tasks > 0
        assert mock_settings.max_file_size_mb > 0
        assert 0 < mock_settings.max_vram_usage <= 1.0
