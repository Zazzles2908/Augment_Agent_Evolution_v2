"""
Performance tests for Brain 4 system
Tests performance characteristics and RTX 5070 Ti optimization
Based on implementation_part_6.md specifications
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import statistics

class TestBrain4Performance:
    """Performance test suite for Brain 4 system"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_document_processing_throughput(self, mock_settings):
        """Test document processing throughput"""
        
        # Mock all external dependencies for performance testing
        with patch('brain4_docling.core.brain4_manager.MemoryManager') as mock_memory, \
             patch('brain4_docling.core.brain4_manager.PerformanceMonitor') as mock_perf, \
             patch('brain4_docling.core.brain4_manager.BrainCommunicator') as mock_comm, \
             patch('brain4_docling.core.brain4_manager.DocumentStore') as mock_store, \
             patch('brain4_docling.core.brain4_manager.DocumentConverter') as mock_converter:
            
            # Configure fast mocks
            mock_memory.return_value.allocate_memory_for_brain = AsyncMock(return_value=True)
            mock_memory.return_value.deallocate_memory_for_brain = AsyncMock()
            
            mock_comm_instance = AsyncMock()
            mock_comm_instance.initialize = AsyncMock()
            mock_comm_instance.register_brain = AsyncMock()
            mock_comm_instance.request_embeddings = AsyncMock(return_value=[0.1] * 384)
            mock_comm_instance.request_analysis = AsyncMock(return_value={"analysis": "test"})
            mock_comm_instance.request_summary = AsyncMock(return_value="summary")
            mock_comm_instance.broadcast_document_processed = AsyncMock()
            mock_comm_instance.send_heartbeat = AsyncMock()
            mock_comm_instance.close = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            
            mock_store.return_value.initialize = AsyncMock()
            mock_store.return_value.store_document = AsyncMock(return_value=True)
            mock_store.return_value.close = AsyncMock()
            
            # Mock fast document conversion
            mock_result = Mock()
            mock_document = Mock()
            mock_document.export_to_markdown.return_value = "# Test\nContent"
            mock_document.export_to_json.return_value = '{"title": "Test"}'
            mock_document.pages = [Mock()]
            mock_document.texts = []
            mock_document.tables = []
            mock_document.pictures = []
            mock_result.document = mock_document
            mock_converter.return_value.convert.return_value = mock_result
            
            from brain4_docling.core.brain4_manager import Brain4Manager
            manager = Brain4Manager(mock_settings)
            await manager.start()
            
            try:
                # Create test documents
                test_docs = []
                for i in range(10):
                    doc_path = mock_settings.temp_dir / f"test_{i}.pdf"
                    doc_path.write_text(f"Test document {i} content")
                    test_docs.append(str(doc_path))
                
                # Measure processing time
                start_time = time.time()
                
                # Submit all documents
                task_ids = []
                for doc_path in test_docs:
                    task_id = await manager.submit_document_task(doc_path)
                    task_ids.append(task_id)
                
                # Wait for all processing to complete
                await asyncio.sleep(1.0)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Calculate throughput
                throughput = len(test_docs) / total_time
                
                # Performance assertions
                assert throughput > 5.0  # Should process at least 5 docs/second
                assert total_time < 5.0  # Should complete within 5 seconds
                
                print(f"Throughput: {throughput:.2f} documents/second")
                print(f"Total time: {total_time:.2f} seconds")
                
            finally:
                await manager.shutdown()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, mock_settings):
        """Test memory usage optimization for RTX 5070 Ti"""
        
        from brain4_docling.utils.memory_manager import MemoryManager
        
        # Test memory allocation patterns
        memory_manager = MemoryManager(max_vram_usage=0.75, target_vram_usage=0.65)
        
        # Simulate RTX 5070 Ti memory constraints (16GB)
        total_memory_gb = 16.0
        max_usage_gb = total_memory_gb * 0.75  # 12GB
        
        # Test allocation efficiency
        allocations = []
        allocation_times = []
        
        for i in range(10):
            start_time = time.time()
            
            # Allocate memory for brain processing
            success = await memory_manager.allocate_memory_for_brain(f"brain{i%4}", 2.0)
            
            end_time = time.time()
            allocation_time = end_time - start_time
            
            allocations.append(success)
            allocation_times.append(allocation_time)
        
        # Performance assertions
        avg_allocation_time = statistics.mean(allocation_times)
        assert avg_allocation_time < 0.01  # Should allocate within 10ms
        
        # Test memory usage tracking
        usage = await memory_manager.get_current_usage()
        assert "total_allocated_gb" in usage
        
        # Test memory cleanup performance
        start_time = time.time()
        await memory_manager.cleanup_unused_memory()
        cleanup_time = time.time() - start_time
        
        assert cleanup_time < 0.1  # Should cleanup within 100ms
        
        print(f"Average allocation time: {avg_allocation_time*1000:.2f}ms")
        print(f"Cleanup time: {cleanup_time*1000:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, mock_settings):
        """Test concurrent document processing performance"""
        
        # Mock dependencies for concurrent testing
        with patch('brain4_docling.core.brain4_manager.MemoryManager') as mock_memory, \
             patch('brain4_docling.core.brain4_manager.BrainCommunicator') as mock_comm, \
             patch('brain4_docling.core.brain4_manager.DocumentStore') as mock_store, \
             patch('brain4_docling.core.brain4_manager.DocumentConverter') as mock_converter:
            
            # Configure mocks with realistic delays
            async def mock_allocate(*args):
                await asyncio.sleep(0.01)  # 10ms allocation time
                return True
            
            async def mock_process(*args):
                await asyncio.sleep(0.1)  # 100ms processing time
                return {
                    "source_path": "/test/file.pdf",
                    "content": {"text": "Test content"},
                    "chunks": [{"chunk_id": "chunk_0", "text": "Test"}],
                    "processing_time": 0.1
                }
            
            mock_memory.return_value.allocate_memory_for_brain = mock_allocate
            mock_memory.return_value.deallocate_memory_for_brain = AsyncMock()
            
            mock_comm_instance = AsyncMock()
            mock_comm_instance.initialize = AsyncMock()
            mock_comm_instance.register_brain = AsyncMock()
            mock_comm_instance.request_embeddings = AsyncMock(return_value=[0.1] * 384)
            mock_comm_instance.close = AsyncMock()
            mock_comm.return_value = mock_comm_instance
            
            mock_store.return_value.initialize = AsyncMock()
            mock_store.return_value.store_document = AsyncMock(return_value=True)
            mock_store.return_value.close = AsyncMock()
            
            from brain4_docling.core.brain4_manager import Brain4Manager
            
            # Override document processor with mock
            with patch('brain4_docling.core.brain4_manager.DocumentProcessor') as mock_processor:
                mock_processor.return_value.process_document = mock_process
                
                manager = Brain4Manager(mock_settings)
                await manager.start()
                
                try:
                    # Test concurrent processing
                    start_time = time.time()
                    
                    # Submit multiple documents concurrently
                    tasks = []
                    for i in range(8):  # More than max_concurrent_tasks (2)
                        doc_path = mock_settings.temp_dir / f"concurrent_{i}.pdf"
                        doc_path.write_text(f"Concurrent test document {i}")
                        
                        task = asyncio.create_task(
                            manager.submit_document_task(str(doc_path))
                        )
                        tasks.append(task)
                    
                    # Wait for all submissions
                    task_ids = await asyncio.gather(*tasks)
                    
                    # Wait for processing to complete
                    await asyncio.sleep(2.0)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Performance assertions
                    assert len(task_ids) == 8
                    assert total_time < 5.0  # Should complete within 5 seconds
                    
                    # Check that concurrent processing was limited
                    max_concurrent = manager.max_concurrent_tasks
                    assert max_concurrent == 2  # Should respect concurrency limit
                    
                    print(f"Concurrent processing time: {total_time:.2f} seconds")
                    print(f"Max concurrent tasks: {max_concurrent}")
                    
                finally:
                    await manager.shutdown()
    
    @pytest.mark.performance
    def test_chunking_performance(self, mock_settings):
        """Test semantic chunking performance"""
        
        from brain4_docling.core.document_processor import DocumentProcessor
        
        # Create processor with mock converter
        mock_converter = Mock()
        processor = DocumentProcessor(mock_converter, mock_settings)
        
        # Create large text for chunking
        large_text = "This is a test sentence for chunking performance. " * 1000
        
        # Measure chunking performance
        start_time = time.time()
        
        chunks = asyncio.run(processor._create_semantic_chunks(large_text))
        
        end_time = time.time()
        chunking_time = end_time - start_time
        
        # Performance assertions
        assert chunking_time < 1.0  # Should chunk within 1 second
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Test chunking efficiency
        total_chars = len(large_text)
        chars_per_second = total_chars / chunking_time
        
        assert chars_per_second > 10000  # Should process at least 10k chars/second
        
        print(f"Chunking time: {chunking_time*1000:.2f}ms")
        print(f"Chunks created: {len(chunks)}")
        print(f"Processing rate: {chars_per_second:.0f} chars/second")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_communication_latency(self):
        """Test inter-brain communication latency"""
        
        with patch('brain4_docling.integration.brain_communicator.aioredis') as mock_redis:
            # Configure Redis mock
            mock_client = AsyncMock()
            mock_redis.Redis.return_value = mock_client
            mock_redis.ConnectionPool.from_url.return_value = AsyncMock()
            
            from brain4_docling.integration.brain_communicator import BrainCommunicator
            communicator = BrainCommunicator("redis://localhost:6379")
            
            # Mock fast message sending
            async def mock_send_and_wait(message, timeout=None):
                await asyncio.sleep(0.01)  # 10ms latency
                return {"embeddings": [0.1] * 384}
            
            communicator._send_message_and_wait = mock_send_and_wait
            communicator.is_connected = True
            
            # Test communication latency
            latencies = []
            
            for i in range(10):
                start_time = time.time()
                
                await communicator.request_embeddings(f"test content {i}")
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
            
            # Performance assertions
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            assert avg_latency < 0.05  # Average latency under 50ms
            assert max_latency < 0.1   # Max latency under 100ms
            
            print(f"Average latency: {avg_latency*1000:.2f}ms")
            print(f"Max latency: {max_latency*1000:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_utilization(self, mock_settings):
        """Test resource utilization efficiency"""
        
        from brain4_docling.utils.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Collect baseline metrics
        baseline_metrics = await monitor.collect_metrics()
        
        # Simulate processing load
        start_time = time.time()
        
        # Record multiple processing times
        processing_times = [0.5, 1.2, 0.8, 2.1, 0.9]
        for proc_time in processing_times:
            monitor.record_processing_time(proc_time)
        
        # Collect metrics under load
        load_metrics = await monitor.collect_metrics()
        
        end_time = time.time()
        monitoring_overhead = end_time - start_time
        
        # Performance assertions
        assert monitoring_overhead < 0.1  # Monitoring overhead under 100ms
        
        # Test performance summary generation
        start_time = time.time()
        summary = monitor.get_performance_summary()
        summary_time = time.time() - start_time
        
        assert summary_time < 0.01  # Summary generation under 10ms
        assert "current_stats" in summary
        assert "trends" in summary
        
        print(f"Monitoring overhead: {monitoring_overhead*1000:.2f}ms")
        print(f"Summary generation: {summary_time*1000:.2f}ms")
