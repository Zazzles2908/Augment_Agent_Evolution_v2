"""
Brain 4 Manager - Modular Architecture
Docling Document Processing System with Clean Modular Design

This replaces the monolithic brain4_manager.py (400+ lines) with a clean,
modular architecture using focused components following Brain 1/2 pattern.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular design
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import modular components
from .modules.document_engine import DocumentEngine
from .modules.docling_manager import DoclingManager
from .modules.config_manager import ConfigManager
from .modules.memory_optimizer import MemoryOptimizer
from .modules.performance_monitor import PerformanceMonitor

# Import shared components
# Import shared components with absolute package path (module launched via -m)
from shared.redis_client import RedisStreamsClient
from shared.error_handling.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class Brain4Manager:
    """
    Brain 4 Manager - Modular Architecture
    Docling Document Processing System with focused, testable components
    
    Replaces monolithic 400+ line brain4_manager.py with clean modular design.
    """
    
    def __init__(self, settings=None):
        """Initialize Brain 4 Manager with modular components"""
        self.settings = settings or self._get_default_settings()
        self.brain_id = "brain4"
        self.status = "initializing"
        
        # Initialize modular components
        self.config_manager = ConfigManager(self.settings)
        self.docling_manager = DoclingManager(self.config_manager)
        self.document_engine = DocumentEngine(self.docling_manager, self.config_manager)
        self.memory_optimizer = MemoryOptimizer(self.config_manager)
        self.performance_monitor = PerformanceMonitor()
        
        # Redis Streams for inter-brain communication
        self.redis_client = None
        if hasattr(self.settings, 'redis_url'):
            self.redis_client = RedisStreamsClient(
                redis_url=self.settings.redis_url,
                brain_id=self.brain_id
            )
        
        # Health monitoring
        self.health_monitor = HealthMonitor(self.brain_id)
        
        # Performance tracking
        self.initialization_time = time.time()
        self.total_requests = 0
        self.total_processing_time = 0.0
        
        logger.info("🧠 Brain 4 Manager (Modular) initialized")
        logger.info("📄 Docling Document Processing with 40% GPU allocation")
    
    async def start(self) -> bool:
        """Start Brain 4 Manager and all components"""
        try:
            logger.info("🚀 Starting Brain 4 Manager (Modular Architecture)...")
            
            # Load configuration
            config = self.config_manager.load_config()
            logger.info("✅ Configuration loaded")
            
            # Optimize memory settings for document processing
            await self.memory_optimizer.optimize_for_document_processing()
            logger.info("✅ Memory optimization applied for document processing")
            
            # Load Docling DocumentConverter
            converter_loaded = self.docling_manager.load_converter()
            
            if not converter_loaded:
                logger.error("❌ Failed to load Docling DocumentConverter")
                self.status = "error"
                return False
            
            logger.info("✅ Docling DocumentConverter loaded successfully")
            
            # Start Redis Streams if available
            if self.redis_client:
                # Align with RedisStreamsClient interface
                await self.redis_client.connect()
                # Register handler to receive embedding results
                from shared.streams import StreamNames
                self.redis_client.register_handler(StreamNames.EMBEDDING_RESULTS, self._handle_embedding_result)
                await self.redis_client.start_consuming()
                logger.info("✅ Redis Streams client started")
            
            # Start performance monitoring
            await self.performance_monitor.start()
            logger.info("✅ Performance monitoring started")
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            logger.info("✅ Health monitoring started")
            
            self.status = "operational"
            logger.info("🎯 Brain 4 Manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Brain 4 Manager: {e}")
            self.status = "error"
            return False
    
    async def stop(self):
        """Stop Brain 4 Manager and cleanup resources"""
        try:
            logger.info("🛑 Stopping Brain 4 Manager...")
            
            # Stop monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            # Stop Redis client
            if self.redis_client:
                await self.redis_client.disconnect()
            
            # Unload converter and free memory
            self.docling_manager.unload_converter()
            await self.memory_optimizer.cleanup()
            
            self.status = "stopped"
            logger.info("✅ Brain 4 Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Brain 4 Manager: {e}")
    
    async def _enqueue_embedding_batch(self, doc_id: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Send a batch embedding request with chunk refs (includes small text excerpts)."""
        try:
            if not self.redis_client:
                logger.warning("Redis client not initialized; skipping embedding enqueue")
                return None
            from shared.streams import StreamNames, EmbeddingBatchRequest
            import uuid as _uuid
            chunk_batch_id = str(_uuid.uuid4())
            # Build compact refs; include short text excerpts (<= 500 chars) for Phase 1
            refs = []
            for idx, ch in enumerate(chunks):
                txt = (ch.get('text') or '')
                refs.append({
                    'chunk_id': ch.get('chunk_id', idx),
                    'page_no': ch.get('page_no', None),
                    'token_count': len(txt.split()),
                    'text_excerpt': txt[:500]
                })
            msg = EmbeddingBatchRequest(doc_id=doc_id, chunk_batch_id=chunk_batch_id, chunk_refs=refs, target_dim=2000)
            await self.redis_client.send_message(StreamNames.EMBEDDING_REQUESTS, msg)
            logger.info(f"📤 Enqueued embedding batch: {chunk_batch_id} ({len(refs)} chunks)")
            return chunk_batch_id
        except Exception as e:
            logger.error(f"Failed to enqueue embedding batch: {e}")
            return None

    async def _handle_embedding_result(self, message):
        """Handle embedding results: update document status and log stats."""
        try:
            data = message.data
            doc_id = data.get('doc_id')
            stats = data.get('stats', {})
            import asyncpg
            conn = await asyncpg.connect(self.settings.database_url, timeout=10.0)
            try:
                await conn.execute(
                    """
                    UPDATE augment_agent.documents
                    SET processing_status = 'embedded', processing_timestamp = NOW()
                    WHERE id = $1
                    """,
                    doc_id
                )
            finally:
                await conn.close()
            logger.info(f"📥 Embedding results received for doc {doc_id}: {stats}")
        except Exception as e:
            logger.error(f"Failed to handle embedding result: {e}")

    async def process_document(self, file_path: Union[str, Path],
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process document using modular document engine
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            Dictionary with all extracted information
        """
        start_time = time.time()
        self.total_requests += 1
        file_path = Path(file_path)
        
        try:
            # Allocate memory for document processing
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            memory_allocated = await self.memory_optimizer.allocate_document_memory(
                document_id=str(file_path),
                estimated_size_mb=file_size_mb
            )
            
            if not memory_allocated:
                logger.warning(f"⚠️ Memory allocation failed for {file_path.name}")
            
            # Use modular document engine
            result = await self.document_engine.process_document(
                file_path=file_path,
                metadata=metadata
            )

            # Enqueue embedding batch if chunks exist (vector-first bus)
            try:
                doc_id = result.get('metadata', {}).get('document_id') or result.get('document_id')
                if not doc_id:
                    doc_id = str(file_path)
                chunks = result.get('chunks', [])
                if chunks:
                    await self._enqueue_embedding_batch(doc_id, chunks)
            except Exception as ee:
                logger.warning(f"Embedding enqueue failed: {ee}")

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            # Track performance
            await self.performance_monitor.record_document_processing(
                processing_time=processing_time,
                file_size_mb=file_size_mb,
                chunk_count=len(result.get('chunks', [])),
                success=result.get('success', False)
            )

            # Deallocate memory
            await self.memory_optimizer.deallocate_document_memory(str(file_path))

            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.error(f"❌ Document processing failed: {e}")
            
            # Track failed request
            await self.performance_monitor.record_document_processing(
                processing_time=processing_time,
                file_size_mb=file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
                chunk_count=0,
                success=False,
                error=str(e)
            )
            
            # Deallocate memory
            await self.memory_optimizer.deallocate_document_memory(str(file_path))
            
            return {
                "document_id": None,
                "file_path": str(file_path),
                "metadata": metadata or {},
                "content": {},
                "chunks": [],
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
    
    async def process_document_batch(self, file_paths: List[Union[str, Path]], 
                                   metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        results = []
        
        if metadata_list is None:
            metadata_list = [None] * len(file_paths)
        
        for file_path, metadata in zip(file_paths, metadata_list):
            result = await self.process_document(file_path, metadata)
            results.append(result)
        
        return results
    
    async def process_document_from_redis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document task from Redis Stream
        
        This is the main integration point with other brains via Redis Streams
        """
        try:
            file_path = task_data.get('file_path')
            metadata = task_data.get('metadata', {})
            
            if not file_path:
                logger.error("❌ No file path provided in task data")
                return {
                    "success": False,
                    "error": "No file path provided"
                }
            
            # Process document
            result = await self.process_document(file_path, metadata)
            
            # Add task metadata
            result['task_id'] = task_data.get('task_id')
            result['requested_by'] = task_data.get('requested_by')
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing document from Redis: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_data.get('task_id')
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check using modular components"""
        try:
            # Docling health check
            docling_health = self.docling_manager.health_check()
            
            # Memory health check
            memory_health = await self.memory_optimizer.health_check()
            
            # Performance health check
            performance_health = await self.performance_monitor.health_check()
            
            # Overall health assessment
            healthy = (
                docling_health.get("healthy", False) and
                memory_health.get("healthy", False) and
                performance_health.get("healthy", False)
            )
            
            return {
                "healthy": healthy,
                "status": self.status,
                "brain_id": self.brain_id,
                "uptime_seconds": time.time() - self.initialization_time,
                "total_requests": self.total_requests,
                "average_processing_time": (
                    self.total_processing_time / self.total_requests
                    if self.total_requests > 0 else 0
                ),
                "components": {
                    "docling": docling_health,
                    "memory": memory_health,
                    "performance": performance_health
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        try:
            return {
                "brain_id": self.brain_id,
                "status": self.status,
                "docling_info": self.docling_manager.get_converter_info(),
                "document_stats": self.document_engine.get_processing_stats(),
                "memory_stats": await self.memory_optimizer.get_memory_stats(),
                "performance_stats": await self.performance_monitor.get_performance_stats(),
                "uptime_seconds": time.time() - self.initialization_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return {"error": str(e)}
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for Brain 4"""
        return {
            "brain_id": "brain4",
            "service_name": "Docling Document Processor",
            "model_cache_dir": "/workspace/models/cache",
            "temp_dir": "/workspace/temp",
            "max_vram_usage": 0.40,  # 40% of GPU memory (largest allocation)
            "max_concurrent_tasks": 4,
            "batch_size_documents": 2,
            "max_file_size_mb": 100,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "enable_ocr": True,
            "enable_table_extraction": True,
            "enable_image_extraction": True,
            "enable_semantic_chunking": True,
            "supported_formats": ["pdf", "docx", "pptx", "html", "md", "txt"],
            "redis_url": "redis://redis:6379/0"
        }
