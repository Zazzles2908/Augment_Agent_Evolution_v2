"""
Brain 2 Manager - Modular Architecture
Qwen3-Reranker-4B Document Relevance Ranking with Clean Modular Design

This replaces the monolithic brain2_manager.py (433 lines) with a clean,
modular architecture using focused components following Brain 1 pattern.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular design
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional

# Import modular components
from .modules.reranking_engine import RerankingEngine
from .modules.model_manager import ModelManager
from .modules.config_manager import ConfigManager
from .modules.memory_optimizer import MemoryOptimizer
from .modules.performance_monitor import PerformanceMonitor

# Import shared components using absolute imports
import sys
import os
sys.path.append('/workspace/src')
from shared.redis_client import RedisStreamsClient
from shared.error_handling.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class Brain2Manager:
    """
    Brain 2 Manager - Modular Architecture
    Qwen3-Reranker-4B Document Relevance Ranking with focused, testable components
    
    Replaces monolithic 433-line brain2_manager.py with clean modular design.
    """
    
    def __init__(self, settings=None):
        """Initialize Brain 2 Manager with modular components"""
        self.settings = settings or self._get_default_settings()
        self.brain_id = "brain2"
        self.status = "initializing"
        
        # Initialize modular components
        self.config_manager = ConfigManager(self.settings)
        self.model_manager = ModelManager(self.config_manager)
        self.reranking_engine = RerankingEngine(self.model_manager, self.config_manager)
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
        
        logger.info("🧠 Brain 2 Manager (Modular) initialized")
        logger.info("🔄 Qwen3-Reranker-4B with MoE efficiency (3B/30B active)")

    async def initialize(self) -> bool:
        """Initialize Brain 2 Manager - alias for start() method for compatibility"""
        return await self.start()

    async def start(self) -> bool:
        """Start Brain 2 Manager and all components"""
        try:
            logger.info("🚀 Starting Brain 2 Manager (Modular Architecture)...")
            
            # Load configuration
            config = self.config_manager.load_config()
            logger.info("✅ Configuration loaded")
            
            # Optimize memory settings for reranking
            await self.memory_optimizer.optimize_for_reranking()
            logger.info("✅ Memory optimization applied for reranking")
            
            # Load Qwen3-Reranker-4B model
            model_loaded = self.model_manager.load_model(
                use_blackwell=config.get('use_blackwell_quantization', True)
            )
            
            if not model_loaded:
                logger.error("❌ Failed to load Qwen3-Reranker-4B model")
                self.status = "error"
                return False
            
            logger.info("✅ Qwen3-Reranker-4B model loaded successfully")
            
            # Start Redis Streams if available
            if self.redis_client:
                await self.redis_client.start()
                logger.info("✅ Redis Streams client started")
            
            # Start performance monitoring
            await self.performance_monitor.start()
            logger.info("✅ Performance monitoring started")
            
            # Start health monitoring
            await self.health_monitor.start()
            logger.info("✅ Health monitoring started")
            
            self.status = "operational"
            logger.info("🎯 Brain 2 Manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Brain 2 Manager: {e}")
            self.status = "error"
            return False
    
    async def stop(self):
        """Stop Brain 2 Manager and cleanup resources"""
        try:
            logger.info("🛑 Stopping Brain 2 Manager...")
            
            # Stop monitoring
            if self.health_monitor:
                await self.health_monitor.stop()
            
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            # Stop Redis client
            if self.redis_client:
                await self.redis_client.stop()
            
            # Unload model and free memory
            self.model_manager.unload_model()
            await self.memory_optimizer.cleanup()
            
            self.status = "stopped"
            logger.info("✅ Brain 2 Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Brain 2 Manager: {e}")
    
    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]], 
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents using modular reranking engine
        
        Args:
            query: Search query string
            documents: List of document dictionaries with 'text' field
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Use modular reranking engine
            reranked_docs = await self.reranking_engine.rerank_documents(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Track performance
            await self.performance_monitor.record_reranking_request(
                processing_time=processing_time,
                document_count=len(documents),
                batch_size=self.config_manager.get_config('batch_size'),
                success=True
            )
            
            return reranked_docs
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.error(f"❌ Document reranking failed: {e}")
            
            # Track failed request
            await self.performance_monitor.record_reranking_request(
                processing_time=processing_time,
                document_count=len(documents),
                batch_size=self.config_manager.get_config('batch_size'),
                success=False,
                error=str(e)
            )
            
            # Return original order as fallback
            return documents[:top_k]
    
    async def process_embedding_results(self, embedding_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process embedding results from Brain 1 and perform reranking
        
        This is the main integration point with Brain 1 via Redis Streams
        """
        try:
            query = embedding_results.get('query', '')
            candidates = embedding_results.get('candidates', [])
            top_k = embedding_results.get('top_k', 10)
            
            if not candidates:
                logger.warning("⚠️ No candidates received from Brain 1")
                return {
                    "query": query,
                    "reranked_results": [],
                    "success": False,
                    "error": "No candidates to rerank"
                }
            
            # Perform reranking
            reranked_docs = await self.rerank_documents(query, candidates, top_k)
            
            return {
                "query": query,
                "reranked_results": reranked_docs,
                "original_count": len(candidates),
                "reranked_count": len(reranked_docs),
                "success": True,
                "processing_time": time.time() - time.time()  # Will be updated by caller
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing embedding results: {e}")
            return {
                "query": embedding_results.get('query', ''),
                "reranked_results": [],
                "success": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check using modular components"""
        try:
            # Model health check
            model_health = self.model_manager.health_check()
            
            # Memory health check
            memory_health = await self.memory_optimizer.health_check()
            
            # Performance health check
            performance_health = await self.performance_monitor.health_check()
            
            # Overall health assessment
            healthy = (
                model_health.get("healthy", False) and
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
                    "model": model_health,
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
                "model_info": self.model_manager.get_model_info(),
                "reranking_stats": self.reranking_engine.get_reranking_stats(),
                "memory_stats": await self.memory_optimizer.get_memory_stats(),
                "performance_stats": await self.performance_monitor.get_performance_stats(),
                "uptime_seconds": time.time() - self.initialization_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return {"error": str(e)}
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for Brain 2"""
        return {
            "brain_id": "brain2",
            "model_name": "Qwen/Qwen3-Reranker-4B",
            "model_path": "/workspace/models/qwen3/reranker-4b",
            "cache_dir": "/workspace/models/cache",
            "use_blackwell_quantization": True,
            "enable_4bit_quantization": True,
            "enable_8bit_quantization": True,
            "max_vram_usage": 0.20,  # 20% of GPU memory
            "batch_size": 16,
            "max_length": 512,
            "top_k_default": 10,
            "enable_moe_efficiency": True,
            "active_experts": 3,
            "redis_url": "redis://redis:6379/0"
        }
