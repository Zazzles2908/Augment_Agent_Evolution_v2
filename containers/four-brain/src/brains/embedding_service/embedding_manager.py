"""
Brain 1 Manager - Modular Architecture
Qwen3-4B Embedding System with Clean Modular Design

This replaces the monolithic brain1_manager.py (807 lines) with a clean,
modular architecture using focused components.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular design
"""

import logging
import time
import os
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

# Import modular components
from .modules.embedding_engine import EmbeddingEngine
from .modules.model_manager import ModelManager
from .modules.config_manager import ConfigManager
from .modules.memory_optimizer import MemoryOptimizer
from .modules.performance_monitor import PerformanceMonitor
from .modules.triton_client import TritonEmbeddingClient
from .modules.inference_router import InferenceRouter
from .modules.streams_handler import StreamsHandler
from .modules.persistence import PersistenceClient

# Import shared components (absolute imports using PYTHONPATH=/workspace/src)
from shared.redis_client import RedisStreamsClient
from shared.error_handling.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class Brain1Manager:
    """
    Brain 1 Manager - Modular Architecture
    Qwen3-4B Embedding System with focused, testable components

    Replaces monolithic 807-line brain1_manager.py with clean modular design.
    """

    def __init__(self, settings=None):
        """Initialize Brain 1 Manager with modular components"""
        self.settings = settings or self._get_default_settings()
        self.brain_id = "brain1"
        self.status = "initializing"

        # Initialize modular components
        self.config_manager = ConfigManager(self.settings)
        self.model_manager = ModelManager(self.config_manager)
        self.embedding_engine = EmbeddingEngine(self.model_manager, self.config_manager)
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

        logger.info("🧠 Brain 1 Manager (Modular) initialized")

    async def start(self) -> bool:
        """Start Brain 1 Manager and all components"""
        try:
            logger.info("🚀 Starting Brain 1 Manager (Modular Architecture)...")

            # Load configuration
            config = self.config_manager.load_config()
            logger.info("✅ Configuration loaded")

            # Optimize memory settings
            await self.memory_optimizer.optimize_for_embedding()
            logger.info("✅ Memory optimization applied")

            # Triton-first startup: if enabled, skip local model loading
            self.triton_enabled = bool(config.get("use_triton", True))
            self.triton_client = None
            if self.triton_enabled:
                logger.info("🚀 USE_TRITON=true: initializing Triton client and skipping local model load")
                triton_url = os.getenv("TRITON_URL", config.get("triton_url", "http://triton:8000"))
                model_name = os.getenv("TRITON_MODEL_NAME", config.get("triton_model_name", "qwen3_embedding_trt"))
                self.triton_client = TritonEmbeddingClient(
                    url=triton_url,
                    model_name=model_name,
                    timeout_s=int(config.get("triton_timeout_s", 30))
                )
                if not self.triton_client.is_ready():
                    logger.warning("⚠️ Triton model not ready yet; health endpoint may be degraded until ready")
            else:
                # Load Qwen3-4B model locally
                model_loaded = self.model_manager.load_model(
                    use_blackwall=config.get('use_blackwall_quantization', True)
                )
                if not model_loaded:
                    logger.error("❌ Failed to load Qwen3-4B model")
                    self.status = "error"
                    return False
                logger.info("✅ Qwen3-4B model loaded successfully")

            # Initialize inference router
            try:
                from transformers import AutoTokenizer
                cfg = self.config_manager.get_config()
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.get("model_path"), trust_remote_code=True, local_files_only=True
                )
            except Exception:
                tokenizer = None
            self.inference_router = InferenceRouter(
                triton_enabled=self.triton_enabled,
                triton_client=self.triton_client,
                embedding_engine=self.embedding_engine,
                tokenizer=tokenizer,
                config=self.config_manager.get_config(),
            )

            # Start Redis Streams if available
            if self.redis_client:
                # Connect, register handlers below, then begin consuming
                connected = await self.redis_client.connect()
                if not connected:
                    logger.warning("⚠️ Redis client failed to connect; continuing without streams")
                else:
                    logger.info("✅ Redis Streams client connected")

            # Register Redis stream handlers via StreamsHandler and begin consuming
            if self.redis_client and self.redis_client.is_connected:
                persistence = PersistenceClient(self.config_manager.settings.database_url)
                self.streams_handler = StreamsHandler(
                    redis_client=self.redis_client,
                    persistence_client=persistence,
                    embed_batch_func=self.generate_batch_embeddings,
                    target_dimensions=int(self.config_manager.get_config().get("target_dimensions", 2000))
                )
                await self.streams_handler.register_embedding_batch_handler()
                await self.streams_handler.begin_consuming()
                logger.info("✅ Redis Streams consuming started via StreamsHandler")
            else:
                logger.warning("⚠️ Redis client not connected; skipping StreamsHandler setup")

            # Start performance monitoring
            await self.performance_monitor.start()
            logger.info("✅ Performance monitoring started")

            # Start health monitoring
            await self.health_monitor.start_monitoring()
            logger.info("✅ Health monitoring started")

            self.status = "operational"
            logger.info("🎯 Brain 1 Manager started successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start Brain 1 Manager: {e}")
            self.status = "error"
            return False

    async def stop(self):
        """Stop Brain 1 Manager and cleanup resources"""
        try:
            logger.info("🛑 Stopping Brain 1 Manager...")

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
            logger.info("✅ Brain 1 Manager stopped successfully")

        except Exception as e:
            logger.error(f"❌ Error stopping Brain 1 Manager: {e}")

    async def generate_embedding(self, text: str, truncate_to_2000: bool = True) -> Optional[np.ndarray]:
        """
        Generate embedding using modular embedding engine

        Args:
            text: Input text to embed
            truncate_to_2000: Apply MRL truncation for Supabase compatibility

        Returns:
            Embedding vector (2000-dim if truncated, 2560-dim if not)
        """
        start_time = time.time()
        self.total_requests += 1

        try:
            # Delegate to inference router
            embedding = self.inference_router.embed_one(text, truncate_to_2000=truncate_to_2000)

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            # Track performance
            await self.performance_monitor.record_embedding_request(
                processing_time=processing_time,
                success=embedding is not None
            )

            return embedding

        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            logger.error(f"❌ Embedding generation failed: {e}")

            # Track failed request
            await self.performance_monitor.record_embedding_request(
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

            return None

    async def generate_batch_embeddings(self, texts: List[str],
                                      truncate_to_2000: bool = True) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts"""
        # Delegate to inference router for batch
        return self.inference_router.embed_batch(texts, truncate_to_2000=truncate_to_2000)

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check using modular components"""
        try:
            # Model/Triton health
            if getattr(self, "triton_enabled", False) and self.triton_client is not None:
                model_health = {
                    "healthy": self.triton_client.is_ready(),
                    "status": "triton_ready" if self.triton_client.is_ready() else "triton_not_ready"
                }
            else:
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
                "embedding_stats": self.embedding_engine.get_embedding_stats(),
                "memory_stats": await self.memory_optimizer.get_memory_stats(),
                "performance_stats": await self.performance_monitor.get_performance_stats(),
                "uptime_seconds": time.time() - self.initialization_time
            }

        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return {"error": str(e)}

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for Brain 1"""
        return {
            "brain_id": "brain1",
            "model_path": "/workspace/models/qwen3/embedding-4b",
            "cache_dir": "/workspace/models/cache",
            "use_blackwall_quantization": True,
            "mrl_truncation_enabled": True,
            "target_dimensions": 2000,
            "native_dimensions": 2560,
            "thinking_mode_enabled": True,
            "redis_url": "redis://redis:6379/0"
        }
