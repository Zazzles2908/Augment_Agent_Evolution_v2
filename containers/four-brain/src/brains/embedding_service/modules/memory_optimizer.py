"""
Memory Optimizer Module for Brain-1
Handles GPU/CPU memory optimization for Qwen3-4B model

Extracted from brain1_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import asyncio
import torch
import psutil
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory Optimizer for Brain-1
    Handles memory management and optimization for modular architecture
    """
    
    def __init__(self, config_manager=None):
        """Initialize Memory Optimizer"""
        self.config_manager = config_manager

        # Initialize Four-Brain GPU Allocator
        try:
            from ...shared.gpu_allocator import gpu_allocator, BrainType, configure_brain_gpu

            # Configure GPU for Brain-1 (Embedding)
            success = configure_brain_gpu(BrainType.BRAIN1_EMBEDDING)
            if success:
                logger.info("✅ Four-Brain GPU allocation configured for Brain-1")
            else:
                logger.warning("⚠️ Four-Brain GPU allocation failed, using fallback")
        except ImportError:
            logger.warning("⚠️ Four-Brain GPU allocator not available, using manual configuration")

        # Read memory allocation from environment variables (Four-Brain architecture)
        import os

        # Use consistent environment variable mapping with settings.py
        self.max_vram_usage = float(os.getenv("MAX_VRAM_USAGE",
                                            os.getenv("TORCH_CUDA_MEMORY_FRACTION", "0.35")))  # 35% for Brain-1
        self.cuda_memory_fraction = float(os.getenv("CUDA_MEMORY_FRACTION", "0.35"))
        self.target_vram_usage = float(os.getenv("TARGET_VRAM_USAGE", "0.30"))

        # Log the actual values being used for debugging
        logger.info(f"🔧 Memory Optimizer Configuration:")
        logger.info(f"  MAX_VRAM_USAGE: {self.max_vram_usage} ({self.max_vram_usage * 100}%)")
        logger.info(f"  CUDA_MEMORY_FRACTION: {self.cuda_memory_fraction} ({self.cuda_memory_fraction * 100}%)")
        logger.info(f"  TARGET_VRAM_USAGE: {self.target_vram_usage} ({self.target_vram_usage * 100}%)")

        self.memory_pressure_threshold = 0.85
        
        # Memory tracking
        self.initial_gpu_memory = 0
        self.initial_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        
        self.initial_cpu_memory = psutil.virtual_memory().used
        
        logger.info("🔧 Memory Optimizer initialized")
    
    async def optimize_for_embedding(self):
        """Optimize memory settings for embedding workload"""
        try:
            logger.info("🔧 Optimizing memory for embedding workload...")
            
            # Set CUDA memory fraction if available
            if torch.cuda.is_available():
                # Set memory fraction for Brain-1 (35% of total GPU memory as per Four-Brain architecture)
                torch.cuda.set_per_process_memory_fraction(self.max_vram_usage)
                logger.info(f"✅ GPU memory fraction set to {self.max_vram_usage * 100}% (Brain-1 allocation)")
                logger.info(f"📊 Target VRAM usage: {self.target_vram_usage * 100}%")
                logger.info(f"🧠 Four-Brain Memory Architecture: Brain-1 gets {self.max_vram_usage * 100}% of 16GB = {self.max_vram_usage * 16:.1f}GB")
                
                # Enable memory optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ CUDNN optimization enabled")
            
            # Set CPU memory optimization
            torch.set_num_threads(min(4, torch.get_num_threads()))
            logger.info("✅ CPU thread optimization applied")
            
            # Clear any existing cache
            await self._clear_memory_cache()
            
            logger.info("✅ Memory optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
    
    async def _clear_memory_cache(self):
        """Clear memory caches"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 GPU cache cleared")
            
            # Force garbage collection
            import gc
            gc.collect()
            logger.debug("🧹 Garbage collection completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Cache clearing failed: {e}")
    
    async def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            # Check GPU memory pressure
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                gpu_usage = allocated / total
                if gpu_usage > self.memory_pressure_threshold:
                    logger.warning(f"⚠️ High GPU memory usage: {gpu_usage:.2%}")
                    return True
            
            # Check CPU memory pressure
            cpu_memory = psutil.virtual_memory()
            if cpu_memory.percent > self.memory_pressure_threshold * 100:
                logger.warning(f"⚠️ High CPU memory usage: {cpu_memory.percent:.1f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Memory pressure check failed: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            stats = {
                "cpu": {
                    "total_gb": psutil.virtual_memory().total / 1024**3,
                    "used_gb": psutil.virtual_memory().used / 1024**3,
                    "available_gb": psutil.virtual_memory().available / 1024**3,
                    "percent_used": psutil.virtual_memory().percent
                }
            }
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                stats["gpu"] = {
                    "total_gb": total / 1024**3,
                    "allocated_gb": allocated / 1024**3,
                    "reserved_gb": reserved / 1024**3,
                    "free_gb": (total - reserved) / 1024**3,
                    "allocated_percent": (allocated / total) * 100,
                    "reserved_percent": (reserved / total) * 100,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
                }
                
                # Update peak memory tracking
                self.peak_gpu_memory = max(self.peak_gpu_memory, allocated)
            
            # Update peak CPU memory tracking
            current_cpu = psutil.virtual_memory().used
            self.peak_cpu_memory = max(self.peak_cpu_memory, current_cpu)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting memory stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform memory health check"""
        try:
            memory_pressure = await self.check_memory_pressure()
            memory_stats = await self.get_memory_stats()
            
            # Determine health status
            healthy = not memory_pressure
            
            if memory_pressure:
                status = "memory_pressure"
            else:
                status = "optimal"
            
            return {
                "healthy": healthy,
                "status": status,
                "memory_pressure": memory_pressure,
                "memory_stats": memory_stats
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup memory resources"""
        try:
            logger.info("🧹 Cleaning up memory resources...")
            
            # Clear caches
            await self._clear_memory_cache()
            
            # Reset CUDA memory fraction
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1.0)
                logger.info("✅ GPU memory fraction reset")
            
            logger.info("✅ Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
    
    async def optimize_for_inference(self):
        """Optimize memory specifically for inference workload"""
        try:
            # Enable inference mode optimizations
            torch.inference_mode(True)
            
            # Disable gradient computation globally
            torch.set_grad_enabled(False)
            
            logger.info("✅ Inference mode optimizations enabled")
            
        except Exception as e:
            logger.error(f"❌ Inference optimization failed: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        try:
            summary = {
                "initial_gpu_memory_gb": self.initial_gpu_memory / 1024**3,
                "peak_gpu_memory_gb": self.peak_gpu_memory / 1024**3,
                "initial_cpu_memory_gb": self.initial_cpu_memory / 1024**3,
                "peak_cpu_memory_gb": self.peak_cpu_memory / 1024**3,
                "max_vram_usage_percent": self.max_vram_usage * 100
            }
            
            if torch.cuda.is_available():
                current_gpu = torch.cuda.memory_allocated()
                summary["current_gpu_memory_gb"] = current_gpu / 1024**3
                summary["gpu_memory_growth_gb"] = (current_gpu - self.initial_gpu_memory) / 1024**3
            
            current_cpu = psutil.virtual_memory().used
            summary["current_cpu_memory_gb"] = current_cpu / 1024**3
            summary["cpu_memory_growth_gb"] = (current_cpu - self.initial_cpu_memory) / 1024**3
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting memory summary: {e}")
            return {"error": str(e)}
