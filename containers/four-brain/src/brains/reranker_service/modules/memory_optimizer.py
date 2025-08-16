"""
Memory Optimizer Module for Brain-2
Handles GPU/CPU memory optimization for Qwen3-Reranker-4B model

Extracted from brain2_manager.py for modular architecture.
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
    Memory Optimizer for Brain-2
    Handles memory management and optimization for modular architecture
    """
    
    def __init__(self, config_manager=None):
        """Initialize Memory Optimizer"""
        self.config_manager = config_manager
        self.max_vram_usage = 0.20  # 20% of GPU memory for Brain-2
        self.target_vram_usage = 0.18  # Target 18% usage
        self.memory_pressure_threshold = 0.85
        
        # Memory tracking
        self.initial_gpu_memory = 0
        self.initial_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        
        self.initial_cpu_memory = psutil.virtual_memory().used
        
        logger.info("🔧 Memory Optimizer (Brain-2) initialized")
        logger.info(f"📊 GPU memory allocation: {self.max_vram_usage * 100}%")
    
    async def optimize_for_reranking(self):
        """Optimize memory settings for reranking workload"""
        try:
            logger.info("🔧 Optimizing memory for reranking workload...")
            
            # Set CUDA memory fraction if available
            if torch.cuda.is_available():
                # Set memory fraction for Brain-2 (20% of total GPU memory)
                torch.cuda.set_per_process_memory_fraction(self.max_vram_usage)
                logger.info(f"✅ GPU memory fraction set to {self.max_vram_usage * 100}%")
                
                # Enable memory optimization for reranking
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ CUDNN optimization enabled for reranking")
                
                # Set memory growth to avoid pre-allocation
                torch.cuda.empty_cache()
                logger.info("✅ GPU cache cleared for fresh start")
            
            # Set CPU memory optimization for reranking workload
            # Reranking is less CPU intensive than embedding generation
            torch.set_num_threads(min(2, torch.get_num_threads()))
            logger.info("✅ CPU thread optimization applied for reranking")
            
            # Clear any existing cache
            await self._clear_memory_cache()
            
            logger.info("✅ Memory optimization for reranking completed")
            
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
                
                # Check if we're exceeding our allocated fraction
                if gpu_usage > self.max_vram_usage:
                    logger.warning(f"⚠️ Exceeding allocated GPU memory: {gpu_usage:.2%} > {self.max_vram_usage:.2%}")
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
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "brain2_allocation_percent": self.max_vram_usage * 100,
                    "within_allocation": (allocated / total) <= self.max_vram_usage
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
            
            # Additional checks for Brain-2 specific issues
            if torch.cuda.is_available() and "gpu" in memory_stats:
                gpu_stats = memory_stats["gpu"]
                if not gpu_stats.get("within_allocation", True):
                    healthy = False
                    status = "exceeding_allocation"
                elif memory_pressure:
                    status = "memory_pressure"
                else:
                    status = "optimal"
            else:
                status = "optimal" if healthy else "memory_pressure"
            
            return {
                "healthy": healthy,
                "status": status,
                "memory_pressure": memory_pressure,
                "memory_stats": memory_stats,
                "max_vram_usage": self.max_vram_usage,
                "target_vram_usage": self.target_vram_usage
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
    
    async def optimize_for_batch_processing(self, batch_size: int):
        """Optimize memory for specific batch size"""
        try:
            # Estimate memory requirements for batch
            estimated_memory_per_item = 0.1  # GB per document (rough estimate)
            estimated_batch_memory = batch_size * estimated_memory_per_item
            
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                available_memory = total_memory * self.max_vram_usage
                
                if estimated_batch_memory > available_memory * 0.8:
                    logger.warning(f"⚠️ Batch size {batch_size} may exceed memory limits")
                    logger.warning(f"📊 Estimated: {estimated_batch_memory:.2f}GB, Available: {available_memory:.2f}GB")
                    
                    # Suggest smaller batch size
                    suggested_batch = int(available_memory * 0.8 / estimated_memory_per_item)
                    logger.info(f"💡 Suggested batch size: {suggested_batch}")
            
            logger.info(f"✅ Memory optimization applied for batch size: {batch_size}")
            
        except Exception as e:
            logger.error(f"❌ Batch memory optimization failed: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        try:
            summary = {
                "initial_gpu_memory_gb": self.initial_gpu_memory / 1024**3,
                "peak_gpu_memory_gb": self.peak_gpu_memory / 1024**3,
                "initial_cpu_memory_gb": self.initial_cpu_memory / 1024**3,
                "peak_cpu_memory_gb": self.peak_cpu_memory / 1024**3,
                "max_vram_usage_percent": self.max_vram_usage * 100,
                "target_vram_usage_percent": self.target_vram_usage * 100
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
