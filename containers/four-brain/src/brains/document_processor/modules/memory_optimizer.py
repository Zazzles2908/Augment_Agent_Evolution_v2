"""
Memory Optimizer Module for Brain-4
Handles GPU/CPU memory optimization for document processing workloads

Extracted from brain4_manager.py for modular architecture.
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
    Memory Optimizer for Brain-4
    Handles memory management and optimization for document processing workloads
    """
    
    def __init__(self, config_manager=None):
        """Initialize Memory Optimizer"""
        self.config_manager = config_manager
        self.max_vram_usage = 0.40  # 40% of GPU memory for Brain-4 (largest allocation)
        self.target_vram_usage = 0.35  # Target 35% usage
        self.memory_pressure_threshold = 0.85
        
        # Memory tracking
        self.initial_gpu_memory = 0
        self.initial_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        
        # Document processing specific tracking
        self.document_memory_usage = {}
        self.active_document_count = 0
        
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        
        self.initial_cpu_memory = psutil.virtual_memory().used
        
        logger.info("🔧 Memory Optimizer (Brain-4) initialized")
        logger.info(f"📊 GPU memory allocation: {self.max_vram_usage * 100}% (largest)")
    
    async def optimize_for_document_processing(self):
        """Optimize memory settings for document processing workload"""
        try:
            logger.info("🔧 Optimizing memory for document processing workload...")
            
            # Set CUDA memory fraction if available
            if torch.cuda.is_available():
                # Set memory fraction for Brain-4 (40% of total GPU memory - largest allocation)
                torch.cuda.set_per_process_memory_fraction(self.max_vram_usage)
                logger.info(f"✅ GPU memory fraction set to {self.max_vram_usage * 100}%")
                
                # Enable memory optimization for document processing
                # Document processing is less GPU intensive than AI models
                torch.backends.cudnn.benchmark = False  # Disable for document processing
                torch.backends.cudnn.deterministic = True  # Enable for consistent results
                logger.info("✅ CUDNN optimization configured for document processing")
                
                # Set memory growth to avoid pre-allocation
                torch.cuda.empty_cache()
                logger.info("✅ GPU cache cleared for fresh start")
            
            # Set CPU memory optimization for document processing workload
            # Document processing is CPU intensive (parsing, OCR, etc.)
            torch.set_num_threads(min(6, torch.get_num_threads()))  # More threads for document processing
            logger.info("✅ CPU thread optimization applied for document processing")
            
            # Clear any existing cache
            await self._clear_memory_cache()
            
            logger.info("✅ Memory optimization for document processing completed")
            
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
    
    async def allocate_document_memory(self, document_id: str, estimated_size_mb: float) -> bool:
        """Allocate memory for document processing"""
        try:
            # Check if we have enough memory available
            memory_pressure = await self.check_memory_pressure()
            if memory_pressure:
                logger.warning(f"⚠️ Memory pressure detected, cannot allocate for document {document_id}")
                return False
            
            # Track document memory usage
            self.document_memory_usage[document_id] = {
                "allocated_mb": estimated_size_mb,
                "allocated_at": asyncio.get_event_loop().time()
            }
            
            self.active_document_count += 1
            
            logger.debug(f"📊 Memory allocated for document {document_id}: {estimated_size_mb:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document memory allocation failed: {e}")
            return False
    
    async def deallocate_document_memory(self, document_id: str):
        """Deallocate memory for completed document"""
        try:
            if document_id in self.document_memory_usage:
                allocated_mb = self.document_memory_usage[document_id]["allocated_mb"]
                del self.document_memory_usage[document_id]
                self.active_document_count = max(0, self.active_document_count - 1)
                
                # Clear caches after document processing
                await self._clear_memory_cache()
                
                logger.debug(f"📊 Memory deallocated for document {document_id}: {allocated_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Document memory deallocation failed: {e}")
    
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
            
            # Check document-specific memory pressure
            if self.active_document_count > 10:  # Arbitrary threshold
                logger.warning(f"⚠️ High active document count: {self.active_document_count}")
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
                },
                "document_processing": {
                    "active_document_count": self.active_document_count,
                    "total_allocated_mb": sum(doc["allocated_mb"] for doc in self.document_memory_usage.values()),
                    "active_documents": list(self.document_memory_usage.keys())
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
                    "brain4_allocation_percent": self.max_vram_usage * 100,
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
            
            # Additional checks for Brain-4 specific issues
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
                "target_vram_usage": self.target_vram_usage,
                "active_document_count": self.active_document_count
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
            
            # Clear document memory tracking
            self.document_memory_usage.clear()
            self.active_document_count = 0
            
            # Clear caches
            await self._clear_memory_cache()
            
            # Reset CUDA memory fraction
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1.0)
                logger.info("✅ GPU memory fraction reset")
            
            logger.info("✅ Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        try:
            summary = {
                "initial_gpu_memory_gb": self.initial_gpu_memory / 1024**3,
                "peak_gpu_memory_gb": self.peak_gpu_memory / 1024**3,
                "initial_cpu_memory_gb": self.initial_cpu_memory / 1024**3,
                "peak_cpu_memory_gb": self.peak_cpu_memory / 1024**3,
                "max_vram_usage_percent": self.max_vram_usage * 100,
                "target_vram_usage_percent": self.target_vram_usage * 100,
                "active_document_count": self.active_document_count,
                "total_documents_tracked": len(self.document_memory_usage)
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
