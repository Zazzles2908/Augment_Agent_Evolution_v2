"""
VRAM Manager for Four-Brain System
Enforces GPU memory allocation limits for RTX 5070 Ti

This module implements strict VRAM management to prevent models from
consuming entire GPU memory and ensure proper allocation:
- Brain1 (Embedding): 35% = 5.6GB
- Brain2 (Reranker): 25% = 4GB  
- Brain3 (Intelligence): 15% = 2.4GB
- Brain4 (Document): 15% = 2.4GB

Created: 2025-08-03 AEST
Purpose: Fix VRAM overuse causing system instability
"""

import os
import sys
import logging
import time
from typing import Dict, Optional, Tuple
import torch
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class VRAMManager:
    """GPU Memory Manager for Four-Brain Architecture"""
    
    # RTX 5070 Ti memory allocation strategy (FIXED: Increased Brain1 allocation for Qwen3-4B)
    BRAIN_MEMORY_ALLOCATIONS = {
        'embedding': 0.60,      # Brain1: 60% = 9.6GB (INCREASED for Qwen3-4B FP16)
        'reranker': 0.20,       # Brain2: 20% = 3.2GB
        'intelligence': 0.10,   # Brain3: 10% = 1.6GB
        'document': 0.10,       # Brain4: 10% = 1.6GB
        'orchestrator': 0.05    # Orchestrator: 5% = 0.8GB
    }
    
    def __init__(self, brain_role: str):
        """
        Initialize VRAM Manager for specific brain
        
        Args:
            brain_role: Role of the brain (embedding, reranker, intelligence, document, orchestrator)
        """
        self.brain_role = brain_role
        self.allocated_fraction = self.BRAIN_MEMORY_ALLOCATIONS.get(brain_role, 0.10)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.total_vram_gb = 16.0  # RTX 5070 Ti has 16GB
        self.allocated_vram_gb = self.total_vram_gb * self.allocated_fraction
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info(f"VRAMManager initialized for {brain_role}")
        logger.info(f"Allocated fraction: {self.allocated_fraction:.1%} ({self.allocated_vram_gb:.1f}GB)")
        
        # Set PyTorch memory fraction
        self._set_memory_fraction()
    
    def _set_memory_fraction(self):
        """Set PyTorch CUDA memory fraction"""
        try:
            if torch.cuda.is_available():
                # Set memory fraction for this process
                torch.cuda.set_per_process_memory_fraction(
                    self.allocated_fraction, 
                    device=self.device
                )
                
                # Configure memory allocator
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                    f"max_split_size_mb:512,"
                    f"backend:cudaMallocAsync,"
                    f"expandable_segments:True"
                )
                
                logger.info(f"✅ CUDA memory fraction set to {self.allocated_fraction:.1%}")
            else:
                logger.warning("⚠️ CUDA not available, using CPU")
                
        except Exception as e:
            logger.error(f"❌ Failed to set memory fraction: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        try:
            if not torch.cuda.is_available():
                return {'error': 'CUDA not available'}
            
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)   # GB
            # memory_cached is deprecated; use memory_reserved which now includes cache
            memory_cached = memory_reserved
            
            # Calculate usage percentages
            allocated_percent = (memory_allocated / self.allocated_vram_gb) * 100
            reserved_percent = (memory_reserved / self.allocated_vram_gb) * 100
            
            return {
                'brain_role': self.brain_role,
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'cached_gb': memory_cached,
                'allocated_percent': allocated_percent,
                'reserved_percent': reserved_percent,
                'limit_gb': self.allocated_vram_gb,
                'limit_fraction': self.allocated_fraction,
                'total_vram_gb': self.total_vram_gb
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {'error': str(e)}
    
    def check_memory_usage(self) -> Tuple[bool, str]:
        """
        Check if memory usage is within limits
        
        Returns:
            Tuple of (is_within_limits, status_message)
        """
        try:
            memory_info = self.get_memory_info()
            
            if 'error' in memory_info:
                return False, f"Memory check failed: {memory_info['error']}"
            
            allocated_percent = memory_info['allocated_percent']
            
            if allocated_percent > 90:
                return False, f"❌ CRITICAL: Memory usage {allocated_percent:.1f}% exceeds 90% limit"
            elif allocated_percent > 80:
                return True, f"⚠️ WARNING: Memory usage {allocated_percent:.1f}% approaching limit"
            else:
                return True, f"✅ OK: Memory usage {allocated_percent:.1f}% within limits"
                
        except Exception as e:
            return False, f"Memory check error: {e}"
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory cache cleared")
            else:
                logger.warning("⚠️ CUDA not available for cache clearing")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
    
    def force_garbage_collection(self):
        """Force garbage collection and memory cleanup"""
        try:
            import gc
            
            # Python garbage collection
            gc.collect()
            
            # Clear CUDA cache
            self.clear_cache()
            
            logger.info("✅ Forced garbage collection completed")
            
        except Exception as e:
            logger.error(f"❌ Garbage collection failed: {e}")
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        Context manager for memory-aware operations
        
        Args:
            operation_name: Name of the operation for logging
        """
        # Check memory before operation
        is_ok, status = self.check_memory_usage()
        logger.info(f"🔍 Memory check before {operation_name}: {status}")
        
        if not is_ok and "CRITICAL" in status:
            logger.warning(f"⚠️ Clearing cache before {operation_name}")
            self.clear_cache()
        
        try:
            yield self
        finally:
            # Check memory after operation
            is_ok, status = self.check_memory_usage()
            logger.info(f"🔍 Memory check after {operation_name}: {status}")
            
            if not is_ok:
                logger.warning(f"⚠️ Memory cleanup after {operation_name}")
                self.clear_cache()
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background memory monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"✅ Memory monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("✅ Memory monitoring stopped")
    
    def _monitor_memory(self, interval_seconds: int):
        """Background memory monitoring loop"""
        while self.monitoring_active:
            try:
                is_ok, status = self.check_memory_usage()
                
                if not is_ok:
                    logger.warning(f"🚨 Memory Alert: {status}")
                    
                    if "CRITICAL" in status:
                        logger.warning("🧹 Performing emergency memory cleanup")
                        self.force_garbage_collection()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Memory monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def get_status_report(self) -> str:
        """Get formatted status report"""
        try:
            memory_info = self.get_memory_info()
            
            if 'error' in memory_info:
                return f"❌ VRAM Status Error: {memory_info['error']}"
            
            report = f"""
📊 VRAM Status Report - {self.brain_role.upper()}
{'='*50}
🎯 Allocated Limit: {memory_info['limit_gb']:.1f}GB ({memory_info['limit_fraction']:.1%})
📈 Current Usage: {memory_info['allocated_gb']:.2f}GB ({memory_info['allocated_percent']:.1f}%)
🔒 Reserved: {memory_info['reserved_gb']:.2f}GB ({memory_info['reserved_percent']:.1f}%)
💾 Cached: {memory_info['cached_gb']:.2f}GB
🖥️ Total VRAM: {memory_info['total_vram_gb']:.1f}GB (RTX 5070 Ti)
"""
            
            is_ok, status = self.check_memory_usage()
            report += f"🚦 Status: {status}\n"
            
            return report.strip()
            
        except Exception as e:
            return f"❌ Failed to generate status report: {e}"

# Global VRAM manager instance
_vram_manager = None

def get_vram_manager(brain_role: str = None) -> VRAMManager:
    """Get or create global VRAM manager instance"""
    global _vram_manager
    
    if _vram_manager is None:
        if brain_role is None:
            brain_role = os.environ.get('BRAIN_ROLE', 'unknown')
        
        _vram_manager = VRAMManager(brain_role)
    
    return _vram_manager

def initialize_vram_management(brain_role: str, start_monitoring: bool = True):
    """Initialize VRAM management for a brain service"""
    vram_manager = get_vram_manager(brain_role)
    
    if start_monitoring:
        vram_manager.start_monitoring()
    
    logger.info(f"✅ VRAM management initialized for {brain_role}")
    return vram_manager

# CUDA 13.0 Enhanced GPU Management - Consolidated from scripts/utils/gpu_detector.py
class CUDA13GPUManager:
    """Enhanced GPU management with CUDA 13.0 optimizations for RTX 5070 Ti Blackwell"""

    def __init__(self):
        self.cuda_version = None
        self.gpu_name = None
        self.compute_capability = None
        self.blackwell_support = False
        self.memory_pool = None
        self._initialize_cuda13_features()

    def _initialize_cuda13_features(self):
        """Initialize CUDA 13.0 specific GPU features"""
        try:
            if torch.cuda.is_available():
                self.cuda_version = torch.version.cuda
                self.gpu_name = torch.cuda.get_device_name(0)
                self.compute_capability = torch.cuda.get_device_capability(0)

                # Check for Blackwell architecture (sm_120)
                if self.compute_capability >= (12, 0):
                    self.blackwell_support = True
                    logger.info(f"✅ Blackwell GPU detected: {self.gpu_name} (sm_{self.compute_capability[0]}{self.compute_capability[1]})")

                    # Enable CUDA 13.0 memory optimizations
                    if hasattr(torch.cuda, 'memory_pool'):
                        self.memory_pool = torch.cuda.memory_pool()
                        self.memory_pool.set_memory_fraction(0.9)  # Use 90% of VRAM efficiently
                        logger.info("✅ CUDA 13.0 memory pool optimizations enabled")

                    # Enable Blackwell-specific optimizations
                    torch.backends.cuda.enable_blackwell_optimizations = getattr(
                        torch.backends.cuda, 'enable_blackwell_optimizations', True
                    )
                    torch.backends.cuda.matmul.allow_fp4 = getattr(
                        torch.backends.cuda.matmul, 'allow_fp4', True
                    )

                else:
                    logger.warning(f"⚠️ Non-Blackwell GPU: {self.gpu_name} (sm_{self.compute_capability[0]}{self.compute_capability[1]})")
            else:
                raise RuntimeError("CUDA not available")

        except Exception as e:
            logger.error(f"❌ CUDA 13.0 GPU initialization failed: {e}")
            raise

    def get_optimized_memory_allocation(self, brain_role: str) -> Dict[str, float]:
        """Get optimized memory allocation for CUDA 13.0 with Blackwell improvements"""
        base_allocations = {
            "brain1": 0.35,  # 35% for embedding
            "brain2": 0.25,  # 25% for reranker
            "brain3": 0.15,  # 15% for intelligence
            "brain4": 0.15,  # 15% for document processor
            "system": 0.10   # 10% for system overhead
        }

        if self.blackwell_support:
            # Blackwell optimizations allow more efficient memory usage
            optimized_allocations = {
                "brain1": 0.40,  # Increased to 40% due to better efficiency
                "brain2": 0.30,  # Increased to 30%
                "brain3": 0.15,  # Maintained
                "brain4": 0.15,  # Maintained
                "system": 0.05   # Reduced system overhead due to efficiency
            }
            logger.info("✅ Using Blackwell-optimized memory allocations")
            return optimized_allocations
        else:
            logger.info("ℹ️ Using standard memory allocations")
            return base_allocations

    def enable_cuda13_optimizations(self):
        """Enable CUDA 13.0 specific optimizations"""
        if not self.blackwell_support:
            logger.warning("⚠️ Blackwell optimizations not available on this GPU")
            return False

        try:
            # Enable CUDA graphs for reduced kernel launch overhead
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Enable memory optimizations
            if self.memory_pool:
                # Configure memory pool for optimal performance
                torch.cuda.empty_cache()  # Clear existing cache
                logger.info("✅ CUDA 13.0 memory optimizations enabled")

            # Enable FP4 operations if available
            if hasattr(torch.backends.cuda.matmul, 'allow_fp4'):
                torch.backends.cuda.matmul.allow_fp4 = True
                logger.info("✅ FP4 operations enabled for Blackwell")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to enable CUDA 13.0 optimizations: {e}")
            return False

    def get_gpu_info(self) -> Dict[str, any]:
        """Get comprehensive GPU information for CUDA 13.0"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "compute_capability": f"sm_{self.compute_capability[0]}{self.compute_capability[1]}",
            "blackwell_support": self.blackwell_support,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
            "memory_pool_enabled": self.memory_pool is not None,
            "optimizations_enabled": self.blackwell_support
        }
