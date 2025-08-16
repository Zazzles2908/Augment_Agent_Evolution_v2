"""
GPU Memory Manager for Four-Brain System
Provides comprehensive GPU memory management and optimization for RTX 5070 Ti

Created: 2025-08-03 AEST
Purpose: Fix missing MemoryManager import errors and provide proper GPU memory management
"""

import torch
import psutil
import logging
import gc
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory information structure"""
    total_memory: int
    allocated_memory: int
    cached_memory: int
    free_memory: int
    memory_fraction: float
    device_name: str


class MemoryManager:
    """
    GPU Memory Management for Four-Brain System
    Optimized for RTX 5070 Ti Blackwell architecture with 16GB VRAM
    """
    
    def __init__(self, memory_fraction: float = 0.9):
        """
        Initialize Memory Manager
        
        Args:
            memory_fraction: Fraction of GPU memory to allocate (default: 0.9 = 90%)
        """
        self.logger = logging.getLogger(__name__)
        self.memory_fraction = memory_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.device_properties = None
        
        # Memory optimization settings
        self.enable_memory_efficient_attention = True
        self.enable_gradient_checkpointing = True
        self.enable_mixed_precision = True
        
    def initialize(self) -> bool:
        """
        Initialize memory management system
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                self.logger.warning("⚠️ CUDA not available - GPU memory management disabled")
                return False
            
            # Get device properties
            self.device_properties = torch.cuda.get_device_properties(0)
            
            # Set memory allocation configuration
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory optimization features
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                self.logger.info("✅ Flash Attention enabled")
            
            # Clear any existing cache
            torch.cuda.empty_cache()
            
            # Enable TF32 for RTX 5070 Ti Blackwell
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("✅ TF32 optimizations enabled for Blackwell")
            
            self.initialized = True
            self.logger.info(f"✅ MemoryManager initialized successfully")
            self.logger.info(f"🎯 GPU: {self.device_properties.name}")
            self.logger.info(f"📊 Total VRAM: {self.device_properties.total_memory / 1024**3:.1f} GB")
            self.logger.info(f"🔧 Memory fraction: {self.memory_fraction * 100:.0f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManager initialization failed: {e}")
            return False
    
    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory information
        
        Returns:
            MemoryInfo object with current memory statistics
        """
        if not torch.cuda.is_available():
            return MemoryInfo(
                total_memory=0,
                allocated_memory=0,
                cached_memory=0,
                free_memory=0,
                memory_fraction=0.0,
                device_name="CPU"
            )
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory
            
            return MemoryInfo(
                total_memory=total_memory,
                allocated_memory=allocated_memory,
                cached_memory=cached_memory,
                free_memory=free_memory,
                memory_fraction=allocated_memory / total_memory,
                device_name=torch.cuda.get_device_properties(0).name
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory info: {e}")
            return MemoryInfo(
                total_memory=0,
                allocated_memory=0,
                cached_memory=0,
                free_memory=0,
                memory_fraction=0.0,
                device_name="Error"
            )
    
    def optimize_memory(self) -> bool:
        """
        Optimize memory usage by clearing cache and running garbage collection
        
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Run Python garbage collection
                gc.collect()
                
                self.logger.info("✅ Memory optimization completed")
                return True
            else:
                self.logger.warning("⚠️ CUDA not available - memory optimization skipped")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Memory optimization failed: {e}")
            return False
    
    def check_memory_pressure(self, threshold: float = 0.85) -> bool:
        """
        Check if memory usage is above threshold
        
        Args:
            threshold: Memory usage threshold (default: 0.85 = 85%)
            
        Returns:
            True if memory pressure detected, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                return False
            
            memory_info = self.get_memory_info()
            memory_usage = memory_info.memory_fraction
            
            if memory_usage > threshold:
                self.logger.warning(f"⚠️ Memory pressure detected: {memory_usage:.1%} usage")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Memory pressure check failed: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Dictionary with detailed memory statistics
        """
        memory_info = self.get_memory_info()
        
        stats = {
            "device_name": memory_info.device_name,
            "total_memory_gb": memory_info.total_memory / 1024**3,
            "allocated_memory_gb": memory_info.allocated_memory / 1024**3,
            "cached_memory_gb": memory_info.cached_memory / 1024**3,
            "free_memory_gb": memory_info.free_memory / 1024**3,
            "memory_usage_percent": memory_info.memory_fraction * 100,
            "memory_fraction_setting": self.memory_fraction,
            "cuda_available": torch.cuda.is_available(),
            "initialized": self.initialized
        }
        
        # Add system memory info
        try:
            system_memory = psutil.virtual_memory()
            stats.update({
                "system_memory_total_gb": system_memory.total / 1024**3,
                "system_memory_available_gb": system_memory.available / 1024**3,
                "system_memory_usage_percent": system_memory.percent
            })
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get system memory info: {e}")
        
        return stats
    
    def set_memory_fraction(self, fraction: float) -> bool:
        """
        Update memory fraction setting
        
        Args:
            fraction: New memory fraction (0.0 to 1.0)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not 0.0 <= fraction <= 1.0:
                self.logger.error(f"❌ Invalid memory fraction: {fraction}")
                return False
            
            self.memory_fraction = fraction
            
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(fraction)
                self.logger.info(f"✅ Memory fraction updated to {fraction:.1%}")
                return True
            else:
                self.logger.warning("⚠️ CUDA not available - memory fraction not applied")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to set memory fraction: {e}")
            return False


# Global memory manager instance
memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager


def initialize_memory_management(memory_fraction: float = 0.9) -> bool:
    """
    Initialize global memory management
    
    Args:
        memory_fraction: Fraction of GPU memory to allocate
        
    Returns:
        True if initialization successful, False otherwise
    """
    global memory_manager
    memory_manager = MemoryManager(memory_fraction)
    return memory_manager.initialize()
