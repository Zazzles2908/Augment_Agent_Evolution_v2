#!/usr/bin/env python3
"""
RTX 5070 Ti Memory Optimizer
Optimized memory management for 16GB GDDR7 with TensorRT FP4

Created: 2025-08-04 AEST
Author: AugmentAI - RTX 5070 Ti Optimization
"""

import os
import gc
import logging
import time
import torch
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics for RTX 5070 Ti"""
    total_vram_gb: float
    allocated_vram_gb: float
    cached_vram_gb: float
    free_vram_gb: float
    utilization_percent: float
    system_ram_gb: float
    system_ram_used_gb: float

class RTX5070TiMemoryOptimizer:
    """
    Memory Optimizer for RTX 5070 Ti Blackwell (16GB GDDR7)
    Implements dynamic VRAM allocation and model pre-warming
    """
    
    def __init__(self, 
                 max_vram_fraction: float = 0.9,
                 enable_memory_pool: bool = True,
                 enable_prewarming: bool = True):
        """
        Initialize RTX 5070 Ti memory optimizer
        
        Args:
            max_vram_fraction: Maximum VRAM fraction to use (0.9 = 14.4GB of 16GB)
            enable_memory_pool: Enable CUDA memory pool for efficiency
            enable_prewarming: Enable model pre-warming to reduce cold start
        """
        self.max_vram_fraction = max_vram_fraction
        self.enable_memory_pool = enable_memory_pool
        self.enable_prewarming = enable_prewarming
        
        # RTX 5070 Ti specifications
        self.total_vram_gb = 16.0  # 16GB GDDR7
        self.memory_bandwidth_gbps = 672  # GB/s
        self.cuda_cores = 8960
        
        self._validate_hardware()
        self._configure_memory_settings()
        
        # Memory tracking
        self.prewarmed_models = set()
        self.memory_pools = {}
        
        logger.info("✅ RTX 5070 Ti Memory Optimizer initialized")
        logger.info(f"📊 Max VRAM allocation: {self.max_vram_fraction * 16:.1f}GB")
    
    def _validate_hardware(self) -> None:
        """Validate RTX 5070 Ti hardware"""
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA not available")
        
        device_name = torch.cuda.get_device_name(0)
        if "RTX 5070 Ti" not in device_name:
            logger.warning(f"⚠️ Optimized for RTX 5070 Ti - found: {device_name}")
        
        # Check VRAM capacity
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_memory / (1024**3)
        
        if abs(total_gb - 16.0) > 1.0:  # Allow 1GB tolerance
            logger.warning(f"⚠️ Expected 16GB VRAM - found: {total_gb:.1f}GB")
        
        logger.info(f"🎯 Target GPU: {device_name} ({total_gb:.1f}GB VRAM)")
    
    def _configure_memory_settings(self) -> None:
        """Configure optimal memory settings for RTX 5070 Ti"""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.max_vram_fraction)
        
        # Enable memory pool if requested
        if self.enable_memory_pool:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:16'
        
        # Optimize for Blackwell architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        logger.info("🔧 Memory settings configured for RTX 5070 Ti Blackwell")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # GPU memory stats
        total_vram = torch.cuda.get_device_properties(0).total_memory
        allocated_vram = torch.cuda.memory_allocated(0)
        cached_vram = torch.cuda.memory_reserved(0)
        free_vram = total_vram - cached_vram
        
        # Convert to GB
        total_vram_gb = total_vram / (1024**3)
        allocated_vram_gb = allocated_vram / (1024**3)
        cached_vram_gb = cached_vram / (1024**3)
        free_vram_gb = free_vram / (1024**3)
        
        utilization_percent = (allocated_vram / total_vram) * 100
        
        # System RAM stats
        system_memory = psutil.virtual_memory()
        system_ram_gb = system_memory.total / (1024**3)
        system_ram_used_gb = system_memory.used / (1024**3)
        
        return MemoryStats(
            total_vram_gb=total_vram_gb,
            allocated_vram_gb=allocated_vram_gb,
            cached_vram_gb=cached_vram_gb,
            free_vram_gb=free_vram_gb,
            utilization_percent=utilization_percent,
            system_ram_gb=system_ram_gb,
            system_ram_used_gb=system_ram_used_gb
        )
    
    def optimize_for_model_loading(self, model_size_gb: float) -> bool:
        """Optimize memory for model loading"""
        logger.info(f"🔧 Optimizing memory for {model_size_gb:.1f}GB model...")
        
        stats = self.get_memory_stats()
        required_vram = model_size_gb * 1.5  # 50% overhead for safety
        
        if stats.free_vram_gb < required_vram:
            logger.warning(f"⚠️ Insufficient VRAM: need {required_vram:.1f}GB, have {stats.free_vram_gb:.1f}GB")
            
            # Try to free memory
            freed_gb = self.free_memory()
            stats = self.get_memory_stats()
            
            if stats.free_vram_gb < required_vram:
                logger.error(f"❌ Still insufficient VRAM after cleanup: {stats.free_vram_gb:.1f}GB")
                return False
        
        logger.info(f"✅ Memory optimized - {stats.free_vram_gb:.1f}GB available")
        return True
    
    def free_memory(self) -> float:
        """Free GPU memory and return amount freed in GB"""
        initial_stats = self.get_memory_stats()
        
        logger.info("🧹 Freeing GPU memory...")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear memory pools if enabled
        if self.enable_memory_pool:
            torch.cuda.memory.empty_cache()
        
        final_stats = self.get_memory_stats()
        freed_gb = final_stats.free_vram_gb - initial_stats.free_vram_gb
        
        logger.info(f"♻️ Freed {freed_gb:.2f}GB VRAM")
        return freed_gb
    
    def prewarm_model(self, model: torch.nn.Module, input_shape: Tuple[int, ...], 
                     model_name: str = "unknown") -> bool:
        """Pre-warm model to reduce cold start latency"""
        if not self.enable_prewarming:
            return True
        
        if model_name in self.prewarmed_models:
            logger.info(f"🔥 Model {model_name} already prewarmed")
            return True
        
        logger.info(f"🔥 Pre-warming model: {model_name}")
        
        try:
            # Move model to GPU if not already
            if next(model.parameters()).device.type != 'cuda':
                model = model.cuda()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape, device='cuda', dtype=torch.float16)
            
            # Warmup iterations
            model.eval()
            with torch.no_grad():
                for i in range(5):
                    _ = model(dummy_input)
                    if i == 0:
                        torch.cuda.synchronize()  # Ensure first run completes
            
            torch.cuda.synchronize()
            
            self.prewarmed_models.add(model_name)
            logger.info(f"✅ Model {model_name} prewarmed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prewarm model {model_name}: {e}")
            return False
    
    def create_memory_pool(self, pool_name: str, size_gb: float) -> bool:
        """Create dedicated memory pool for specific operations"""
        if not self.enable_memory_pool:
            return True
        
        try:
            size_bytes = int(size_gb * 1024**3)
            
            # Allocate memory pool
            pool_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device='cuda')
            self.memory_pools[pool_name] = pool_tensor
            
            logger.info(f"📦 Created memory pool '{pool_name}': {size_gb:.1f}GB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create memory pool '{pool_name}': {e}")
            return False
    
    def monitor_memory_usage(self, operation_name: str = "operation") -> None:
        """Monitor memory usage during operation"""
        stats = self.get_memory_stats()
        
        logger.info(f"📊 Memory usage during {operation_name}:")
        logger.info(f"  - VRAM: {stats.allocated_vram_gb:.2f}GB / {stats.total_vram_gb:.1f}GB ({stats.utilization_percent:.1f}%)")
        logger.info(f"  - Free VRAM: {stats.free_vram_gb:.2f}GB")
        logger.info(f"  - System RAM: {stats.system_ram_used_gb:.1f}GB / {stats.system_ram_gb:.1f}GB")
        
        # Warning thresholds
        if stats.utilization_percent > 90:
            logger.warning("⚠️ High VRAM usage - consider freeing memory")
        
        if stats.free_vram_gb < 1.0:
            logger.warning("⚠️ Low free VRAM - performance may degrade")
    
    def optimize_for_inference(self) -> None:
        """Optimize memory settings for inference workloads"""
        logger.info("⚡ Optimizing for inference workloads...")
        
        # Set optimal settings for inference
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # Set memory growth strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        logger.info("✅ Memory optimized for inference")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats.utilization_percent > 85:
            recommendations.append("Consider reducing batch size or model precision")
        
        if stats.free_vram_gb < 2.0:
            recommendations.append("Free memory using free_memory() method")
        
        if len(self.prewarmed_models) == 0:
            recommendations.append("Use prewarm_model() to reduce cold start latency")
        
        if not self.enable_memory_pool:
            recommendations.append("Enable memory pooling for better allocation efficiency")
        
        if stats.utilization_percent < 50:
            recommendations.append("VRAM underutilized - consider larger batch sizes")
        
        return recommendations
    
    def cleanup(self) -> None:
        """Cleanup memory optimizer resources"""
        logger.info("🧹 Cleaning up memory optimizer...")
        
        # Clear memory pools
        self.memory_pools.clear()
        
        # Clear prewarmed models tracking
        self.prewarmed_models.clear()
        
        # Free GPU memory
        self.free_memory()
        
        logger.info("✅ Memory optimizer cleanup completed")


def main():
    """Example usage of RTX 5070 Ti Memory Optimizer"""
    optimizer = RTX5070TiMemoryOptimizer()
    
    # Display memory stats
    stats = optimizer.get_memory_stats()
    logger.info("📊 Current Memory Stats:")
    logger.info(f"  - Total VRAM: {stats.total_vram_gb:.1f}GB")
    logger.info(f"  - Used VRAM: {stats.allocated_vram_gb:.2f}GB ({stats.utilization_percent:.1f}%)")
    logger.info(f"  - Free VRAM: {stats.free_vram_gb:.2f}GB")
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        logger.info("💡 Optimization Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")
    
    logger.info("🎉 RTX 5070 Ti Memory Optimizer ready for use")


if __name__ == "__main__":
    main()
