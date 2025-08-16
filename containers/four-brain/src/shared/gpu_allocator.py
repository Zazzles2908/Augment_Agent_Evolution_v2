"""
Four-Brain GPU Memory Allocator
Implements dynamic GPU memory allocation for RTX 5070 Ti (16GB)
Brain-1: 35% (5.6GB) - Primary embedding model
Brain-2: 20% (3.2GB) - Reranker service  
Brain-3: 15% (2.4GB) - Intelligence service
Brain-4: 15% (2.4GB) - Document processor
Remaining: 15% (2.4GB) - System overhead and buffers
"""

import torch
import logging
import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BrainType(Enum):
    BRAIN1_EMBEDDING = "brain1"
    BRAIN2_RERANKER = "brain2" 
    BRAIN3_INTELLIGENCE = "brain3"
    BRAIN4_DOCUMENT = "brain4"

@dataclass
class GPUAllocation:
    brain_type: BrainType
    memory_fraction: float
    memory_gb: float
    target_fraction: float
    target_gb: float
    max_fraction: float
    max_gb: float

class FourBrainGPUAllocator:
    """
    Dynamic GPU Memory Allocator for Four-Brain Architecture
    """
    
    def __init__(self, total_gpu_memory_gb: float = 16.0):
        self.total_gpu_memory_gb = total_gpu_memory_gb
        self.total_gpu_memory_bytes = int(total_gpu_memory_gb * 1024**3)
        
        # Four-Brain allocation strategy
        self.allocations = {
            BrainType.BRAIN1_EMBEDDING: GPUAllocation(
                brain_type=BrainType.BRAIN1_EMBEDDING,
                memory_fraction=0.35,
                memory_gb=5.6,
                target_fraction=0.30,
                target_gb=4.8,
                max_fraction=0.40,
                max_gb=6.4
            ),
            BrainType.BRAIN2_RERANKER: GPUAllocation(
                brain_type=BrainType.BRAIN2_RERANKER,
                memory_fraction=0.20,
                memory_gb=3.2,
                target_fraction=0.18,
                target_gb=2.9,
                max_fraction=0.25,
                max_gb=4.0
            ),
            BrainType.BRAIN3_INTELLIGENCE: GPUAllocation(
                brain_type=BrainType.BRAIN3_INTELLIGENCE,
                memory_fraction=0.15,
                memory_gb=2.4,
                target_fraction=0.13,
                target_gb=2.1,
                max_fraction=0.20,
                max_gb=3.2
            ),
            BrainType.BRAIN4_DOCUMENT: GPUAllocation(
                brain_type=BrainType.BRAIN4_DOCUMENT,
                memory_fraction=0.15,
                memory_gb=2.4,
                target_fraction=0.13,
                target_gb=2.1,
                max_fraction=0.20,
                max_gb=3.2
            )
        }
        
        # System overhead: 15% (2.4GB)
        self.system_overhead_fraction = 0.15
        self.system_overhead_gb = 2.4
        
        logger.info("🧠 Four-Brain GPU Allocator initialized")
        logger.info(f"📊 Total GPU Memory: {total_gpu_memory_gb}GB")
        self._log_allocation_strategy()
    
    def _log_allocation_strategy(self):
        """Log the allocation strategy"""
        logger.info("🎯 Four-Brain GPU Allocation Strategy:")
        for brain_type, allocation in self.allocations.items():
            logger.info(f"  {brain_type.value}: {allocation.memory_fraction*100:.0f}% ({allocation.memory_gb:.1f}GB)")
        logger.info(f"  System Overhead: {self.system_overhead_fraction*100:.0f}% ({self.system_overhead_gb:.1f}GB)")
    
    def get_allocation(self, brain_type: BrainType) -> GPUAllocation:
        """Get allocation for specific brain type"""
        return self.allocations.get(brain_type)
    
    def set_memory_fraction(self, brain_type: BrainType, device: str = "cuda:0") -> bool:
        """Set CUDA memory fraction for specific brain"""
        try:
            if not torch.cuda.is_available():
                logger.warning("❌ CUDA not available")
                return False
            
            allocation = self.get_allocation(brain_type)
            if not allocation:
                logger.error(f"❌ No allocation found for {brain_type}")
                return False
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(
                allocation.memory_fraction, 
                device=device
            )
            
            logger.info(f"✅ {brain_type.value} memory fraction set to {allocation.memory_fraction*100:.0f}% ({allocation.memory_gb:.1f}GB)")
            
            # Verify allocation
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(device)
                total_memory = props.total_memory / 1024**3
                allocated_memory = allocation.memory_fraction * total_memory
                
                logger.info(f"📊 {brain_type.value} allocated: {allocated_memory:.1f}GB of {total_memory:.1f}GB total")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting memory fraction for {brain_type}: {e}")
            return False
    
    def configure_brain_memory(self, brain_type: BrainType) -> Dict[str, float]:
        """Configure memory settings for a specific brain"""
        allocation = self.get_allocation(brain_type)
        if not allocation:
            logger.error(f"❌ No allocation found for {brain_type}")
            return {}
        
        # Set environment variables for the brain
        env_vars = {
            "TORCH_CUDA_MEMORY_FRACTION": str(allocation.memory_fraction),
            "CUDA_MEMORY_FRACTION": str(allocation.memory_fraction),
            "MAX_VRAM_USAGE": str(allocation.memory_fraction),
            "TARGET_VRAM_USAGE": str(allocation.target_fraction),
            "MAX_MEMORY_GB": str(allocation.max_gb),
            "TARGET_MEMORY_GB": str(allocation.target_gb)
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info(f"🔧 {brain_type.value} memory configuration:")
        logger.info(f"  Memory Fraction: {allocation.memory_fraction*100:.0f}%")
        logger.info(f"  Target Memory: {allocation.target_gb:.1f}GB")
        logger.info(f"  Max Memory: {allocation.max_gb:.1f}GB")
        
        return {
            "memory_fraction": allocation.memory_fraction,
            "target_fraction": allocation.target_fraction,
            "max_fraction": allocation.max_fraction,
            "memory_gb": allocation.memory_gb,
            "target_gb": allocation.target_gb,
            "max_gb": allocation.max_gb
        }
    
    def check_memory_usage(self, brain_type: BrainType) -> Dict[str, float]:
        """Check current memory usage for a brain"""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
            
            allocation = self.get_allocation(brain_type)
            if not allocation:
                return {"error": f"No allocation found for {brain_type}"}
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Calculate usage percentages
            allocated_percent = (memory_allocated / memory_total) * 100
            reserved_percent = (memory_reserved / memory_total) * 100
            
            # Check if within limits
            within_target = memory_allocated <= allocation.target_gb
            within_max = memory_allocated <= allocation.max_gb
            
            usage_info = {
                "brain_type": brain_type.value,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_total_gb": memory_total,
                "allocated_percent": allocated_percent,
                "reserved_percent": reserved_percent,
                "target_gb": allocation.target_gb,
                "max_gb": allocation.max_gb,
                "within_target": within_target,
                "within_max": within_max,
                "allocation_fraction": allocation.memory_fraction
            }
            
            # Log status
            status = "✅" if within_target else "⚠️" if within_max else "❌"
            logger.info(f"{status} {brain_type.value} memory: {memory_allocated:.1f}GB/{allocation.max_gb:.1f}GB ({allocated_percent:.1f}%)")
            
            return usage_info
            
        except Exception as e:
            logger.error(f"❌ Error checking memory usage for {brain_type}: {e}")
            return {"error": str(e)}
    
    def optimize_memory_allocation(self) -> bool:
        """Optimize memory allocation across all brains"""
        try:
            logger.info("🔧 Optimizing Four-Brain memory allocation...")
            
            if not torch.cuda.is_available():
                logger.warning("❌ CUDA not available for optimization")
                return False
            
            # Clear cache
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
            
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("⚡ CUDNN optimization enabled")
            
            # Set memory allocation configuration
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync,max_split_size_mb:512"
            logger.info("🔧 PyTorch CUDA allocation configured")
            
            logger.info("✅ Four-Brain memory optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error optimizing memory allocation: {e}")
            return False
    
    def get_brain_type_from_env(self) -> Optional[BrainType]:
        """Determine brain type from environment variables"""
        brain_id = os.getenv("BRAIN_ID", "").lower()
        service_name = os.getenv("SERVICE_NAME", "").lower()
        
        if "brain1" in brain_id or "embedding" in service_name:
            return BrainType.BRAIN1_EMBEDDING
        elif "brain2" in brain_id or "reranker" in service_name:
            return BrainType.BRAIN2_RERANKER
        elif "brain3" in brain_id or "intelligence" in service_name:
            return BrainType.BRAIN3_INTELLIGENCE
        elif "brain4" in brain_id or "document" in service_name:
            return BrainType.BRAIN4_DOCUMENT
        
        logger.warning("⚠️ Could not determine brain type from environment")
        return None


# Global allocator instance
gpu_allocator = FourBrainGPUAllocator()

def configure_brain_gpu(brain_type: BrainType = None) -> bool:
    """Configure GPU for current brain service"""
    try:
        if brain_type is None:
            brain_type = gpu_allocator.get_brain_type_from_env()
        
        if brain_type is None:
            logger.warning("⚠️ Could not determine brain type, using default allocation")
            return False
        
        # Configure memory
        memory_config = gpu_allocator.configure_brain_memory(brain_type)
        
        # Set memory fraction
        success = gpu_allocator.set_memory_fraction(brain_type)
        
        # Optimize allocation
        gpu_allocator.optimize_memory_allocation()
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error configuring brain GPU: {e}")
        return False
