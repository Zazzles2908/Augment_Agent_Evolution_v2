#!/usr/bin/env python3.11
"""
GPU Memory Management for Four-Brain TensorRT System
Dynamic GPU memory allocation with TensorRT engine caching for RTX 5070 Ti Blackwell

Author: AugmentAI
Date: 2025-08-02
Purpose: Manage GPU memory allocation across Brain services with TensorRT optimization
"""

import os
import sys
import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MemoryAllocation:
    """GPU memory allocation for a Brain service"""
    service_name: str
    allocated_mb: float
    used_mb: float
    reserved_mb: float
    tensorrt_engines: List[str]
    allocation_time: float
    last_used: float

@dataclass
class GPUMemoryStats:
    """GPU memory statistics"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    tensorrt_memory_gb: float
    cache_memory_gb: float
    fragmentation_ratio: float
    allocation_efficiency: float

class MemoryPriority(Enum):
    """Memory allocation priority levels"""
    CRITICAL = 1    # Brain-3 Intelligence (40%)
    HIGH = 2        # Brain-1 Embedding (25%)
    MEDIUM = 3      # Brain-2 Reranker (20%)
    LOW = 4         # Brain-4 Docling (15%)

class GPUMemoryManager:
    """Dynamic GPU memory manager for Four-Brain TensorRT system"""
    
    def __init__(self, total_memory_gb: float = 16.0):
        self.total_memory_gb = total_memory_gb
        self.total_memory_mb = total_memory_gb * 1024
        self.allocations = {}
        self.engine_cache = {}
        self.memory_lock = threading.Lock()
        
        # RTX 5070 Ti Blackwell allocation strategy
        self.allocation_strategy = {
            "brain1_embedding": {
                "percentage": 0.25,  # 25% = 4GB
                "priority": MemoryPriority.HIGH,
                "min_mb": 2048,      # Minimum 2GB
                "max_mb": 6144       # Maximum 6GB
            },
            "brain2_reranker": {
                "percentage": 0.20,  # 20% = 3.2GB
                "priority": MemoryPriority.MEDIUM,
                "min_mb": 1536,      # Minimum 1.5GB
                "max_mb": 4096       # Maximum 4GB
            },
            "brain4_docling": {
                "percentage": 0.15,  # 15% = 2.4GB
                "priority": MemoryPriority.LOW,
                "min_mb": 1024,      # Minimum 1GB
                "max_mb": 3072       # Maximum 3GB
            },
            "brain3_intelligence": {
                "percentage": 0.40,  # 40% = 6.4GB (external)
                "priority": MemoryPriority.CRITICAL,
                "min_mb": 4096,      # Minimum 4GB
                "max_mb": 8192       # Maximum 8GB
            }
        }
        
        # TensorRT engine cache configuration
        self.engine_cache_mb = 512  # 512MB for engine cache
        self.max_cached_engines = 10
        
        # Performance tracking
        self.allocation_history = []
        self.memory_stats = GPUMemoryStats(
            total_memory_gb=total_memory_gb,
            available_memory_gb=total_memory_gb,
            used_memory_gb=0.0,
            tensorrt_memory_gb=0.0,
            cache_memory_gb=0.0,
            fragmentation_ratio=0.0,
            allocation_efficiency=1.0
        )
        
        # Initialize CUDA if available
        self.cuda_available = self._initialize_cuda()
        
        logger.info(f"🧠 GPU Memory Manager initialized for RTX 5070 Ti ({total_memory_gb}GB)")
    
    def _initialize_cuda(self) -> bool:
        """Initialize CUDA for memory management"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"✅ CUDA initialized: {device_name} ({memory_gb:.1f}GB)")
                
                # Update actual memory if different from configured
                if abs(memory_gb - self.total_memory_gb) > 1.0:
                    logger.info(f"📊 Updating memory from {self.total_memory_gb}GB to {memory_gb:.1f}GB")
                    self.total_memory_gb = memory_gb
                    self.total_memory_mb = memory_gb * 1024
                    self.memory_stats.total_memory_gb = memory_gb
                    self.memory_stats.available_memory_gb = memory_gb
                
                return True
            else:
                logger.warning("⚠️ CUDA not available - using CPU memory simulation")
                return False
        except ImportError:
            logger.warning("⚠️ PyTorch not available - using memory simulation")
            return False
    
    async def allocate_memory(self, service_name: str, requested_mb: Optional[float] = None) -> bool:
        """Allocate GPU memory for a Brain service"""
        with self.memory_lock:
            try:
                # Get allocation configuration
                if service_name not in self.allocation_strategy:
                    logger.error(f"❌ Unknown service: {service_name}")
                    return False
                
                config = self.allocation_strategy[service_name]
                
                # Calculate allocation amount
                if requested_mb is None:
                    allocated_mb = self.total_memory_mb * config["percentage"]
                else:
                    allocated_mb = max(config["min_mb"], min(requested_mb, config["max_mb"]))
                
                # Check if memory is available
                current_used = sum(alloc.allocated_mb for alloc in self.allocations.values())
                available_mb = self.total_memory_mb - current_used - self.engine_cache_mb
                
                if allocated_mb > available_mb:
                    logger.warning(f"⚠️ Insufficient memory for {service_name}: requested {allocated_mb:.0f}MB, available {available_mb:.0f}MB")
                    
                    # Try to free memory from lower priority services
                    if await self._free_memory_for_priority(config["priority"], allocated_mb - available_mb):
                        available_mb = self.total_memory_mb - sum(alloc.allocated_mb for alloc in self.allocations.values()) - self.engine_cache_mb
                    
                    if allocated_mb > available_mb:
                        logger.error(f"❌ Cannot allocate {allocated_mb:.0f}MB for {service_name}")
                        return False
                
                # Perform allocation
                allocation = MemoryAllocation(
                    service_name=service_name,
                    allocated_mb=allocated_mb,
                    used_mb=0.0,
                    reserved_mb=allocated_mb * 0.1,  # 10% reserved
                    tensorrt_engines=[],
                    allocation_time=time.time(),
                    last_used=time.time()
                )
                
                self.allocations[service_name] = allocation
                
                # Update memory stats
                self._update_memory_stats()
                
                # Allocate CUDA memory if available
                if self.cuda_available:
                    await self._allocate_cuda_memory(service_name, allocated_mb)
                
                logger.info(f"✅ Allocated {allocated_mb:.0f}MB for {service_name}")
                logger.info(f"📊 Memory usage: {self.memory_stats.used_memory_gb:.1f}GB / {self.memory_stats.total_memory_gb:.1f}GB")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Memory allocation failed for {service_name}: {str(e)}")
                return False
    
    async def deallocate_memory(self, service_name: str) -> bool:
        """Deallocate GPU memory for a Brain service"""
        with self.memory_lock:
            try:
                if service_name not in self.allocations:
                    logger.warning(f"⚠️ No allocation found for {service_name}")
                    return False
                
                allocation = self.allocations[service_name]
                
                # Clear TensorRT engines from cache
                for engine_name in allocation.tensorrt_engines:
                    if engine_name in self.engine_cache:
                        del self.engine_cache[engine_name]
                        logger.debug(f"🗑️ Removed engine {engine_name} from cache")
                
                # Deallocate CUDA memory if available
                if self.cuda_available:
                    await self._deallocate_cuda_memory(service_name)
                
                # Remove allocation
                del self.allocations[service_name]
                
                # Update memory stats
                self._update_memory_stats()
                
                logger.info(f"✅ Deallocated memory for {service_name}")
                logger.info(f"📊 Memory usage: {self.memory_stats.used_memory_gb:.1f}GB / {self.memory_stats.total_memory_gb:.1f}GB")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Memory deallocation failed for {service_name}: {str(e)}")
                return False
    
    async def cache_tensorrt_engine(self, service_name: str, engine_name: str, engine_data: bytes) -> bool:
        """Cache TensorRT engine in GPU memory"""
        try:
            with self.memory_lock:
                # Check if service has allocation
                if service_name not in self.allocations:
                    logger.error(f"❌ No memory allocation for {service_name}")
                    return False
                
                # Check cache space
                engine_size_mb = len(engine_data) / (1024 * 1024)
                current_cache_size = sum(len(data) for data in self.engine_cache.values()) / (1024 * 1024)
                
                if current_cache_size + engine_size_mb > self.engine_cache_mb:
                    # Evict oldest engines
                    await self._evict_cached_engines(engine_size_mb)
                
                # Cache engine
                self.engine_cache[engine_name] = engine_data
                self.allocations[service_name].tensorrt_engines.append(engine_name)
                
                logger.info(f"✅ Cached TensorRT engine {engine_name} ({engine_size_mb:.1f}MB) for {service_name}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Engine caching failed: {str(e)}")
            return False
    
    async def get_cached_engine(self, engine_name: str) -> Optional[bytes]:
        """Get cached TensorRT engine"""
        with self.memory_lock:
            return self.engine_cache.get(engine_name)
    
    async def _free_memory_for_priority(self, priority: MemoryPriority, required_mb: float) -> bool:
        """Free memory from lower priority services"""
        try:
            freed_mb = 0.0
            services_to_deallocate = []
            
            # Find lower priority services
            for service_name, allocation in self.allocations.items():
                service_config = self.allocation_strategy[service_name]
                if service_config["priority"].value > priority.value:  # Lower priority (higher number)
                    services_to_deallocate.append((service_name, allocation.allocated_mb))
            
            # Sort by priority (lowest first)
            services_to_deallocate.sort(key=lambda x: self.allocation_strategy[x[0]]["priority"].value, reverse=True)
            
            # Deallocate services until enough memory is freed
            for service_name, allocated_mb in services_to_deallocate:
                if freed_mb >= required_mb:
                    break
                
                logger.info(f"🔄 Freeing memory from {service_name} for higher priority allocation")
                if await self.deallocate_memory(service_name):
                    freed_mb += allocated_mb
            
            return freed_mb >= required_mb
            
        except Exception as e:
            logger.error(f"❌ Priority-based memory freeing failed: {str(e)}")
            return False
    
    async def _allocate_cuda_memory(self, service_name: str, size_mb: float):
        """Allocate actual CUDA memory"""
        try:
            import torch
            
            # Allocate tensor to reserve memory
            size_bytes = int(size_mb * 1024 * 1024)
            memory_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device='cuda')
            
            # Store reference (in real implementation, this would be managed properly)
            setattr(self, f"_cuda_memory_{service_name}", memory_tensor)
            
            logger.debug(f"🔧 Allocated {size_mb:.0f}MB CUDA memory for {service_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA memory allocation failed for {service_name}: {str(e)}")
    
    async def _deallocate_cuda_memory(self, service_name: str):
        """Deallocate actual CUDA memory"""
        try:
            import torch
            
            # Release memory tensor
            memory_attr = f"_cuda_memory_{service_name}"
            if hasattr(self, memory_attr):
                delattr(self, memory_attr)
                torch.cuda.empty_cache()
                logger.debug(f"🔧 Deallocated CUDA memory for {service_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA memory deallocation failed for {service_name}: {str(e)}")
    
    async def _evict_cached_engines(self, required_mb: float):
        """Evict cached engines to free space"""
        try:
            # Sort engines by last access time (LRU)
            engine_access_times = {}
            for service_name, allocation in self.allocations.items():
                for engine_name in allocation.tensorrt_engines:
                    engine_access_times[engine_name] = allocation.last_used
            
            sorted_engines = sorted(engine_access_times.items(), key=lambda x: x[1])
            
            freed_mb = 0.0
            for engine_name, _ in sorted_engines:
                if freed_mb >= required_mb:
                    break
                
                if engine_name in self.engine_cache:
                    engine_size_mb = len(self.engine_cache[engine_name]) / (1024 * 1024)
                    del self.engine_cache[engine_name]
                    freed_mb += engine_size_mb
                    
                    # Remove from service allocation
                    for allocation in self.allocations.values():
                        if engine_name in allocation.tensorrt_engines:
                            allocation.tensorrt_engines.remove(engine_name)
                            break
                    
                    logger.debug(f"🗑️ Evicted engine {engine_name} ({engine_size_mb:.1f}MB)")
            
            logger.info(f"✅ Evicted {freed_mb:.1f}MB from engine cache")
            
        except Exception as e:
            logger.error(f"❌ Engine eviction failed: {str(e)}")
    
    def _update_memory_stats(self):
        """Update memory statistics"""
        try:
            total_allocated = sum(alloc.allocated_mb for alloc in self.allocations.values())
            total_used = sum(alloc.used_mb for alloc in self.allocations.values())
            cache_size = sum(len(data) for data in self.engine_cache.values()) / (1024 * 1024)
            
            self.memory_stats.used_memory_gb = total_allocated / 1024
            self.memory_stats.available_memory_gb = self.total_memory_gb - (total_allocated / 1024)
            self.memory_stats.tensorrt_memory_gb = total_used / 1024
            self.memory_stats.cache_memory_gb = cache_size / 1024
            
            # Calculate fragmentation ratio
            if total_allocated > 0:
                self.memory_stats.fragmentation_ratio = (total_allocated - total_used) / total_allocated
            else:
                self.memory_stats.fragmentation_ratio = 0.0
            
            # Calculate allocation efficiency
            if self.total_memory_mb > 0:
                self.memory_stats.allocation_efficiency = total_used / self.total_memory_mb
            else:
                self.memory_stats.allocation_efficiency = 0.0
            
        except Exception as e:
            logger.error(f"❌ Memory stats update failed: {str(e)}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.memory_lock:
            self._update_memory_stats()
            
            return {
                "total_memory_gb": self.memory_stats.total_memory_gb,
                "available_memory_gb": self.memory_stats.available_memory_gb,
                "used_memory_gb": self.memory_stats.used_memory_gb,
                "tensorrt_memory_gb": self.memory_stats.tensorrt_memory_gb,
                "cache_memory_gb": self.memory_stats.cache_memory_gb,
                "fragmentation_ratio": self.memory_stats.fragmentation_ratio,
                "allocation_efficiency": self.memory_stats.allocation_efficiency,
                "cuda_available": self.cuda_available,
                "allocations": {
                    name: {
                        "allocated_mb": alloc.allocated_mb,
                        "used_mb": alloc.used_mb,
                        "reserved_mb": alloc.reserved_mb,
                        "engines_cached": len(alloc.tensorrt_engines),
                        "last_used": alloc.last_used
                    }
                    for name, alloc in self.allocations.items()
                },
                "engine_cache": {
                    "total_engines": len(self.engine_cache),
                    "total_size_mb": sum(len(data) for data in self.engine_cache.values()) / (1024 * 1024),
                    "max_size_mb": self.engine_cache_mb,
                    "engines": list(self.engine_cache.keys())
                },
                "allocation_strategy": self.allocation_strategy
            }
    
    async def optimize_memory_layout(self) -> bool:
        """Optimize memory layout for better performance"""
        try:
            logger.info("🔧 Optimizing GPU memory layout...")
            
            # Defragment memory by reallocating in priority order
            current_allocations = list(self.allocations.items())
            
            # Sort by priority
            priority_order = sorted(
                current_allocations,
                key=lambda x: self.allocation_strategy[x[0]]["priority"].value
            )
            
            # Temporarily store allocations
            temp_allocations = {}
            for service_name, allocation in priority_order:
                temp_allocations[service_name] = allocation.allocated_mb
                await self.deallocate_memory(service_name)
            
            # Reallocate in priority order
            for service_name, allocated_mb in temp_allocations.items():
                await self.allocate_memory(service_name, allocated_mb)
            
            logger.info("✅ Memory layout optimized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory layout optimization failed: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown memory manager and cleanup"""
        logger.info("🔄 Shutting down GPU Memory Manager...")
        
        # Get final stats
        final_stats = self.get_memory_stats()
        logger.info(f"📊 Final memory stats: {final_stats['used_memory_gb']:.1f}GB used, "
                   f"{final_stats['allocation_efficiency']:.2f} efficiency")
        
        # Deallocate all services
        services_to_deallocate = list(self.allocations.keys())
        for service_name in services_to_deallocate:
            await self.deallocate_memory(service_name)
        
        # Clear engine cache
        self.engine_cache.clear()
        
        # Clear CUDA cache if available
        if self.cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("✅ CUDA cache cleared")
            except:
                pass
        
        logger.info("✅ GPU Memory Manager shutdown complete")

# Global memory manager instance
_memory_manager = None

def get_gpu_memory_manager(total_memory_gb: float = 16.0) -> GPUMemoryManager:
    """Get global GPU memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager(total_memory_gb)
    return _memory_manager
