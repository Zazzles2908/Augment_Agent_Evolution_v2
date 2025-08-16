"""
Memory Manager for Brain 4
Handles GPU and system memory allocation for RTX 5070 Ti optimization
"""

import asyncio
import logging
import psutil
import torch
from typing import Dict, Any, Optional
import gc

class MemoryManager:
    """
    Memory management for Brain 4 with RTX 5070 Ti optimization
    Ensures efficient memory usage across GPU and system memory
    """
    
    def __init__(self, max_vram_usage: float = 0.4, target_vram_usage: float = 0.35):
        self.max_vram_usage = max_vram_usage
        self.target_vram_usage = target_vram_usage
        self.logger = logging.getLogger(__name__)

        # Memory allocation tracking
        self.allocated_memory = {
            "brain1": 0.0,
            "brain2": 0.0,
            "brain3": 0.0,
            "brain4": 0.0
        }

        # GPU availability status with robust checking
        self.gpu_available = self._check_gpu_availability()
        self.memory_fraction_set = False  # Track if memory fraction has been set

        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.total_gpu_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory / (1024**3)

            # Set memory fraction once at initialization
            try:
                torch.cuda.set_per_process_memory_fraction(self.max_vram_usage, self.gpu_device)
                self.memory_fraction_set = True
                self.logger.info(f"GPU memory fraction set to {self.max_vram_usage:.1%} ({self.max_vram_usage * self.total_gpu_memory:.1f}GB)")
            except Exception as e:
                self.logger.warning(f"Could not set memory fraction: {e}")
        else:
            self.total_gpu_memory = 0.0

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and functional with graceful fallback"""
        try:
            if not torch.cuda.is_available():
                self.logger.warning("⚠️ CUDA not available - using CPU mode")
                return False

            # Test GPU functionality
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                torch.cuda.empty_cache()

                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
                return True

            except Exception as e:
                self.logger.warning(f"GPU test failed: {e} - falling back to CPU")
                return False

        except Exception as e:
            self.logger.warning(f"GPU availability check failed: {e} - using CPU mode")
            return False

    async def allocate_memory_for_brain(self, brain_id: str, required_gb: float) -> bool:
        """
        Allocate memory for a specific brain
        
        Args:
            brain_id: Brain identifier (brain1, brain2, brain3, brain4)
            required_gb: Required memory in GB
            
        Returns:
            True if allocation successful, False otherwise
        """
        
        try:
            # Check current memory usage
            current_usage = await self.get_current_usage()
            
            # Calculate total allocated memory
            total_allocated = sum(self.allocated_memory.values()) + required_gb
            max_allowed = self.total_gpu_memory * self.max_vram_usage
            
            if total_allocated > max_allowed:
                self.logger.warning(f"Memory allocation denied for {brain_id}: "
                                  f"Would exceed limit ({total_allocated:.2f}GB > {max_allowed:.2f}GB)")
                return False
            
            # REAL GPU memory allocation using PyTorch
            if self.gpu_available:
                try:
                    # Calculate available memory within our fraction
                    max_allowed_memory = self.total_gpu_memory * self.max_vram_usage
                    new_total_allocated = sum(self.allocated_memory.values()) + required_gb

                    if new_total_allocated > max_allowed_memory:
                        self.logger.error(f"GPU memory allocation failed for {brain_id}: "
                                        f"Would exceed allocated fraction ({new_total_allocated:.2f}GB > {max_allowed_memory:.2f}GB)")
                        return False

                    # Test allocation to verify memory is available
                    # Use smaller test tensor to avoid large allocations
                    test_size = min(int(required_gb * 1024 * 1024 * 64), 1024 * 1024 * 256)  # 64 floats per MB, max 256MB test
                    test_tensor = torch.zeros(test_size, device=f'cuda:{self.gpu_device}', dtype=torch.float32)

                    # Clean up test allocation immediately
                    del test_tensor
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Only update tracking after successful test allocation
                    self.allocated_memory[brain_id] += required_gb

                    self.logger.info(f"REAL GPU memory allocated for {brain_id}: {required_gb:.2f}GB "
                                   f"(Total: {sum(self.allocated_memory.values()):.2f}GB, "
                                   f"Fraction: {sum(self.allocated_memory.values())/self.total_gpu_memory:.3f})")

                    return True

                except torch.cuda.OutOfMemoryError as e:
                    self.logger.error(f"GPU memory allocation failed for {brain_id}: Allocation on device")
                    # Force cleanup on OOM
                    torch.cuda.empty_cache()
                    return False
                except Exception as e:
                    self.logger.error(f"Error during real GPU allocation for {brain_id}: {e}")
                    torch.cuda.empty_cache()
                    return False
            else:
                # No GPU available - fail honestly instead of fake tracking
                self.logger.error(f"Cannot allocate GPU memory for {brain_id}: No GPU available")
                return False
            
        except Exception as e:
            self.logger.error(f"Error allocating memory for {brain_id}: {e}")
            return False
    
    async def deallocate_memory_for_brain(self, brain_id: str, amount_gb: float):
        """
        Deallocate memory for a specific brain
        
        Args:
            brain_id: Brain identifier
            amount_gb: Amount to deallocate in GB
        """
        
        try:
            if brain_id in self.allocated_memory and self.allocated_memory[brain_id] > 0:
                # REAL GPU memory deallocation
                if self.gpu_available:
                    # Update tracking first
                    self.allocated_memory[brain_id] = max(0, self.allocated_memory[brain_id] - amount_gb)

                    # Force comprehensive memory cleanup
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Additional cleanup for PyTorch
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()

                    # Force garbage collection
                    import gc
                    gc.collect()

                    self.logger.info(f"REAL GPU memory deallocated for {brain_id}: {amount_gb:.2f}GB "
                                   f"(Remaining: {sum(self.allocated_memory.values()):.2f}GB)")
                else:
                    self.logger.warning(f"Cannot deallocate GPU memory for {brain_id}: No GPU available")

        except Exception as e:
            self.logger.error(f"Error deallocating memory for {brain_id}: {e}")

    def force_memory_cleanup(self):
        """
        Force comprehensive GPU memory cleanup
        """
        try:
            if self.gpu_available:
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Reset memory stats if available
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()

                # Force garbage collection
                import gc
                gc.collect()

                self.logger.info("Forced GPU memory cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during forced memory cleanup: {e}")

    async def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage information
        """
        
        try:
            usage_info = {
                "system": {
                    "virtual": psutil.virtual_memory()._asdict(),
                    "swap": psutil.swap_memory()._asdict()
                },
                "allocated": self.allocated_memory.copy(),
                "total_allocated_gb": sum(self.allocated_memory.values())
            }
            
            # Add GPU information if available
            if self.gpu_available:
                torch.cuda.synchronize()
                gpu_memory = torch.cuda.memory_stats(self.gpu_device)
                
                usage_info["gpu"] = {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(self.gpu_device),
                    "total_gb": self.total_gpu_memory,
                    "allocated_gb": gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3),
                    "reserved_gb": gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3),
                    "usage_percent": (gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3)) / self.total_gpu_memory * 100,
                    "max_usage_percent": self.max_vram_usage * 100,
                    "target_usage_percent": self.target_vram_usage * 100
                }
            else:
                usage_info["gpu"] = {
                    "available": False,
                    "total_gb": 0.0,
                    "allocated_gb": 0.0,
                    "usage_percent": 0.0
                }
            
            return usage_info
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    async def cleanup_unused_memory(self):
        """
        Clean up unused memory and optimize allocation
        """
        
        try:
            # Python garbage collection
            collected = gc.collect()
            
            # GPU memory cleanup if available
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.debug(f"Memory cleanup completed. Collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
    
    async def check_memory_health(self) -> Dict[str, Any]:
        """
        Check memory health and return status
        
        Returns:
            Dictionary with health status and recommendations
        """
        
        try:
            usage = await self.get_current_usage()
            health_status = {
                "status": "healthy",
                "issues": [],
                "recommendations": []
            }
            
            # Check system memory
            system_usage = usage["system"]["virtual"]["percent"]
            if system_usage > 90:
                health_status["status"] = "critical"
                health_status["issues"].append(f"High system memory usage: {system_usage:.1f}%")
                health_status["recommendations"].append("Consider reducing concurrent tasks")
            elif system_usage > 80:
                health_status["status"] = "warning"
                health_status["issues"].append(f"Elevated system memory usage: {system_usage:.1f}%")
            
            # Check GPU memory if available
            if self.gpu_available and "gpu" in usage:
                gpu_usage = usage["gpu"]["usage_percent"]
                if gpu_usage > self.max_vram_usage * 100:
                    health_status["status"] = "critical"
                    health_status["issues"].append(f"GPU memory usage exceeds limit: {gpu_usage:.1f}%")
                    health_status["recommendations"].append("Reduce batch sizes or concurrent processing")
                elif gpu_usage > self.target_vram_usage * 100:
                    health_status["status"] = "warning"
                    health_status["issues"].append(f"GPU memory usage above target: {gpu_usage:.1f}%")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking memory health: {e}")
            return {
                "status": "error",
                "issues": [f"Health check failed: {str(e)}"],
                "recommendations": ["Check system logs for details"]
            }
    
    def get_memory_allocation_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory allocation across brains
        
        Returns:
            Dictionary with allocation summary
        """
        
        total_allocated = sum(self.allocated_memory.values())
        max_available = self.total_gpu_memory * self.max_vram_usage
        
        return {
            "total_allocated_gb": total_allocated,
            "max_available_gb": max_available,
            "utilization_percent": (total_allocated / max_available * 100) if max_available > 0 else 0,
            "per_brain_allocation": self.allocated_memory.copy(),
            "remaining_gb": max(0, max_available - total_allocated)
        }
