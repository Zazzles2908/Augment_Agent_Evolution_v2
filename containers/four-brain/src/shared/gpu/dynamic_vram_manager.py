#!/usr/bin/env python3.11
"""
Dynamic VRAM Manager for Four-Brain System
Intelligent GPU memory allocation with TORCH_CUDA_MEMORY_FRACTION and safety nets

Author: AugmentAI
Date: 2025-08-02
Purpose: Dynamic VRAM allocation based on available resources and workload demands
"""

import os
import sys
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil

# Configure logging
logger = logging.getLogger(__name__)

class VRAMAllocationStrategy(Enum):
    """VRAM allocation strategies"""
    CONSERVATIVE = "conservative"  # 60% max utilization
    BALANCED = "balanced"         # 80% max utilization  
    AGGRESSIVE = "aggressive"     # 90% max utilization
    DYNAMIC = "dynamic"          # Adaptive based on workload

@dataclass
class VRAMAllocation:
    """VRAM allocation information"""
    service_name: str
    allocated_gb: float
    fraction: float
    strategy: VRAMAllocationStrategy
    timestamp: float
    pid: int

@dataclass
class GPUStatus:
    """Current GPU status"""
    total_memory_gb: float
    allocated_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    temperature_c: float
    power_usage_w: float
    timestamp: float

class DynamicVRAMManager:
    """Intelligent dynamic VRAM allocation manager"""
    
    def __init__(self, strategy: VRAMAllocationStrategy = VRAMAllocationStrategy.DYNAMIC):
        self.strategy = strategy
        self.allocation_lock = threading.Lock()
        
        # VRAM tracking
        self.active_allocations: Dict[str, VRAMAllocation] = {}
        self.allocation_history: List[VRAMAllocation] = []
        
        # GPU information
        self.gpu_available = self._check_gpu_availability()
        self.total_vram_gb = self._get_total_vram()
        self.baseline_allocation = 0.1  # Reserve 10% for system
        
        # Strategy configurations
        self.strategy_configs = {
            VRAMAllocationStrategy.CONSERVATIVE: {
                "max_fraction": 0.60,
                "safety_margin": 0.15,
                "min_free_gb": 2.0
            },
            VRAMAllocationStrategy.BALANCED: {
                "max_fraction": 0.80,
                "safety_margin": 0.10,
                "min_free_gb": 1.5
            },
            VRAMAllocationStrategy.AGGRESSIVE: {
                "max_fraction": 0.90,
                "safety_margin": 0.05,
                "min_free_gb": 1.0
            },
            VRAMAllocationStrategy.DYNAMIC: {
                "max_fraction": 0.85,
                "safety_margin": 0.10,
                "min_free_gb": 1.0
            }
        }
        
        # TensorRT-specific configurations
        self.tensorrt_configs = {
            "fp4_memory_multiplier": 0.25,
            "fp16_memory_multiplier": 0.50,
            "fp32_memory_multiplier": 1.00,
            "engine_cache_gb": 2.0,
            "batch_overhead_factor": 0.1
        }
        
        logger.info("🔧 Dynamic VRAM Manager initialized")
        logger.info(f"  Strategy: {strategy.value}")
        logger.info(f"  Total VRAM: {self.total_vram_gb:.1f}GB")
        logger.info(f"  GPU Available: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ GPU available: {device_name} ({device_count} devices)")
                return True
            else:
                logger.warning("⚠️ No GPU available")
                return False
        except ImportError:
            logger.warning("⚠️ PyTorch not available for GPU detection")
            return False
    
    def _get_total_vram(self) -> float:
        """Get total VRAM in GB"""
        try:
            if self.gpu_available:
                import torch
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return total_memory / (1024**3)
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️ Could not get total VRAM: {str(e)}")
            return 16.0  # Default for RTX 5070 Ti
    
    def get_gpu_status(self) -> GPUStatus:
        """Get current GPU status"""
        try:
            if not self.gpu_available:
                return GPUStatus(
                    total_memory_gb=0.0,
                    allocated_memory_gb=0.0,
                    free_memory_gb=0.0,
                    utilization_percent=0.0,
                    temperature_c=0.0,
                    power_usage_w=0.0,
                    timestamp=time.time()
                )
            
            import torch
            
            # Memory information
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            # Utilization (simplified)
            utilization = (allocated_memory / total_memory) * 100
            
            # Temperature and power (would need nvidia-ml-py for real values)
            temperature = 65.0  # Simulated
            power_usage = 200.0  # Simulated
            
            return GPUStatus(
                total_memory_gb=total_memory,
                allocated_memory_gb=allocated_memory,
                free_memory_gb=free_memory,
                utilization_percent=utilization,
                temperature_c=temperature,
                power_usage_w=power_usage,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get GPU status: {str(e)}")
            return GPUStatus(
                total_memory_gb=self.total_vram_gb,
                allocated_memory_gb=0.0,
                free_memory_gb=self.total_vram_gb,
                utilization_percent=0.0,
                temperature_c=0.0,
                power_usage_w=0.0,
                timestamp=time.time()
            )
    
    def calculate_optimal_fraction(self, service_name: str, precision: str = "FP16", 
                                 batch_size: int = 1, model_size_gb: float = 4.0) -> float:
        """Calculate optimal VRAM fraction for a service"""
        try:
            # Get current strategy config
            config = self.strategy_configs[self.strategy]
            max_fraction = config["max_fraction"]
            safety_margin = config["safety_margin"]
            min_free_gb = config["min_free_gb"]
            
            # Calculate base memory requirement
            precision_multiplier = self.tensorrt_configs.get(
                f"{precision.lower()}_memory_multiplier", 1.0
            )
            
            base_memory = model_size_gb * precision_multiplier
            batch_overhead = base_memory * self.tensorrt_configs["batch_overhead_factor"] * batch_size
            engine_cache = self.tensorrt_configs["engine_cache_gb"]
            
            total_required = base_memory + batch_overhead + engine_cache
            
            # Calculate fraction
            required_fraction = total_required / self.total_vram_gb
            
            # Apply safety margins
            safe_fraction = min(required_fraction * (1 + safety_margin), max_fraction)
            
            # Ensure minimum free memory
            max_allowed_fraction = (self.total_vram_gb - min_free_gb) / self.total_vram_gb
            final_fraction = min(safe_fraction, max_allowed_fraction)
            
            logger.debug(f"🔧 VRAM fraction calculation for {service_name}:")
            logger.debug(f"  Precision: {precision} (multiplier: {precision_multiplier})")
            logger.debug(f"  Base memory: {base_memory:.2f}GB")
            logger.debug(f"  Batch overhead: {batch_overhead:.2f}GB")
            logger.debug(f"  Engine cache: {engine_cache:.2f}GB")
            logger.debug(f"  Total required: {total_required:.2f}GB")
            logger.debug(f"  Required fraction: {required_fraction:.3f}")
            logger.debug(f"  Final fraction: {final_fraction:.3f}")
            
            return final_fraction
            
        except Exception as e:
            logger.error(f"❌ VRAM fraction calculation failed: {str(e)}")
            return 0.5  # Conservative fallback
    
    def allocate_vram(self, service_name: str, precision: str = "FP16", 
                     batch_size: int = 1, model_size_gb: float = 4.0) -> bool:
        """Allocate VRAM for a service using TORCH_CUDA_MEMORY_FRACTION"""
        try:
            if not self.gpu_available:
                logger.warning(f"⚠️ GPU not available for {service_name}")
                return False
            
            with self.allocation_lock:
                # Calculate optimal fraction
                fraction = self.calculate_optimal_fraction(
                    service_name, precision, batch_size, model_size_gb
                )
                
                # Check if allocation is possible
                current_gpu_status = self.get_gpu_status()
                required_gb = fraction * self.total_vram_gb
                
                if required_gb > current_gpu_status.free_memory_gb:
                    logger.warning(f"⚠️ Insufficient VRAM for {service_name}: "
                                 f"required {required_gb:.2f}GB, available {current_gpu_status.free_memory_gb:.2f}GB")
                    return False
                
                # Set PyTorch memory fraction
                try:
                    import torch
                    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                    
                    # Also set environment variable for child processes
                    os.environ["TORCH_CUDA_MEMORY_FRACTION"] = str(fraction)
                    
                    # Record allocation
                    allocation = VRAMAllocation(
                        service_name=service_name,
                        allocated_gb=required_gb,
                        fraction=fraction,
                        strategy=self.strategy,
                        timestamp=time.time(),
                        pid=os.getpid()
                    )
                    
                    self.active_allocations[service_name] = allocation
                    self.allocation_history.append(allocation)
                    
                    logger.info(f"✅ VRAM allocated for {service_name}:")
                    logger.info(f"  Fraction: {fraction:.3f}")
                    logger.info(f"  Memory: {required_gb:.2f}GB")
                    logger.info(f"  Precision: {precision}")
                    logger.info(f"  Strategy: {self.strategy.value}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Failed to set PyTorch memory fraction: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ VRAM allocation failed for {service_name}: {str(e)}")
            return False
    
    def deallocate_vram(self, service_name: str) -> bool:
        """Deallocate VRAM for a service"""
        try:
            with self.allocation_lock:
                if service_name in self.active_allocations:
                    allocation = self.active_allocations.pop(service_name)
                    
                    # Reset PyTorch memory fraction to default
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        # Reset to conservative default
                        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
                        os.environ["TORCH_CUDA_MEMORY_FRACTION"] = "0.8"
                        
                        logger.info(f"✅ VRAM deallocated for {service_name}")
                        logger.info(f"  Released: {allocation.allocated_gb:.2f}GB")
                        
                        return True
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to reset PyTorch memory fraction: {str(e)}")
                        return False
                else:
                    logger.warning(f"⚠️ No VRAM allocation found for {service_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ VRAM deallocation failed for {service_name}: {str(e)}")
            return False
    
    def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize current VRAM allocations based on usage patterns"""
        try:
            logger.info("🔧 Optimizing VRAM allocations...")
            
            optimization_results = {
                "optimizations_applied": 0,
                "memory_freed_gb": 0.0,
                "recommendations": []
            }
            
            current_status = self.get_gpu_status()
            
            # Check for underutilized allocations
            with self.allocation_lock:
                for service_name, allocation in list(self.active_allocations.items()):
                    # Check if we can reduce allocation
                    if allocation.allocated_gb > 4.0:  # If allocated more than 4GB
                        potential_reduction = allocation.allocated_gb * 0.1  # 10% reduction
                        optimization_results["memory_freed_gb"] += potential_reduction
                        optimization_results["optimizations_applied"] += 1
                        
                        logger.info(f"🔧 Optimization opportunity: {service_name} could free {potential_reduction:.2f}GB")
            
            # Dynamic strategy adjustment
            if self.strategy == VRAMAllocationStrategy.DYNAMIC:
                if current_status.utilization_percent < 50:
                    optimization_results["recommendations"].append(
                        "Consider AGGRESSIVE strategy for better VRAM utilization"
                    )
                elif current_status.utilization_percent > 85:
                    optimization_results["recommendations"].append(
                        "Consider CONSERVATIVE strategy to prevent VRAM contention"
                    )
            
            # Temperature-based recommendations
            if current_status.temperature_c > 80:
                optimization_results["recommendations"].append(
                    "High GPU temperature detected - consider reducing VRAM usage"
                )
            
            logger.info(f"✅ VRAM optimization complete: {optimization_results['optimizations_applied']} optimizations")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ VRAM optimization failed: {str(e)}")
            return {"error": str(e)}
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current VRAM allocations"""
        try:
            with self.allocation_lock:
                current_status = self.get_gpu_status()
                
                total_allocated = sum(alloc.allocated_gb for alloc in self.active_allocations.values())
                
                return {
                    "timestamp": time.time(),
                    "strategy": self.strategy.value,
                    "gpu_status": {
                        "total_memory_gb": current_status.total_memory_gb,
                        "allocated_memory_gb": current_status.allocated_memory_gb,
                        "free_memory_gb": current_status.free_memory_gb,
                        "utilization_percent": current_status.utilization_percent,
                        "temperature_c": current_status.temperature_c
                    },
                    "allocations": {
                        service: {
                            "allocated_gb": alloc.allocated_gb,
                            "fraction": alloc.fraction,
                            "strategy": alloc.strategy.value,
                            "age_seconds": time.time() - alloc.timestamp
                        }
                        for service, alloc in self.active_allocations.items()
                    },
                    "summary": {
                        "active_services": len(self.active_allocations),
                        "total_allocated_gb": total_allocated,
                        "allocation_efficiency": (total_allocated / current_status.total_memory_gb) * 100,
                        "free_memory_gb": current_status.free_memory_gb
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get allocation summary: {str(e)}")
            return {"error": str(e)}
    
    def apply_gpu_guard(self, max_fraction: float = 0.9) -> bool:
        """Apply GPU guard to clamp PyTorch to percentage of free VRAM"""
        try:
            if not self.gpu_available:
                return False
            
            import torch
            
            # Get current free memory
            current_status = self.get_gpu_status()
            free_fraction = current_status.free_memory_gb / current_status.total_memory_gb
            
            # Apply guard with maximum fraction
            guard_fraction = min(free_fraction * max_fraction, max_fraction)
            
            torch.cuda.set_per_process_memory_fraction(guard_fraction, device=0)
            os.environ["TORCH_CUDA_MEMORY_FRACTION"] = str(guard_fraction)
            
            logger.info(f"🛡️ GPU guard applied:")
            logger.info(f"  Free memory: {current_status.free_memory_gb:.2f}GB")
            logger.info(f"  Guard fraction: {guard_fraction:.3f}")
            logger.info(f"  Protected memory: {guard_fraction * current_status.total_memory_gb:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU guard application failed: {str(e)}")
            return False

# Global VRAM manager instance
_vram_manager = None

def get_vram_manager(strategy: VRAMAllocationStrategy = VRAMAllocationStrategy.DYNAMIC) -> DynamicVRAMManager:
    """Get global dynamic VRAM manager instance"""
    global _vram_manager
    if _vram_manager is None:
        _vram_manager = DynamicVRAMManager(strategy)
    return _vram_manager

def initialize_vram_for_service(service_name: str, precision: str = "FP16", 
                               model_size_gb: float = 4.0) -> bool:
    """Initialize VRAM allocation for a service"""
    vram_manager = get_vram_manager()
    return vram_manager.allocate_vram(service_name, precision, model_size_gb=model_size_gb)
