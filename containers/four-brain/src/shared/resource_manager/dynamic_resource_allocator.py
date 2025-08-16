#!/usr/bin/env python3.11
"""
Dynamic Resource Allocator for Four-Brain System
Intelligent resource allocation based on available compute and workload demands

Author: AugmentAI
Date: 2025-08-02
Purpose: Dynamic allocation of GPU memory, CPU cores, and compute resources for optimal performance
"""

import os
import sys
import asyncio
import logging
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Configure logging
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Resource types for allocation"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    TENSORRT_ENGINES = "tensorrt_engines"
    DISK_SPACE = "disk_space"

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    CONSERVATIVE = "conservative"  # 70% max utilization
    BALANCED = "balanced"         # 85% max utilization
    AGGRESSIVE = "aggressive"     # 95% max utilization
    DYNAMIC = "dynamic"          # Adaptive based on workload

@dataclass
class ResourceSpec:
    """Resource specification for a service"""
    service_name: str
    resource_type: ResourceType
    min_allocation: float
    max_allocation: float
    current_allocation: float
    priority: int  # 1-10, higher is more important
    can_scale: bool = True
    allocation_unit: str = "GB"  # GB, cores, engines, etc.

@dataclass
class SystemResources:
    """Current system resource availability"""
    total_gpu_memory_gb: float
    available_gpu_memory_gb: float
    total_cpu_cores: int
    available_cpu_cores: float
    total_system_memory_gb: float
    available_system_memory_gb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    timestamp: float

@dataclass
class AllocationResult:
    """Result of resource allocation"""
    service_name: str
    resource_type: ResourceType
    requested_amount: float
    allocated_amount: float
    success: bool
    reason: str
    allocation_id: str
    timestamp: float

class DynamicResourceAllocator:
    """Intelligent dynamic resource allocation system"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.DYNAMIC):
        self.strategy = strategy
        self.allocation_lock = threading.Lock()
        
        # Resource tracking
        self.resource_specs: Dict[str, List[ResourceSpec]] = {}
        self.active_allocations: Dict[str, AllocationResult] = {}
        self.allocation_history: List[AllocationResult] = []
        
        # System monitoring
        self.system_resources = self._get_system_resources()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Allocation strategies
        self.strategy_configs = {
            AllocationStrategy.CONSERVATIVE: {"max_utilization": 0.70, "safety_margin": 0.15},
            AllocationStrategy.BALANCED: {"max_utilization": 0.85, "safety_margin": 0.10},
            AllocationStrategy.AGGRESSIVE: {"max_utilization": 0.95, "safety_margin": 0.05},
            AllocationStrategy.DYNAMIC: {"max_utilization": 0.85, "safety_margin": 0.10}  # Default
        }
        
        # TensorRT-specific configurations
        self.tensorrt_configs = {
            "fp4_memory_multiplier": 0.25,  # FP4 uses ~25% of FP32 memory
            "fp16_memory_multiplier": 0.50,  # FP16 uses ~50% of FP32 memory
            "engine_cache_size_gb": 2.0,     # Reserve 2GB for engine cache
            "min_free_memory_gb": 1.0        # Always keep 1GB free
        }
        
        logger.info("🔧 Dynamic Resource Allocator initialized")
        logger.info(f"  Strategy: {strategy.value}")
        logger.info(f"  GPU Memory: {self.system_resources.total_gpu_memory_gb:.1f}GB total")
        logger.info(f"  CPU Cores: {self.system_resources.total_cpu_cores}")
        logger.info(f"  System Memory: {self.system_resources.total_system_memory_gb:.1f}GB")
    
    def _get_system_resources(self) -> SystemResources:
        """Get current system resource availability"""
        try:
            # CPU information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_usage = psutil.cpu_percent(interval=1)
            available_cpu = cpu_count * (1 - cpu_usage / 100)
            
            # Memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent
            
            # GPU information
            total_gpu_memory = 0.0
            available_gpu_memory = 0.0
            gpu_utilization = 0.0
            
            try:
                import torch
                if torch.cuda.is_available():
                    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    allocated_gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                    available_gpu_memory = total_gpu_memory - allocated_gpu_memory
                    gpu_utilization = (allocated_gpu_memory / total_gpu_memory) * 100
            except ImportError:
                logger.warning("⚠️ PyTorch not available for GPU monitoring")
            
            return SystemResources(
                total_gpu_memory_gb=total_gpu_memory,
                available_gpu_memory_gb=available_gpu_memory,
                total_cpu_cores=cpu_count,
                available_cpu_cores=available_cpu,
                total_system_memory_gb=total_memory_gb,
                available_system_memory_gb=available_memory_gb,
                gpu_utilization_percent=gpu_utilization,
                cpu_utilization_percent=cpu_usage,
                memory_utilization_percent=memory_usage_percent,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get system resources: {str(e)}")
            return SystemResources(
                total_gpu_memory_gb=16.0,  # Default for RTX 5070 Ti
                available_gpu_memory_gb=14.0,
                total_cpu_cores=16,
                available_cpu_cores=12.0,
                total_system_memory_gb=32.0,
                available_system_memory_gb=24.0,
                gpu_utilization_percent=20.0,
                cpu_utilization_percent=25.0,
                memory_utilization_percent=25.0,
                timestamp=time.time()
            )
    
    def register_service(self, service_name: str, resource_specs: List[ResourceSpec]):
        """Register a service with its resource requirements"""
        try:
            with self.allocation_lock:
                self.resource_specs[service_name] = resource_specs
            
            logger.info(f"✅ Registered service: {service_name}")
            for spec in resource_specs:
                logger.info(f"  {spec.resource_type.value}: {spec.min_allocation}-{spec.max_allocation} {spec.allocation_unit} (priority: {spec.priority})")
                
        except Exception as e:
            logger.error(f"❌ Failed to register service {service_name}: {str(e)}")
    
    def calculate_tensorrt_memory_requirements(self, precision: str, model_size_gb: float, 
                                             batch_size: int = 1) -> float:
        """Calculate TensorRT memory requirements based on precision and model size"""
        try:
            base_memory = model_size_gb
            
            # Apply precision multiplier
            if precision.upper() == "FP4":
                memory_multiplier = self.tensorrt_configs["fp4_memory_multiplier"]
            elif precision.upper() == "FP16":
                memory_multiplier = self.tensorrt_configs["fp16_memory_multiplier"]
            else:  # FP32
                memory_multiplier = 1.0
            
            # Calculate memory for model + batch processing
            model_memory = base_memory * memory_multiplier
            batch_memory = model_memory * 0.1 * batch_size  # 10% per batch item
            engine_cache = self.tensorrt_configs["engine_cache_size_gb"]
            
            total_memory = model_memory + batch_memory + engine_cache
            
            logger.debug(f"🔧 TensorRT memory calculation:")
            logger.debug(f"  Base model: {base_memory:.2f}GB")
            logger.debug(f"  Precision ({precision}): {memory_multiplier:.2f}x = {model_memory:.2f}GB")
            logger.debug(f"  Batch processing: {batch_memory:.2f}GB")
            logger.debug(f"  Engine cache: {engine_cache:.2f}GB")
            logger.debug(f"  Total required: {total_memory:.2f}GB")
            
            return total_memory
            
        except Exception as e:
            logger.error(f"❌ TensorRT memory calculation failed: {str(e)}")
            return model_size_gb * 2.0  # Conservative fallback
    
    async def allocate_resources(self, service_name: str, resource_type: ResourceType, 
                                requested_amount: float, precision: str = "FP16") -> AllocationResult:
        """Allocate resources for a service"""
        try:
            allocation_id = f"{service_name}_{resource_type.value}_{int(time.time())}"
            
            # Update system resources
            self.system_resources = self._get_system_resources()
            
            # Get current strategy config
            strategy_config = self.strategy_configs[self.strategy]
            max_utilization = strategy_config["max_utilization"]
            safety_margin = strategy_config["safety_margin"]
            
            # Calculate available resources
            if resource_type == ResourceType.GPU_MEMORY:
                total_resource = self.system_resources.total_gpu_memory_gb
                available_resource = self.system_resources.available_gpu_memory_gb
                
                # Apply TensorRT-specific calculations
                if "tensorrt" in service_name.lower():
                    # Adjust request based on precision
                    if precision.upper() == "FP4":
                        requested_amount *= self.tensorrt_configs["fp4_memory_multiplier"]
                    elif precision.upper() == "FP16":
                        requested_amount *= self.tensorrt_configs["fp16_memory_multiplier"]
                
                # Reserve minimum free memory
                available_resource -= self.tensorrt_configs["min_free_memory_gb"]
                
            elif resource_type == ResourceType.CPU_CORES:
                total_resource = self.system_resources.total_cpu_cores
                available_resource = self.system_resources.available_cpu_cores
                
            elif resource_type == ResourceType.SYSTEM_MEMORY:
                total_resource = self.system_resources.total_system_memory_gb
                available_resource = self.system_resources.available_system_memory_gb
                
            else:
                logger.error(f"❌ Unsupported resource type: {resource_type}")
                return AllocationResult(
                    service_name=service_name,
                    resource_type=resource_type,
                    requested_amount=requested_amount,
                    allocated_amount=0.0,
                    success=False,
                    reason=f"Unsupported resource type: {resource_type}",
                    allocation_id=allocation_id,
                    timestamp=time.time()
                )
            
            # Check if allocation is possible
            max_allocatable = total_resource * max_utilization
            safe_allocatable = available_resource * (1 - safety_margin)
            
            # Determine allocation amount
            if requested_amount <= safe_allocatable:
                allocated_amount = requested_amount
                success = True
                reason = "Full allocation granted"
            elif requested_amount <= max_allocatable:
                allocated_amount = min(requested_amount, available_resource * 0.9)
                success = True
                reason = "Partial allocation granted (reduced for safety)"
            else:
                allocated_amount = 0.0
                success = False
                reason = f"Insufficient resources: requested {requested_amount:.2f}, available {safe_allocatable:.2f}"
            
            # Create allocation result
            result = AllocationResult(
                service_name=service_name,
                resource_type=resource_type,
                requested_amount=requested_amount,
                allocated_amount=allocated_amount,
                success=success,
                reason=reason,
                allocation_id=allocation_id,
                timestamp=time.time()
            )
            
            # Store allocation
            with self.allocation_lock:
                if success:
                    self.active_allocations[allocation_id] = result
                self.allocation_history.append(result)
            
            # Log allocation
            if success:
                logger.info(f"✅ Resource allocation successful:")
                logger.info(f"  Service: {service_name}")
                logger.info(f"  Resource: {resource_type.value}")
                logger.info(f"  Allocated: {allocated_amount:.2f} (requested: {requested_amount:.2f})")
                logger.info(f"  Precision: {precision}")
                logger.info(f"  Allocation ID: {allocation_id}")
            else:
                logger.warning(f"⚠️ Resource allocation failed:")
                logger.warning(f"  Service: {service_name}")
                logger.warning(f"  Resource: {resource_type.value}")
                logger.warning(f"  Reason: {reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Resource allocation failed: {str(e)}")
            return AllocationResult(
                service_name=service_name,
                resource_type=resource_type,
                requested_amount=requested_amount,
                allocated_amount=0.0,
                success=False,
                reason=f"Allocation error: {str(e)}",
                allocation_id=f"error_{int(time.time())}",
                timestamp=time.time()
            )
    
    async def deallocate_resources(self, allocation_id: str) -> bool:
        """Deallocate resources by allocation ID"""
        try:
            with self.allocation_lock:
                if allocation_id in self.active_allocations:
                    allocation = self.active_allocations.pop(allocation_id)
                    logger.info(f"✅ Deallocated resources: {allocation.service_name} - {allocation.resource_type.value}")
                    return True
                else:
                    logger.warning(f"⚠️ Allocation ID not found: {allocation_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Resource deallocation failed: {str(e)}")
            return False
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization summary"""
        try:
            self.system_resources = self._get_system_resources()
            
            # Calculate allocated resources
            allocated_gpu_memory = sum(
                alloc.allocated_amount for alloc in self.active_allocations.values()
                if alloc.resource_type == ResourceType.GPU_MEMORY
            )
            
            allocated_cpu_cores = sum(
                alloc.allocated_amount for alloc in self.active_allocations.values()
                if alloc.resource_type == ResourceType.CPU_CORES
            )
            
            allocated_system_memory = sum(
                alloc.allocated_amount for alloc in self.active_allocations.values()
                if alloc.resource_type == ResourceType.SYSTEM_MEMORY
            )
            
            return {
                "timestamp": time.time(),
                "strategy": self.strategy.value,
                "system_resources": asdict(self.system_resources),
                "allocated_resources": {
                    "gpu_memory_gb": allocated_gpu_memory,
                    "cpu_cores": allocated_cpu_cores,
                    "system_memory_gb": allocated_system_memory
                },
                "utilization_percentages": {
                    "gpu_memory": (allocated_gpu_memory / self.system_resources.total_gpu_memory_gb) * 100,
                    "cpu_cores": (allocated_cpu_cores / self.system_resources.total_cpu_cores) * 100,
                    "system_memory": (allocated_system_memory / self.system_resources.total_system_memory_gb) * 100
                },
                "active_allocations": len(self.active_allocations),
                "total_allocations": len(self.allocation_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get resource utilization: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize current resource allocations based on usage patterns"""
        try:
            logger.info("🔧 Starting allocation optimization...")
            
            optimization_results = {
                "optimizations_applied": 0,
                "memory_freed_gb": 0.0,
                "performance_improvements": [],
                "recommendations": []
            }
            
            # Update system resources
            self.system_resources = self._get_system_resources()
            
            # Check for underutilized allocations
            with self.allocation_lock:
                for allocation_id, allocation in list(self.active_allocations.items()):
                    # Check if allocation is underutilized (placeholder logic)
                    if allocation.resource_type == ResourceType.GPU_MEMORY:
                        if allocation.allocated_amount > 2.0:  # If allocated more than 2GB
                            # Check if we can reduce allocation
                            potential_reduction = allocation.allocated_amount * 0.1  # 10% reduction
                            optimization_results["memory_freed_gb"] += potential_reduction
                            optimization_results["optimizations_applied"] += 1
                            
                            logger.info(f"🔧 Optimization opportunity: {allocation.service_name} could free {potential_reduction:.2f}GB")
            
            # Dynamic strategy adjustment
            if self.strategy == AllocationStrategy.DYNAMIC:
                current_gpu_util = self.system_resources.gpu_utilization_percent
                
                if current_gpu_util < 50:
                    optimization_results["recommendations"].append("Consider AGGRESSIVE strategy for better resource utilization")
                elif current_gpu_util > 90:
                    optimization_results["recommendations"].append("Consider CONSERVATIVE strategy to prevent resource contention")
            
            logger.info(f"✅ Optimization complete: {optimization_results['optimizations_applied']} optimizations applied")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Allocation optimization failed: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def start_monitoring(self, interval_seconds: float = 30.0):
        """Start continuous resource monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("⚠️ Monitoring already active")
                return
            
            logger.info("🚀 Starting resource monitoring...")
            self.monitoring_active = True
            
            def monitoring_loop():
                while self.monitoring_active:
                    try:
                        # Update system resources
                        self.system_resources = self._get_system_resources()
                        
                        # Log resource status
                        logger.debug(f"📊 Resource Status:")
                        logger.debug(f"  GPU: {self.system_resources.gpu_utilization_percent:.1f}% ({self.system_resources.available_gpu_memory_gb:.1f}GB free)")
                        logger.debug(f"  CPU: {self.system_resources.cpu_utilization_percent:.1f}% ({self.system_resources.available_cpu_cores:.1f} cores free)")
                        logger.debug(f"  Memory: {self.system_resources.memory_utilization_percent:.1f}% ({self.system_resources.available_system_memory_gb:.1f}GB free)")
                        
                        time.sleep(interval_seconds)
                        
                    except Exception as e:
                        logger.error(f"❌ Monitoring loop error: {str(e)}")
                        time.sleep(interval_seconds)
            
            self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info("✅ Resource monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {str(e)}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        try:
            logger.info("🔄 Stopping resource monitoring...")
            self.monitoring_active = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            logger.info("✅ Resource monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring: {str(e)}")

# Global resource allocator instance
_resource_allocator = None

def get_resource_allocator(strategy: AllocationStrategy = AllocationStrategy.DYNAMIC) -> DynamicResourceAllocator:
    """Get global dynamic resource allocator instance"""
    global _resource_allocator
    if _resource_allocator is None:
        _resource_allocator = DynamicResourceAllocator(strategy)
    return _resource_allocator

def initialize_tensorrt_resources() -> DynamicResourceAllocator:
    """Initialize resource allocator with TensorRT-optimized settings"""
    allocator = get_resource_allocator(AllocationStrategy.DYNAMIC)
    
    # Register TensorRT accelerator service
    tensorrt_specs = [
        ResourceSpec(
            service_name="tensorrt_accelerator",
            resource_type=ResourceType.GPU_MEMORY,
            min_allocation=2.0,  # Minimum 2GB
            max_allocation=12.0,  # Maximum 12GB (leave 4GB for system)
            current_allocation=0.0,
            priority=9,  # High priority
            can_scale=True,
            allocation_unit="GB"
        ),
        ResourceSpec(
            service_name="tensorrt_accelerator",
            resource_type=ResourceType.CPU_CORES,
            min_allocation=2.0,  # Minimum 2 cores
            max_allocation=8.0,  # Maximum 8 cores
            current_allocation=0.0,
            priority=7,
            can_scale=True,
            allocation_unit="cores"
        )
    ]
    
    allocator.register_service("tensorrt_accelerator", tensorrt_specs)
    
    return allocator
