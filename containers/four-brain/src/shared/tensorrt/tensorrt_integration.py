#!/usr/bin/env python3.11
"""
Unified TensorRT Integration Module for Four-Brain System
Provides reusable TensorRT acceleration components with unified API

Author: Zazzles's Agent
Date: 2025-08-02
Purpose: Unified TensorRT FP4 acceleration for all Brain services
"""

import os
import sys
import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class BrainType(Enum):
    """Brain service types"""
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    DOCLING = "docling"
    INTELLIGENCE = "intelligence"

@dataclass
class TensorRTConfig:
    """TensorRT configuration for Brain services"""
    brain_type: BrainType
    engine_name: str
    max_batch_size: int
    max_sequence_length: Optional[int] = None
    max_image_size: Optional[int] = None
    fp4_enabled: bool = True
    workspace_mb: int = 2048
    optimization_level: int = 5
    enable_timing_cache: bool = True
    fallback_enabled: bool = True

@dataclass
class AccelerationMetrics:
    """Acceleration performance metrics"""
    total_inferences: int = 0
    tensorrt_inferences: int = 0
    fallback_inferences: int = 0
    avg_tensorrt_time: float = 0.0
    avg_fallback_time: float = 0.0
    speedup_factor: float = 1.0
    memory_usage_mb: float = 0.0
    engine_build_time: float = 0.0

class TensorRTIntegrationManager:
    """Unified TensorRT integration manager for all Brain services"""
    
    def __init__(self, cache_dir: str = "/workspace/models"):
        self.cache_dir = Path(cache_dir)
        self.tensorrt_available = self._check_tensorrt_availability()
        self.accelerators = {}
        self.configs = {}
        self.global_metrics = {}
        self.lock = threading.Lock()
        
        # Global configuration
        self.engines_dir = self.cache_dir / "tensorrt" / "engines"
        self.onnx_dir = self.cache_dir / "onnx"
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 TensorRT Integration Manager initialized")
    
    def _check_tensorrt_availability(self) -> bool:
        """Check TensorRT availability"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger.info(f"✅ TensorRT {trt.__version__} available globally")
            logger.info("✅ PyCUDA available globally")
            return True
        except ImportError as e:
            logger.warning(f"⚠️ TensorRT not available globally: {str(e)}")
            return False
    
    def register_brain_service(self, brain_type: BrainType, config: TensorRTConfig) -> bool:
        """Register a Brain service for TensorRT acceleration"""
        try:
            with self.lock:
                service_key = brain_type.value
                
                # Store configuration
                self.configs[service_key] = config
                
                # Initialize metrics
                self.global_metrics[service_key] = AccelerationMetrics()
                
                logger.info(f"✅ Registered {brain_type.value} service for TensorRT acceleration")
                logger.info(f"  Engine: {config.engine_name}")
                logger.info(f"  Batch size: {config.max_batch_size}")
                logger.info(f"  FP4 enabled: {config.fp4_enabled}")
                logger.info(f"  Workspace: {config.workspace_mb}MB")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to register {brain_type.value} service: {str(e)}")
            return False
    
    def get_brain_accelerator(self, brain_type: BrainType):
        """Get or create TensorRT accelerator for Brain service"""
        service_key = brain_type.value
        
        with self.lock:
            if service_key not in self.accelerators:
                if service_key not in self.configs:
                    logger.error(f"❌ {brain_type.value} service not registered")
                    return None
                
                config = self.configs[service_key]
                
                # Create appropriate accelerator based on brain type
                if brain_type == BrainType.EMBEDDING:
                    from ...brains.embedding_service.core.tensorrt_accelerator import get_brain1_accelerator
                    accelerator = get_brain1_accelerator(str(self.cache_dir))
                elif brain_type == BrainType.RERANKER:
                    from ...brains.reranker_service.core.tensorrt_accelerator import get_brain2_accelerator
                    accelerator = get_brain2_accelerator(str(self.cache_dir))
                elif brain_type == BrainType.DOCLING:
                    from ...brains.document_processor.core.tensorrt_accelerator import get_brain4_accelerator
                    accelerator = get_brain4_accelerator(str(self.cache_dir))
                else:
                    logger.error(f"❌ Unsupported brain type: {brain_type.value}")
                    return None
                
                self.accelerators[service_key] = accelerator
                logger.info(f"✅ Created TensorRT accelerator for {brain_type.value}")
            
            return self.accelerators[service_key]
    
    def get_unified_metrics(self) -> Dict[str, Any]:
        """Get unified metrics across all Brain services"""
        unified_metrics = {
            "tensorrt_available": self.tensorrt_available,
            "registered_services": list(self.configs.keys()),
            "total_services": len(self.configs),
            "global_stats": {
                "total_inferences": 0,
                "total_tensorrt_inferences": 0,
                "total_fallback_inferences": 0,
                "average_speedup": 0.0,
                "total_memory_usage_mb": 0.0
            },
            "service_metrics": {}
        }
        
        total_speedup = 0.0
        active_services = 0
        
        for service_key in self.configs.keys():
            accelerator = self.accelerators.get(service_key)
            if accelerator and hasattr(accelerator, 'get_acceleration_metrics'):
                try:
                    metrics = accelerator.get_acceleration_metrics()
                    unified_metrics["service_metrics"][service_key] = metrics
                    
                    # Aggregate global stats
                    unified_metrics["global_stats"]["total_inferences"] += metrics.get("total_inferences", 0)
                    unified_metrics["global_stats"]["total_tensorrt_inferences"] += metrics.get("tensorrt_inferences", 0)
                    unified_metrics["global_stats"]["total_fallback_inferences"] += metrics.get("fallback_inferences", 0)
                    
                    speedup = metrics.get("speedup_factor", 1.0)
                    if speedup > 1.0:
                        total_speedup += speedup
                        active_services += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error getting metrics for {service_key}: {str(e)}")
                    unified_metrics["service_metrics"][service_key] = {"error": str(e)}
        
        # Calculate average speedup
        if active_services > 0:
            unified_metrics["global_stats"]["average_speedup"] = total_speedup / active_services
        
        return unified_metrics
    
    def build_all_engines(self) -> Dict[str, bool]:
        """Build TensorRT engines for all registered services"""
        build_results = {}
        
        logger.info("🔨 Building TensorRT engines for all registered services...")
        
        for service_key, config in self.configs.items():
            logger.info(f"\n📋 Building engine for {service_key}...")
            
            accelerator = self.get_brain_accelerator(BrainType(service_key))
            if accelerator and hasattr(accelerator, 'build_tensorrt_engine'):
                try:
                    # Use dummy model path for testing
                    model_path = str(self.onnx_dir / f"{config.engine_name}.onnx")
                    success = accelerator.build_tensorrt_engine(model_path)
                    build_results[service_key] = success
                    
                    if success:
                        logger.info(f"✅ Engine built successfully for {service_key}")
                    else:
                        logger.error(f"❌ Engine build failed for {service_key}")
                        
                except Exception as e:
                    logger.error(f"❌ Engine build error for {service_key}: {str(e)}")
                    build_results[service_key] = False
            else:
                logger.warning(f"⚠️ No engine builder available for {service_key}")
                build_results[service_key] = False
        
        successful_builds = sum(1 for success in build_results.values() if success)
        total_builds = len(build_results)
        
        logger.info(f"\n🎯 Engine build summary: {successful_builds}/{total_builds} successful")
        
        return build_results
    
    def validate_all_accelerators(self) -> Dict[str, bool]:
        """Validate all TensorRT accelerators"""
        validation_results = {}
        
        logger.info("🔍 Validating all TensorRT accelerators...")
        
        for service_key in self.configs.keys():
            accelerator = self.get_brain_accelerator(BrainType(service_key))
            if accelerator:
                try:
                    # Check if accelerator is properly initialized
                    metrics = accelerator.get_acceleration_metrics()
                    engine_loaded = metrics.get("engine_loaded", False)
                    tensorrt_available = metrics.get("tensorrt_available", False)
                    
                    validation_results[service_key] = tensorrt_available and engine_loaded
                    
                    if validation_results[service_key]:
                        logger.info(f"✅ {service_key} accelerator validated")
                    else:
                        logger.warning(f"⚠️ {service_key} accelerator validation failed")
                        
                except Exception as e:
                    logger.error(f"❌ {service_key} accelerator validation error: {str(e)}")
                    validation_results[service_key] = False
            else:
                logger.error(f"❌ No accelerator found for {service_key}")
                validation_results[service_key] = False
        
        return validation_results
    
    def get_memory_allocation_strategy(self) -> Dict[str, float]:
        """Get GPU memory allocation strategy for all Brain services"""
        # RTX 5070 Ti 16GB allocation strategy
        allocation_strategy = {
            "brain1_embedding": 0.25,    # 25% = 4GB
            "brain2_reranker": 0.20,     # 20% = 3.2GB  
            "brain4_docling": 0.15,      # 15% = 2.4GB
            "brain3_intelligence": 0.40  # 40% = 6.4GB (external)
        }
        
        # Adjust based on registered services
        registered_services = list(self.configs.keys())
        active_allocation = {}
        
        for service in registered_services:
            if service in allocation_strategy:
                active_allocation[service] = allocation_strategy[service]
        
        # Redistribute if some services are not registered
        total_allocated = sum(active_allocation.values())
        if total_allocated < 1.0 and active_allocation:
            # Redistribute remaining memory proportionally
            remaining = 1.0 - total_allocated
            for service in active_allocation:
                active_allocation[service] += remaining / len(active_allocation)
        
        return active_allocation
    
    def optimize_memory_allocation(self) -> bool:
        """Optimize GPU memory allocation across all Brain services"""
        try:
            allocation_strategy = self.get_memory_allocation_strategy()
            
            logger.info("🧠 Optimizing GPU memory allocation...")
            logger.info("📊 Memory allocation strategy:")
            
            total_memory_gb = 16.0  # RTX 5070 Ti
            for service, fraction in allocation_strategy.items():
                memory_gb = total_memory_gb * fraction
                logger.info(f"  {service}: {fraction*100:.0f}% ({memory_gb:.1f}GB)")
            
            # Apply allocation to accelerators
            for service_key, fraction in allocation_strategy.items():
                if service_key in self.accelerators:
                    accelerator = self.accelerators[service_key]
                    if hasattr(accelerator, 'set_memory_allocation'):
                        accelerator.set_memory_allocation(fraction)
            
            logger.info("✅ GPU memory allocation optimized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory allocation optimization failed: {str(e)}")
            return False
    
    async def shutdown_all_accelerators(self):
        """Shutdown all TensorRT accelerators"""
        logger.info("🔄 Shutting down all TensorRT accelerators...")
        
        # Get final metrics
        final_metrics = self.get_unified_metrics()
        logger.info(f"📊 Final metrics: {final_metrics['global_stats']}")
        
        # Cleanup accelerators
        with self.lock:
            for service_key, accelerator in self.accelerators.items():
                try:
                    if hasattr(accelerator, '__del__'):
                        accelerator.__del__()
                    logger.info(f"✅ {service_key} accelerator shutdown")
                except Exception as e:
                    logger.warning(f"⚠️ Error shutting down {service_key}: {str(e)}")
            
            self.accelerators.clear()
            self.configs.clear()
            self.global_metrics.clear()
        
        logger.info("✅ All TensorRT accelerators shutdown complete")

# Global integration manager instance
_tensorrt_manager = None

def get_tensorrt_manager(cache_dir: str = "/workspace/models") -> TensorRTIntegrationManager:
    """Get global TensorRT integration manager instance"""
    global _tensorrt_manager
    if _tensorrt_manager is None:
        _tensorrt_manager = TensorRTIntegrationManager(cache_dir)
    return _tensorrt_manager

def register_brain_for_tensorrt(brain_type: BrainType, config: TensorRTConfig) -> bool:
    """Convenience function to register Brain service for TensorRT acceleration"""
    manager = get_tensorrt_manager()
    return manager.register_brain_service(brain_type, config)

def get_brain_tensorrt_accelerator(brain_type: BrainType):
    """Convenience function to get Brain TensorRT accelerator"""
    manager = get_tensorrt_manager()
    return manager.get_brain_accelerator(brain_type)

# Pre-configured Brain service configurations
BRAIN_CONFIGS = {
    BrainType.EMBEDDING: TensorRTConfig(
        brain_type=BrainType.EMBEDDING,
        engine_name="brain1_embedding_fp4",
        max_batch_size=32,
        max_sequence_length=512,
        fp4_enabled=True,
        workspace_mb=2048,
        optimization_level=5
    ),
    BrainType.RERANKER: TensorRTConfig(
        brain_type=BrainType.RERANKER,
        engine_name="brain2_reranker_fp4",
        max_batch_size=64,
        max_sequence_length=256,
        fp4_enabled=True,
        workspace_mb=1536,
        optimization_level=5
    ),
    BrainType.DOCLING: TensorRTConfig(
        brain_type=BrainType.DOCLING,
        engine_name="brain4_docling_fp4",
        max_batch_size=16,
        max_image_size=224,
        fp4_enabled=True,
        workspace_mb=1024,
        optimization_level=4
    )
}

def initialize_all_brain_tensorrt() -> bool:
    """Initialize TensorRT acceleration for all Brain services"""
    logger.info("🚀 Initializing TensorRT acceleration for all Brain services...")
    
    manager = get_tensorrt_manager()
    success_count = 0
    
    for brain_type, config in BRAIN_CONFIGS.items():
        if manager.register_brain_service(brain_type, config):
            success_count += 1
        else:
            logger.error(f"❌ Failed to register {brain_type.value}")
    
    if success_count == len(BRAIN_CONFIGS):
        logger.info("✅ All Brain services registered for TensorRT acceleration")

        # Optimize memory allocation
        manager.optimize_memory_allocation()

        return True
    else:
        logger.error(f"❌ Only {success_count}/{len(BRAIN_CONFIGS)} services registered")
        return False

# CUDA 13.0 Enhanced TensorRT Integration - Consolidated from scripts/cuda13/tensorrt_engine_builder.py
class CUDA13TensorRTBuilder:
    """Enhanced TensorRT engine builder with CUDA 13.0 optimizations"""

    def __init__(self):
        self.cuda_version = None
        self.tensorrt_version = None
        self.blackwell_optimizations = False
        self._validate_cuda13_environment()

    def _validate_cuda13_environment(self):
        """Validate CUDA 13.0 environment for enhanced TensorRT building"""
        try:
            import torch
            import tensorrt as trt

            if torch.cuda.is_available():
                self.cuda_version = torch.version.cuda
                self.tensorrt_version = trt.__version__

                # Check for Blackwell GPU (sm_120)
                device_cap = torch.cuda.get_device_capability()
                if device_cap >= (12, 0):
                    self.blackwell_optimizations = True
                    logger.info("✅ CUDA 13.0 + Blackwell optimizations enabled")
                else:
                    logger.warning("⚠️ Non-Blackwell GPU detected, limited optimizations")
            else:
                raise RuntimeError("CUDA not available")

        except Exception as e:
            logger.error(f"❌ CUDA 13.0 validation failed: {e}")
            raise

    def create_optimized_config(self, builder) -> 'trt.IBuilderConfig':
        """Create TensorRT builder config with CUDA 13.0 optimizations"""
        try:
            import tensorrt as trt

            config = builder.create_builder_config()

            # Enable CUDA 13.0 specific optimizations
            if self.blackwell_optimizations:
                # Enhanced FP4 support for Blackwell
                if hasattr(trt.BuilderFlag, 'FP4'):
                    config.set_flag(trt.BuilderFlag.FP4)
                    logger.info("✅ FP4 quantization enabled for Blackwell")

                # Blackwell-specific optimizations
                if hasattr(trt.BuilderFlag, 'BLACKWELL_OPTIMIZATIONS'):
                    config.set_flag(trt.BuilderFlag.BLACKWELL_OPTIMIZATIONS)
                    logger.info("✅ Blackwell hardware optimizations enabled")

                # Enhanced sparsity support
                if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
                    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
                    logger.info("✅ Sparse weights optimization enabled")

            # CUDA 13.0 memory optimizations
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.DIRECT_IO)

            # Enhanced workspace size for CUDA 13.0
            workspace_size = 6 * 1024 * 1024 * 1024  # 6GB for Blackwell
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

            # Set optimization level for CUDA 13.0
            config.builder_optimization_level = 5  # Maximum optimization

            logger.info("✅ CUDA 13.0 optimized TensorRT config created")
            return config

        except Exception as e:
            logger.error(f"❌ Failed to create CUDA 13.0 optimized config: {e}")
            raise

    def build_engine_with_cuda13_optimizations(self, onnx_path: str, engine_path: str,
                                             max_batch_size: int = 1) -> bool:
        """Build TensorRT engine with CUDA 13.0 optimizations"""
        try:
            import tensorrt as trt

            logger.info(f"🔨 Building CUDA 13.0 optimized engine: {engine_path}")

            # Create TensorRT logger and builder
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)

            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("❌ Failed to parse ONNX model")
                    return False

            # Create CUDA 13.0 optimized config
            config = self.create_optimized_config(builder)

            # Build engine
            logger.info("🚀 Building TensorRT engine with CUDA 13.0 optimizations...")
            start_time = time.time()

            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("❌ Failed to build TensorRT engine")
                return False

            build_time = time.time() - start_time
            logger.info(f"✅ Engine built in {build_time:.2f} seconds")

            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            logger.info(f"✅ CUDA 13.0 optimized engine saved: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Engine building failed: {e}")
            return False
