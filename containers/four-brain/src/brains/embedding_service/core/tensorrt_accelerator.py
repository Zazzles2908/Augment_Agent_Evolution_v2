#!/usr/bin/env python3.11
"""
Brain-1 TensorRT FP4 Accelerator Integration
Integrates TensorRT FP4 acceleration into Qwen3-4B Embedding service with 8-bit fallback

Author: AugmentAI
Date: 2025-08-02
Purpose: Accelerate Brain-1 embedding generation with TensorRT FP4 optimization
"""

import os
import sys
import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class Brain1TensorRTAccelerator:
    """TensorRT FP4 accelerator for Brain-1 Qwen3-4B embedding service"""
    
    def __init__(self, cache_dir: str = "/workspace/models"):
        self.cache_dir = Path(cache_dir)
        self.tensorrt_available = self._check_tensorrt_availability()
        self.engine_cache = {}
        self.engine_lock = threading.Lock()
        
        # Configuration
        self.engine_path = self.cache_dir / "tensorrt" / "engines" / "brain1_embedding_fp4.engine"
        self.onnx_path = self.cache_dir / "onnx" / "brain1_embedding_fp4.onnx"
        self.max_batch_size = int(os.getenv("BRAIN1_TENSORRT_BATCH_SIZE", "32"))
        self.max_sequence_length = int(os.getenv("BRAIN1_TENSORRT_SEQ_LEN", "512"))
        self.fp4_enabled = os.getenv("BRAIN1_TENSORRT_FP4", "true").lower() == "true"
        self.fallback_to_8bit = os.getenv("BRAIN1_TENSORRT_FALLBACK", "true").lower() == "true"
        
        # Performance metrics
        self.inference_times = []
        self.acceleration_metrics = {
            "total_inferences": 0,
            "tensorrt_inferences": 0,
            "fallback_inferences": 0,
            "avg_tensorrt_time": 0.0,
            "avg_fallback_time": 0.0,
            "speedup_factor": 1.0
        }
        
        # Initialize TensorRT components
        self.runtime = None
        self.engine = None
        self.context = None
        self.cuda_context = None
        
        if self.tensorrt_available:
            self._initialize_tensorrt()
    
    def _check_tensorrt_availability(self) -> bool:
        """Check TensorRT availability"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger.info(f"✅ TensorRT {trt.__version__} available")
            logger.info("✅ PyCUDA available")
            return True
        except ImportError as e:
            logger.warning(f"⚠️ TensorRT not available: {str(e)}")
            return False
    
    def _initialize_tensorrt(self) -> bool:
        """Initialize TensorRT runtime and engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create runtime
            self.runtime = trt.Runtime(trt_logger)
            if self.runtime is None:
                logger.error("❌ Failed to create TensorRT runtime")
                return False
            
            # Load engine if it exists
            if self.engine_path.exists():
                logger.info(f"🔧 Loading TensorRT engine: {self.engine_path}")
                with open(self.engine_path, 'rb') as f:
                    engine_data = f.read()
                
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    logger.error("❌ Failed to deserialize TensorRT engine")
                    return False
                
                # Create execution context
                self.context = self.engine.create_execution_context()
                if self.context is None:
                    logger.error("❌ Failed to create execution context")
                    return False
                
                logger.info("✅ TensorRT engine loaded successfully")
                return True
            else:
                logger.warning(f"⚠️ TensorRT engine not found: {self.engine_path}")
                logger.info("💡 Engine will be built on first use")
                return False
                
        except Exception as e:
            logger.error(f"❌ TensorRT initialization failed: {str(e)}")
            return False
    
    def _allocate_cuda_memory(self) -> bool:
        """Allocate CUDA memory for inference"""
        try:
            import pycuda.driver as cuda
            
            # Calculate memory sizes
            input_size = self.max_batch_size * self.max_sequence_length * 4  # float32
            output_size = self.max_batch_size * 768 * 4  # embedding dimension
            
            # Allocate device memory
            self.d_input = cuda.mem_alloc(input_size)
            self.d_output = cuda.mem_alloc(output_size)
            
            # Allocate host memory
            self.h_input = cuda.pagelocked_empty((self.max_batch_size, self.max_sequence_length), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty((self.max_batch_size, 768), dtype=np.float32)
            
            logger.info("✅ CUDA memory allocated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ CUDA memory allocation failed: {str(e)}")
            return False
    
    def build_tensorrt_engine(self, model_path: str) -> bool:
        """Build TensorRT engine from ONNX model"""
        if not self.tensorrt_available:
            logger.error("❌ TensorRT not available for engine building")
            return False
        
        logger.info("🔧 Building TensorRT engine for Brain-1 embedding...")
        
        try:
            import tensorrt as trt
            
            # Create builder
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            if builder is None:
                logger.error("❌ Failed to create TensorRT builder")
                return False
            
            # Create network
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            if network is None:
                logger.error("❌ Failed to create TensorRT network")
                return False
            
            # Create ONNX parser
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            if parser is None:
                logger.error("❌ Failed to create ONNX parser")
                return False
            
            # Parse ONNX model - MUST BE REAL
            if not self.onnx_path.exists():
                logger.error(f"❌ PROCESSING FAILED: ONNX model not found: {self.onnx_path}")
                logger.info("🔍 Attempting to load real ONNX model...")
                if not self._load_real_onnx_model():
                    logger.error("❌ PROCESSING FAILED: No real ONNX model available")
                    return False
            
            with open(self.onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("❌ ONNX parsing failed")
                    return False
            
            # Create builder config
            config = builder.create_builder_config()
            if config is None:
                logger.error("❌ Failed to create builder config")
                return False
            
            # Configure FP4 precision
            if self.fp4_enabled:
                try:
                    # Enable FP4 if available
                    if hasattr(trt.BuilderFlag, 'FP4'):
                        config.set_flag(trt.BuilderFlag.FP4)
                        logger.info("✅ FP4 precision enabled")
                    else:
                        # Fallback to FP16
                        config.set_flag(trt.BuilderFlag.FP16)
                        logger.info("✅ FP16 precision enabled (FP4 fallback)")
                except:
                    logger.warning("⚠️ FP4/FP16 not supported, using FP32")
            
            # Set workspace size (2GB)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
            
            # Create optimization profile
            profile = builder.create_optimization_profile()
            
            # Configure input shapes for embedding model
            input_name = "input_ids"
            min_shape = (1, 1)
            opt_shape = (self.max_batch_size // 2, self.max_sequence_length // 2)
            max_shape = (self.max_batch_size, self.max_sequence_length)
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("🔨 Building TensorRT engine (this may take several minutes)...")
            start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("❌ Engine building failed")
                return False
            
            build_time = time.time() - start_time
            
            # Save engine
            self.engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"✅ TensorRT engine built successfully in {build_time:.1f}s")
            logger.info(f"📄 Engine saved: {self.engine_path}")
            
            # Initialize the engine
            return self._initialize_tensorrt()
            
        except Exception as e:
            logger.error(f"❌ Engine building failed: {str(e)}")
            return False
    
    def _load_real_onnx_model(self):
        """Load real ONNX model from Brain-1 service - NO FABRICATION"""
        try:
            # CRITICAL: Must use real ONNX model from actual Brain-1 service
            # Check if real ONNX model exists from Brain-1 export

            # Look for actual Brain-1 ONNX model
            potential_paths = [
                self.cache_dir / "brain1" / "qwen3-4b-embedding.onnx",
                self.cache_dir / "models" / "brain1_embedding.onnx",
                Path("/workspace/models/brain1/embedding.onnx")
            ]

            for model_path in potential_paths:
                if model_path.exists():
                    # Copy real model to expected location
                    self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(model_path, self.onnx_path)
                    logger.info(f"✅ Loaded real ONNX model from: {model_path}")
                    return True

            # If no real model found, report failure clearly
            logger.error("❌ PROCESSING FAILED: No real ONNX model found for Brain-1")
            logger.error(f"❌ Searched paths: {[str(p) for p in potential_paths]}")
            logger.error("❌ TensorRT acceleration requires real exported ONNX model")
            return False

        except Exception as e:
            logger.error(f"❌ PROCESSING FAILED: Real ONNX model loading failed: {str(e)}")
            return False
    
    async def accelerated_embedding(self, input_ids: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using TensorRT acceleration"""
        if not self.tensorrt_available or self.engine is None or self.context is None:
            return None
        
        try:
            import pycuda.driver as cuda
            
            start_time = time.time()
            
            # Prepare input
            batch_size, seq_len = input_ids.shape
            if batch_size > self.max_batch_size or seq_len > self.max_sequence_length:
                logger.warning(f"⚠️ Input size ({batch_size}, {seq_len}) exceeds limits")
                return None
            
            # Allocate memory if not done
            if not hasattr(self, 'd_input'):
                if not self._allocate_cuda_memory():
                    return None
            
            # Copy input to host memory
            self.h_input[:batch_size, :seq_len] = input_ids.astype(np.float32)
            
            # Copy to device
            cuda.memcpy_htod(self.d_input, self.h_input)
            
            # Set input shape
            self.context.set_binding_shape(0, (batch_size, seq_len))
            
            # Run inference
            bindings = [int(self.d_input), int(self.d_output)]
            success = self.context.execute_async_v2(bindings, cuda.Stream())
            
            if not success:
                logger.error("❌ TensorRT inference failed")
                return None
            
            # Copy result back
            cuda.memcpy_dtoh(self.h_output, self.d_output)
            
            # Extract result
            result = self.h_output[:batch_size].copy()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Update metrics
            self.acceleration_metrics["total_inferences"] += 1
            self.acceleration_metrics["tensorrt_inferences"] += 1
            self.acceleration_metrics["avg_tensorrt_time"] = np.mean(self.inference_times[-100:])
            
            logger.debug(f"✅ TensorRT inference completed in {inference_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TensorRT inference failed: {str(e)}")
            return None
    
    async def generate_embedding_with_acceleration(self, text: str, model, tokenizer, 
                                                  truncate_to_2000: bool = True) -> Optional[np.ndarray]:
        """Generate embedding with TensorRT acceleration and fallback"""
        try:
            # Tokenize input
            if hasattr(tokenizer, 'encode'):
                input_ids = tokenizer.encode(text, return_tensors='np', max_length=self.max_sequence_length, truncation=True)
            else:
                # PROCESSING FAILED: No real tokenizer available
                logger.error("❌ PROCESSING FAILED: No real tokenizer available for Brain-1")
                return None
            
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            
            # Try TensorRT acceleration first
            if self.tensorrt_available and self.engine is not None:
                result = await self.accelerated_embedding(input_ids)
                if result is not None:
                    # Apply MRL truncation if requested
                    if truncate_to_2000 and result.shape[-1] > 2000:
                        result = result[:, :2000]
                    
                    return result[0] if result.shape[0] == 1 else result
            
            # Fallback to original model
            if self.fallback_to_8bit:
                logger.debug("🔄 Using fallback to original model")
                start_time = time.time()
                
                # Use original model inference
                if hasattr(model, 'encode'):
                    embedding = model.encode(text, convert_to_numpy=True)
                else:
                    # PROCESSING FAILED: No real embedding model available
                    logger.error("❌ PROCESSING FAILED: No real embedding model available")
                    return None
                
                fallback_time = time.time() - start_time
                
                # Update metrics
                self.acceleration_metrics["total_inferences"] += 1
                self.acceleration_metrics["fallback_inferences"] += 1
                self.acceleration_metrics["avg_fallback_time"] = (
                    self.acceleration_metrics["avg_fallback_time"] * 0.9 + fallback_time * 0.1
                )
                
                # Apply MRL truncation if requested
                if truncate_to_2000 and len(embedding) > 2000:
                    embedding = embedding[:2000]
                
                logger.debug(f"✅ Fallback inference completed in {fallback_time*1000:.1f}ms")
                return embedding
            
            logger.error("❌ Both TensorRT and fallback failed")
            return None
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            return None
    
    def get_acceleration_metrics(self) -> Dict[str, Any]:
        """Get acceleration performance metrics"""
        if self.acceleration_metrics["avg_fallback_time"] > 0:
            self.acceleration_metrics["speedup_factor"] = (
                self.acceleration_metrics["avg_fallback_time"] / 
                max(self.acceleration_metrics["avg_tensorrt_time"], 0.001)
            )
        
        return {
            **self.acceleration_metrics,
            "tensorrt_available": self.tensorrt_available,
            "engine_loaded": self.engine is not None,
            "fp4_enabled": self.fp4_enabled,
            "recent_inference_times": self.inference_times[-10:],
            "engine_path": str(self.engine_path),
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length
        }
    
    def __del__(self):
        """Cleanup CUDA memory"""
        try:
            if hasattr(self, 'd_input') and self.d_input:
                self.d_input.free()
            if hasattr(self, 'd_output') and self.d_output:
                self.d_output.free()
        except:
            pass

# Global accelerator instance
_brain1_accelerator = None

def get_brain1_accelerator(cache_dir: str = "/workspace/models") -> Brain1TensorRTAccelerator:
    """Get global Brain-1 TensorRT accelerator instance"""
    global _brain1_accelerator
    if _brain1_accelerator is None:
        _brain1_accelerator = Brain1TensorRTAccelerator(cache_dir)
    return _brain1_accelerator
