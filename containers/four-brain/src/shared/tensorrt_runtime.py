"""
TensorRT Runtime for RTX 5070 Ti Optimized Inference
Real TensorRT engine loading and execution
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Initialize CUDA context

logger = logging.getLogger(__name__)

class TensorRTEngine:
    """TensorRT engine wrapper for optimized inference"""
    
    def __init__(self, engine_path: str, max_batch_size: int = 16):
        """Initialize TensorRT engine"""
        self.engine_path = Path(engine_path)
        self.max_batch_size = max_batch_size
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        
        logger.info(f"🔧 Loading TensorRT engine: {self.engine_path}")
        
        # Create TensorRT runtime
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        # Load engine
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        logger.info(f"✅ TensorRT engine loaded successfully")
        logger.info(f"📊 Engine info: {self.engine.num_bindings} bindings, {self.engine.num_layers} layers")
    
    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers for inference"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(i)) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'name': binding_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': self.engine.get_binding_shape(i),
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': binding_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': self.engine.get_binding_shape(i),
                    'dtype': dtype
                })
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        logger.info(f"📦 Allocated buffers: {len(self.inputs)} inputs, {len(self.outputs)} outputs")
    
    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with TensorRT engine"""
        try:
            # Set dynamic shapes if needed
            for i, input_info in enumerate(self.inputs):
                input_name = input_info['name']
                if input_name in input_data:
                    actual_shape = input_data[input_name].shape
                    self.context.set_binding_shape(i, actual_shape)
            
            # Copy input data to GPU
            for input_info in self.inputs:
                input_name = input_info['name']
                if input_name in input_data:
                    np.copyto(input_info['host'], input_data[input_name].ravel())
                    cuda.memcpy_htod_async(input_info['device'], input_info['host'], self.stream)
            
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Copy output data from GPU
            outputs = {}
            for output_info in self.outputs:
                cuda.memcpy_dtoh_async(output_info['host'], output_info['device'], self.stream)
                self.stream.synchronize()
                
                # Reshape output
                output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_info['name']))
                outputs[output_info['name']] = output_info['host'][:np.prod(output_shape)].reshape(output_shape)
            
            return outputs
            
        except Exception as e:
            logger.error(f"❌ TensorRT inference failed: {e}")
            raise
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.synchronize()

class TensorRTModelManager:
    """Manager for TensorRT optimized models"""
    
    def __init__(self, engines_dir: str = "/workspace/models/tensorrt/engines"):
        """Initialize TensorRT model manager"""
        self.engines_dir = Path(engines_dir)
        self.engines = {}
        self.performance_stats = {}
        
        logger.info(f"🚀 TensorRT Model Manager initialized")
        logger.info(f"📁 Engines directory: {self.engines_dir}")
    
    def load_embedding_engine(self) -> bool:
        """Load embedding TensorRT engine"""
        engine_path = self.engines_dir / "brain1_embedding_fp16.engine"
        
        try:
            if not engine_path.exists():
                logger.warning(f"⚠️ TensorRT embedding engine not found: {engine_path}")
                return False
            
            logger.info("🔧 Loading embedding TensorRT engine...")
            self.engines['embedding'] = TensorRTEngine(str(engine_path), max_batch_size=16)
            logger.info("✅ Embedding TensorRT engine loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding TensorRT engine: {e}")
            return False
    
    def load_reranker_engine(self) -> bool:
        """Load reranker TensorRT engine"""
        engine_path = self.engines_dir / "brain2_reranker_fp16.engine"
        
        try:
            if not engine_path.exists():
                logger.warning(f"⚠️ TensorRT reranker engine not found: {engine_path}")
                return False
            
            logger.info("🔧 Loading reranker TensorRT engine...")
            self.engines['reranker'] = TensorRTEngine(str(engine_path), max_batch_size=8)
            logger.info("✅ Reranker TensorRT engine loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load reranker TensorRT engine: {e}")
            return False
    
    def is_tensorrt_available(self, model_type: str) -> bool:
        """Check if TensorRT engine is available for model type"""
        return model_type in self.engines
    
    def infer_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run embedding inference with TensorRT"""
        if 'embedding' not in self.engines:
            raise RuntimeError("Embedding TensorRT engine not loaded")
        
        start_time = time.time()
        
        # Convert PyTorch tensors to numpy
        input_data = {
            'input_ids': input_ids.cpu().numpy().astype(np.int64),
            'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
        }
        
        # Run TensorRT inference
        outputs = self.engines['embedding'].infer(input_data)
        
        # Convert back to PyTorch tensor
        embeddings = torch.from_numpy(outputs['last_hidden_state']).to(input_ids.device)
        
        # Track performance
        inference_time = time.time() - start_time
        self._update_performance_stats('embedding', inference_time, input_ids.shape[0])
        
        return embeddings
    
    def infer_reranker(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run reranker inference with TensorRT"""
        if 'reranker' not in self.engines:
            raise RuntimeError("Reranker TensorRT engine not loaded")
        
        start_time = time.time()
        
        # Convert PyTorch tensors to numpy
        input_data = {
            'input_ids': input_ids.cpu().numpy().astype(np.int64),
            'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
        }
        
        # Run TensorRT inference
        outputs = self.engines['reranker'].infer(input_data)
        
        # Convert back to PyTorch tensor
        scores = torch.from_numpy(outputs['last_hidden_state']).to(input_ids.device)
        
        # Track performance
        inference_time = time.time() - start_time
        self._update_performance_stats('reranker', inference_time, input_ids.shape[0])
        
        return scores
    
    def _update_performance_stats(self, model_type: str, inference_time: float, batch_size: int):
        """Update performance statistics"""
        if model_type not in self.performance_stats:
            self.performance_stats[model_type] = {
                'total_inferences': 0,
                'total_time': 0.0,
                'total_samples': 0
            }
        
        stats = self.performance_stats[model_type]
        stats['total_inferences'] += 1
        stats['total_time'] += inference_time
        stats['total_samples'] += batch_size
        
        # Log performance every 10 inferences
        if stats['total_inferences'] % 10 == 0:
            avg_time = stats['total_time'] / stats['total_inferences']
            avg_throughput = stats['total_samples'] / stats['total_time']
            logger.info(f"📊 {model_type} TensorRT performance: {avg_time*1000:.1f}ms/inference, {avg_throughput:.1f} samples/sec")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        formatted_stats = {}
        
        for model_type, stats in self.performance_stats.items():
            if stats['total_inferences'] > 0:
                formatted_stats[model_type] = {
                    'avg_inference_time_ms': (stats['total_time'] / stats['total_inferences']) * 1000,
                    'throughput_samples_per_sec': stats['total_samples'] / stats['total_time'],
                    'total_inferences': stats['total_inferences'],
                    'total_samples': stats['total_samples']
                }
        
        return formatted_stats
    
    def load_all_engines(self) -> Dict[str, bool]:
        """Load all available TensorRT engines"""
        logger.info("🚀 Loading all TensorRT engines...")
        
        results = {
            'embedding': self.load_embedding_engine(),
            'reranker': self.load_reranker_engine()
        }
        
        loaded_count = sum(results.values())
        logger.info(f"📊 TensorRT engines loaded: {loaded_count}/{len(results)}")
        
        return results

# Global TensorRT manager instance
tensorrt_manager = None

def get_tensorrt_manager() -> TensorRTModelManager:
    """Get global TensorRT manager instance"""
    global tensorrt_manager
    if tensorrt_manager is None:
        tensorrt_manager = TensorRTModelManager()
    return tensorrt_manager

def initialize_tensorrt() -> Dict[str, bool]:
    """Initialize TensorRT engines"""
    manager = get_tensorrt_manager()
    return manager.load_all_engines()
