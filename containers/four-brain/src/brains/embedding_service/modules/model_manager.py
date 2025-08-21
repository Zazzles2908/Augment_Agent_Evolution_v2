"""
Model Manager Module for Brain-1
Handles Qwen3-4B model loading, caching, and lifecycle management

Extracted from brain1_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import torch
from typing import Optional, Tuple, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model Manager for Brain-1 - Handles Qwen3-4B model lifecycle
    Extracted from brain1_manager.py for modular architecture
    """
    
    def __init__(self, config_manager=None):
        """Initialize Model Manager with configuration"""
        self.config_manager = config_manager
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.loading_time = 0.0
        self.initialization_time = time.time()
        
        # Configuration
        self.model_path = "/workspace/models/qwen3/embedding-4b"
        self.cache_dir = "/workspace/models/cache"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Quantization settings
        self.use_blackwall_quantization = True
        self.quantization_enabled = False
        self.quantization_type = None
        
        logger.info("🔧 Model Manager initialized")
    
    def load_model(self, use_blackwall: bool = True, force_reload: bool = False) -> bool:
        """
        Load Qwen3-4B embedding model with quantization optimization
        
        Args:
            use_blackwall: Enable Blackwall quantization for RTX 5070 Ti
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded and not force_reload:
            logger.info("✅ Model already loaded")
            return True
        
        logger.info("🧠 Loading Qwen3-4B Embedding model...")
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"🔢 Blackwell quantization: {use_blackwall}")
        
        start_time = time.time()
        
        try:
            # Import model loading utilities
            from ..core.model_loader import ModelLoader
            
            # Initialize model loader
            model_loader = ModelLoader()
            
            # Load model with quantization
            result = model_loader.load_embedding_model(
                model_name=self.model_path,
                cache_dir=self.cache_dir,
                use_blackwell=use_blackwall
            )
            
            if result:
                self.model, self.tokenizer = result

                # Apply TensorRT FP4 optimization if enabled
                if use_blackwall:
                    try:
                        from .tensorrt_optimizer import TensorRTFP4Optimizer
                        logger.info("🔧 Applying TensorRT FP4 optimization...")
                        optimizer = TensorRTFP4Optimizer()
                        self.model = optimizer.optimize_model_for_inference(self.model, (1, 512))
                        self.quantization_type = "tensorrt_fp4"
                        logger.info("✅ TensorRT FP4 optimization applied")
                    except Exception as e:
                        logger.warning(f"⚠️ TensorRT FP4 optimization failed, using standard: {e}")
                        self.quantization_type = "standard"
                else:
                    self.quantization_type = "standard"

                self.model_loaded = True
                self.loading_time = time.time() - start_time
                self.quantization_enabled = use_blackwall
                
                logger.info(f"✅ Model loaded successfully in {self.loading_time:.2f} seconds")
                logger.info(f"📊 Model specs: Qwen3-4B, 2560-dimensional embeddings")
                logger.info(f"🔧 Quantization: {self.quantization_type}")
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"📊 GPU memory used: {memory_used:.2f} GB")
                
                return True
            else:
                logger.error("❌ Model loading failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading exception: {e}")
            return False
    
    def unload_model(self):
        """Unload model and free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model_loaded = False
            logger.info("✅ Model unloaded and memory freed")
            
        except Exception as e:
            logger.error(f"❌ Error unloading model: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model_loaded and self.model is not None
    
    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Get model and tokenizer (for embedding engine)"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        memory_usage = {}
        
        if torch.cuda.is_available():
            memory_usage = {
                "gpu": {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
                }
            }
        
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "device": self.device,
            "loading_time_seconds": self.loading_time,
            "quantization_enabled": self.quantization_enabled,
            "quantization_type": self.quantization_type,
            "uptime_seconds": time.time() - self.initialization_time,
            "memory_usage": memory_usage
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform model health check"""
        try:
            if not self.is_model_loaded():
                return {
                    "healthy": False,
                    "status": "model_not_loaded",
                    "error": "Model not loaded"
                }
            
            # Basic model test
            if hasattr(self.model, 'encode'):
                # Test with simple text
                test_embedding = self.model.encode("test", convert_to_numpy=True)
                if test_embedding is not None and len(test_embedding) > 0:
                    return {
                        "healthy": True,
                        "status": "operational",
                        "model_loaded": True,
                        "test_embedding_shape": test_embedding.shape
                    }
            
            return {
                "healthy": True,
                "status": "loaded_but_untested",
                "model_loaded": True
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    def reload_model(self, use_blackwall: bool = True) -> bool:
        """Reload model (unload and load again)"""
        logger.info("🔄 Reloading model...")
        self.unload_model()
        return self.load_model(use_blackwall=use_blackwall, force_reload=True)

    def export_to_onnx(self, onnx_path: str = None, max_sequence_length: int = 512) -> bool:
        """
        Export loaded model to ONNX format for TensorRT optimization

        Args:
            onnx_path: Path to save ONNX model (default: /workspace/models/onnx/brain1_embedding.onnx)
            max_sequence_length: Maximum sequence length for ONNX export

        Returns:
            True if export successful, False otherwise
        """
        try:
            if not self.is_model_loaded():
                logger.error("❌ Cannot export ONNX: Model not loaded")
                return False

            # Set default ONNX path
            if onnx_path is None:
                onnx_dir = Path("/workspace/models/onnx")
                onnx_dir.mkdir(parents=True, exist_ok=True)
                onnx_path = str(onnx_dir / "brain1_embedding.onnx")

            logger.info(f"🔄 Exporting model to ONNX format...")
            logger.info(f"📁 ONNX path: {onnx_path}")
            logger.info(f"📊 Max sequence length: {max_sequence_length}")

            # Import required libraries
            import torch
            import onnx
            from pathlib import Path

            # Prepare dummy input for tracing
            dummy_input_ids = torch.randint(0, 1000, (1, max_sequence_length), dtype=torch.long)
            dummy_attention_mask = torch.ones((1, max_sequence_length), dtype=torch.long)

            # Move to same device as model
            if hasattr(self.model, 'device'):
                device = self.model.device
                dummy_input_ids = dummy_input_ids.to(device)
                dummy_attention_mask = dummy_attention_mask.to(device)

            # For SentenceTransformer, we need to access the underlying model
            if hasattr(self.model, '_modules') and '0' in self.model._modules:
                # Get the transformer model (usually the first module)
                transformer_model = self.model._modules['0'].auto_model
                logger.info("✅ Found transformer model in SentenceTransformer")
            else:
                logger.error("❌ Could not find transformer model in SentenceTransformer")
                return False

            # Set model to evaluation mode
            transformer_model.eval()

            # Export to ONNX
            logger.info("🔄 Performing ONNX export...")
            torch.onnx.export(
                transformer_model,
                (dummy_input_ids, dummy_attention_mask),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                }
            )

            # Validate ONNX model
            logger.info("🔍 Validating ONNX model...")
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Get model info
            model_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ ONNX export successful!")
            logger.info(f"📊 ONNX model size: {model_size:.2f} MB")
            logger.info(f"📁 ONNX model saved to: {onnx_path}")

            return True

        except Exception as e:
            logger.error(f"❌ ONNX export failed: {str(e)}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False

    def get_onnx_model_path(self) -> Optional[str]:
        """Get path to ONNX model if it exists"""
        onnx_path = "/workspace/models/onnx/brain1_embedding.onnx"
        if Path(onnx_path).exists():
            return onnx_path
        return None
