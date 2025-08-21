"""
Model Manager Module for Brain-2
Handles Qwen3-Reranker-4B model loading, caching, and lifecycle management

Extracted from brain2_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import torch
from typing import Optional, Tuple, Any, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model Manager for Brain-2 - Handles Qwen3-Reranker-4B model lifecycle
    Extracted from brain2_manager.py for modular architecture
    """
    
    def __init__(self, config_manager=None):
        """Initialize Model Manager with configuration"""
        self.config_manager = config_manager
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.triton_mode = False
        self.loading_time = 0.0
        self.initialization_time = time.time()
        
        # Configuration
        self.model_name = "Qwen/Qwen3-Reranker-4B"
        self.model_path = "/workspace/models/qwen3/reranker-4b"
        self.cache_dir = "/workspace/models/cache"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Quantization settings
        self.use_blackwell_quantization = True
        self.quantization_enabled = False
        self.quantization_type = None
        
        # Memory allocation (20% for Brain-2)
        self.max_vram_usage = 0.20
        
        logger.info("🔧 Model Manager (Brain-2) initialized")
        logger.info(f"📊 GPU memory allocation: {self.max_vram_usage * 100}%")
    
    def load_model(self, use_blackwell: bool = True, force_reload: bool = False) -> bool:
        """
        Load Qwen3-Reranker-4B model with quantization optimization

        Args:
            use_blackwell: Enable Blackwell quantization for RTX 5070 Ti
            force_reload: Force reload even if model is already loaded

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded and not force_reload:
            logger.info("✅ Model already loaded")
            return True
        
        # If configured to use Triton for reranking, only load tokenizer and skip heavy model
        try:
            if self.config_manager and self.config_manager.get_config("use_triton"):
                self.triton_mode = True
                logger.info("🚀 USE_TRITON=true: loading tokenizer only for reranker")
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False
                )
                self.model = None
                self.model_loaded = True  # Tokenizer is sufficient in Triton mode
                self.loading_time = time.time() - start_time
                logger.info("✅ Tokenizer loaded for Triton reranker path; skipping local model load")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Failed Triton-mode tokenizer-only load path, falling back to local model: {e}")

        logger.info("🧠 Loading Qwen3-Reranker-4B model...")
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"🔢 Blackwell quantization: {use_blackwell}")
        
        start_time = time.time()
        
        try:
            # Set GPU memory fraction for Brain-2 (20%)
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.max_vram_usage)
                logger.info(f"✅ GPU memory fraction set to {self.max_vram_usage * 100}%")
            
            # Load tokenizer first
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=False  # Allow download if not cached
            )
            logger.info("✅ Tokenizer loaded successfully")
            
            # Load model with quantization
            logger.info("🧠 Loading Qwen3-Reranker-4B model...")
            
            if use_blackwell:
                # Load with Blackwell quantization (FP16 direct GPU loading)
                logger.info("🔧 Loading with Blackwell quantization (direct GPU loading)...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="auto",  # Direct GPU loading
                    torch_dtype=torch.float16,  # Use FP16 to reduce memory
                    trust_remote_code=True,
                    local_files_only=False,
                    low_cpu_mem_usage=True  # Optimize CPU memory usage
                )

                # Apply Blackwell quantization if available
                try:
                    from ....core.quantization import blackwell_quantizer
                    self.model = blackwell_quantizer.quantize_model(self.model)
                    logger.info("✅ Blackwell quantization applied successfully")
                    self.quantization_type = "blackwell"
                except ImportError:
                    logger.warning("⚠️ Blackwell quantization not available, using FP16")
                    self.quantization_type = "fp16"
            else:
                # Load with standard FP16 optimization
                logger.info("🔧 Loading with standard FP16 quantization...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=False,
                    low_cpu_mem_usage=True
                )
                self.quantization_type = "fp16"
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cleaned up after model loading")
            
            self.model_loaded = True
            self.loading_time = time.time() - start_time
            self.quantization_enabled = use_blackwell
            
            logger.info(f"✅ Model loaded successfully in {self.loading_time:.2f} seconds")
            logger.info(f"📊 Model specs: Qwen3-Reranker-4B, document relevance scoring")
            logger.info(f"🔧 Quantization: {self.quantization_type}")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"📊 GPU memory used: {memory_used:.2f} GB")
            
            return True
            
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
        return self.model_loaded and self.model is not None and self.tokenizer is not None
    
    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Get model and tokenizer (for reranking engine)"""
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
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "fraction_allocated": self.max_vram_usage
                }
            }
        
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loading_time_seconds": self.loading_time,
            "quantization_enabled": self.quantization_enabled,
            "quantization_type": self.quantization_type,
            "uptime_seconds": time.time() - self.initialization_time,
            "memory_usage": memory_usage,
            "max_vram_usage": self.max_vram_usage
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
            
            # Basic model test with simple input
            test_input = "Query: test\nDocument: test document"
            inputs = self.tokenizer(
                test_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if outputs is not None and hasattr(outputs, 'last_hidden_state'):
                    return {
                        "healthy": True,
                        "status": "operational",
                        "model_loaded": True,
                        "test_output_shape": outputs.last_hidden_state.shape
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
    
    def reload_model(self, use_blackwell: bool = True) -> bool:
        """Reload model (unload and load again)"""
        logger.info("🔄 Reloading model...")
        self.unload_model()
        return self.load_model(use_blackwell=use_blackwell, force_reload=True)
