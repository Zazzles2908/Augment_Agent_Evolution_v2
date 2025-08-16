"""
Embedding Engine Module for Brain-1
Handles Qwen3-4B embedding generation with MRL truncation

Extracted from brain1_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Embedding Engine for Brain-1 - Handles Qwen3-4B embedding generation
    Extracted from brain1_manager.py for modular architecture
    """
    
    def __init__(self, model_manager=None, config_manager=None):
        """Initialize Embedding Engine with dependencies"""
        self.model_manager = model_manager
        self.config_manager = config_manager
        
        # Configuration from config manager
        self.embedding_dimensions = 2560  # Qwen3-4B native dimensions
        self.target_dimensions = 2000     # MRL truncation for Supabase
        self.use_mrl_truncation = True
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        
        # Thinking mode configuration
        self.enable_thinking = True
        self.thinking_iterations = 3
        self.thinking_temperature = 0.7
        
        logger.info("🔧 Embedding Engine initialized")
    
    def generate_embedding(self, text: str, truncate_to_2000: bool = True, 
                          use_thinking_mode: Optional[bool] = None) -> Optional[np.ndarray]:
        """
        Generate embedding using Qwen3-4B with optional MRL truncation
        
        Args:
            text: Input text to embed
            truncate_to_2000: Apply MRL truncation to 2000 dimensions for Supabase
            use_thinking_mode: Override thinking mode setting
            
        Returns:
            Embedding vector (2000-dim if truncated, 2560-dim if not)
        """
        if not self.model_manager or not self.model_manager.is_model_loaded():
            logger.error("❌ Model not loaded in model manager")
            return None
        
        start_time = time.time()
        self.total_embeddings_generated += 1
        
        try:
            # Use thinking mode if enabled
            if use_thinking_mode or (use_thinking_mode is None and self.enable_thinking):
                return self._generate_embedding_with_thinking(text, truncate_to_2000)
            else:
                return self._generate_embedding_direct(text, truncate_to_2000)
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return None
        finally:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
    
    def _generate_embedding_direct(self, text: str, truncate_to_2000: bool) -> Optional[np.ndarray]:
        """Generate embedding directly using the model"""
        try:
            model, tokenizer = self.model_manager.get_model_and_tokenizer()
            
            # Use SentenceTransformer encode method if available
            if hasattr(model, 'encode'):
                embedding = model.encode(text, convert_to_numpy=True)
                logger.debug(f"✅ Generated embedding using SentenceTransformer: {embedding.shape}")
            else:
                # Fallback to manual tokenization and forward pass
                inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # Get embedding (pooler output or mean pooling)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output
                    else:
                        embedding = outputs.last_hidden_state.mean(dim=1)
                    
                    embedding = embedding.cpu().numpy().flatten()
                    logger.debug(f"✅ Generated embedding using transformers: {embedding.shape}")
            
            # Apply MRL truncation if requested
            if truncate_to_2000 and len(embedding) > self.target_dimensions:
                embedding = embedding[:self.target_dimensions]
                logger.debug(f"🔧 Applied MRL truncation: {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Direct embedding generation failed: {e}")
            return None
    
    def _generate_embedding_with_thinking(self, text: str, truncate_to_2000: bool) -> Optional[np.ndarray]:
        """Generate embedding with thinking mode for deeper reasoning"""
        try:
            logger.debug(f"🤔 Using thinking mode with {self.thinking_iterations} iterations")
            
            # Generate multiple embeddings with different approaches
            embeddings = []
            
            for i in range(self.thinking_iterations):
                # Modify text slightly for each iteration to get diverse representations
                thinking_text = f"[Iteration {i+1}] {text}"
                embedding = self._generate_embedding_direct(thinking_text, truncate_to_2000=False)
                
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                logger.warning("⚠️ No embeddings generated in thinking mode")
                return None
            
            # Combine embeddings (mean pooling)
            combined_embedding = np.mean(embeddings, axis=0)
            
            # Apply MRL truncation if requested
            if truncate_to_2000 and len(combined_embedding) > self.target_dimensions:
                combined_embedding = combined_embedding[:self.target_dimensions]
                logger.debug(f"🔧 Applied MRL truncation after thinking: {len(combined_embedding)} dimensions")
            
            logger.debug(f"🤔 Thinking mode embedding generated: {combined_embedding.shape}")
            return combined_embedding
            
        except Exception as e:
            logger.error(f"❌ Thinking mode embedding generation failed: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str], truncate_to_2000: bool = True) -> List[Optional[np.ndarray]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        for text in texts:
            embedding = self.generate_embedding(text, truncate_to_2000)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding engine statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_embeddings_generated
            if self.total_embeddings_generated > 0 else 0
        )
        
        return {
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "embedding_dimensions": self.embedding_dimensions,
            "target_dimensions": self.target_dimensions,
            "mrl_truncation_enabled": self.use_mrl_truncation,
            "thinking_mode_enabled": self.enable_thinking,
            "thinking_iterations": self.thinking_iterations
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        logger.info("📊 Embedding engine statistics reset")
