"""
Smart Model Manager - On-Demand Model Loading and Memory Optimization
Prevents RAM spiking by loading models only when needed and unloading when idle
"""

import asyncio
import logging
import gc
import torch
import psutil
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import weakref

logger = logging.getLogger(__name__)

class SmartModelManager:
    """
    Intelligent model management system that:
    1. Loads models to VRAM only when needed
    2. Unloads models when idle to free memory
    3. Prevents RAM spiking and system strain
    4. Manages model lifecycle efficiently
    """
    
    def __init__(self, max_idle_time: int = 300, max_vram_usage: float = 0.75):
        self.max_idle_time = max_idle_time  # 5 minutes default
        self.max_vram_usage = max_vram_usage
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, datetime] = {}
        self.model_loaders: Dict[str, Callable] = {}
        self.cleanup_task = None
        self.lock = threading.Lock()
        
        # Cleanup task will be started when first model is loaded
        self.cleanup_task = None
        
        logger.info("Smart Model Manager initialized")
    
    def register_model_loader(self, model_name: str, loader_func: Callable):
        """Register a function to load a specific model"""
        self.model_loaders[model_name] = loader_func
        logger.info(f"Registered loader for model: {model_name}")
    
    async def get_model(self, model_name: str):
        """Get model, loading it if necessary"""
        # Start cleanup task if not already running
        if self.cleanup_task is None:
            self._start_cleanup_task()

        with self.lock:
            # Check if model is already loaded
            if model_name in self.loaded_models:
                self.model_last_used[model_name] = datetime.now()
                logger.debug(f"Using cached model: {model_name}")
                return self.loaded_models[model_name]
            
            # Check VRAM availability before loading
            if not self._check_vram_availability():
                await self._free_memory()
            
            # Load model if loader is registered
            if model_name in self.model_loaders:
                logger.info(f"Loading model: {model_name}")
                try:
                    model = await self._load_model_safely(model_name)
                    self.loaded_models[model_name] = model
                    self.model_last_used[model_name] = datetime.now()
                    logger.info(f"Successfully loaded model: {model_name}")
                    return model
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    raise
            else:
                raise ValueError(f"No loader registered for model: {model_name}")
    
    async def _load_model_safely(self, model_name: str):
        """Load model with proper error handling and memory management"""
        try:
            # Clear cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model using registered loader
            loader_func = self.model_loaders[model_name]
            if asyncio.iscoroutinefunction(loader_func):
                model = await loader_func()
            else:
                model = loader_func()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Clean up on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise
    
    def _check_vram_availability(self) -> bool:
        """Check if there's enough VRAM available"""
        if not torch.cuda.is_available():
            return True
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            usage_ratio = allocated_memory / total_memory
            
            return usage_ratio < self.max_vram_usage
        except Exception:
            return False
    
    async def _free_memory(self):
        """Free memory by unloading idle models"""
        logger.info("Freeing memory by unloading idle models")
        
        current_time = datetime.now()
        models_to_unload = []
        
        for model_name, last_used in self.model_last_used.items():
            idle_time = (current_time - last_used).total_seconds()
            if idle_time > self.max_idle_time:
                models_to_unload.append(model_name)
        
        for model_name in models_to_unload:
            await self.unload_model(model_name)
    
    async def unload_model(self, model_name: str):
        """Unload a specific model from memory"""
        with self.lock:
            if model_name in self.loaded_models:
                logger.info(f"Unloading model: {model_name}")
                
                # Delete model reference
                del self.loaded_models[model_name]
                del self.model_last_used[model_name]
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Successfully unloaded model: {model_name}")
    
    async def unload_all_models(self):
        """Unload all models from memory"""
        logger.info("Unloading all models")
        
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            await self.unload_model(model_name)
    
    def _start_cleanup_task(self):
        """Start background task for automatic cleanup"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background loop for automatic model cleanup"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._free_memory()
                
                # Log memory status
                self._log_memory_status()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def _log_memory_status(self):
        """Log current memory status"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            system_usage = memory.percent
            
            # GPU memory
            gpu_usage = 0
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = (allocated / total) * 100
            
            logger.debug(f"Memory Status - System: {system_usage:.1f}%, GPU: {gpu_usage:.1f}%, Loaded Models: {len(self.loaded_models)}")
            
        except Exception as e:
            logger.error(f"Error logging memory status: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of model manager"""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "model_count": len(self.loaded_models),
            "last_used": {name: time.isoformat() for name, time in self.model_last_used.items()},
            "max_idle_time": self.max_idle_time,
            "max_vram_usage": self.max_vram_usage
        }
    
    async def shutdown(self):
        """Graceful shutdown of model manager"""
        logger.info("Shutting down Smart Model Manager")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Unload all models
        await self.unload_all_models()
        
        logger.info("Smart Model Manager shutdown complete")

# Global instance
smart_model_manager = SmartModelManager()

# Convenience functions
async def get_model(model_name: str):
    """Get model using global smart model manager"""
    return await smart_model_manager.get_model(model_name)

def register_model_loader(model_name: str, loader_func: Callable):
    """Register model loader using global smart model manager"""
    smart_model_manager.register_model_loader(model_name, loader_func)

async def unload_model(model_name: str):
    """Unload model using global smart model manager"""
    await smart_model_manager.unload_model(model_name)

async def unload_all_models():
    """Unload all models using global smart model manager"""
    await smart_model_manager.unload_all_models()
