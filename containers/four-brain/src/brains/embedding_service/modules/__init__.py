"""
Brain-1 Modular Components
Modular architecture for Brain-1 Qwen3-4B Embedding System

This package contains modular components that replace the monolithic
brain1_manager.py (807 lines) with focused, testable modules.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular architecture
"""

__version__ = "2.0.0"
__author__ = "AugmentAI - Modular Refactoring"

# Core modules
from .embedding_engine import EmbeddingEngine
from .model_manager import ModelManager
from .memory_optimizer import MemoryOptimizer
from .performance_monitor import PerformanceMonitor
from .config_manager import ConfigManager

__all__ = [
    "EmbeddingEngine",
    "ModelManager", 
    "MemoryOptimizer",
    "PerformanceMonitor",
    "ConfigManager"
]
