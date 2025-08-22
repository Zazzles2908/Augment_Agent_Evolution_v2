"""
Brain-2 Modular Components
Modular architecture for Brain-2 Qwen3-Reranker-4B System

This package contains modular components that replace the monolithic
brain2_manager.py (433 lines) with focused, testable modules.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular architecture following Brain 1 pattern
"""

__version__ = "2.0.0"
__author__ = "AugmentAI - Modular Refactoring"

# Core modules
from .reranking_engine import RerankingEngine
from .model_manager import ModelManager
from .config_manager import ConfigManager
from .memory_optimizer import MemoryOptimizer
from .performance_monitor import PerformanceMonitor

__all__ = [
    "RerankingEngine",
    "ModelManager", 
    "MemoryOptimizer",
    "PerformanceMonitor",
    "ConfigManager"
]
