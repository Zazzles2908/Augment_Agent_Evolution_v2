"""
Brain1 Embedding Module - Qwen3-4B Embedding System
Four-Brain Architecture - Standardized Structure

This module provides embedding generation capabilities using Qwen3-4B model
with optimized performance and container-native storage.

Created: 2025-07-13 AEST
Author: Augment Agent Evolution - Brain Architecture Standardization
"""

__version__ = "1.0.0"
__author__ = "Augment Agent Evolution"

# Core imports (modular Brain1Manager)
from .embedding_manager import Brain1Manager
from .core.model_loader import ModelLoader
from .core.database_manager import HybridDatabaseManager, get_database_manager
from .core.resource_monitor import ResourceMonitor

__all__ = [
    "Brain1Manager",
    "ModelLoader",
    "HybridDatabaseManager",
    "get_database_manager",
    "ResourceMonitor"
]
