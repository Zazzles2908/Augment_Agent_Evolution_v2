"""
Brain-4 Modular Components
Modular architecture for Brain-4 Docling Document Processing System

This package contains modular components that replace the monolithic
brain4_manager.py (400+ lines) with focused, testable modules.

Created: 2025-08-02 AEST
Purpose: Replace monolithic manager with modular architecture following Brain 1/2 pattern
"""

__version__ = "2.0.0"
__author__ = "Zazzles's Agent - Modular Refactoring"

# Core modules
from .document_engine import DocumentEngine
from .docling_manager import DoclingManager
from .config_manager import ConfigManager
from .memory_optimizer import MemoryOptimizer
from .performance_monitor import PerformanceMonitor

__all__ = [
    "DocumentEngine",
    "DoclingManager", 
    "MemoryOptimizer",
    "PerformanceMonitor",
    "ConfigManager"
]
