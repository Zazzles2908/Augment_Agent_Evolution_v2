"""
Brain-3 Modular Components
Modular architecture for Brain-3 Augment Intelligence

This package contains modular components that replace the monolithic
brain3_manager.py with focused, testable modules.

Created: 2025-07-29 AEST
Purpose: Replace hardcoded AI responses with real intelligence
"""

__version__ = "1.0.0"
__author__ = "AugmentAI - Modular Refactoring"

# Core modules
from .ai_interface import AIInterface
from .response_generator import ResponseGenerator
from .fallback_handler import FallbackHandler
from .config_manager import ConfigManager

__all__ = [
    "AIInterface",
    "ResponseGenerator", 
    "FallbackHandler",
    "ConfigManager"
]
