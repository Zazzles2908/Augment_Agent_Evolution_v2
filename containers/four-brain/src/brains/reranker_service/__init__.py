"""
Brain 2 (Qwen3-Reranker-4B) Module
Document Relevance Ranking for Four-Brain Architecture

This module implements Brain 2 of the Four-Brain AI System, responsible for
document relevance ranking using the Qwen3-Reranker-4B model with quantization
optimization for RTX 5070 Ti hardware.

Components:
- Brain2Manager: Main manager class following Brain 1 patterns
- RerankerService: FastAPI service wrapper
- ModelHandler: Qwen3-Reranker-4B model handling with quantization
- TaskProcessor: Background task processing via Redis

Hardware Optimization:
- RTX 5070 Ti Blackwell architecture support
- 4-bit/8-bit quantization with Unsloth
- Memory management for 16GB VRAM
- Flash attention optimization

Zero Fabrication Policy: ENFORCED
All implementations use real models, actual configurations, and verified functionality.
"""

from .brain2_manager import Brain2Manager
from .reranker_service import RerankerService
from .model_handler import Qwen3RerankerHandler
from .task_processor import Brain2TaskProcessor

__version__ = "1.0.0"
__author__ = "Augment Agent"
__description__ = "Brain 2 (Qwen3-Reranker-4B) for Four-Brain Architecture"

# Export main components
__all__ = [
    "Brain2Manager",
    "RerankerService", 
    "Qwen3RerankerHandler",
    "Brain2TaskProcessor"
]
