"""
Brain2Manager Compatibility Module
Provides backward compatibility for import paths

This module ensures that existing imports for brain2_manager continue to work
while the actual implementation has been moved to reranker_manager.py as part
of the modular architecture refactoring.

Created: 2025-08-03 AEST
Purpose: Fix import path issues for Brain2Manager
"""

# Import the actual Brain2Manager from the new modular location
from .reranker_manager import Brain2Manager

# Re-export for backward compatibility
__all__ = ["Brain2Manager"]
