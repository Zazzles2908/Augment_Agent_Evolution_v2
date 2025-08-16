"""
Compatibility shim for Brain‑1 manager

Purpose:
- Provide a stable import path for legacy code importing from core.brain1_manager
- Re-export the modular Brain1Manager implemented in embedding_manager.py

Usage:
from brains.embedding_service.core.manager_alias import Brain1Manager

This allows us to deprecate the monolithic core/brain1_manager.py gradually without breaking imports.
"""

from brains.embedding_service.embedding_manager import Brain1Manager  # re-export

__all__ = ["Brain1Manager"]

