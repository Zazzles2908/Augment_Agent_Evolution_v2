"""
Brain1 Core Module
Contains core business logic and main components.
"""

# Prefer modular manager; legacy monolith kept only for reference
try:
    from .manager_alias import Brain1Manager  # re-export modular manager
except Exception:
    from .brain1_manager import Brain1Manager  # fallback if alias not present
from .model_loader import ModelLoader
from .database_manager import HybridDatabaseManager, get_database_manager
from .resource_monitor import ResourceMonitor

__all__ = [
    "Brain1Manager",
    "ModelLoader",
    "HybridDatabaseManager",
    "get_database_manager",
    "ResourceMonitor"
]
