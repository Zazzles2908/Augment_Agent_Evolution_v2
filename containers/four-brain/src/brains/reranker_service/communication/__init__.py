"""
Brain 2 Communication Module
Real Redis-based inter-brain communication for Four-Brain Architecture

This module provides authentic Redis communication capabilities for Brain 2,
enabling real message passing, task coordination, and workflow management.

Zero Fabrication Policy: ENFORCED
All communication uses real Redis infrastructure with authentic message handling.
"""

from .brain_communicator import (
    Brain2Communicator,
    BrainMessage,
    MessageType,
    BrainType
)

__version__ = "1.0.0"
__author__ = "Augment Agent"
__description__ = "Brain 2 Redis Communication Module"

# Export main components
__all__ = [
    "Brain2Communicator",
    "BrainMessage", 
    "MessageType",
    "BrainType"
]
