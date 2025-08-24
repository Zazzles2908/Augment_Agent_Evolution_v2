"""
Brain 3 Communication Module
Inter-brain communication components for Zazzles's Agent integration
"""

from .brain_communicator import BrainCommunicator

# Create alias for backward compatibility
Brain3Communicator = BrainCommunicator

__all__ = [
    "BrainCommunicator",
    "Brain3Communicator"
]
