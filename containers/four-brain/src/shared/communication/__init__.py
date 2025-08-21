"""
Shared Communication Module
Standardized inter-brain communication components

This package provides standardized communication interfaces and utilities
for the Four-Brain system, fixing interface inconsistencies.

Created: 2025-07-29 AEST
Purpose: Centralized communication standards
"""

from .brain_communicator import (
    StandardizedBrainCommunicator,
    BrainMessage,
    MessageType,
    BrainType,
    create_brain_communicator
)
from .message_formatter import (
    MessageFormatter,
    MessageMetadata,
    MessageHeaders,
    create_message_formatter
)
from .error_handler import (
    CommunicationErrorHandler,
    CommunicationError,
    ErrorType,
    ErrorSeverity,
    create_error_handler
)
from .retry_logic import (
    RetryManager,
    RetryConfig,
    RetryStrategy,
    RetryResult,
    create_retry_manager,
    simple_retry
)

__all__ = [
    "StandardizedBrainCommunicator",
    "BrainMessage",
    "MessageType",
    "BrainType",
    "create_brain_communicator",
    "MessageFormatter",
    "MessageMetadata",
    "MessageHeaders",
    "create_message_formatter",
    "CommunicationErrorHandler",
    "CommunicationError",
    "ErrorType",
    "ErrorSeverity",
    "create_error_handler",
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryResult",
    "create_retry_manager",
    "simple_retry"
]
