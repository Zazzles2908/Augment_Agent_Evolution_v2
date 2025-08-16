"""
Shared utilities and modules for Four-Brain Architecture
Common utilities and interfaces for the Four-Brain system

This package contains shared components that are used across
multiple brains to ensure consistency and reduce duplication.

Created: 2025-07-29 AEST
Purpose: Centralized shared components
"""

# Import key shared components
from .communication import (
    StandardizedBrainCommunicator,
    BrainMessage,
    MessageType,
    BrainType,
    create_brain_communicator
)
from .database import (
    DatabaseConnectionManager,
    DatabaseAuthenticationHandler,
    DatabaseConfigValidator,
    create_connection_manager,
    create_auth_handler,
    create_config_validator
)
from .redis_client import (
    RedisStreamsClient
)
from .streams import (
    StreamNames,
    StreamMessage,
    StreamConsumerGroup,
    StreamDefinitions,
    MessageType,
    DoclingRequest,
    EmbeddingRequest,
    RerankRequest,
    AgenticTask,
    MemoryUpdate
)
from .memory_store import (
    MemoryStore,
    TaskScore,
    PatternMatch
)
from .self_grading import (
    SelfGradingSystem,
    PerformanceScore,
    get_self_grading_system
)

__all__ = [
    "StandardizedBrainCommunicator",
    "BrainMessage",
    "MessageType",
    "BrainType",
    "create_brain_communicator",
    "DatabaseConnectionManager",
    "DatabaseAuthenticationHandler",
    "DatabaseConfigValidator",
    "create_connection_manager",
    "create_auth_handler",
    "create_config_validator",
    "RedisStreamsClient",
    "StreamNames",
    "StreamMessage",
    "StreamConsumerGroup",
    "StreamDefinitions",
    "DoclingRequest",
    "EmbeddingRequest",
    "RerankRequest",
    "AgenticTask",
    "MemoryUpdate",
    "MemoryStore",
    "TaskScore",
    "PatternMatch",
    "SelfGradingSystem",
    "PerformanceScore",
    "get_self_grading_system"
]
