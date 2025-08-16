"""
Shared Error Handling Module
Centralized error handling system for the Four-Brain architecture

This package provides unified error handling capabilities including centralized
error management, recovery mechanisms, circuit breakers, and health monitoring.

Created: 2025-07-29 AEST
Purpose: Centralized error handling for all Four-Brain components
"""

from .centralized_error_handler import (
    CentralizedErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    RecoveryAction,
    ErrorContext,
    ErrorRecord,
    create_centralized_error_handler
)
from .recovery_manager import (
    RecoveryManager,
    RecoveryStatus,
    RecoveryStrategy,
    RecoveryPlan,
    RecoveryExecution,
    create_recovery_manager
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    CircuitBreakerConfig,
    create_circuit_breaker,
    create_circuit_breaker_manager
)
from .retry_engine import (
    RetryEngine,
    RetryConfig,
    BackoffStrategy,
    create_retry_engine,
    exponential_backoff,
    linear_backoff,
    fixed_delay
)
from .fallback_manager import (
    FallbackManager,
    FallbackStrategy,
    FallbackConfig,
    create_fallback_manager
)
from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    HealthCheck,
    HealthResult,
    create_health_monitor
)
from .error_recovery_integration import (
    ErrorRecoveryIntegration,
    create_error_recovery_integration,
    get_global_error_handling
)

__all__ = [
    "CentralizedErrorHandler",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryAction",
    "ErrorContext",
    "ErrorRecord",
    "create_centralized_error_handler",
    "RecoveryManager",
    "RecoveryStatus",
    "RecoveryStrategy",
    "RecoveryPlan",
    "RecoveryExecution",
    "create_recovery_manager",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitState",
    "CircuitBreakerConfig",
    "create_circuit_breaker",
    "create_circuit_breaker_manager",
    "RetryEngine",
    "RetryConfig",
    "BackoffStrategy",
    "create_retry_engine",
    "exponential_backoff",
    "linear_backoff",
    "fixed_delay",
    "FallbackManager",
    "FallbackStrategy",
    "FallbackConfig",
    "create_fallback_manager",
    "HealthMonitor",
    "HealthStatus",
    "HealthCheck",
    "HealthResult",
    "create_health_monitor",
    "ErrorRecoveryIntegration",
    "create_error_recovery_integration",
    "get_global_error_handling"
]
