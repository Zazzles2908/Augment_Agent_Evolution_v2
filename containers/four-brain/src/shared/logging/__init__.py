"""
Shared Logging Module
Centralized logging system for the Four-Brain architecture

This package provides unified logging capabilities including centralized
configuration, log aggregation, analysis, and visualization.

Created: 2025-07-29 AEST
Purpose: Centralized logging for all Four-Brain components
"""

from .centralized_logger import (
    CentralizedLogger,
    LogLevel,
    LogFormat,
    create_centralized_logger,
    get_global_logger,
    get_logger
)
from .log_aggregator import (
    LogAggregator,
    LogEntry,
    create_log_aggregator
)
from .log_analyzer import (
    LogAnalyzer,
    LogPattern,
    LogAnomaly,
    LogInsight,
    create_log_analyzer
)
from .log_dashboard import (
    LogDashboardManager,
    LogWidget,
    LogDashboard,
    create_log_dashboard_manager
)
from .log_retention import (
    LogRetentionManager,
    RetentionPolicy,
    RetentionStats,
    create_log_retention_manager
)
from .logging_integration import (
    LoggingIntegration,
    create_logging_integration,
    get_global_logging
)

__all__ = [
    "CentralizedLogger",
    "LogLevel",
    "LogFormat",
    "create_centralized_logger",
    "get_global_logger",
    "get_logger",
    "LogAggregator",
    "LogEntry",
    "create_log_aggregator",
    "LogAnalyzer",
    "LogPattern",
    "LogAnomaly",
    "LogInsight",
    "create_log_analyzer",
    "LogDashboardManager",
    "LogWidget",
    "LogDashboard",
    "create_log_dashboard_manager",
    "LogRetentionManager",
    "RetentionPolicy",
    "RetentionStats",
    "create_log_retention_manager",
    "LoggingIntegration",
    "create_logging_integration",
    "get_global_logging"
]
