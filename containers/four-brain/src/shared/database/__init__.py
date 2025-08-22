"""
Shared Database Module
Centralized database connection and authentication management

This package provides standardized database connection management
for the Four-Brain system, fixing authentication issues.

Created: 2025-07-29 AEST
Purpose: Centralized database management
"""

from .connection_manager import (
    DatabaseConnectionManager,
    create_connection_manager
)
from .auth_handler import (
    DatabaseAuthenticationHandler,
    create_auth_handler
)
from .config_validator import (
    DatabaseConfigValidator,
    create_config_validator
)

__all__ = [
    "DatabaseConnectionManager",
    "create_connection_manager",
    "DatabaseAuthenticationHandler",
    "create_auth_handler",
    "DatabaseConfigValidator",
    "create_config_validator"
]
