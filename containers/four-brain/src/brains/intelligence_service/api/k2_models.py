"""
Compatibility shim for legacy imports.
This module re-exports the Orchestrator Hub API models from hub_models.py.
"""
# Do not remove this file; it preserves backward compatibility for existing imports
from .hub_models import (
    StrategyRequest as K2StrategyRequest,
    StrategyResponse as K2StrategyResponse,
    HealthResponse as K2HealthResponse,
    VectorRequest as K2VectorRequest,
    VectorResponse as K2VectorResponse,
    MetricsResponse as K2MetricsResponse,
    ErrorResponse as K2ErrorResponse,
)
