#!/usr/bin/env python3
"""
HRM (Hierarchical Reasoning Module) Package
Implements the HRM high & low module system for the Four-Brain architecture.

This package provides:
- HRM H-Module: Strategic planning with FP8 precision (always loaded)
- HRM L-Module: Fast execution with NVFP4 precision (always loaded)
- HRM Orchestrator: Hierarchical convergence pattern coordination
- Blackwell Optimizer: GPU optimizations for maximum performance
"""

from .hrm_module import (
    HRMModule,
    HRMHModule,
    HRMLModule,
    HRMModuleType,
    HRMPrecision,
    HRMTaskType,
    HRMRequest,
    HRMResponse,
    HRMConvergenceState
)

from .hrm_orchestrator import HRMOrchestrator

from .blackwell_optimizer import (
    BlackwellOptimizer,
    BlackwellOptimizationType,
    BlackwellPrecisionMode,
    BlackwellOptimizationConfig
)

__version__ = "1.0.0"
__author__ = "Four-Brain System"
__description__ = "HRM (Hierarchical Reasoning Module) Implementation"

__all__ = [
    # Core HRM modules
    "HRMModule",
    "HRMHModule", 
    "HRMLModule",
    "HRMOrchestrator",
    
    # Enums and types
    "HRMModuleType",
    "HRMPrecision",
    "HRMTaskType",
    
    # Data structures
    "HRMRequest",
    "HRMResponse", 
    "HRMConvergenceState",
    
    # Blackwell optimization
    "BlackwellOptimizer",
    "BlackwellOptimizationType",
    "BlackwellPrecisionMode",
    "BlackwellOptimizationConfig"
]
