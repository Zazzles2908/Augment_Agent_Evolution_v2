"""
Core Quantization Module for Four-Brain System
Provides quantization utilities for RTX 5070 Ti Blackwell architecture
"""

from .blackwell_quantization import (
    BlackwellQuantizationManager,
    blackwell_quantizer,
    blackwall_quantizer,  # DEPRECATED: Use blackwell_quantizer instead
    FOUR_BRAIN_QUANTIZATION_CONFIG
)

__all__ = [
    'BlackwellQuantizationManager',
    'blackwell_quantizer',
    'blackwall_quantizer',  # DEPRECATED: Incorrect terminology, use blackwell_quantizer
    'FOUR_BRAIN_QUANTIZATION_CONFIG'
]
