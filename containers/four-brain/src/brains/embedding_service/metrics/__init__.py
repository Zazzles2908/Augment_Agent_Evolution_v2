"""
Brain 1 Embedding Metrics Package
Prometheus metrics collection for Qwen3-4B embedding service
"""

from .prometheus_metrics import get_brain1_metrics, Brain1Metrics

__all__ = ["get_brain1_metrics", "Brain1Metrics"]
