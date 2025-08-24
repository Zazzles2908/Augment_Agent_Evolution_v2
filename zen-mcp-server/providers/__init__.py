"""Provider package exports.

Follow the documented provider pattern:
- Export base types and the registry here
- Import specific provider implementations lazily in server.py based on env keys
This avoids optional dependency imports (e.g., Google/Gemini) when not configured.
"""

from .base import ModelCapabilities, ModelProvider, ModelResponse  # noqa: F401
from .registry import ModelProviderRegistry  # noqa: F401

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
]
