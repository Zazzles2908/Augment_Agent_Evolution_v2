"""
API endpoints for Brain 4 document processing system
"""

from .health import router as health_router
from .documents import router as documents_router  
from .monitoring import router as monitoring_router

__all__ = ['health_router', 'documents_router', 'monitoring_router']
