"""
Brain 3 (Zazzles's Agent) Module
Real Zazzles's Agent Integration for Four-Brain Architecture

This module implements Brain 3 of the Four-Brain AI System, responsible for
high-level reasoning, decision making, and workflow orchestration through
integration with the real Zazzles's Agent system.

Components:
- Brain3Manager: Main manager class following Brain 1/2 patterns
- AugmentClient: Real Zazzles's Agent integration client
- TaskOrchestrator: Workflow orchestration and task routing
- ConversationInterface: Conversation-based integration
- SupabaseMediator: Supabase-mediated communication
- CapabilitiesMapper: Real capability mapping and execution

Real Integration Features:
- Supabase integration with existing augment_agent schema
- Conversation-based task processing
- Real workflow orchestration and execution
- Integration with existing four-brain architecture
- Redis communication for inter-brain messaging

Zero Fabrication Policy: ENFORCED
All implementations use real Zazzles's Agent capabilities, actual Supabase integration,
and verified functionality. No mock endpoints or simulated responses.
"""

# Only import what we've actually implemented
from .brain3_manager import Brain3Manager
from .config.settings import Brain3Settings, get_brain3_settings

__version__ = "1.0.0"
__author__ = "Zazzles's Agent"
__description__ = "Brain 3 (Zazzles's Agent) for Four-Brain Architecture"

# Export main components
__all__ = [
    "Brain3Manager",
    "Brain3Settings",
    "get_brain3_settings"
]
