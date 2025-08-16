"""
Brain 3 Configuration Settings
Configuration for real Augment Agent integration with zero fabrication policy
"""

import os
from typing import Optional, List
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Brain3Settings(BaseSettings):
    """Configuration settings for Brain 3 (Augment Agent Integration)"""
    
    # Brain Configuration
    brain_id: str = Field(
        default="brain3",
        description="Brain identifier for inter-brain communication"
    )
    brain_name: str = Field(
        default="Augment Agent Brain",
        description="Human-readable brain name"
    )
    
    # Real Supabase Integration (VERIFIED ENDPOINTS)
    supabase_url: str = Field(
        default="https://ustcfwmonegxeoqeixgg.supabase.co",
        description="Real Supabase URL for Augment Agent integration"
    )
    supabase_service_role_key: str = Field(
        default="",
        description="Supabase service role key for database access",
        env="SUPABASE_SERVICE_ROLE_KEY"
    )
    supabase_anon_key: str = Field(
        default="",
        description="Supabase anonymous key for client access",
        env="SUPABASE_ANON_KEY"
    )
    
    # Real Augment Agent Capabilities
    capabilities: List[str] = Field(
        default=[
            "conversation_analysis",
            "task_management", 
            "code_generation",
            "system_integration",
            "documentation_creation",
            "problem_solving",
            "workflow_orchestration",
            "sequential_thinking",
            "tool_execution",
            "file_operations",
            "web_search",
            "github_integration",
            "supabase_operations"
        ],
        description="Real Augment Agent capabilities"
    )
    
    # Service Configuration
    service_host: str = Field(
        default="0.0.0.0",
        description="Brain 3 service host"
    )
    service_port: int = Field(
        default=8003,
        description="Brain 3 service port"
    )
    
    # Redis Configuration for Inter-Brain Communication
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis URL for inter-brain communication"
    )
    redis_timeout: int = Field(
        default=30,
        description="Redis operation timeout in seconds"
    )
    
    # GLM API Configuration
    glm_api_key: str = Field(
        default="",
        description="GLM API key for external model access"
    )

    # Task Processing Configuration
    max_concurrent_tasks: int = Field(
        default=5,
        description="Maximum concurrent tasks for Brain 3"
    )
    task_timeout_seconds: int = Field(
        default=300,
        description="Task processing timeout in seconds (5 minutes)"
    )
    max_task_queue_size: int = Field(
        default=100,
        description="Maximum task queue size"
    )
    
    # Conversation Interface Configuration
    conversation_timeout: int = Field(
        default=60,
        description="Conversation interface timeout in seconds"
    )
    max_conversation_length: int = Field(
        default=10000,
        description="Maximum conversation length in characters"
    )
    
    # Workflow Orchestration
    max_workflow_steps: int = Field(
        default=50,
        description="Maximum steps in a workflow"
    )
    workflow_step_timeout: int = Field(
        default=60,
        description="Timeout for individual workflow steps"
    )
    
    # Performance and Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    metrics_port: int = Field(
        default=9092,
        description="Metrics endpoint port"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Health Check Configuration
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(
        default=10,
        description="Health check timeout in seconds"
    )
    
    # Security Configuration
    enable_authentication: bool = Field(
        default=False,
        description="Enable API authentication"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    
    # Integration Modes
    integration_mode: str = Field(
        default="supabase_mediated",
        description="Integration mode: 'supabase_mediated' or 'conversation_based'"
    )
    enable_conversation_interface: bool = Field(
        default=True,
        description="Enable conversation-based interface"
    )
    enable_supabase_mediation: bool = Field(
        default=True,
        description="Enable Supabase-mediated communication"
    )
    
    # Real System Integration
    augment_agent_schema: str = Field(
        default="augment_agent",
        description="Supabase schema for Augment Agent data"
    )
    sessions_table: str = Field(
        default="sessions",
        description="Sessions table name"
    )
    knowledge_table: str = Field(
        default="knowledge",
        description="Knowledge table name"
    )
    learning_patterns_table: str = Field(
        default="learning_patterns",
        description="Learning patterns table name"
    )
    
    # Zero Fabrication Policy Enforcement
    enforce_zero_fabrication: bool = Field(
        default=True,
        description="Enforce zero fabrication policy - no mock data or endpoints"
    )
    validate_real_endpoints: bool = Field(
        default=True,
        description="Validate that all endpoints are real and functional"
    )
    
    class Config:
        env_prefix = "BRAIN3_"
        case_sensitive = False


def get_brain3_settings() -> Brain3Settings:
    """Get Brain 3 settings with environment variable override"""
    return Brain3Settings()
