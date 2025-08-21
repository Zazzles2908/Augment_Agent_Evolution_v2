"""
Brain 3 API Models
Pydantic models for Augment Agent API endpoints
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class AugmentRequest(BaseModel):
    """Request model for Augment Agent processing"""
    task_type: str = Field(..., description="Type of task to process")
    conversation: Optional[Dict[str, Any]] = Field(None, description="Conversation data")
    task_data: Optional[Dict[str, Any]] = Field(None, description="Task management data")
    code_request: Optional[Dict[str, Any]] = Field(None, description="Code generation request")
    integration: Optional[Dict[str, Any]] = Field(None, description="System integration request")
    workflow: Optional[Dict[str, Any]] = Field(None, description="Workflow orchestration data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AugmentResponse(BaseModel):
    """Response model for Augment Agent processing"""
    result: Dict[str, Any] = Field(..., description="Processing result")
    task_type: str = Field(..., description="Type of task processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_info: Dict[str, Any] = Field(..., description="Agent information")


class ConversationRequest(BaseModel):
    """Request model for conversation processing"""
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
    context: Optional[Dict[str, Any]] = Field(None, description="Conversation context")
    max_length: Optional[int] = Field(None, description="Maximum conversation length")


class ConversationResponse(BaseModel):
    """Response model for conversation processing"""
    conversation_id: str = Field(..., description="Conversation ID")
    stored_in_supabase: bool = Field(..., description="Whether stored in Supabase")
    message_count: int = Field(..., description="Number of messages")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class TaskRequest(BaseModel):
    """Request model for task management"""
    task_id: Optional[str] = Field(None, description="Task ID")
    action: str = Field(..., description="Task action (create, update, delete, get)")
    title: Optional[str] = Field(None, description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: Optional[str] = Field(None, description="Task status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Task metadata")


class TaskResponse(BaseModel):
    """Response model for task management"""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    results: Optional[Dict[str, Any]] = Field(None, description="Task results")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class WorkflowRequest(BaseModel):
    """Request model for workflow orchestration"""
    workflow_id: Optional[str] = Field(None, description="Workflow ID")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    context: Optional[Dict[str, Any]] = Field(None, description="Workflow context")
    max_steps: Optional[int] = Field(None, description="Maximum workflow steps")


class WorkflowResponse(BaseModel):
    """Response model for workflow orchestration"""
    workflow_id: str = Field(..., description="Workflow ID")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    max_steps: int = Field(..., description="Maximum steps allowed")
    orchestration_ready: bool = Field(..., description="Whether orchestration is ready")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    brain_id: str = Field(..., description="Brain identifier")
    agent_initialized: bool = Field(..., description="Whether agent is initialized")
    supabase_connected: bool = Field(..., description="Supabase connection status")
    conversation_active: bool = Field(..., description="Conversation interface status")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    brain_name: str = Field(..., description="Brain name")
    agent_initialized: bool = Field(..., description="Whether agent is initialized")
    supabase_connected: bool = Field(..., description="Supabase connection status")
    integration_mode: str = Field(..., description="Integration mode")
    capabilities: List[str] = Field(..., description="Available capabilities")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")


class CodeGenerationRequest(BaseModel):
    """Request model for GLM code generation"""
    requirements: str = Field(..., description="Code requirements and specifications")
    context: Optional[str] = Field(None, description="Additional context about the system/project")
    verify_code: bool = Field(True, description="Whether to verify the generated code")


class CodeGenerationResponse(BaseModel):
    """Response model for GLM code generation"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="Whether generation was successful")
    generated_code: Optional[str] = Field(None, description="The generated code")
    verification_result: Optional[Dict[str, Any]] = Field(None, description="Code verification results")
    model_used: Optional[str] = Field(None, description="GLM model used for generation")
    thinking_enabled: Optional[bool] = Field(None, description="Whether thinking mode was enabled")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")


class SystemIntegrationRequest(BaseModel):
    """Request model for system integration"""
    integration_type: str = Field(..., description="Type of integration")
    target_system: str = Field(..., description="Target system")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Integration configuration")
    credentials: Optional[Dict[str, Any]] = Field(None, description="System credentials")


class SystemIntegrationResponse(BaseModel):
    """Response model for system integration"""
    success: bool = Field(..., description="Whether integration was successful")
    integration_type: str = Field(..., description="Type of integration")
    target_system: str = Field(..., description="Target system")
    capabilities_available: List[str] = Field(..., description="Available capabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
