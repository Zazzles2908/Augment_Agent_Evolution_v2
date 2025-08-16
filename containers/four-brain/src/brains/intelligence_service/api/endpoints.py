"""
Brain 3 FastAPI Endpoints
API endpoints for Augment Agent Integration service
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..brain3_manager import Brain3Manager
from ..config.settings import get_brain3_settings
from .models import (
    AugmentRequest, AugmentResponse, HealthResponse, 
    ModelStatusResponse, TaskRequest, TaskResponse,
    ConversationRequest, ConversationResponse,
    WorkflowRequest, WorkflowResponse,
    CodeGenerationRequest, CodeGenerationResponse,
    SystemIntegrationRequest, SystemIntegrationResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/brain3", tags=["Brain 3 - Augment Agent"])

# Global Brain 3 manager instance
brain3_manager: Brain3Manager = None


async def get_brain3_manager() -> Brain3Manager:
    """Dependency to get Brain 3 manager instance"""
    global brain3_manager
    if brain3_manager is None:
        settings = get_brain3_settings()
        brain3_manager = Brain3Manager(settings)
        # Initialize the manager
        await brain3_manager.initialize()
    return brain3_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(manager: Brain3Manager = Depends(get_brain3_manager)):
    """
    Health check endpoint for Brain 3 service
    Returns comprehensive health status including agent initialization and Supabase connection
    """
    try:
        health_data = await manager.health_check()
        
        return HealthResponse(
            status="healthy" if health_data.get("healthy", False) else "unhealthy",
            brain_id=manager.settings.brain_id,
            agent_initialized=health_data.get("agent_initialized", False),
            supabase_connected=health_data.get("supabase_connected", False),
            conversation_active=health_data.get("conversation_active", False),
            uptime_seconds=health_data.get("uptime_seconds", 0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/agent/status", response_model=ModelStatusResponse)
async def agent_status(manager: Brain3Manager = Depends(get_brain3_manager)):
    """
    Get detailed agent status and performance metrics
    """
    try:
        status_data = manager.get_status()
        
        return ModelStatusResponse(
            brain_name=status_data.get("brain_name", ""),
            agent_initialized=status_data.get("agent_initialized", False),
            supabase_connected=status_data.get("supabase_connected", False),
            integration_mode=status_data.get("integration_mode", ""),
            capabilities=status_data.get("capabilities", []),
            performance_metrics={
                "total_requests": status_data.get("total_requests", 0),
                "average_processing_time_ms": status_data.get("average_processing_time_ms", 0),
                "uptime_seconds": status_data.get("uptime_seconds", 0),
                "task_orchestrator": status_data.get("task_orchestrator", {})
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Agent status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent status check failed: {str(e)}")


@router.post("/process", response_model=AugmentResponse)
async def process_augment_request(
    request: AugmentRequest,
    manager: Brain3Manager = Depends(get_brain3_manager)
):
    """
    Process Augment Agent request
    
    This is the main functionality endpoint for Brain 3.
    Handles various types of Augment Agent tasks including conversation,
    task management, code generation, and workflow orchestration.
    """
    try:
        logger.info(f"🔄 Augment Agent request: {request.task_type}")
        
        # Validate agent is initialized
        if not manager.agent_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Brain 3 agent not initialized. Service unavailable."
            )
        
        # Convert request to dict format expected by manager
        request_dict = {
            "task_type": request.task_type,
            "conversation": request.conversation,
            "task_data": request.task_data,
            "code_request": request.code_request,
            "integration": request.integration,
            "workflow": request.workflow,
            "metadata": request.metadata or {}
        }
        
        # Process the request
        result = await manager.process_augment_request(request_dict)
        
        return AugmentResponse(
            result=result["result"],
            task_type=result["task_type"],
            processing_time_ms=result["processing_time_ms"],
            agent_info=result["agent_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Augment Agent processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/conversation", response_model=ConversationResponse)
async def process_conversation(
    request: ConversationRequest,
    manager: Brain3Manager = Depends(get_brain3_manager)
):
    """
    Process conversation request
    
    Specialized endpoint for conversation-based interactions with Augment Agent.
    Stores conversations in Supabase if connected.
    """
    try:
        logger.info(f"💬 Conversation request: {len(request.messages)} messages")
        
        # Validate agent is initialized
        if not manager.agent_initialized:
            raise HTTPException(
                status_code=503,
                detail="Brain 3 agent not initialized. Service unavailable."
            )
        
        # Create conversation request
        conversation_request = {
            "task_type": "conversation",
            "conversation": {
                "id": request.conversation_id,
                "messages": request.messages,
                "context": request.context,
                "max_length": request.max_length
            }
        }
        
        # Process the conversation
        result = await manager.process_augment_request(conversation_request)
        
        return ConversationResponse(
            conversation_id=result["result"]["conversation_id"],
            stored_in_supabase=result["result"]["stored_in_supabase"],
            message_count=result["result"]["message_count"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Conversation processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")


@router.post("/tasks", response_model=TaskResponse)
async def manage_task(
    request: TaskRequest,
    manager: Brain3Manager = Depends(get_brain3_manager)
):
    """
    Task management endpoint
    
    Handles task creation, updates, and management through Augment Agent.
    """
    try:
        logger.info(f"📋 Task management request: {request.action}")
        
        # Validate agent is initialized
        if not manager.agent_initialized:
            raise HTTPException(
                status_code=503,
                detail="Brain 3 agent not initialized. Service unavailable."
            )
        
        # Create task management request
        task_request = {
            "task_type": "task_management",
            "task_data": {
                "action": request.action,
                "task_id": request.task_id,
                "title": request.title,
                "description": request.description,
                "status": request.status,
                "metadata": request.metadata
            }
        }
        
        # Process the task
        result = await manager.process_augment_request(task_request)
        
        task_info = result["result"].get("task_info", {})
        
        return TaskResponse(
            task_id=result["result"]["task_id"],
            status=task_info.get("status", "unknown"),
            created_at=datetime.fromtimestamp(task_info.get("created_at", time.time())).isoformat(),
            completed_at=None,
            results=result["result"],
            error_message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task management failed: {str(e)}")


@router.post("/workflow", response_model=WorkflowResponse)
async def orchestrate_workflow(
    request: WorkflowRequest,
    manager: Brain3Manager = Depends(get_brain3_manager)
):
    """
    Workflow orchestration endpoint
    
    Handles complex workflow orchestration through Augment Agent.
    """
    try:
        logger.info(f"🔄 Workflow orchestration: {len(request.steps)} steps")
        
        # Validate agent is initialized
        if not manager.agent_initialized:
            raise HTTPException(
                status_code=503,
                detail="Brain 3 agent not initialized. Service unavailable."
            )
        
        # Create workflow request
        workflow_request = {
            "task_type": "workflow_orchestration",
            "workflow": {
                "id": request.workflow_id,
                "steps": request.steps,
                "context": request.context,
                "max_steps": request.max_steps
            }
        }
        
        # Process the workflow
        result = await manager.process_augment_request(workflow_request)
        
        return WorkflowResponse(
            workflow_id=result["result"]["workflow_id"],
            steps=result["result"]["steps"],
            max_steps=result["result"]["max_steps"],
            orchestration_ready=result["result"]["orchestration_ready"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Workflow orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow orchestration failed: {str(e)}")


@router.get("/metrics")
async def get_metrics(manager: Brain3Manager = Depends(get_brain3_manager)):
    """
    Get Brain 3 performance metrics in Prometheus format
    """
    try:
        status = manager.get_status()
        
        # Simple metrics in Prometheus format
        metrics = f"""# HELP brain3_requests_total Total number of Augment Agent requests
# TYPE brain3_requests_total counter
brain3_requests_total {status.get('total_requests', 0)}

# HELP brain3_processing_time_ms Average processing time in milliseconds
# TYPE brain3_processing_time_ms gauge
brain3_processing_time_ms {status.get('average_processing_time_ms', 0)}

# HELP brain3_agent_initialized Whether the agent is initialized
# TYPE brain3_agent_initialized gauge
brain3_agent_initialized {1 if status.get('agent_initialized', False) else 0}

# HELP brain3_supabase_connected Whether Supabase is connected
# TYPE brain3_supabase_connected gauge
brain3_supabase_connected {1 if status.get('supabase_connected', False) else 0}

# HELP brain3_active_tasks Number of active tasks
# TYPE brain3_active_tasks gauge
brain3_active_tasks {status.get('task_orchestrator', {}).get('active_tasks', 0)}
"""
        
        return JSONResponse(
            content=metrics,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@router.post("/generate-code", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    manager: Brain3Manager = Depends(get_brain3_manager)
):
    """
    Generate code using GLM-4.5 with verification

    This endpoint uses the integrated GLM-4.5 model to generate code based on
    requirements and optionally verify the generated code for correctness,
    security, and best practices.
    """
    try:
        logger.info(f"🔄 Code generation request: {request.requirements[:100]}...")

        # Validate agent is initialized
        if not manager.agent_initialized:
            raise HTTPException(status_code=503, detail="Brain 3 agent not initialized")

        # Check if GLM client is available
        if not manager.glm_client:
            raise HTTPException(status_code=503, detail="GLM client not available")

        # Convert request to dict
        request_dict = {
            "requirements": request.requirements,
            "context": request.context,
            "verify": request.verify_code
        }

        # Generate code using GLM
        result = await manager.generate_code_with_glm(request_dict)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Code generation failed"))

        return CodeGenerationResponse(
            success=True,
            generated_code=result["code"],
            verification_result=result.get("verification"),
            model_used=result["model_used"],
            thinking_enabled=result["thinking_enabled"],
            processing_time_ms=0,  # Could add timing if needed
            usage=result.get("usage", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")
