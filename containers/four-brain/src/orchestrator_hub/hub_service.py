#!/usr/bin/env python3
"""
Orchestrator Hub Service
Global Strategy Coordinator for Four-Brain Architecture

This service coordinates the multi-brain stack (Docling, Embedding, Reranker, Generation),
reads from Redis channels, and publishes strategy plans.

Zero Fabrication Policy: ENFORCED.
"""

import os
import sys
import time
import asyncio
import logging
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

# Add paths for imports
sys.path.append('/workspace/src')

from orchestrator_hub.core.strategy_manager import StrategyManager
from orchestrator_hub.core.moonshot_client import MoonshotClient
from orchestrator_hub.communication.redis_coordinator import RedisCoordinator
from orchestrator_hub.core.supabase_manager import supabase_manager, UserProfile
from shared.redis_client import RedisStreamsClient, send_docling_request
# HRM removed from project; orchestration no longer depends on HRM

# (Optional) Import Blackwell Quantization System for model optimizations (if used)
try:
    import sys
    import os
    sys.path.append('/workspace/src')
    from core.quantization import blackwell_quantizer, FOUR_BRAIN_QUANTIZATION_CONFIG
    BLACKWELL_AVAILABLE = True
    logging.getLogger(__name__).info("✅ Blackwell quantization system imported successfully for orchestrator")
except ImportError as e:
    BLACKWELL_AVAILABLE = False
    logging.getLogger(__name__).warning(f"⚠️ Blackwell quantization not available: {e}")

# Pydantic models for API requests/responses
class DocumentProcessRequest(BaseModel):
    file_info: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class DocumentProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None

class SemanticSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None

class SemanticSearchResponse(BaseModel):
    task_id: str
    results: List[Dict[str, Any]]
    total_found: int
    processing_time_ms: float

class ChatEnhanceRequest(BaseModel):
    query: str
    context: Optional[str] = None
    user_id: Optional[str] = None
    personality_traits: Optional[Dict[str, Any]] = None

class ChatEnhanceResponse(BaseModel):
    task_id: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time_ms: float

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
strategy_manager: StrategyManager = None
redis_coordinator: RedisCoordinator = None
redis_streams_client: RedisStreamsClient = None
# HRM removed
# hrm_orchestrator removed
# blackwell_optimizer removed

# Task tracking
active_tasks: Dict[str, Dict[str, Any]] = {}

# Helper function to extract user_id from JWT token
async def get_user_id(authorization: str = Header(None)) -> Optional[str]:
    """Extract user_id from JWT token in Authorization header"""
    if not authorization:
        return None

    try:
        # Remove 'Bearer ' prefix if present
        token = authorization.replace('Bearer ', '') if authorization.startswith('Bearer ') else authorization

        # Decode JWT without verification for now (in production, verify with Supabase JWT secret)
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get('sub')  # 'sub' contains the user_id in Supabase JWTs
    except Exception as e:
        logger.warning(f"Failed to decode JWT token: {e}")
        return None

# Brain orchestration helper functions
async def orchestrate_document_processing(task_id: str, file_info: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrate document processing: Brain-4 -> Brain-1 -> Storage"""
    try:
        # Step 1: Send to Brain-4 (Docling) for document processing
        brain4_request = {
            "task_id": task_id,
            "file_info": file_info,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update task status
        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0.1,
            "stage": "document_extraction",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Send to Brain-4 via Redis Streams for document processing
        if redis_streams_client:
            try:
                # Extract file path from file_info (assuming it contains path or name)
                file_path = file_info.get("name", "unknown_document")
                document_type = file_info.get("type", "application/octet-stream")

                # Send document processing request to Brain-4
                message_id = await send_docling_request(
                    redis_streams_client,
                    document_path=file_path,
                    document_type=document_type,
                    processing_options={
                        "task_id": task_id,
                        "extract_tables": True,
                        "extract_images": True,
                        "generate_embeddings": True,
                        "metadata": metadata
                    }
                )

                logger.info(f"📄 Task {task_id}: Sent to Brain-4 via Redis Streams (message_id: {message_id})")

                # Update task status to indicate delegation successful
                active_tasks[task_id]["stage"] = "delegated_to_brain4"
                active_tasks[task_id]["progress"] = 0.2
                active_tasks[task_id]["brain4_message_id"] = message_id

            except Exception as delegation_error:
                logger.error(f"❌ Task {task_id}: Failed to delegate to Brain-4: {delegation_error}")
                active_tasks[task_id]["status"] = "delegation_failed"
                active_tasks[task_id]["error"] = f"Brain-4 delegation failed: {str(delegation_error)}"
                raise
        else:
            logger.warning(f"⚠️ Task {task_id}: Redis Streams client not available - cannot delegate to Brain-4")

        logger.info(f"📄 Task {task_id}: Document processing initiated")
        return {"status": "initiated", "stage": "document_extraction"}

    except Exception as e:
        logger.error(f"❌ Task {task_id}: Document processing failed: {e}")
        active_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat()
        }
        raise

async def orchestrate_semantic_search(task_id: str, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrate semantic search: Brain-1 (embedding) -> Brain-2 (reranking)"""
    try:
        # Step 1: Generate query embedding with Brain-1
        brain1_request = {
            "task_id": task_id,
            "text": query,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update task status
        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0.2,
            "stage": "embedding_generation",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # TODO: Send to Redis Stream for Brain-1
        # await redis_coordinator.send_to_brain1(brain1_request)

        logger.info(f"🔍 Task {task_id}: Semantic search initiated")
        return {"status": "initiated", "stage": "embedding_generation"}

    except Exception as e:
        logger.error(f"❌ Task {task_id}: Semantic search failed: {e}")
        active_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat()
        }
        raise

async def orchestrate_chat_enhancement(task_id: str, query: str, context: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Orchestrate chat enhancement: Brain-3 (Augment) with personalization"""
    try:
        # Prepare request for Brain-3 (Augment Intelligence)
        brain3_request = {
            "task_id": task_id,
            "query": query,
            "context": context,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update task status
        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0.3,
            "stage": "augment_processing",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Send request to Brain-3 (Augment Intelligence) using the /process endpoint
        import httpx
        brain3_url = "http://four-brain-brain3:8003/brain3/process"
        brain3_payload = {
            "task_type": "conversation",
            "conversation": {
                "id": task_id,
                "messages": [{"role": "user", "content": query}],
                "context": {"user_context": context, "user_id": user_id}
            },
            "metadata": {"source": "orchestrator_hub", "user_id": user_id}
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"💬 Task {task_id}: Sending request to Brain-3...")
                response = await client.post(brain3_url, json=brain3_payload)
                response.raise_for_status()
                brain3_result = response.json()
                logger.info(f"💬 Task {task_id}: Brain-3 response received (confidence: {brain3_result.get('confidence', 'N/A')})")
                return {"status": "completed", "stage": "augment_processing", "brain3_result": brain3_result}
        except Exception as brain3_error:
            logger.error(f"❌ Task {task_id}: Brain-3 communication failed: {brain3_error}")
            # Fallback to mock response if Brain-3 fails
            logger.info(f"💬 Task {task_id}: Using fallback response")
            return {"status": "completed", "stage": "augment_processing", "brain3_result": None}

    except Exception as e:
        logger.error(f"❌ Task {task_id}: Chat enhancement failed: {e}")
        active_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat()
        }
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for Orchestrator Hub Service"""
    global strategy_manager, redis_coordinator, redis_streams_client, hrm_orchestrator, blackwell_optimizer

    # Phase 1 smoke mode: skip heavy dependencies
    if os.getenv("PHASE1_SMOKE") == "1":
        logger.info("🧪 PHASE1_SMOKE=1: Skipping Moonshot/Redis/Supabase init")
        yield
        return

    # Startup
    logger.info("Starting Orchestrator Hub Service...")
    from datetime import datetime
    logger.info(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M AEST')}")

    try:
        # Initialize Moonshot client
        moonshot_client = MoonshotClient()
        await moonshot_client.initialize()

        # Initialize strategy manager
        strategy_manager = StrategyManager(moonshot_client)

        # Initialize Redis coordinator
        redis_coordinator = RedisCoordinator()
        await redis_coordinator.connect()

        # Initialize Redis Streams client for Brain communication
        redis_streams_client = RedisStreamsClient()
        await redis_streams_client.connect()
        logger.info("✅ Redis Streams client initialized for Brain communication")

        # Initialize Supabase database manager
        supabase_success = await supabase_manager.initialize()
        if supabase_success:
            logger.info("✅ Supabase database connection established")
        else:
            logger.warning("⚠️ Supabase database connection failed - continuing without database features")

        # HRM removed: Orchestrator no longer initializes HRM or Blackwell-specific optimizations here

        # Start the main coordination loop
        asyncio.create_task(coordination_loop())

        # Orchestrator initialized

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize Orchestrator Hub service: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Orchestrator Hub Service...")
        if redis_coordinator:
            await redis_coordinator.disconnect()
        if redis_streams_client:
            await redis_streams_client.disconnect()
        await supabase_manager.close()

async def coordination_loop():
    """Main coordination loop - The Mayor's decision-making process"""
    logger.info("🔄 Starting Orchestrator coordination loop...")

    while True:
        try:
            # Check for new vector jobs
            job = await redis_coordinator.get_next_vector_job()

            if job:
                logger.info(f"📋 Mayor received new job: {job.get('job_id')}")

                # Make strategy decision using Moonshot Kimi API
                strategy_plan = await strategy_manager.create_strategy_plan(job)

                # Publish strategy plan
                await redis_coordinator.publish_strategy_plan(strategy_plan)

                logger.info(f"📤 Mayor published strategy plan for job {job.get('job_id')}")

            # Brief pause to prevent busy waiting
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Coordination loop error: {e}")
            await asyncio.sleep(5)  # Longer pause on error

# Create FastAPI application
app = FastAPI(
    title="Orchestrator Hub Service",
    description="Global Strategy Coordinator for Four-Brain Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Integrate Triton-centric ResourceManager
try:
    from shared.triton_repository_client import TritonRepositoryClient
    from shared.resource_manager.triton_resource_manager import TritonResourceManager, ResourceManagerConfig
    _triton_client = TritonRepositoryClient(base_url=os.getenv("TRITON_URL", "http://triton:8000"))
    if os.getenv("PHASE1_SMOKE") == "1":
        cfg = ResourceManagerConfig(total_vram_gb=4.0, reserved_gb=1.0)
        _rm = TritonResourceManager(_triton_client, cfg)
    else:
        _rm = TritonResourceManager(_triton_client)
    logger.info("✅ Triton ResourceManager initialized")
except Exception as e:
    _triton_client = None
    _rm = None
    logger.error(f"❌ Failed to initialize Triton ResourceManager: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin endpoints for Triton model management
@app.get("/admin/models/status")
async def admin_models_status():
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    return _rm.status()

@app.post("/admin/models/load")
async def admin_models_load(payload: dict):
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    models = payload.get("models", [])
    _rm.ensure_loaded(models)
    return {"ok": True, "status": _rm.status()}

@app.post("/admin/models/unload")
async def admin_models_unload(payload: dict):
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model'")
    ok = _rm.unload(model)
    return {"ok": ok, "status": _rm.status()}


# Orchestrator endpoints (additive; keep /admin/* for backward compatibility)
@app.post("/orchestrator/models/ensure_loaded")
async def orchestrator_models_ensure_loaded(payload: dict):
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise HTTPException(status_code=400, detail="'models' must be a list")
    before = _rm.status()
    before_loaded = set(before.get("loaded", []))
    # Ensure requested models
    _rm.ensure_loaded(models)
    after = _rm.status()
    after_loaded = set(after.get("loaded", []))
    # Evicted = models that were previously loaded and are in registry but no longer loaded
    try:
        registry_models = set(_rm.cfg.registry.keys())
    except Exception:
        registry_models = set()
    evicted = sorted(list((before_loaded & registry_models) - after_loaded))
    return {
        "loaded": sorted(list(after_loaded)),
        "evicted": evicted,
        "available_gb": after.get("available_gb")
    }

@app.post("/orchestrator/models/unload")
async def orchestrator_models_unload(payload: dict):
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model'")
    if model in getattr(_rm, "cfg", {}).always_loaded if hasattr(_rm, "cfg") else {}:
        return {"status": "skipped", "reason": "always_on", "current": _rm.status()}
    ok = _rm.unload(model)
    return {"status": "unloaded" if ok else "failed", "current": _rm.status()}

@app.get("/orchestrator/metrics")
async def orchestrator_metrics():
    if _rm is None:
        raise HTTPException(status_code=503, detail="ResourceManager unavailable")
    try:
        total = _rm.cfg.total_vram_gb
        reserved = _rm.cfg.reserved_gb
        always_sum = sum(_rm.cfg.always_loaded.values())
        available = _rm.available_gb()
        used_dynamic = max(0.0, total - reserved - always_sum - available)
        vram_used_gb = max(0.0, reserved + always_sum + used_dynamic)
        util = (vram_used_gb / total) if total > 0 else 0.0
        soft_pressure = util >= _rm.cfg.soft_threshold
        hard_pressure = util >= _rm.cfg.hard_threshold
        return {
            "vram_used_gb": round(vram_used_gb, 2),
            "available_gb": round(available, 2),
            "utilization": round(util, 3),
            "soft_pressure": soft_pressure,
            "hard_pressure": hard_pressure,
            "loaded": _rm.status().get("loaded", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {e}")

# HRM admin endpoints removed

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Orchestrator Hub Service",
        "version": "1.0.0",
        "status": "operational",
        "port": 9098,
        "role": "Global Strategy Coordinator",
        "capabilities": [
            "strategy_decision_making",
            
            "moonshot_kimi_integration",
            "redis_coordination",
            "brain_allocation_optimization",
            "workflow_orchestration"
        ],
        "channels": {
            "input": "vector_jobs",
            "output": "strategy_plans"
        },
        "zero_fabrication": True
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    global strategy_manager, redis_coordinator

    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "orchestrator_hub",
        "version": "1.0.0",
        "components": {}
    }

    try:
        # Check Strategy Manager
        if strategy_manager:
            manager_health = await strategy_manager.health_check()
            health_status["components"]["strategy_manager"] = manager_health
        else:
            health_status["components"]["strategy_manager"] = {
                "status": "not_initialized",
                "healthy": False
            }

        # Check Redis Coordinator
        if redis_coordinator:
            coordinator_health = await redis_coordinator.health_check()
            health_status["components"]["redis_coordinator"] = coordinator_health
        else:
            health_status["components"]["redis_coordinator"] = {
                "status": "not_initialized",
                "healthy": False
            }

        # Overall health assessment
        component_health = [comp.get("healthy", False) for comp in health_status["components"].values()]
        overall_healthy = all(component_health) and len(component_health) > 0

        health_status["status"] = "healthy" if overall_healthy else "degraded"
        health_status["healthy"] = overall_healthy

        return health_status

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }

# High-Level Orchestration API Endpoints

@app.post("/api/v1/process-document", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    """
    Process document through Four-Brain pipeline: Brain-4 -> Brain-1 -> Storage
    Accepts file info and orchestrates complete document processing workflow.
    """
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"📄 Starting document processing task: {task_id}")

        # Orchestrate document processing
        result = await orchestrate_document_processing(
            task_id=task_id,
            file_info=request.file_info,
            metadata=request.metadata
        )

        return DocumentProcessResponse(
            task_id=task_id,
            status="processing",
            message="Document processing initiated successfully",
            estimated_completion="2-5 minutes"
        )

    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/api/v1/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest, user_id: str = Depends(get_user_id)):
    """
    Perform semantic search: Brain-1 (embedding) -> Brain-2 (reranking)
    Returns ranked results based on semantic similarity from user's knowledge base.
    """
    try:
        task_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"🔍 Starting semantic search task: {task_id} for user: {user_id}")

        # Log task to database
        if user_id:
            await supabase_manager.log_task(
                task_id=task_id,
                user_id=user_id,
                task_type="semantic_search",
                input_data={"query": request.query, "limit": request.limit},
                status="processing"
            )

        # Generate query embedding with Brain-1
        try:
            # Attempt to get real embedding from Brain-1
            from ..brains.embedding_service.core.brain1_manager import Brain1Manager
            brain1_manager = Brain1Manager()

            # Generate real embedding
            embedding_result = await brain1_manager.generate_embedding(request.query)

            if not embedding_result or 'embedding' not in embedding_result:
                logger.error(f"❌ Task {task_id}: Brain-1 embedding generation failed")
                raise HTTPException(
                    status_code=503,
                    detail="PROCESSING FAILED: Brain-1 embedding service unavailable"
                )

            query_embedding = embedding_result['embedding']
            logger.info(f"✅ Task {task_id}: Generated {len(query_embedding)}-dim embedding")

        except ImportError:
            logger.error(f"❌ Task {task_id}: Brain-1 service not available")
            raise HTTPException(
                status_code=503,
                detail="PROCESSING FAILED: Brain-1 embedding service not available"
            )
        except Exception as e:
            logger.error(f"❌ Task {task_id}: Embedding generation failed: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"PROCESSING FAILED: Embedding generation error - {str(e)}"
            )

        # Search similar embeddings in database
        similar_results = []
        if user_id:
            similar_results = await supabase_manager.search_similar_embeddings(
                user_id=user_id,
                query_embedding=query_embedding,
                limit=request.limit or 10
            )

        # Convert database results to API response format
        results = []
        for i, result in enumerate(similar_results):
            results.append({
                "id": f"result_{i}",
                "title": f"Document Chunk {i+1}",
                "content": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "score": result["similarity"],
                "metadata": {
                    "created_at": result["created_at"],
                    "full_text": result["text"]
                }
            })

        processing_time = (time.time() - start_time) * 1000

        # Update task status
        if user_id:
            await supabase_manager.update_task_status(
                task_id=task_id,
                status="completed",
                output_data={"results_count": len(results)},
                processing_time_ms=int(processing_time)
            )

        return SemanticSearchResponse(
            task_id=task_id,
            results=results,
            total_found=len(results),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"❌ Semantic search failed: {e}")
        if user_id:
            await supabase_manager.update_task_status(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.post("/api/v1/chat-enhance", response_model=ChatEnhanceResponse)
async def chat_enhance(request: ChatEnhanceRequest):
    """
    Enhance chat interaction: Brain-3 (Augment) with personalization
    Provides intelligent, context-aware responses with user personalization.
    """
    try:
        task_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"💬 Starting chat enhancement task: {task_id}")

        # Orchestrate chat enhancement
        result = await orchestrate_chat_enhancement(
            task_id=task_id,
            query=request.query,
            context=request.context,
            user_id=request.user_id
        )

        processing_time = (time.time() - start_time) * 1000

        # Use actual Brain-3 results if available, otherwise fallback
        brain3_result = result.get("brain3_result")
        if brain3_result:
            # Extract response from Brain-3 AugmentResponse format
            brain3_data = brain3_result.get("result", {})
            brain3_response = brain3_data.get("ai_response", brain3_data.get("response", "Brain-3 processed request successfully"))
            brain3_confidence = brain3_data.get("confidence", 0.8)
            brain3_sources = brain3_data.get("sources", [{"type": "brain3", "title": "Augment Intelligence", "confidence": 0.8}])

            return ChatEnhanceResponse(
                task_id=task_id,
                response=brain3_response,
                confidence=brain3_confidence,
                sources=brain3_sources,
                processing_time_ms=processing_time
            )
        else:
            # Fallback response if Brain-3 communication failed
            return ChatEnhanceResponse(
                task_id=task_id,
                response="Brain-3 communication failed - using fallback response",
                confidence=0.3,
                sources=[
                    {
                        "type": "fallback",
                        "title": "System Fallback",
                        "confidence": 0.3
                    }
                ],
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.error(f"❌ Chat enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat enhancement failed: {str(e)}")

@app.get("/api/v1/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of asynchronous task by task ID.
    Returns current progress, results, or error information.
    """
    try:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task_info = active_tasks[task_id]

        return TaskStatusResponse(
            task_id=task_id,
            status=task_info.get("status", "unknown"),
            progress=task_info.get("progress", 0.0),
            result=task_info.get("result"),
            error=task_info.get("error"),
            created_at=task_info.get("created_at", ""),
            updated_at=task_info.get("updated_at", "")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task status retrieval failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Service metrics endpoint in Prometheus format"""
    global strategy_manager, redis_coordinator

    try:
        # Get current timestamp
        current_time = time.time()

        # Default values
        total_jobs_processed = 0
        avg_decision_time = 0
        uptime_seconds = 0
        active_strategies = 0
        moonshot_api_calls = 0
        redis_messages_sent = 0
        redis_messages_received = 0
        connected = 0

        # Strategy manager metrics
        if strategy_manager:
            status = strategy_manager.get_status()
            total_jobs_processed = status.get("total_jobs_processed", 0)
            avg_decision_time = status.get("average_decision_time_ms", 0)
            uptime_seconds = status.get("uptime_seconds", 0)
            active_strategies = status.get("active_strategies", 0)
            moonshot_api_calls = status.get("moonshot_api_calls", 0)

        # Redis coordinator metrics
        if redis_coordinator:
            redis_stats = redis_coordinator.get_stats()
            redis_messages_sent = redis_stats.get("messages_sent", 0)
            redis_messages_received = redis_stats.get("messages_received", 0)
            connected = 1 if redis_stats.get("connected", False) else 0

        # Prometheus format metrics (renamed from legacy k2_hub_* to orchestrator_hub_*)
        metrics = f"""# HELP orchestrator_hub_uptime_seconds Service uptime in seconds
# TYPE orchestrator_hub_uptime_seconds gauge
orchestrator_hub_uptime_seconds {uptime_seconds}

# HELP orchestrator_hub_jobs_processed_total Total number of jobs processed
# TYPE orchestrator_hub_jobs_processed_total counter
orchestrator_hub_jobs_processed_total {total_jobs_processed}

# HELP orchestrator_hub_decision_time_ms Average decision time in milliseconds
# TYPE orchestrator_hub_decision_time_ms gauge
orchestrator_hub_decision_time_ms {avg_decision_time}

# HELP orchestrator_hub_active_strategies Number of active strategies
# TYPE orchestrator_hub_active_strategies gauge
orchestrator_hub_active_strategies {active_strategies}

# HELP orchestrator_hub_moonshot_api_calls_total Total Moonshot API calls
# TYPE orchestrator_hub_moonshot_api_calls_total counter
orchestrator_hub_moonshot_api_calls_total {moonshot_api_calls}

# HELP orchestrator_hub_redis_messages_sent_total Total Redis messages sent
# TYPE orchestrator_hub_redis_messages_sent_total counter
orchestrator_hub_redis_messages_sent_total {redis_messages_sent}

# HELP orchestrator_hub_redis_messages_received_total Total Redis messages received
# TYPE orchestrator_hub_redis_messages_received_total counter
orchestrator_hub_redis_messages_received_total {redis_messages_received}

# HELP orchestrator_hub_connected Connection status (1=connected, 0=disconnected)
# TYPE orchestrator_hub_connected gauge
orchestrator_hub_connected {connected}
"""

        return Response(
            content=metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"❌ Metrics collection failed: {e}")
        error_metrics = f"""# HELP orchestrator_hub_error Error status
# TYPE orchestrator_hub_error gauge
orchestrator_hub_error 1
"""
        return Response(
            content=error_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Orchestrator Hub Service...")

    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9098"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )
