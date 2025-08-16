"""
Main entry point for Brain 4 (Docling) integration
FastAPI application with comprehensive document processing capabilities
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn
import time

# Support running as script (python src/brains/document_processor/main.py)
import sys as _sys, os as _os
_pkg_root = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_root not in _sys.path:
    _sys.path.append(_pkg_root)
from brains.document_processor.config.settings import settings, configure_logging
from brains.document_processor.document_manager import Brain4Manager
from brains.document_processor.core.end_to_end_workflow import EndToEndWorkflowOrchestrator
from .api.health import router as health_router
from .api.documents import router as documents_router
from .api.monitoring import router as monitoring_router
from .api.prometheus_metrics import router as prometheus_router
from .models.document_models import HealthCheck, ProcessingResponse

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Global Brain 4 manager instance
brain4_manager: Brain4Manager = None
brain4_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    global brain4_manager
    
    # Startup
    logger.info("Starting Brain 4 application...")

    # Initialize VRAM management first
    logger.info("🎮 Initializing VRAM management for Brain 4 (15% allocation)...")
    import sys
    sys.path.append('/workspace/src')
    from shared.gpu.vram_manager import initialize_vram_management
    vram_manager = initialize_vram_management('document', start_monitoring=True)
    logger.info(f"✅ VRAM management initialized: {vram_manager.allocated_vram_gb:.1f}GB allocated")

    try:
        # Initialize Brain 4 manager
        # Optional dev auto-migration to ensure local DB schema exists
        import os
        if os.getenv('MIGRATE_ON_START', 'true').lower() in ('1','true','yes'):
            try:
                from brains.document_processor.db.auto_migrate import run_auto_migrations
                ok = await run_auto_migrations(settings.database_url)
                if ok:
                    logger.info("✅ Auto-migrations applied (augment_agent schema)")
                else:
                    logger.warning("⚠️ Auto-migrations failed; proceeding anyway")
            except Exception as me:
                logger.warning(f"⚠️ Auto-migration error: {me}")

        brain4_manager = Brain4Manager(settings)
        await brain4_manager.start()

        logger.info("Brain 4 application started successfully")

        yield
        
    except Exception as e:
        logger.error(f"Failed to start Brain 4 application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Brain 4 application...")
        
        if brain4_manager:
            await brain4_manager.stop()
        
        logger.info("Brain 4 application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Brain 4 - Document Processing Intelligence",
    description="Advanced document processing with Docling integration for Four-Brain AI System",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(documents_router, prefix="/api/v1", tags=["documents"])
app.include_router(monitoring_router, prefix="/api/v1", tags=["monitoring"])
app.include_router(prometheus_router, tags=["metrics"])

# Dependency to get Brain 4 manager
async def get_brain4_manager() -> Brain4Manager:
    """Dependency to get the Brain 4 manager instance"""
    
    if brain4_manager is None:
        raise HTTPException(status_code=503, detail="Brain 4 manager not initialized")
    
    return brain4_manager

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    
    return {
        "service": "Brain 4 - Document Processing Intelligence",
        "version": "1.0.0",
        "status": "operational",
        "brain_id": "brain4",
        "capabilities": [
            "document_processing",
            "multi_format_parsing", 
            "structure_extraction",
            "vector_generation",
            "inter_brain_communication"
        ],
        "supported_formats": settings.supported_formats,
        "api_docs": "/docs" if settings.debug else "disabled"
    }

# Root health endpoint (for compatibility) - FIXED: Proper health verification
@app.get("/health")
async def root_health():
    """Root health endpoint with proper health verification"""
    try:
        # Use the proper health check from the health router
        from .api.health import _check_brain4_manager

        # Quick health check
        brain4_ready = await _check_brain4_manager()

        if brain4_ready:
            return {
                "status": "healthy",
                "service": "Brain 4 - Document Processing Intelligence",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": True,
                "message": "For detailed health information, use /api/v1/health"
            }
        else:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready",
                "service": "Brain 4 - Document Processing Intelligence",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": False,
                "message": "Brain 4 manager not ready"
            })
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "status": "error",
            "service": "Brain 4 - Document Processing Intelligence",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })

# Note: Prometheus metrics are now handled by the prometheus_router at /prometheus-metrics

# Quick document processing endpoint
@app.post("/api/v1/process", response_model=ProcessingResponse)
async def process_document(
    file: UploadFile = File(...),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=True),
    generate_embeddings: bool = Form(default=True),
    manager: Brain4Manager = Depends(get_brain4_manager)
):
    """
    Quick document processing endpoint
    
    Args:
        file: Document file to process
        extract_tables: Whether to extract tables
        extract_images: Whether to extract images
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        Processing response with task ID
    """
    
    try:
        # Validate file size
        if file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.max_file_size_mb}MB"
            )
        
        # Validate file format
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.supported_formats:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format: {file_extension}"
            )
        
        # Save uploaded file
        temp_file_path = settings.temp_dir / f"upload_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Submit processing task
        task_id = await manager.submit_document_task(
            file_path=str(temp_file_path),
            extract_tables=extract_tables,
            extract_images=extract_images,
            generate_embeddings=generate_embeddings
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status="submitted",
            message="Document processing task submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# End-to-End Four-Brain Workflow endpoint - AUTHENTIC IMPLEMENTATION
@app.post("/api/v1/workflow/end-to-end")
async def process_document_end_to_end(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}")
):
    """
    Process document through complete Four-Brain pipeline - AUTHENTIC IMPLEMENTATION

    This endpoint orchestrates the complete Four-Brain processing workflow:
    1. Brain 4 (Docling) - Document processing and content extraction
    2. Brain 1 (Embedding) - Qwen3-4B embedding generation
    3. Brain 2 (Wisdom) - Wisdom analysis and knowledge extraction
    4. Brain 3 (Execution) - Action planning and execution coordination
    5. Data storage and finalization

    Returns real processing results with honest status reporting
    """
    try:
        # Parse metadata
        import json
        try:
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            doc_metadata = {}

        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Save file temporarily for workflow processing
        import tempfile
        from pathlib import Path

        temp_dir = Path(settings.temp_dir)
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / f"workflow_{int(time.time())}_{file.filename}"

        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        # Initialize workflow orchestrator
        workflow_orchestrator = EndToEndWorkflowOrchestrator(settings)
        await workflow_orchestrator.initialize()

        try:
            # Process document through complete Four-Brain pipeline
            workflow_result = await workflow_orchestrator.process_document_end_to_end(
                file_path=str(temp_file_path),
                metadata=doc_metadata
            )

            # Return authentic workflow results
            return {
                "workflow_id": workflow_result["workflow_id"],
                "status": workflow_result["status"],
                "total_processing_time": workflow_result.get("total_processing_time", 0),
                "brain_contributions": workflow_result["brain_contributions"],
                "final_results": workflow_result["final_results"],
                "stages_completed": len(workflow_result["stages"]),
                "fabrication_check": workflow_result["fabrication_check"],
                "timestamp": datetime.now().isoformat(),
                "message": "End-to-end Four-Brain processing completed"
            }

        finally:
            # Cleanup
            await workflow_orchestrator.close()
            if temp_file_path.exists():
                temp_file_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in end-to-end workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow processing failed: {str(e)}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=True
    )
