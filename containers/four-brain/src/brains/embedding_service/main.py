#!/usr/bin/env python3
"""
Brain 1 Embedding Service - Main Entry Point
Docker container entry point for the Four-Brain embedding service.

This file serves as the main entry point for the Docker container,
initializing and starting the Brain 1 embedding service with proper
error handling and graceful shutdown.

Created: 2025-08-06 AEST
Author: Zazzles's Agent - Four-Brain System v2
"""

import asyncio
import logging
import os
import signal
import sys
import time
import uvicorn
from pathlib import Path

# Add the workspace to Python path
sys.path.insert(0, '/workspace/src')
sys.path.insert(0, '/workspace')

# Configure logging early
# Ensure logs directory exists and is writable
import os
os.makedirs('/workspace/logs', exist_ok=True)

# Try to create log file handler, fall back to stdout only if permission denied
log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    log_handlers.append(logging.FileHandler('/workspace/logs/embedding_service.log', mode='a'))
except PermissionError:
    print("Warning: Cannot write to log file, using stdout only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def validate_environment():
    """Validate required environment variables and paths"""
    logger.info("🔍 Validating environment...")
    
    # Check required environment variables
    required_env_vars = [
        'BRAIN_ROLE',
        'PORT',
        'POSTGRES_URL',
        'REDIS_URL',
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    # Check required directories
    required_dirs = [
        '/workspace/src',
        '/workspace/logs',
        '/workspace/.cache',
        '/workspace/models'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"⚠️ Creating missing directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    logger.info("✅ Environment validation completed")
    return True

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"✅ GPU available: {device_name} (devices: {device_count})")
            
            # Check VRAM
            if device_count > 0:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_gb = total_memory / (1024**3)
                logger.info(f"📊 GPU Memory: {total_gb:.1f} GB total")
            
            return True
        else:
            logger.warning("⚠️ CUDA not available, running in CPU mode")
            return False
    except ImportError:
        logger.warning("⚠️ PyTorch not available, GPU check skipped")
        return False
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False

def main():
    """Main entry point for the embedding service"""
    logger.info("🚀 Starting Brain 1 Embedding Service...")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🔧 Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed, exiting...")
        sys.exit(1)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Get configuration from environment
    brain_role = os.getenv('BRAIN_ROLE', 'embedding')
    port = int(os.getenv('PORT', 8001))
    host = os.getenv('HOST', '0.0.0.0')
    log_level = os.getenv('LOG_LEVEL', 'INFO').lower()
    workers = int(os.getenv('WORKERS', 1))
    
    logger.info(f"🧠 Brain Role: {brain_role}")
    logger.info(f"🌐 Host: {host}:{port}")
    logger.info(f"📝 Log Level: {log_level}")
    logger.info(f"👥 Workers: {workers}")
    
    try:
        # Import and start the FastAPI application
        logger.info("📦 Importing embedding service...")
        from brains.embedding_service.embedding_service import app
        
        logger.info("🎯 Starting FastAPI server...")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
            workers=workers,
            reload=False,  # Disable reload in production
            access_log=True,
            use_colors=True,
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        
        # Start the server
        logger.info(f"🎉 Brain 1 Embedding Service starting on {host}:{port}")
        server.run()
        
    except ImportError as e:
        logger.error(f"❌ Failed to import embedding service: {e}")
        logger.error("💡 Check if all dependencies are installed and paths are correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start embedding service: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
