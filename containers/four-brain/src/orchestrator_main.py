#!/usr/bin/env python3
"""
Orchestrator Hub Main Entry Point
Starts the Orchestrator Hub service for Four-Brain coordination
"""

import os
import sys

# Add paths for imports
sys.path.insert(0, '/workspace/src')
sys.path.insert(0, '/workspace')

# Check if this is the Orchestrator container
brain_role = os.getenv("BRAIN_ROLE", "")

if brain_role == "orchestrator":
    # Run Orchestrator service
    from orchestrator_hub.hub_service import app
    import uvicorn
    import logging

    logger = logging.getLogger(__name__)
    logger.info("🏛️ Starting Orchestrator Hub Service (Four-Brain Coordinator)...")
    
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
else:
    # Run unified main.py for other containers
    from main import app
    import uvicorn
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Enhanced Four-Brain System...")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
