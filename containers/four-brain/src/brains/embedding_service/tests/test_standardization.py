#!/usr/bin/env python3
"""
Brain1 Standardization Test
Quick validation script to test the new standardized structure.

Created: 2025-07-13 AEST
Author: Augment Agent Evolution - Brain Architecture Standardization
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all the new import paths."""
    logger.info("🧪 Testing Brain1 standardized imports...")
    
    try:
        # Test core imports
        from brains.embedding_service.embedding_manager import Brain1Manager
        logger.info("✅ Brain1Manager (modular) import successful")

        from brains.embedding_service.core.model_loader import ModelLoader
        logger.info("✅ ModelLoader import successful")

        from brains.embedding_service.core.database_manager import get_database_manager
        logger.info("✅ DatabaseManager import successful")

        from brains.embedding_service.core.resource_monitor import ResourceMonitor
        logger.info("✅ ResourceMonitor import successful")

        # Test config imports
        from brains.embedding_service.config.settings import brain1_settings
        logger.info("✅ Brain1Settings import successful")

        # Test API imports
        from brains.embedding_service.api.models import EmbeddingRequest, EmbeddingResponse
        logger.info("✅ API models import successful")
        
        # Test communication imports
        from communication.brain_communicator import BrainCommunicator
        logger.info("✅ BrainCommunicator import successful")
        
        logger.info("🎯 All imports successful! Brain1 standardization complete.")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_brain1_manager_instantiation():
    """Test Brain1Manager can be instantiated."""
    try:
        from brains.embedding_service.embedding_manager import Brain1Manager

        # Test instantiation with default settings (no start)
        brain1 = Brain1Manager()
        logger.info("✅ Brain1Manager instantiation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Brain1Manager instantiation failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Brain1 standardization validation...")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test instantiation
    instantiation_ok = test_brain1_manager_instantiation()
    
    if imports_ok and instantiation_ok:
        logger.info("🎉 Brain1 standardization validation PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 Brain1 standardization validation FAILED!")
        sys.exit(1)
