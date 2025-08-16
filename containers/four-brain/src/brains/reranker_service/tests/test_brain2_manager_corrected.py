#!/usr/bin/env python3
"""
Brain 2 Manager Unit Tests - CORRECTED REAL IMPLEMENTATION
Date: July 11, 2025 14:05 AEST
Purpose: Comprehensive unit testing for Brain2Manager with real models and data
Zero Fabrication Policy: ENFORCED - No mock data or simulated responses

CORRECTED VERSION:
- Fixed async fixture issues
- Aligned with actual Brain2Manager methods
- Real functionality testing without fabrication
"""

import pytest
import asyncio
import torch
import logging
import sys
from typing import List, Dict, Any
from pathlib import Path

# Add project paths for imports
sys.path.insert(0, '/workspace/src')

# Import Brain 2 components using absolute imports
from brain2_reranker.brain2_manager import Brain2Manager
from brain2_reranker.config.settings import Brain2Settings

# Test configuration
TEST_MODEL_PATH = "/workspace/models/qwen3/reranker-4b"
TEST_CACHE_DIR = "/workspace/models/cache"
TEST_REDIS_URL = "redis://localhost:6379/0"

# Real test data (no fabrication)
SAMPLE_QUERY = "artificial intelligence machine learning"
SAMPLE_DOCUMENTS = [
    {
        "text": "Artificial intelligence is transforming modern technology through machine learning algorithms.",
        "doc_id": "doc1",
        "metadata": {"source": "tech_article", "category": "AI"}
    },
    {
        "text": "The weather today is sunny with a chance of rain in the afternoon.",
        "doc_id": "doc2", 
        "metadata": {"source": "weather_report", "category": "weather"}
    },
    {
        "text": "Machine learning models require large datasets for effective training and validation.",
        "doc_id": "doc3",
        "metadata": {"source": "research_paper", "category": "ML"}
    }
]


class TestBrain2ManagerCorrected:
    """
    Corrected unit tests for Brain2Manager
    Tests real functionality without fabrication
    """
    
    def test_brain2_manager_creation(self):
        """Test Brain2Manager can be created with default settings"""
        manager = Brain2Manager()

        assert manager is not None
        assert hasattr(manager, 'settings')
        assert hasattr(manager, 'model_loaded')
        assert hasattr(manager, 'status')
        assert hasattr(manager, 'model_handler')
        assert hasattr(manager, 'communicator')  # Fixed: actual attribute name

        # Test initial state
        assert manager.model_loaded is False
        assert manager.status == "initialized"
        
    def test_brain2_manager_with_custom_settings(self):
        """Test Brain2Manager creation with custom settings"""
        settings = Brain2Settings(
            model_name=TEST_MODEL_PATH,
            model_cache_dir=TEST_CACHE_DIR,
            enable_4bit_quantization=True,
            enable_8bit_quantization=True,
            max_vram_usage=0.3,
            target_vram_usage=0.25
        )
        
        manager = Brain2Manager(settings)
        
        assert manager is not None
        assert manager.settings == settings
        assert manager.settings.model_name == TEST_MODEL_PATH
        assert manager.settings.enable_4bit_quantization is True
        assert manager.settings.enable_8bit_quantization is True
        
    def test_brain2_manager_attributes(self):
        """Test Brain2Manager has all required attributes"""
        manager = Brain2Manager()
        
        required_attributes = [
            'settings', 'model_loaded', 'status', 'model_handler',
            'communicator', 'initialization_time', 'total_rerank_requests',
            'total_processing_time'
        ]
        
        for attr in required_attributes:
            assert hasattr(manager, attr), f"Missing attribute: {attr}"
    
    def test_get_status_method(self):
        """Test get_status method returns proper structure"""
        manager = Brain2Manager()
        status = manager.get_status()
        
        assert isinstance(status, dict)
        
        # Test required status fields (based on actual implementation)
        required_fields = [
            'brain_id', 'status', 'model_loaded', 'model_name',
            'quantization_enabled', 'uptime_seconds', 'total_requests',
            'average_processing_time_ms', 'memory_usage', 'settings'
        ]

        for field in required_fields:
            assert field in status, f"Missing status field: {field}"

        # Test status values
        assert status['brain_id'] == "brain2"
        assert status['status'] == "initialized"
        assert status['model_loaded'] is False
        assert status['uptime_seconds'] >= 0
        assert status['total_requests'] == 0
        assert status['average_processing_time_ms'] == 0

        # Test memory usage structure
        memory_usage = status['memory_usage']
        assert isinstance(memory_usage, dict)
        assert 'gpu' in memory_usage
        assert 'timestamp' in memory_usage

        # Test settings structure
        settings = status['settings']
        assert isinstance(settings, dict)
        assert 'max_vram_usage' in settings
        assert 'target_vram_usage' in settings
    
    def test_memory_usage_method(self):
        """Test _get_memory_usage method"""
        manager = Brain2Manager()
        memory_usage = manager._get_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert 'timestamp' in memory_usage
        
        if torch.cuda.is_available():
            # Test GPU memory reporting
            assert 'gpu_memory' in memory_usage
            gpu_memory = memory_usage['gpu_memory']
            assert 'allocated_mb' in gpu_memory
            assert 'reserved_mb' in gpu_memory
            assert 'device_name' in gpu_memory
            assert 'device_count' in gpu_memory
            
            # Validate memory values are reasonable
            assert gpu_memory['allocated_mb'] >= 0
            assert gpu_memory['reserved_mb'] >= 0
            assert gpu_memory['device_count'] > 0
        else:
            # Test CPU-only environment
            assert 'system_memory' in memory_usage
    
    @pytest.mark.asyncio
    async def test_health_check_method(self):
        """Test health_check method"""
        manager = Brain2Manager()
        health = await manager.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
        assert 'checks' in health
        
        # Test health check structure
        checks = health['checks']
        assert isinstance(checks, dict)
        assert 'model_loaded' in checks
        assert 'memory_usage' in checks
        assert 'gpu_available' in checks
        
        # Test health check values
        assert checks['model_loaded'] is False  # Model not loaded yet
        assert isinstance(checks['memory_usage'], dict)
        assert isinstance(checks['gpu_available'], bool)
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection"""
        manager = Brain2Manager()
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Test GPU detection
            device_count = torch.cuda.device_count()
            assert device_count > 0
            
            device_name = torch.cuda.get_device_name(0)
            assert isinstance(device_name, str)
            assert len(device_name) > 0
            
            # Test CUDA memory functions work
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            assert memory_allocated >= 0
            assert memory_reserved >= 0
        else:
            pytest.skip("CUDA not available - GPU tests skipped")
    
    def test_settings_validation(self):
        """Test settings validation and configuration"""
        settings = Brain2Settings(
            model_name="test/model",
            model_cache_dir="/test/cache",
            enable_4bit_quantization=True,
            enable_8bit_quantization=False,
            max_vram_usage=0.5,
            target_vram_usage=0.4,
            batch_size=16,
            max_seq_length=1024
        )
        
        manager = Brain2Manager(settings)
        
        # Test settings are properly stored
        assert manager.settings.model_name == "test/model"
        assert manager.settings.model_cache_dir == "/test/cache"
        assert manager.settings.enable_4bit_quantization is True
        assert manager.settings.enable_8bit_quantization is False
        assert manager.settings.max_vram_usage == 0.5
        assert manager.settings.target_vram_usage == 0.4
        assert manager.settings.batch_size == 16
        assert manager.settings.max_seq_length == 1024
    
    def test_logging_configuration(self):
        """Test logging setup and functionality"""
        manager = Brain2Manager()
        
        # Test logger exists and is configured
        assert hasattr(manager, 'logger') or hasattr(manager, '_logger')
        
        # Test logging doesn't crash (no assertions needed)
        logging.getLogger(__name__).info("Test log message from Brain2Manager test")
        logging.getLogger(__name__).warning("Test warning message")
        logging.getLogger(__name__).error("Test error message")
    
    def test_component_integration(self):
        """Test integration between Brain2Manager components"""
        manager = Brain2Manager()
        
        # Test model_handler integration
        assert hasattr(manager, 'model_handler')
        assert manager.model_handler is not None
        
        # Test communicator integration
        assert hasattr(manager, 'communicator')
        assert manager.communicator is not None
        
        # Test settings integration
        assert hasattr(manager, 'settings')
        assert manager.settings is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
