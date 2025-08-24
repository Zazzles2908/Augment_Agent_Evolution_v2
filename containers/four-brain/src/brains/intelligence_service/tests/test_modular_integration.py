"""
Test Modular Integration - Verify Brain-3 AI Fix
Tests that hardcoded responses have been replaced with real AI

This test verifies that the modular integration successfully
replaces hardcoded if-else statements with real AI processing.

Created: 2025-07-29 AEST
Purpose: Verify Priority 1.1 fix is working correctly
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the modules we're testing
from ..modules.ai_interface import AIInterface
from ..modules.response_generator import ResponseGenerator
from ..modules.fallback_handler import FallbackHandler
from ..modules.config_manager import ConfigManager
from ..modules.brain3_integration import Brain3Integration

logger = logging.getLogger(__name__)


class TestModularIntegration:
    """Test suite for modular Brain-3 integration"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'augment_api': {
                'api_key': 'test_key_12345',
                'api_url': 'https://api.test.com/v1',
                'model': 'test-model',
                'timeout_seconds': 30,
                'max_tokens': 1000,
                'temperature': 0.7
            },
            'brain3': {
                'fallback_enabled': True,
                'max_context_length': 5,
                'response_cache_enabled': True
            }
        }
    
    @pytest.fixture
    def mock_ai_interface(self, mock_config):
        """Mock AI interface for testing"""
        ai_interface = Mock(spec=AIInterface)
        ai_interface.initialize = AsyncMock(return_value=True)
        ai_interface.generate_response = AsyncMock(return_value={
            'response': 'This is a real AI response, not hardcoded!',
            'confidence': 0.9,
            'sources': [{'type': 'real_ai', 'title': 'AI Processing', 'confidence': 0.9}],
            'reasoning': 'Generated using real AI capabilities',
            'processing_time_ms': 500,
            'model_used': 'test-model'
        })
        ai_interface.get_metrics = Mock(return_value={
            'initialized': True,
            'request_count': 1,
            'average_response_time_ms': 500
        })
        return ai_interface
    
    @pytest.mark.asyncio
    async def test_ai_interface_initialization(self, mock_config):
        """Test that AI interface initializes correctly"""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful API test
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            ai_interface = AIInterface(mock_config['augment_api'])
            result = await ai_interface.initialize()
            
            assert result is True
            assert ai_interface.initialized is True
            logger.info("✅ AI Interface initialization test passed")
    
    @pytest.mark.asyncio
    async def test_response_generator_intelligence(self, mock_ai_interface):
        """Test that response generator uses AI instead of hardcoded responses"""
        response_generator = ResponseGenerator(mock_ai_interface)
        
        # Test with a message that would trigger hardcoded response in old system
        test_message = "hello there"
        result = await response_generator.generate_response(test_message)
        
        # Verify it's using AI, not hardcoded responses
        assert result['response'] == 'This is a real AI response, not hardcoded!'
        assert result['confidence'] > 0.8
        assert result['response_type'] == 'intelligent_ai_generated'
        assert 'intent' in result
        
        # Verify AI interface was called
        mock_ai_interface.generate_response.assert_called_once()
        
        logger.info("✅ Response Generator intelligence test passed")
    
    @pytest.mark.asyncio
    async def test_fallback_handler_graceful_degradation(self, mock_config):
        """Test that fallback handler provides graceful degradation"""
        fallback_handler = FallbackHandler(mock_config['brain3'])
        
        # Test fallback response for greeting
        result = await fallback_handler.handle_api_failure(
            "hello", "greeting", "API unavailable", {}
        )
        
        assert result['response_type'] == 'intelligent_fallback'
        assert result['confidence'] > 0.5  # Should be reasonable confidence
        assert 'fallback' in result['response'].lower()
        assert result['health_status'] in ['healthy', 'degraded', 'critical']
        
        logger.info("✅ Fallback Handler graceful degradation test passed")
    
    def test_config_manager_validation(self):
        """Test that config manager validates configuration properly"""
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Should have default configuration
        assert 'augment_api' in config
        assert 'brain3' in config
        assert 'performance' in config
        assert 'security' in config
        
        # Test API key handling
        with patch.dict('os.environ', {'AUGMENT_API_KEY': 'test_key_from_env'}):
            api_key = config_manager.get_api_key()
            assert api_key == 'test_key_from_env'
        
        logger.info("✅ Config Manager validation test passed")
    
    @pytest.mark.asyncio
    async def test_brain3_integration_replaces_hardcoded(self, mock_config):
        """Test that Brain3Integration completely replaces hardcoded responses"""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful API responses
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'choices': [{'message': {'content': 'Real AI response from integration test'}}]
            })
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            # Create integration
            integration = Brain3Integration()
            await integration.initialize()
            
            # Test messages that would trigger hardcoded responses in old system
            test_messages = [
                "hello",
                "four-brain system",
                "test",
                "random question about anything"
            ]
            
            for message in test_messages:
                result = await integration.process_message(message)
                
                # Verify it's NOT using hardcoded responses
                assert 'Real AI response' in result['response']
                assert result['processing_method'] == 'modular_ai_integration'
                assert result['brain_integration_version'] == '2.0.0'
                assert 'request_id' in result
                
                # Verify it's not the old hardcoded patterns
                old_patterns = [
                    "✅ Yes! The Four-Brain system is working",
                    "Hello! I'm Brain-3, the Zazzles's Agent Intelligence component",
                    "Test successful! Brain-3 is operational"
                ]
                for pattern in old_patterns:
                    assert pattern not in result['response']
            
            logger.info("✅ Brain3Integration hardcoded replacement test passed")
    
    @pytest.mark.asyncio
    async def test_integration_health_monitoring(self):
        """Test that integration provides comprehensive health monitoring"""
        integration = Brain3Integration()
        await integration.initialize()
        
        # Get health status
        health = await integration.health_check()
        
        # Verify comprehensive health information
        assert 'brain_integration' in health
        assert 'ai_interface' in health
        assert 'fallback_handler' in health
        assert 'configuration' in health
        
        # Verify health metrics
        brain_health = health['brain_integration']
        assert 'status' in brain_health
        assert 'total_requests' in brain_health
        assert 'success_rate' in brain_health
        
        logger.info("✅ Integration health monitoring test passed")
    
    @pytest.mark.asyncio
    async def test_system_info_accuracy(self):
        """Test that system info accurately reflects modular architecture"""
        integration = Brain3Integration()
        await integration.initialize()
        
        system_info = await integration.get_system_info()
        
        # Verify accurate system information
        assert system_info['brain_id'] == 'brain-3'
        assert system_info['integration_version'] == '2.0.0'
        assert system_info['architecture'] == 'modular_ai_integration'
        
        # Verify all components are listed
        components = system_info['components']
        assert 'ai_interface' in components
        assert 'response_generator' in components
        assert 'fallback_handler' in components
        assert 'config_manager' in components
        
        # Verify capabilities reflect real AI
        capabilities = system_info['capabilities']
        assert 'Real AI processing (not hardcoded)' in capabilities
        assert 'Context-aware responses' in capabilities
        assert 'Intent analysis' in capabilities
        
        logger.info("✅ System info accuracy test passed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
    logger.info("🎯 All modular integration tests completed!")
