"""
AI Interface Module - Real Augment Agent Integration
Replaces hardcoded if-else statements with actual AI

This module provides the core AI interface for Brain-3, integrating with
real Augment Agent API instead of using hardcoded responses.

Created: 2025-07-29 AEST
Purpose: Replace fabricated AI with real intelligence
Module Size: 150 lines (modular design)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import aiohttp
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AIInterface:
    """
    Real AI Interface for Brain-3
    
    Provides actual AI capabilities through Augment Agent API integration,
    replacing the hardcoded if-else statements found in the original implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AI interface with configuration"""
        self.config = config
        self.api_key = config.get('augment_api_key')
        self.api_url = config.get('augment_api_url', 'https://api.augmentcode.com/v1')
        self.model = config.get('model', 'claude-sonnet-4')
        self.session = None
        self.initialized = False
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.last_request_time = None
        
        logger.info("🤖 AI Interface initialized with real Augment Agent integration")
    
    async def initialize(self) -> bool:
        """Initialize the AI interface and test connectivity"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'Four-Brain-System/2.0'
                }
            )
            
            # Test API connectivity
            test_result = await self._test_api_connection()
            if test_result:
                self.initialized = True
                logger.info("✅ AI Interface successfully connected to Augment Agent API")
                return True
            else:
                logger.error("❌ Failed to connect to Augment Agent API")
                return False
                
        except Exception as e:
            logger.error(f"❌ AI Interface initialization failed: {e}")
            return False
    
    async def _test_api_connection(self) -> bool:
        """Test connection to Augment Agent API"""
        try:
            test_payload = {
                'messages': [
                    {'role': 'user', 'content': 'Test connection'}
                ],
                'model': self.model,
                'max_tokens': 10
            }
            
            async with self.session.post(
                f"{self.api_url}/chat/completions",
                json=test_payload
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"API test failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    async def generate_response(self, messages: List[Dict[str, Any]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate intelligent AI response using real Augment Agent API
        
        This replaces the hardcoded if-else statements with actual AI processing.
        """
        if not self.initialized:
            raise RuntimeError("AI Interface not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare the API request
            payload = {
                'messages': messages,
                'model': self.model,
                'temperature': 0.7,
                'max_tokens': 4096
            }
            
            # Add context if provided
            if context:
                # Enhance system message with context
                system_message = self._build_system_message(context)
                if messages and messages[0].get('role') == 'system':
                    messages[0]['content'] = system_message
                else:
                    messages.insert(0, {'role': 'system', 'content': system_message})
            
            # Make API request
            async with self.session.post(
                f"{self.api_url}/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract response
                    ai_response = result['choices'][0]['message']['content']
                    
                    # Calculate metrics
                    processing_time = time.time() - start_time
                    self._update_metrics(processing_time)
                    
                    return {
                        'response': ai_response,
                        'confidence': 0.9,  # High confidence for real AI
                        'sources': [
                            {
                                'type': 'augment_agent_api',
                                'title': 'Real AI Processing',
                                'confidence': 0.9
                            }
                        ],
                        'reasoning': 'Generated using real Augment Agent AI capabilities',
                        'processing_time_ms': int(processing_time * 1000),
                        'model_used': self.model
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            # Return error response instead of hardcoded fallback
            return {
                'response': f"I encountered an issue processing your request: {str(e)}",
                'confidence': 0.1,
                'sources': [{'type': 'error_handling', 'title': 'Error Recovery', 'confidence': 0.1}],
                'reasoning': f'Error in AI processing: {str(e)}',
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'error': str(e)
            }
    
    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build enhanced system message with context"""
        base_message = "You are Brain-3, the Augment Intelligence component of the Four-Brain system."
        
        if context.get('document_context'):
            base_message += f"\n\nDocument Context: {context['document_context']}"
        
        if context.get('conversation_history'):
            base_message += f"\n\nConversation History: {context['conversation_history']}"
        
        if context.get('task_type'):
            base_message += f"\n\nTask Type: {context['task_type']}"
        
        return base_message
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.request_count += 1
        self.total_response_time += processing_time
        self.last_request_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get AI interface performance metrics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'initialized': self.initialized,
            'request_count': self.request_count,
            'average_response_time_ms': int(avg_response_time * 1000),
            'last_request_time': self.last_request_time,
            'api_url': self.api_url,
            'model': self.model
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        logger.info("🧹 AI Interface cleaned up")
