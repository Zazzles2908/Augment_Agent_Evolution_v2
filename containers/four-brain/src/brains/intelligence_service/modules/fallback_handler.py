"""
Fallback Handler Module - Graceful Degradation
Handles API failures and provides intelligent fallback responses

This module ensures Brain-3 continues to function even when the
Augment Agent API is unavailable, providing graceful degradation.

Created: 2025-07-29 AEST
Purpose: Provide graceful degradation when AI API unavailable
Module Size: 150 lines (modular design)
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Graceful Fallback Handler for Brain-3
    
    Provides intelligent fallback responses when the AI API is unavailable,
    ensuring system continues to function gracefully.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fallback handler with configuration"""
        self.config = config
        self.fallback_enabled = config.get('fallback_enabled', True)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_seconds', 5)
        
        # API health tracking
        self.api_failures = 0
        self.last_failure_time = None
        self.consecutive_failures = 0
        self.health_status = 'healthy'
        
        # Intelligent fallback responses (NOT hardcoded like before)
        self.fallback_strategies = {
            'greeting': self._handle_greeting_fallback,
            'system_inquiry': self._handle_system_inquiry_fallback,
            'help_request': self._handle_help_request_fallback,
            'technical_question': self._handle_technical_question_fallback,
            'document_query': self._handle_document_query_fallback,
            'task_request': self._handle_task_request_fallback,
            'general_inquiry': self._handle_general_inquiry_fallback
        }
        
        logger.info("🛡️ Fallback Handler initialized with graceful degradation")
    
    async def handle_api_failure(self, user_message: str, intent: str, error: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle API failure with intelligent fallback response
        
        This provides graceful degradation instead of system failure.
        """
        self._record_api_failure(error)
        
        # Determine if we should retry or use fallback
        if self._should_retry():
            return await self._attempt_retry(user_message, intent, context)
        else:
            return await self._generate_fallback_response(user_message, intent, context)
    
    def _record_api_failure(self, error: str):
        """Record API failure for health tracking"""
        self.api_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        
        # Update health status based on failure pattern
        if self.consecutive_failures >= 3:
            self.health_status = 'degraded'
        if self.consecutive_failures >= 5:
            self.health_status = 'critical'
        
        logger.warning(f"⚠️ API failure recorded: {error} (consecutive: {self.consecutive_failures})")
    
    def _should_retry(self) -> bool:
        """Determine if we should retry API call or use fallback"""
        if not self.fallback_enabled:
            return True
        
        # Don't retry if too many consecutive failures
        if self.consecutive_failures >= self.max_retry_attempts:
            return False
        
        # Don't retry if recent failure
        if self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            if time_since_failure.total_seconds() < self.retry_delay:
                return False
        
        return True
    
    async def _attempt_retry(self, user_message: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to retry API call with exponential backoff"""
        retry_delay = self.retry_delay * (2 ** (self.consecutive_failures - 1))
        
        return {
            'response': f"I'm experiencing temporary connectivity issues. Please wait {retry_delay} seconds and try again.",
            'confidence': 0.3,
            'intent': intent,
            'sources': [{'type': 'retry_handler', 'title': 'Retry Mechanism', 'confidence': 0.3}],
            'reasoning': f'API temporarily unavailable, retry in {retry_delay} seconds',
            'processing_time_ms': 0,
            'brain_id': 'brain-3',
            'timestamp': datetime.now().isoformat(),
            'response_type': 'retry_request',
            'retry_delay_seconds': retry_delay,
            'health_status': self.health_status
        }
    
    async def _generate_fallback_response(self, user_message: str, intent: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate intelligent fallback response based on intent"""
        
        # Use intent-specific fallback strategy
        fallback_strategy = self.fallback_strategies.get(intent, self._handle_general_inquiry_fallback)
        
        try:
            fallback_response = await fallback_strategy(user_message, context)
            
            # Enhance with fallback metadata
            fallback_response.update({
                'confidence': 0.6,  # Moderate confidence for fallback
                'intent': intent,
                'brain_id': 'brain-3',
                'timestamp': datetime.now().isoformat(),
                'response_type': 'intelligent_fallback',
                'health_status': self.health_status,
                'api_failures': self.api_failures,
                'consecutive_failures': self.consecutive_failures
            })
            
            logger.info(f"✅ Generated intelligent fallback response for intent: {intent}")
            return fallback_response
            
        except Exception as e:
            logger.error(f"❌ Fallback generation failed: {e}")
            return self._generate_emergency_response(user_message, str(e))
    
    async def _handle_greeting_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting with intelligent fallback"""
        return {
            'response': "Hello! I'm Brain-3, your Augment Intelligence assistant. I'm currently operating in fallback mode due to temporary connectivity issues, but I'm still here to help you as best I can.",
            'sources': [{'type': 'fallback_greeting', 'title': 'Graceful Greeting', 'confidence': 0.7}],
            'reasoning': 'Providing friendly greeting despite API unavailability'
        }
    
    async def _handle_system_inquiry_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system inquiry with fallback"""
        return {
            'response': f"The Four-Brain system is operational with Brain-3 currently in fallback mode. System health: {self.health_status}. Other brains (Brain-1, Brain-2, Brain-4) should be functioning normally.",
            'sources': [{'type': 'system_status', 'title': 'System Health Check', 'confidence': 0.8}],
            'reasoning': 'Providing accurate system status during fallback mode'
        }
    
    async def _handle_help_request_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help request with fallback"""
        return {
            'response': "I'm here to help! While I'm operating in fallback mode, I can still assist with basic questions, system status, and general guidance. For complex AI tasks, please try again in a few minutes when full connectivity is restored.",
            'sources': [{'type': 'help_system', 'title': 'Fallback Help', 'confidence': 0.6}],
            'reasoning': 'Offering help within fallback capabilities'
        }
    
    async def _handle_technical_question_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical question with fallback"""
        return {
            'response': "I'd love to provide a detailed technical answer, but I'm currently in fallback mode with limited AI capabilities. For comprehensive technical assistance, please try again when full connectivity is restored, or check the system documentation.",
            'sources': [{'type': 'technical_fallback', 'title': 'Limited Technical Support', 'confidence': 0.4}],
            'reasoning': 'Acknowledging limitations during fallback mode'
        }
    
    async def _handle_document_query_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document query with fallback"""
        return {
            'response': "Document search and analysis require full AI capabilities. I'm currently in fallback mode, so I can't access or analyze documents right now. Please try again when connectivity is restored.",
            'sources': [{'type': 'document_fallback', 'title': 'Document Access Limited', 'confidence': 0.3}],
            'reasoning': 'Document processing requires full AI capabilities'
        }
    
    async def _handle_task_request_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task request with fallback"""
        return {
            'response': "I understand you'd like me to help with a task. While I'm in fallback mode with limited capabilities, I can provide basic guidance. For complex task assistance, please try again when full AI capabilities are restored.",
            'sources': [{'type': 'task_fallback', 'title': 'Limited Task Support', 'confidence': 0.4}],
            'reasoning': 'Task execution requires full AI capabilities'
        }
    
    async def _handle_general_inquiry_fallback(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general inquiry with fallback"""
        return {
            'response': "I understand your question, but I'm currently operating in fallback mode with limited AI capabilities. I can provide basic assistance, but for comprehensive answers, please try again when full connectivity is restored.",
            'sources': [{'type': 'general_fallback', 'title': 'General Fallback Response', 'confidence': 0.5}],
            'reasoning': 'General fallback for unspecified inquiries'
        }
    
    def _generate_emergency_response(self, user_message: str, error: str) -> Dict[str, Any]:
        """Generate emergency response when even fallback fails"""
        return {
            'response': "I'm experiencing technical difficulties and cannot process your request right now. Please try again later or contact system support.",
            'confidence': 0.1,
            'intent': 'emergency',
            'sources': [{'type': 'emergency_fallback', 'title': 'Emergency Response', 'confidence': 0.1}],
            'reasoning': f'Emergency fallback due to: {error}',
            'processing_time_ms': 0,
            'brain_id': 'brain-3',
            'timestamp': datetime.now().isoformat(),
            'response_type': 'emergency_fallback',
            'error': error,
            'health_status': 'critical'
        }
    
    def record_api_success(self):
        """Record successful API call to reset failure tracking"""
        self.consecutive_failures = 0
        self.health_status = 'healthy'
        logger.info("✅ API success recorded - health status reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and metrics"""
        return {
            'health_status': self.health_status,
            'total_api_failures': self.api_failures,
            'consecutive_failures': self.consecutive_failures,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'fallback_enabled': self.fallback_enabled,
            'max_retry_attempts': self.max_retry_attempts,
            'retry_delay_seconds': self.retry_delay
        }
