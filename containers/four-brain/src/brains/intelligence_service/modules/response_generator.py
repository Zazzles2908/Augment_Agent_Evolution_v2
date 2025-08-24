"""
Response Generator Module - Intelligent Response Generation
Replaces hardcoded responses with context-aware AI responses

This module handles intelligent response generation, replacing the
hardcoded if-else statements with sophisticated AI-driven responses.

Created: 2025-07-29 AEST
Purpose: Generate intelligent, context-aware responses
Module Size: 150 lines (modular design)
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Intelligent Response Generator for Brain-3
    
    Generates context-aware, intelligent responses using AI interface,
    completely replacing the hardcoded if-else logic.
    """
    
    def __init__(self, ai_interface):
        """Initialize response generator with AI interface"""
        self.ai_interface = ai_interface
        self.response_cache = {}
        self.conversation_context = []
        self.max_context_length = 10
        
        # Response templates for different scenarios
        self.response_templates = {
            'greeting': "Hello! I'm Brain-3, your Zazzles's Agent Intelligence assistant. How can I help you today?",
            'system_status': "The Four-Brain system is operational and ready to assist you.",
            'error_recovery': "I encountered an issue but I'm working to resolve it. Let me try a different approach.",
            'clarification': "Could you provide more details about what you're looking for?"
        }
        
        logger.info("🧠 Response Generator initialized with intelligent processing")
    
    async def generate_response(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate intelligent response based on user message and context
        
        This completely replaces the hardcoded if-else statements with
        sophisticated AI-driven response generation.
        """
        start_time = time.time()
        
        try:
            # Analyze message intent
            intent = self._analyze_intent(user_message)
            
            # Build conversation context
            conversation_context = self._build_conversation_context(user_message, context)
            
            # Prepare messages for AI
            messages = self._prepare_ai_messages(user_message, intent, conversation_context)
            
            # Generate AI response
            ai_result = await self.ai_interface.generate_response(messages, context)
            
            # Enhance response with metadata
            enhanced_response = self._enhance_response(ai_result, intent, user_message)
            
            # Update conversation context
            self._update_conversation_context(user_message, enhanced_response['response'])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            enhanced_response['total_processing_time_ms'] = int(processing_time * 1000)
            
            logger.info(f"✅ Generated intelligent response for intent: {intent}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return self._generate_error_response(user_message, str(e))
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user message intent for better response generation"""
        message_lower = message.lower()
        
        # Intent patterns (more sophisticated than hardcoded if-else)
        intent_patterns = {
            'greeting': r'\b(hello|hi|hey|greetings)\b',
            'system_inquiry': r'\b(system|four-brain|status|health)\b',
            'help_request': r'\b(help|assist|support|guide)\b',
            'technical_question': r'\b(how|what|why|when|where)\b',
            'document_query': r'\b(document|file|content|search)\b',
            'task_request': r'\b(create|generate|build|make|do)\b',
            'test_query': r'\b(test|testing|check)\b'
        }
        
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, message_lower):
                return intent
        
        return 'general_inquiry'
    
    def _build_conversation_context(self, current_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build comprehensive conversation context"""
        conversation_context = {
            'current_message': current_message,
            'conversation_history': self.conversation_context[-5:],  # Last 5 exchanges
            'timestamp': datetime.now().isoformat(),
            'brain_id': 'brain-3',
            'system_status': 'operational'
        }
        
        # Add external context if provided
        if context:
            conversation_context.update({
                'document_context': context.get('document_context'),
                'task_context': context.get('task_context'),
                'user_preferences': context.get('user_preferences'),
                'session_id': context.get('session_id')
            })
        
        return conversation_context
    
    def _prepare_ai_messages(self, user_message: str, intent: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare messages for AI processing"""
        # System message with context
        system_message = self._build_system_message(intent, context)
        
        messages = [
            {'role': 'system', 'content': system_message}
        ]
        
        # Add conversation history
        for exchange in context.get('conversation_history', []):
            messages.append({'role': 'user', 'content': exchange.get('user', '')})
            messages.append({'role': 'assistant', 'content': exchange.get('assistant', '')})
        
        # Add current user message
        messages.append({'role': 'user', 'content': user_message})
        
        return messages
    
    def _build_system_message(self, intent: str, context: Dict[str, Any]) -> str:
        """Build sophisticated system message based on intent and context"""
        base_message = """You are Brain-3, the Zazzles's Agent Intelligence component of the Four-Brain system. 
You provide intelligent, helpful responses with reasoning and context awareness."""
        
        # Intent-specific instructions
        intent_instructions = {
            'greeting': "Provide a warm, professional greeting and offer assistance.",
            'system_inquiry': "Provide accurate information about the Four-Brain system status and capabilities.",
            'help_request': "Offer comprehensive help and guidance tailored to the user's needs.",
            'technical_question': "Provide detailed, technical explanations with examples when appropriate.",
            'document_query': "Help users find and understand document content with relevant excerpts.",
            'task_request': "Assist with task completion, providing step-by-step guidance.",
            'test_query': "Respond to test queries with confirmation of system functionality."
        }
        
        instruction = intent_instructions.get(intent, "Provide helpful, accurate responses.")
        system_message = f"{base_message}\n\nCurrent Intent: {intent}\nInstruction: {instruction}"
        
        # Add context information
        if context.get('document_context'):
            system_message += f"\n\nAvailable Document Context: {context['document_context'][:500]}..."
        
        if context.get('system_status'):
            system_message += f"\n\nSystem Status: {context['system_status']}"
        
        return system_message
    
    def _enhance_response(self, ai_result: Dict[str, Any], intent: str, user_message: str) -> Dict[str, Any]:
        """Enhance AI response with additional metadata and context"""
        enhanced = {
            'response': ai_result.get('response', ''),
            'confidence': ai_result.get('confidence', 0.8),
            'intent': intent,
            'sources': ai_result.get('sources', []),
            'reasoning': ai_result.get('reasoning', 'Generated using AI intelligence'),
            'processing_time_ms': ai_result.get('processing_time_ms', 0),
            'model_used': ai_result.get('model_used', 'Zazzles's Agent-agent'),
            'brain_id': 'brain-3',
            'timestamp': datetime.now().isoformat(),
            'user_message_length': len(user_message),
            'response_type': 'intelligent_ai_generated'
        }
        
        # Add error information if present
        if 'error' in ai_result:
            enhanced['error'] = ai_result['error']
            enhanced['confidence'] = 0.3
        
        return enhanced
    
    def _update_conversation_context(self, user_message: str, response: str):
        """Update conversation context for future responses"""
        exchange = {
            'user': user_message,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_context.append(exchange)
        
        # Keep only recent context
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
    
    def _generate_error_response(self, user_message: str, error: str) -> Dict[str, Any]:
        """Generate error response when AI processing fails"""
        return {
            'response': f"I encountered an issue processing your request: {error}. Please try rephrasing your question.",
            'confidence': 0.2,
            'intent': 'error_recovery',
            'sources': [{'type': 'error_handling', 'title': 'Error Recovery', 'confidence': 0.2}],
            'reasoning': f'Error in response generation: {error}',
            'processing_time_ms': 0,
            'brain_id': 'brain-3',
            'timestamp': datetime.now().isoformat(),
            'response_type': 'error_recovery',
            'error': error
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'total_exchanges': len(self.conversation_context),
            'context_length': self.max_context_length,
            'cache_size': len(self.response_cache),
            'last_exchange_time': (
                self.conversation_context[-1]['timestamp'] 
                if self.conversation_context else None
            )
        }
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context.clear()
        self.response_cache.clear()
        logger.info("🧹 Conversation context cleared")
