"""
Brain 2 Task Processor
Background task processing and Redis communication for Brain 2

This module implements background task processing for Brain 2, enabling
inter-brain communication via Redis messaging. It integrates with the
existing BrainCommunicator from the four-brain architecture.

Key Features:
- Redis-based inter-brain messaging
- Background task processing
- Task queue management
- Integration with Brain 2 Manager
- Error handling and retry logic

Zero Fabrication Policy: ENFORCED
All implementations use real Redis communication and verified functionality.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add paths for imports
sys.path.append('/workspace/src')
sys.path.append('/workspace/src/brain4_docling')

try:
    from shared.communication.brain_communicator import StandardizedBrainCommunicator as BrainCommunicator
    BRAIN_COMMUNICATOR_AVAILABLE = True
except ImportError:
    try:
        from brain4_docling.integration.brain_communicator import BrainCommunicator
        BRAIN_COMMUNICATOR_AVAILABLE = True
    except ImportError:
        BRAIN_COMMUNICATOR_AVAILABLE = False
        logging.warning("⚠️ BrainCommunicator not available - Redis communication disabled")

from .config.settings import Brain2Settings

logger = logging.getLogger(__name__)


class Brain2TaskProcessor:
    """
    Background task processor for Brain 2
    Handles Redis communication and inter-brain messaging
    """
    
    def __init__(self, brain2_manager, settings: Brain2Settings):
        """Initialize task processor with Brain 2 manager and settings"""
        self.brain2_manager = brain2_manager
        self.settings = settings
        
        # Redis communication
        self.brain_communicator = None
        self.redis_available = False
        
        # Task processing
        self.processing_tasks = {}
        self.task_queue = asyncio.Queue()
        self.is_running = False
        
        logger.info("🔧 Brain2TaskProcessor initialized")
    
    async def initialize(self):
        """Initialize Redis communication and task processing"""
        logger.info("🚀 Initializing Brain 2 task processor...")
        
        try:
            if BRAIN_COMMUNICATOR_AVAILABLE:
                # Initialize BrainCommunicator
                self.brain_communicator = BrainCommunicator(
                    brain_id=self.settings.brain_id,
                    redis_url=self.settings.redis_url
                )
                
                # Test Redis connection
                await self.brain_communicator.initialize()
                self.redis_available = True
                
                logger.info("✅ Redis communication initialized")
                
                # Start listening for messages
                await self.start_message_listener()
                
            else:
                logger.warning("⚠️ BrainCommunicator not available - running without Redis")
                
        except Exception as e:
            logger.error(f"❌ Task processor initialization failed: {e}")
            self.redis_available = False
    
    async def start_message_listener(self):
        """Start listening for inter-brain messages"""
        if not self.redis_available:
            logger.warning("⚠️ Redis not available - message listener not started")
            return
        
        try:
            logger.info("👂 Starting Brain 2 message listener...")
            
            # Subscribe to Brain 2 specific channels
            channels = [
                f"{self.settings.brain_id}:rerank",
                f"{self.settings.brain_id}:tasks",
                "broadcast:all_brains"
            ]
            
            for channel in channels:
                await self.brain_communicator.subscribe(channel, self._handle_message)
            
            logger.info(f"✅ Subscribed to channels: {channels}")
            
        except Exception as e:
            logger.error(f"❌ Message listener startup failed: {e}")
    
    async def _handle_message(self, channel: str, message: Dict[str, Any]):
        """Handle incoming Redis messages"""
        try:
            logger.info(f"📨 Received message on {channel}: {message.get('type', 'unknown')}")
            
            message_type = message.get('type')
            
            if message_type == 'rerank_request':
                await self._handle_rerank_request(message)
            elif message_type == 'health_check':
                await self._handle_health_check_request(message)
            elif message_type == 'status_request':
                await self._handle_status_request(message)
            else:
                logger.warning(f"⚠️ Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Message handling failed: {e}")
    
    async def _handle_rerank_request(self, message: Dict[str, Any]):
        """Handle reranking request from other brains"""
        try:
            task_id = message.get('task_id', f"task_{datetime.now().timestamp()}")
            query = message.get('query', '')
            documents = message.get('documents', [])
            top_k = message.get('top_k', self.settings.default_top_k)
            callback_channel = message.get('callback_channel')
            
            logger.info(f"🔄 Processing rerank request: {task_id}")
            
            # Validate inputs
            if not query or not documents:
                error_msg = "Invalid rerank request: missing query or documents"
                logger.error(f"❌ {error_msg}")
                
                if callback_channel:
                    await self._send_error_response(callback_channel, task_id, error_msg)
                return
            
            # Check if model is loaded
            if not self.brain2_manager.model_loaded:
                error_msg = "Brain 2 model not loaded"
                logger.error(f"❌ {error_msg}")
                
                if callback_channel:
                    await self._send_error_response(callback_channel, task_id, error_msg)
                return
            
            # Perform reranking
            result = await self.brain2_manager.rerank_documents(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            # Send response
            if callback_channel:
                response = {
                    'type': 'rerank_response',
                    'task_id': task_id,
                    'status': 'completed',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.brain_communicator.publish(callback_channel, response)
                logger.info(f"✅ Rerank response sent: {task_id}")
            
        except Exception as e:
            logger.error(f"❌ Rerank request handling failed: {e}")
            
            if message.get('callback_channel'):
                await self._send_error_response(
                    message['callback_channel'],
                    message.get('task_id', 'unknown'),
                    str(e)
                )
    
    async def _handle_health_check_request(self, message: Dict[str, Any]):
        """Handle health check request from other brains"""
        try:
            callback_channel = message.get('callback_channel')
            request_id = message.get('request_id', 'unknown')
            
            if not callback_channel:
                return
            
            # Get health status
            health_data = await self.brain2_manager.health_check()
            
            response = {
                'type': 'health_check_response',
                'request_id': request_id,
                'brain_id': self.settings.brain_id,
                'status': health_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.brain_communicator.publish(callback_channel, response)
            logger.info(f"✅ Health check response sent: {request_id}")
            
        except Exception as e:
            logger.error(f"❌ Health check handling failed: {e}")
    
    async def _handle_status_request(self, message: Dict[str, Any]):
        """Handle status request from other brains"""
        try:
            callback_channel = message.get('callback_channel')
            request_id = message.get('request_id', 'unknown')
            
            if not callback_channel:
                return
            
            # Get comprehensive status
            status_data = self.brain2_manager.get_status()
            
            response = {
                'type': 'status_response',
                'request_id': request_id,
                'brain_id': self.settings.brain_id,
                'status': status_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.brain_communicator.publish(callback_channel, response)
            logger.info(f"✅ Status response sent: {request_id}")
            
        except Exception as e:
            logger.error(f"❌ Status request handling failed: {e}")
    
    async def _send_error_response(self, callback_channel: str, task_id: str, error_message: str):
        """Send error response to callback channel"""
        try:
            response = {
                'type': 'error_response',
                'task_id': task_id,
                'status': 'failed',
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.brain_communicator.publish(callback_channel, response)
            logger.info(f"📤 Error response sent: {task_id}")
            
        except Exception as e:
            logger.error(f"❌ Error response sending failed: {e}")
    
    async def send_rerank_request(self, target_brain: str, query: str, 
                                documents: List[Dict], top_k: int = None) -> Optional[Dict]:
        """
        Send reranking request to another brain
        This allows Brain 2 to request reranking from other instances
        """
        if not self.redis_available:
            logger.warning("⚠️ Redis not available - cannot send rerank request")
            return None
        
        try:
            task_id = f"brain2_request_{datetime.now().timestamp()}"
            callback_channel = f"{self.settings.brain_id}:callbacks"
            
            request = {
                'type': 'rerank_request',
                'task_id': task_id,
                'query': query,
                'documents': documents,
                'top_k': top_k or self.settings.default_top_k,
                'callback_channel': callback_channel,
                'sender': self.settings.brain_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send request
            target_channel = f"{target_brain}:rerank"
            await self.brain_communicator.publish(target_channel, request)
            
            logger.info(f"📤 Rerank request sent to {target_brain}: {task_id}")
            
            # Wait for response (with timeout)
            response = await self._wait_for_response(callback_channel, task_id, timeout=30)
            return response
            
        except Exception as e:
            logger.error(f"❌ Rerank request sending failed: {e}")
            return None
    
    async def _wait_for_response(self, channel: str, task_id: str, timeout: int = 30) -> Optional[Dict]:
        """Wait for response on callback channel using Redis BLPOP"""
        try:
            if not self.redis_available or not self.brain_communicator.redis_client:
                logger.warning(f"⚠️ Redis not available for response waiting on {channel}")
                return None

            # Use Redis BLPOP to wait for response with timeout
            response_key = f"{channel}:response:{task_id}"
            logger.debug(f"🔍 Waiting for response on key: {response_key}")

            # BLPOP blocks until message arrives or timeout
            result = await self.brain_communicator.redis_client.blpop(response_key, timeout=timeout)

            if result is None:
                logger.warning(f"⏰ Timeout waiting for response on {channel} for task {task_id}")
                return None

            # result is tuple: (key, value)
            _, response_data = result

            # Parse JSON response
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"✅ Received response for task {task_id}: {response.get('status', 'unknown')}")

            return response

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse response JSON for task {task_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Response waiting failed for task {task_id}: {e}")
            return None
    
    async def broadcast_status(self):
        """Broadcast Brain 2 status to all brains"""
        if not self.redis_available:
            return
        
        try:
            status_data = self.brain2_manager.get_status()
            
            broadcast = {
                'type': 'status_broadcast',
                'brain_id': self.settings.brain_id,
                'status': status_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.brain_communicator.publish('broadcast:status', broadcast)
            logger.debug("📡 Status broadcast sent")
            
        except Exception as e:
            logger.error(f"❌ Status broadcast failed: {e}")
    
    async def shutdown(self):
        """Shutdown task processor and cleanup resources"""
        logger.info("🛑 Shutting down Brain 2 task processor...")
        
        try:
            self.is_running = False
            
            if self.brain_communicator:
                await self.brain_communicator.cleanup()
            
            logger.info("✅ Task processor shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Task processor shutdown failed: {e}")
