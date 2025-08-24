"""
Brain1 Communication Module
Handles inter-brain communication and messaging.

Created: 2025-07-13 AEST
Author: Zazzles's Agent - Brain Architecture Standardization
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
from datetime import datetime

logger = logging.getLogger(__name__)

class BrainCommunicator:
    """Handles communication between Brain1 and other brains."""
    
    def __init__(self, redis_url: str = "redis://phase6-ai-redis:6379/0"):
        """Initialize the brain communicator."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.brain_id = "brain1-embedding"
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info(f"Brain1 communicator initialized with Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Brain1 communicator: {e}")
            raise
    
    async def send_to_brain2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to Brain2 (Reranker) for processing."""
        try:
            message = {
                "from": self.brain_id,
                "to": "brain2-reranker",
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Publish to Brain2 channel
            await self.redis_client.publish("brain2:input", json.dumps(message))
            
            # Wait for response (with timeout)
            response = await self._wait_for_response("brain1:response", timeout=30)
            return response
            
        except Exception as e:
            logger.error(f"Failed to send data to Brain2: {e}")
            raise
    
    async def send_to_brain3(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to Brain3 (Zazzles's Agent) for processing."""
        try:
            message = {
                "from": self.brain_id,
                "to": "brain3-Zazzles's Agent",
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Publish to Brain3 channel
            await self.redis_client.publish("brain3:input", json.dumps(message))
            
            # Wait for response (with timeout)
            response = await self._wait_for_response("brain1:response", timeout=30)
            return response
            
        except Exception as e:
            logger.error(f"Failed to send data to Brain3: {e}")
            raise
    
    async def send_to_brain4(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to Brain4 (Docling) for processing."""
        try:
            message = {
                "from": self.brain_id,
                "to": "brain4-docling",
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Publish to Brain4 channel
            await self.redis_client.publish("brain4:input", json.dumps(message))
            
            # Wait for response (with timeout)
            response = await self._wait_for_response("brain1:response", timeout=30)
            return response
            
        except Exception as e:
            logger.error(f"Failed to send data to Brain4: {e}")
            raise
    
    async def listen_for_requests(self, callback):
        """Listen for incoming requests from other brains."""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("brain1:input")
            
            logger.info("Brain1 listening for incoming requests...")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        response = await callback(data)
                        
                        # Send response back
                        response_channel = f"{data['from']}:response"
                        await self.redis_client.publish(response_channel, json.dumps(response))
                        
                    except Exception as e:
                        logger.error(f"Error processing incoming message: {e}")
                        
        except Exception as e:
            logger.error(f"Error in Brain1 message listener: {e}")
            raise
    
    async def _wait_for_response(self, channel: str, timeout: int = 30) -> Dict[str, Any]:
        """Wait for response on specified channel."""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            
            # Wait for response with timeout
            async with asyncio.timeout(timeout):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        response = json.loads(message["data"])
                        await pubsub.unsubscribe(channel)
                        return response
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response on channel {channel}")
            raise
        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            raise
    
    async def broadcast_status(self, status: Dict[str, Any]):
        """Broadcast Brain1 status to all other brains."""
        try:
            message = {
                "from": self.brain_id,
                "type": "status_update",
                "timestamp": datetime.utcnow().isoformat(),
                "status": status
            }
            
            # Broadcast to all brain channels
            for brain in ["brain2", "brain3", "brain4"]:
                await self.redis_client.publish(f"{brain}:status", json.dumps(message))
                
            logger.debug("Brain1 status broadcasted to all brains")
            
        except Exception as e:
            logger.error(f"Failed to broadcast status: {e}")
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Brain1 communicator closed")
