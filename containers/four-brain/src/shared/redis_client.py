#!/usr/bin/env python3
"""
Redis Streams Client for Four-Brain Architecture
Implements stream-based communication replacing HTTP endpoints

This module provides Redis Streams functionality for the Four-Brain System,
enabling reliable message passing between brains with automatic retries,
consumer groups, and message acknowledgment.

Zero Fabrication Policy: ENFORCED
All implementations use real Redis Streams functionality.
"""

import os
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
import redis.asyncio as redis
import structlog

from .streams import (
    StreamNames, StreamMessage, StreamConsumerGroup, StreamDefinitions,
    STREAM_CONFIG, MessageType, DoclingRequest, EmbeddingRequest,
    RerankRequest, AgenticTask, MemoryUpdate
)

logger = structlog.get_logger(__name__)

class RedisStreamsClient:
    """Redis Streams client for Four-Brain communication"""
    
    def __init__(self, redis_url: Optional[str] = None, brain_id: str = "unknown"):
        """Initialize Redis Streams client"""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.brain_id = brain_id
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self.is_consuming = False
        
        # Consumer groups for this brain
        self.consumer_groups = StreamDefinitions.get_brain_groups(brain_id)
        
        # Message handlers
        self.message_handlers: Dict[str, Callable[[StreamMessage], Awaitable[Any]]] = {}
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_acknowledged = 0
        self.connection_errors = 0
        
        # Background tasks
        self._consumer_tasks: List[asyncio.Task] = []
        
        logger.info("Redis Streams client initialized", 
                   brain_id=brain_id, redis_url=self.redis_url[:50] + "...")
    
    async def connect(self) -> bool:
        """Connect to Redis and initialize streams"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize streams and consumer groups
            await self._initialize_streams()
            await self._initialize_consumer_groups()
            
            self.is_connected = True
            logger.info("✅ Redis Streams connection established", brain_id=self.brain_id)
            return True
            
        except Exception as e:
            self.connection_errors += 1
            logger.error("❌ Redis Streams connection failed", 
                        brain_id=self.brain_id, error=str(e))
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup"""
        try:
            # Stop consumer tasks
            await self._stop_consumers()
            
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            
            self.is_connected = False
            self.is_consuming = False
            
            logger.info("Redis Streams connection closed", brain_id=self.brain_id)
            
        except Exception as e:
            logger.error("Error closing Redis Streams connection", 
                        brain_id=self.brain_id, error=str(e))
    
    async def _initialize_streams(self):
        """Initialize all required streams"""
        for stream_name in StreamDefinitions.get_all_streams():
            try:
                # Create stream if it doesn't exist by adding a dummy message
                await self.redis_client.xadd(
                    stream_name,
                    {"init": "stream_initialized", "timestamp": str(time.time())},
                    maxlen=STREAM_CONFIG['maxlen'],
                    approximate=STREAM_CONFIG['approximate']
                )
                logger.debug("Stream initialized", stream_name=stream_name)
                
            except Exception as e:
                logger.warning("Failed to initialize stream", 
                             stream_name=stream_name, error=str(e))
    
    async def _initialize_consumer_groups(self):
        """Initialize consumer groups for this brain"""
        for group in self.consumer_groups:
            try:
                await self.redis_client.xgroup_create(
                    group.stream_name,
                    group.group_name,
                    id="0",
                    mkstream=True
                )
                logger.debug("Consumer group created", 
                           stream=group.stream_name, group=group.group_name)
                
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists, which is fine
                    logger.debug("Consumer group already exists", 
                               stream=group.stream_name, group=group.group_name)
                else:
                    logger.warning("Failed to create consumer group", 
                                 stream=group.stream_name, group=group.group_name, error=str(e))
    
    async def send_message(self, stream_name: str, message: StreamMessage) -> str:
        """Send a message to a stream"""
        try:
            if not self.is_connected:
                raise ConnectionError("Redis client not connected")
            
            message_id = await self.redis_client.xadd(
                stream_name,
                message.to_dict(),
                maxlen=STREAM_CONFIG['maxlen'],
                approximate=STREAM_CONFIG['approximate']
            )
            
            self.messages_sent += 1
            logger.debug("Message sent to stream", 
                        stream=stream_name, message_id=message_id, 
                        task_id=message.task_id)
            
            return message_id
            
        except Exception as e:
            logger.error("Failed to send message", 
                        stream=stream_name, task_id=message.task_id, error=str(e))
            raise
    
    async def start_consuming(self):
        """Start consuming messages from assigned streams"""
        if self.is_consuming:
            logger.warning("Already consuming messages", brain_id=self.brain_id)
            return
        
        self.is_consuming = True
        
        # Start consumer tasks for each group
        for group in self.consumer_groups:
            task = asyncio.create_task(
                self._consume_stream(group),
                name=f"consumer_{group.stream_name}_{group.group_name}"
            )
            self._consumer_tasks.append(task)
        
        logger.info("Started consuming messages", 
                   brain_id=self.brain_id, groups=len(self.consumer_groups))
    
    async def _consume_stream(self, group: StreamConsumerGroup):
        """Consume messages from a specific stream"""
        while self.is_consuming:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    group.group_name,
                    group.consumer_name,
                    {group.stream_name: ">"},
                    count=STREAM_CONFIG['count'],
                    block=STREAM_CONFIG['block_timeout']
                )
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_message(group, message_id, fields)
                
            except asyncio.TimeoutError:
                # Timeout is expected, continue consuming
                continue
            except Exception as e:
                logger.error("Error consuming from stream", 
                           stream=group.stream_name, group=group.group_name, error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_message(self, group: StreamConsumerGroup, 
                             message_id: str, fields: Dict[str, str]):
        """Process a single message"""
        try:
            # Skip legacy/init messages lacking required fields
            required = {"task_id", "message_type", "timestamp", "brain_id", "data"}
            if not required.issubset(fields.keys()):
                logger.warning("Skipping legacy/invalid message", stream=group.stream_name, message_id=message_id, fields=list(fields.keys()))
                # Ack to clear from pending; they were created during stream init
                await self.redis_client.xack(group.stream_name, group.group_name, message_id)
                return

            # Convert to StreamMessage
            message = StreamMessage.from_dict(fields)

            # Find appropriate handler
            handler = self.message_handlers.get(group.stream_name)
            if handler:
                await handler(message)
                self.messages_received += 1
            else:
                logger.warning("No handler for stream", stream=group.stream_name)

            # Acknowledge message
            await self.redis_client.xack(group.stream_name, group.group_name, message_id)
            self.messages_acknowledged += 1

            logger.debug("Message processed and acknowledged",
                        stream=group.stream_name, message_id=message_id,
                        task_id=message.task_id)

        except Exception as e:
            logger.error("Failed to process message",
                        stream=group.stream_name, message_id=message_id, error=str(e))
    
    async def _stop_consumers(self):
        """Stop all consumer tasks"""
        self.is_consuming = False
        
        # Cancel all consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        self._consumer_tasks.clear()
        logger.info("All consumer tasks stopped", brain_id=self.brain_id)
    
    def register_handler(self, stream_name: str, 
                        handler: Callable[[StreamMessage], Awaitable[Any]]):
        """Register a message handler for a stream"""
        self.message_handlers[stream_name] = handler
        logger.info("Handler registered", stream=stream_name, brain_id=self.brain_id)
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            if not self.redis_client:
                return False

            await self.redis_client.ping()
            return True

        except Exception as e:
            logger.warning("Redis health check failed",
                          brain_id=self.brain_id, error=str(e))
            return False

    async def ping(self) -> bool:
        """Compatibility method for health monitoring - calls health_check()"""
        return await self.health_check()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis Streams client statistics"""
        return {
            "brain_id": self.brain_id,
            "is_connected": self.is_connected,
            "is_consuming": self.is_consuming,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_acknowledged": self.messages_acknowledged,
            "connection_errors": self.connection_errors,
            "consumer_groups": len(self.consumer_groups),
            "active_handlers": len(self.message_handlers),
            "redis_url": self.redis_url[:50] + "..." if len(self.redis_url) > 50 else self.redis_url
        }

# Convenience functions for creating specific message types
async def send_docling_request(client: RedisStreamsClient, document_path: str, 
                              document_type: str, processing_options: Dict[str, Any] = None) -> str:
    """Send a document processing request to Brain 4"""
    request = DoclingRequest(document_path, document_type, processing_options)
    return await client.send_message(StreamNames.DOCLING_REQUESTS, request)

async def send_embedding_request(client: RedisStreamsClient, text: str, 
                               embedding_type: str = "default") -> str:
    """Send an embedding request to Brain 1"""
    request = EmbeddingRequest(text, embedding_type)
    return await client.send_message(StreamNames.EMBEDDING_REQUESTS, request)

async def send_rerank_request(client: RedisStreamsClient, query: str, 
                            documents: List[Dict[str, Any]]) -> str:
    """Send a reranking request to Brain 2"""
    request = RerankRequest(query, documents)
    return await client.send_message(StreamNames.RERANK_REQUESTS, request)

async def send_agentic_task(client: RedisStreamsClient, task_description: str, 
                          context: Dict[str, Any] = None) -> str:
    """Send an agentic task to Brain 3"""
    task = AgenticTask(task_description, context)
    return await client.send_message(StreamNames.AGENTIC_TASKS, task)

async def send_memory_update(client: RedisStreamsClient, operation: str, score: float, 
                           metadata: Dict[str, Any] = None, brain_id: str = "unknown") -> str:
    """Send a memory update for score tracking"""
    update = MemoryUpdate(operation, score, metadata, brain_id)
    return await client.send_message(StreamNames.MEMORY_UPDATES, update)
