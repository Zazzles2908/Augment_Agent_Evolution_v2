#!/usr/bin/env python3
"""
Redis Coordinator for K2-Vector-Hub
Handles Redis communication for vector jobs and strategy plans

This module implements Redis pub/sub communication for K2-Vector-Hub,
reading from vector_jobs channel and publishing to strategy_plans channel.

Zero Fabrication Policy: ENFORCED
All implementations use real Redis infrastructure and verified functionality.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

# Redis imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

class RedisCoordinator:
    """
    Redis Coordinator for K2-Vector-Hub
    Manages Redis communication for the Mayor's Office
    """
    
    def __init__(self, redis_url: str = None):
        """Initialize Redis coordinator"""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        
        # Redis connections
        self.redis_client = None
        self.pubsub = None
        self.connected = False
        
        # Channel names (per fix_containers.md)
        self.vector_jobs_channel = "vector_jobs"      # Input: Brain 3 publishes jobs here
        self.strategy_plans_channel = "strategy_plans" # Output: K2-Vector-Hub publishes strategies here
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_time = None
        self.initialization_time = time.time()
        
        # Job queue
        self.pending_jobs = asyncio.Queue()
        
        logger.info(f"📡 Redis Coordinator initialized for K2-Vector-Hub")
    
    async def connect(self) -> bool:
        """Connect to Redis and setup communication channels"""
        if not REDIS_AVAILABLE:
            logger.error("❌ Redis not available - install redis package")
            return False
        
        try:
            start_time = time.time()
            
            # Create Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            # Setup pub/sub for vector_jobs channel
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(self.vector_jobs_channel)
            
            self.connected = True
            self.connection_time = time.time() - start_time
            
            logger.info(f"✅ K2-Vector-Hub Redis connection established in {self.connection_time:.3f}s")
            logger.info(f"📡 Subscribed to: {self.vector_jobs_channel}")
            logger.info(f"📤 Publishing to: {self.strategy_plans_channel}")
            
            # Start message listener
            asyncio.create_task(self._vector_jobs_listener())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        try:
            if self.pubsub:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.connected = False
            logger.info("🔌 K2-Vector-Hub Redis connection closed")
            
        except Exception as e:
            logger.error(f"❌ Redis disconnection error: {e}")
    
    async def _vector_jobs_listener(self):
        """Listen for vector jobs from Brain 3"""
        logger.info("👂 Starting vector jobs listener...")
        
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    await self._handle_vector_job(message)
        except Exception as e:
            logger.error(f"❌ Vector jobs listener error: {e}")
    
    async def _handle_vector_job(self, redis_message):
        """Handle incoming vector job from Brain 3"""
        try:
            job_data = json.loads(redis_message["data"])
            job_id = job_data.get("job_id", "unknown")
            
            logger.info(f"📥 K2-Vector-Hub received vector job: {job_id}")
            
            # Add to pending jobs queue
            await self.pending_jobs.put(job_data)
            
            self.messages_received += 1
            
        except Exception as e:
            logger.error(f"❌ Vector job handling error: {e}")
    
    async def get_next_vector_job(self) -> Optional[Dict[str, Any]]:
        """Get next vector job from the queue (non-blocking)"""
        try:
            # Non-blocking get with short timeout
            job = await asyncio.wait_for(self.pending_jobs.get(), timeout=0.1)
            return job
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"❌ Error getting vector job: {e}")
            return None
    
    async def publish_strategy_plan(self, strategy_plan: Dict[str, Any]) -> bool:
        """Publish strategy plan to strategy_plans channel"""
        if not self.connected:
            raise RuntimeError("Not connected to Redis")
        
        try:
            job_id = strategy_plan.get("job_id", "unknown")
            
            # Publish strategy plan
            strategy_json = json.dumps(strategy_plan)
            await self.redis_client.publish(self.strategy_plans_channel, strategy_json)
            
            logger.info(f"📤 Published strategy plan for job {job_id} to {self.strategy_plans_channel}")
            self.messages_sent += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish strategy plan: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Redis coordinator"""
        health_status = {
            "healthy": self.connected,
            "redis_url": self.redis_url,
            "connection_time_seconds": self.connection_time,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "pending_jobs": self.pending_jobs.qsize() if self.pending_jobs else 0,
            "uptime_seconds": time.time() - self.initialization_time
        }
        
        # Test Redis connection
        if self.connected and self.redis_client:
            try:
                await self.redis_client.ping()
                health_status["redis_ping"] = "success"
            except Exception as e:
                health_status["redis_ping"] = f"failed: {e}"
                health_status["healthy"] = False
        
        return health_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "connected": self.connected,
            "redis_url": self.redis_url,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "pending_jobs": self.pending_jobs.qsize() if self.pending_jobs else 0,
            "connection_time_seconds": self.connection_time,
            "channels": {
                "input": self.vector_jobs_channel,
                "output": self.strategy_plans_channel
            },
            "uptime_seconds": time.time() - self.initialization_time
        }
