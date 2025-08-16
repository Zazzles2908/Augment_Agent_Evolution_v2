#!/usr/bin/env python3
"""
Memory Store for Four-Brain Architecture
Implements task performance tracking and pattern matching using Redis + Supabase

This module provides memory storage functionality for the Four-Brain System,
enabling tracking of task performance, pattern recognition, and historical
analysis for self-improvement capabilities.

Zero Fabrication Policy: ENFORCED
All implementations use real Redis and Supabase infrastructure.
"""

import os
import time
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis
import structlog

# Supabase imports
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

from .streams import StreamMessage, MemoryUpdate

logger = structlog.get_logger(__name__)

@dataclass
class TaskScore:
    """Task performance score record"""
    task_id: str
    brain_id: str
    operation: str
    score: float
    task_signature: str  # Hash of input parameters
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'task_id': self.task_id,
            'brain_id': self.brain_id,
            'operation': self.operation,
            'score': self.score,
            'task_signature': self.task_signature,
            'timestamp': self.timestamp,
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskScore':
        """Create from dictionary"""
        return cls(
            task_id=data['task_id'],
            brain_id=data['brain_id'],
            operation=data['operation'],
            score=float(data['score']),
            task_signature=data['task_signature'],
            timestamp=float(data['timestamp']),
            metadata=json.loads(data.get('metadata', '{}'))
        )

@dataclass
class PatternMatch:
    """Pattern matching result"""
    task_signature: str
    similar_tasks: List[TaskScore]
    average_score: float
    best_score: float
    worst_score: float
    attempt_count: int
    confidence: float

class MemoryStore:
    """Memory store for task performance tracking and pattern matching"""
    
    def __init__(self, redis_url: Optional[str] = None, 
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None):
        """Initialize memory store"""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        # Redis client for fast access
        self.redis_client: Optional[redis.Redis] = None
        
        # Supabase client for persistent storage
        self.supabase_client: Optional[Client] = None
        
        # Connection status
        self.redis_connected = False
        self.supabase_connected = False
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_cache_size = 10000  # Maximum cached items
        
        # Statistics
        self.scores_stored = 0
        self.patterns_matched = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("Memory store initialized")
    
    async def connect(self) -> bool:
        """Connect to Redis and Supabase"""
        success = True
        
        # Connect to Redis
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True
            )
            
            await self.redis_client.ping()
            self.redis_connected = True
            logger.info("✅ Redis memory store connection established")
            
        except Exception as e:
            logger.error("❌ Redis memory store connection failed", error=str(e))
            self.redis_connected = False
            success = False
        
        # Connect to Supabase
        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                
                # Test connection with a simple query
                result = self.supabase_client.schema('augment_agent').table('task_scores').select('count').limit(1).execute()
                self.supabase_connected = True
                logger.info("✅ Supabase memory store connection established")
                
            except Exception as e:
                logger.error("❌ Supabase memory store connection failed", error=str(e))
                self.supabase_connected = False
                # Don't fail if Supabase is unavailable, Redis can work alone
        else:
            logger.warning("Supabase not available, using Redis-only memory store")
        
        return success
    
    async def disconnect(self):
        """Disconnect from Redis and Supabase"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self.redis_connected = False
        
        # Supabase client doesn't need explicit disconnection
        self.supabase_connected = False
        
        logger.info("Memory store disconnected")
    
    def _generate_task_signature(self, operation: str, inputs: Dict[str, Any]) -> str:
        """Generate a hash signature for task inputs"""
        # Create a normalized representation of inputs
        normalized_inputs = json.dumps(inputs, sort_keys=True, default=str)
        signature_data = f"{operation}:{normalized_inputs}"
        
        # Generate SHA-256 hash
        return hashlib.sha256(signature_data.encode()).hexdigest()
    
    async def store_score(self, task_score: TaskScore) -> bool:
        """Store a task score in both Redis and Supabase"""
        try:
            # Store in Redis for fast access
            if self.redis_connected:
                redis_key = f"score:{task_score.task_id}"
                await self.redis_client.hset(redis_key, mapping=task_score.to_dict())
                await self.redis_client.expire(redis_key, self.cache_ttl)
                
                # Also store by signature for pattern matching
                signature_key = f"pattern:{task_score.task_signature}"
                await self.redis_client.lpush(signature_key, task_score.task_id)
                await self.redis_client.expire(signature_key, self.cache_ttl)
            
            # Store in Supabase for persistence
            if self.supabase_connected:
                data = task_score.to_dict()
                data['created_at'] = datetime.fromtimestamp(task_score.timestamp).isoformat()
                
                self.supabase_client.schema('augment_agent').table('task_scores').insert(data).execute()
            
            self.scores_stored += 1
            logger.debug("Task score stored", task_id=task_score.task_id, 
                        score=task_score.score, operation=task_score.operation)
            
            return True
            
        except Exception as e:
            logger.error("Failed to store task score", 
                        task_id=task_score.task_id, error=str(e))
            return False
    
    async def get_task_score(self, task_id: str) -> Optional[TaskScore]:
        """Retrieve a specific task score"""
        try:
            # Try Redis first (fast cache)
            if self.redis_connected:
                redis_key = f"score:{task_id}"
                data = await self.redis_client.hgetall(redis_key)
                
                if data:
                    self.cache_hits += 1
                    return TaskScore.from_dict(data)
            
            # Fall back to Supabase
            if self.supabase_connected:
                result = self.supabase_client.schema('augment_agent').table('task_scores')\
                    .select('*')\
                    .eq('task_id', task_id)\
                    .limit(1)\
                    .execute()
                
                if result.data:
                    self.cache_misses += 1
                    score_data = result.data[0]
                    
                    # Convert timestamp if needed
                    if 'created_at' in score_data:
                        score_data['timestamp'] = datetime.fromisoformat(
                            score_data['created_at'].replace('Z', '+00:00')
                        ).timestamp()
                    
                    return TaskScore.from_dict(score_data)
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve task score", task_id=task_id, error=str(e))
            return None
    
    async def get_past_attempts(self, operation: str, inputs: Dict[str, Any], 
                              limit: int = 10) -> PatternMatch:
        """Get past attempts for similar tasks"""
        try:
            task_signature = self._generate_task_signature(operation, inputs)
            similar_tasks = []
            
            # Try Redis first for recent attempts
            if self.redis_connected:
                signature_key = f"pattern:{task_signature}"
                task_ids = await self.redis_client.lrange(signature_key, 0, limit - 1)
                
                for task_id in task_ids:
                    score = await self.get_task_score(task_id)
                    if score:
                        similar_tasks.append(score)
            
            # If not enough results, query Supabase
            if len(similar_tasks) < limit and self.supabase_connected:
                result = self.supabase_client.schema('augment_agent').table('task_scores')\
                    .select('*')\
                    .eq('task_signature', task_signature)\
                    .order('timestamp', desc=True)\
                    .limit(limit)\
                    .execute()
                
                for score_data in result.data:
                    if 'created_at' in score_data:
                        score_data['timestamp'] = datetime.fromisoformat(
                            score_data['created_at'].replace('Z', '+00:00')
                        ).timestamp()
                    
                    task_score = TaskScore.from_dict(score_data)
                    
                    # Avoid duplicates
                    if not any(s.task_id == task_score.task_id for s in similar_tasks):
                        similar_tasks.append(task_score)
            
            # Calculate pattern statistics
            if similar_tasks:
                scores = [task.score for task in similar_tasks]
                average_score = sum(scores) / len(scores)
                best_score = max(scores)
                worst_score = min(scores)
                confidence = min(1.0, len(similar_tasks) / 5.0)  # Higher confidence with more samples
            else:
                average_score = best_score = worst_score = 0.0
                confidence = 0.0
            
            pattern_match = PatternMatch(
                task_signature=task_signature,
                similar_tasks=similar_tasks,
                average_score=average_score,
                best_score=best_score,
                worst_score=worst_score,
                attempt_count=len(similar_tasks),
                confidence=confidence
            )
            
            self.patterns_matched += 1
            logger.debug("Pattern match completed", 
                        signature=task_signature[:16] + "...",
                        attempts=len(similar_tasks),
                        avg_score=average_score)
            
            return pattern_match
            
        except Exception as e:
            logger.error("Failed to get past attempts", operation=operation, error=str(e))
            return PatternMatch(
                task_signature=self._generate_task_signature(operation, inputs),
                similar_tasks=[],
                average_score=0.0,
                best_score=0.0,
                worst_score=0.0,
                attempt_count=0,
                confidence=0.0
            )
    
    async def get_brain_performance(self, brain_id: str,
                                  time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for a specific brain"""
        try:
            cutoff_time = time.time() - (time_window_hours * 3600)

            if self.supabase_connected:
                # Convert timestamp to ISO format for Supabase compatibility with UTC timezone
                from datetime import datetime, timezone
                cutoff_datetime = datetime.fromtimestamp(cutoff_time, tz=timezone.utc).isoformat()

                result = self.supabase_client.schema('augment_agent').table('task_scores')\
                    .select('operation, score, timestamp')\
                    .eq('brain_id', brain_id)\
                    .gte('created_at', cutoff_datetime)\
                    .execute()
                
                if result.data:
                    scores_by_operation = {}
                    total_scores = []
                    
                    for record in result.data:
                        operation = record['operation']
                        score = record['score']
                        total_scores.append(score)
                        
                        if operation not in scores_by_operation:
                            scores_by_operation[operation] = []
                        scores_by_operation[operation].append(score)
                    
                    # Calculate statistics
                    performance_stats = {
                        'brain_id': brain_id,
                        'time_window_hours': time_window_hours,
                        'total_tasks': len(total_scores),
                        'average_score': sum(total_scores) / len(total_scores) if total_scores else 0.0,
                        'best_score': max(total_scores) if total_scores else 0.0,
                        'worst_score': min(total_scores) if total_scores else 0.0,
                        'operations': {}
                    }
                    
                    for operation, scores in scores_by_operation.items():
                        performance_stats['operations'][operation] = {
                            'task_count': len(scores),
                            'average_score': sum(scores) / len(scores),
                            'best_score': max(scores),
                            'worst_score': min(scores)
                        }
                    
                    return performance_stats
            
            return {
                'brain_id': brain_id,
                'time_window_hours': time_window_hours,
                'total_tasks': 0,
                'average_score': 0.0,
                'best_score': 0.0,
                'worst_score': 0.0,
                'operations': {}
            }
            
        except Exception as e:
            logger.error("Failed to get brain performance", brain_id=brain_id, error=str(e))
            return {}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for the memory store"""
        try:
            import psutil

            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            # Memory store specific metrics
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "process_memory_rss": process_memory.rss,
                "process_memory_vms": process_memory.vms,
                "process_cpu_percent": process.cpu_percent(),
                "redis_connected": self.redis_connected,
                "supabase_connected": self.supabase_connected,
                "scores_stored": self.scores_stored,
                "patterns_matched": self.patterns_matched,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            }

            return metrics

        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {
                "error": str(e),
                "redis_connected": self.redis_connected,
                "supabase_connected": self.supabase_connected
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check memory store health"""
        health = {
            'redis_connected': self.redis_connected,
            'supabase_connected': self.supabase_connected,
            'scores_stored': self.scores_stored,
            'patterns_matched': self.patterns_matched,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }
        
        # Test Redis connection
        if self.redis_connected:
            try:
                await self.redis_client.ping()
                health['redis_status'] = 'healthy'
            except Exception as e:
                health['redis_status'] = f'error: {str(e)}'
                self.redis_connected = False
        else:
            health['redis_status'] = 'disconnected'
        
        # Test Supabase connection
        if self.supabase_connected:
            try:
                self.supabase_client.schema('augment_agent').table('task_scores').select('count').limit(1).execute()
                health['supabase_status'] = 'healthy'
            except Exception as e:
                health['supabase_status'] = f'error: {str(e)}'
                self.supabase_connected = False
        else:
            health['supabase_status'] = 'disconnected'
        
        return health

# Convenience functions for creating task scores
def create_task_score(task_id: str, brain_id: str, operation: str, 
                     score: float, inputs: Dict[str, Any], 
                     metadata: Dict[str, Any] = None) -> TaskScore:
    """Create a task score with generated signature"""
    memory_store = MemoryStore()
    task_signature = memory_store._generate_task_signature(operation, inputs)
    
    return TaskScore(
        task_id=task_id,
        brain_id=brain_id,
        operation=operation,
        score=score,
        task_signature=task_signature,
        timestamp=time.time(),
        metadata=metadata or {}
    )

# Global memory store instance
_memory_store: Optional[MemoryStore] = None

def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance"""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
