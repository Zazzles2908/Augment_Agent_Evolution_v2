#!/usr/bin/env python3
"""
Redis Streams Definitions for Four-Brain Architecture
Implements all 9 stream definitions for inter-brain communication

This module defines the Redis Streams infrastructure for the Four-Brain System,
replacing HTTP-based communication with reliable stream-based messaging.

Stream Flow:
1. docling_requests -> docling_results (Brain 4)
2. agentic_tasks -> embedding_requests (Brain 3 -> Brain 1)
3. embedding_requests -> embedding_results (Brain 1)
4. embedding_results -> rerank_requests (Brain 1 -> Brain 2)
5. rerank_requests -> rerank_results (Brain 2)
6. rerank_results -> agentic_results (Brain 2 -> Brain 3)
7. memory_updates (All brains for score tracking)

Zero Fabrication Policy: ENFORCED
All stream definitions use real Redis Streams functionality.
"""

import uuid
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)

class StreamNames:
    """Centralized stream name definitions"""

    # Document processing streams (Brain 4)
    DOCLING_REQUESTS = "docling_requests"
    DOCLING_RESULTS = "docling_results"

    # Agentic task coordination (Brain 3)
    AGENTIC_TASKS = "agentic_tasks"
    AGENTIC_RESULTS = "agentic_results"

    # Embedding processing streams (Brain 1)
    EMBEDDING_REQUESTS = "embedding_requests"
    EMBEDDING_RESULTS = "embedding_results"

    # Reranking streams (Brain 2)
    RERANK_REQUESTS = "rerank_requests"
    RERANK_RESULTS = "rerank_results"

    # Memory and scoring
    MEMORY_UPDATES = "memory_updates"

class MessageType(Enum):
    """Message type definitions for stream messages"""
    DOCUMENT_PROCESS = "document_process"
    EMBEDDING_REQUEST = "embedding_request"
    EMBEDDING_BATCH_REQUEST = "embedding_batch_request"
    EMBEDDING_RESULT = "embedding_result"
    RERANK_REQUEST = "rerank_request"
    AGENTIC_TASK = "agentic_task"
    MEMORY_UPDATE = "memory_update"
    HEALTH_CHECK = "health_check"

@dataclass
class StreamMessage:
    """Base class for all stream messages"""
    task_id: str
    message_type: MessageType
    timestamp: float
    brain_id: str
    data: Dict[str, Any]

    def __post_init__(self):
        """Ensure task_id is generated if not provided"""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for Redis"""
        return {
            'task_id': self.task_id,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp,
            'brain_id': self.brain_id,
            'data': json.dumps(self.data)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamMessage':
        """Create message from Redis dictionary"""
        return cls(
            task_id=data['task_id'],
            message_type=MessageType(data['message_type']),
            timestamp=float(data['timestamp']),
            brain_id=data['brain_id'],
            data=json.loads(data['data'])
        )

@dataclass
class DoclingRequest(StreamMessage):
    """Document processing request for Brain 4"""

    def __init__(self, document_path: str, document_type: str,
                 processing_options: Dict[str, Any] = None, task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.DOCUMENT_PROCESS,
            timestamp=time.time(),
            brain_id="brain4",
            data={
                'document_path': document_path,
                'document_type': document_type,
                'processing_options': processing_options or {}
            }
        )

@dataclass
class EmbeddingRequest(StreamMessage):
    """Embedding generation request for Brain 1 (single text)
    Deprecated for inter-brain use; prefer EmbeddingBatchRequest to avoid large payloads.
    """

    def __init__(self, text: str, embedding_type: str = "default",
                 task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.EMBEDDING_REQUEST,
            timestamp=time.time(),
            brain_id="brain1",
            data={
                'text': text,
                'embedding_type': embedding_type
            }
        )

@dataclass
class EmbeddingBatchRequest(StreamMessage):
    """Batch embedding request carrying only refs, not raw text.
    data schema:
      doc_id: str,
      chunk_batch_id: str,
      chunk_refs: List[{chunk_id, page_no, storage_ref, token_count}]
      target_dim: int
    """
    def __init__(self, doc_id: str, chunk_batch_id: str, chunk_refs: List[Dict[str, Any]],
                 target_dim: int = 2000, task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.EMBEDDING_BATCH_REQUEST,
            timestamp=time.time(),
            brain_id="brain1",
            data={
                'doc_id': doc_id,
                'chunk_batch_id': chunk_batch_id,
                'chunk_refs': chunk_refs,
                'target_dim': target_dim
            }
        )

@dataclass
class EmbeddingResult(StreamMessage):
    """Embedding result publishing vector refs stored in DB, not raw floats.
    data schema:
      doc_id: str,
      chunk_batch_id: str,
      vectors: List[{chunk_id, vector_ref, dim}]
      stats: Dict[str, Any]
    """
    def __init__(self, doc_id: str, chunk_batch_id: str, vectors: List[Dict[str, Any]],
                 stats: Dict[str, Any], task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.EMBEDDING_RESULT,
            timestamp=time.time(),
            brain_id="brain1",
            data={
                'doc_id': doc_id,
                'chunk_batch_id': chunk_batch_id,
                'vectors': vectors,
                'stats': stats
            }
        )


@dataclass
class RerankRequest(StreamMessage):
    """Reranking request for Brain 2"""

    def __init__(self, query: str, documents: List[Dict[str, Any]],
                 task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.RERANK_REQUEST,
            timestamp=time.time(),
            brain_id="brain2",
            data={
                'query': query,
                'documents': documents
            }
        )

@dataclass
class AgenticTask(StreamMessage):
    """Agentic task for Brain 3"""

    def __init__(self, task_description: str, context: Dict[str, Any] = None,
                 task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.AGENTIC_TASK,
            timestamp=time.time(),
            brain_id="brain3",
            data={
                'task_description': task_description,
                'context': context or {}
            }
        )

@dataclass
class MemoryUpdate(StreamMessage):
    """Memory update for score tracking"""

    def __init__(self, operation: str, score: float, metadata: Dict[str, Any] = None,
                 brain_id: str = "unknown", task_id: str = None):
        super().__init__(
            task_id=task_id or str(uuid.uuid4()),
            message_type=MessageType.MEMORY_UPDATE,
            timestamp=time.time(),
            brain_id=brain_id,
            data={
                'operation': operation,
                'score': score,
                'metadata': metadata or {}
            }
        )

class StreamConsumerGroup:
    """Consumer group configuration for Redis Streams"""

    def __init__(self, stream_name: str, group_name: str, consumer_name: str):
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name

    def get_group_key(self) -> str:
        """Get the consumer group key"""
        return f"{self.stream_name}:{self.group_name}"

class StreamDefinitions:
    """Stream definitions and consumer group configurations"""

    # Consumer groups for each brain
    BRAIN1_GROUPS = [
        StreamConsumerGroup(StreamNames.EMBEDDING_REQUESTS, "brain1_processors", "brain1_worker")
    ]

    BRAIN2_GROUPS = [
        StreamConsumerGroup(StreamNames.RERANK_REQUESTS, "brain2_processors", "brain2_worker")
    ]

    BRAIN3_GROUPS = [
        StreamConsumerGroup(StreamNames.AGENTIC_TASKS, "brain3_processors", "brain3_worker"),
        StreamConsumerGroup(StreamNames.EMBEDDING_RESULTS, "brain3_coordinators", "brain3_coordinator"),
        StreamConsumerGroup(StreamNames.RERANK_RESULTS, "brain3_coordinators", "brain3_coordinator")
    ]

    BRAIN4_GROUPS = [
        StreamConsumerGroup(StreamNames.DOCLING_REQUESTS, "brain4_processors", "brain4_worker"),
        StreamConsumerGroup(StreamNames.EMBEDDING_RESULTS, "brain4_embeddings", "brain4_worker")
    ]

    # Orchestrator Hub groups (legacy K2 naming replaced)
    ORCHESTRATOR_HUB_GROUPS = [
        StreamConsumerGroup(StreamNames.DOCLING_RESULTS, "orchestrator_coordinators", "orchestrator_hub"),
        StreamConsumerGroup(StreamNames.AGENTIC_RESULTS, "orchestrator_coordinators", "orchestrator_hub")
    ]
    # Backwards compatibility alias
    K2_HUB_GROUPS = ORCHESTRATOR_HUB_GROUPS

    # Memory system groups
    MEMORY_GROUPS = [
        StreamConsumerGroup(StreamNames.MEMORY_UPDATES, "memory_processors", "memory_worker")
    ]

    @classmethod
    def get_all_streams(cls) -> List[str]:
        """Get all stream names"""
        return [
            StreamNames.DOCLING_REQUESTS,
            StreamNames.DOCLING_RESULTS,
            StreamNames.AGENTIC_TASKS,
            StreamNames.AGENTIC_RESULTS,
            StreamNames.EMBEDDING_REQUESTS,
            StreamNames.EMBEDDING_RESULTS,
            StreamNames.RERANK_REQUESTS,
            StreamNames.RERANK_RESULTS,
            StreamNames.MEMORY_UPDATES
        ]

    @classmethod
    def get_brain_groups(cls, brain_id: str) -> List[StreamConsumerGroup]:
        """Get consumer groups for a specific brain"""
        brain_groups = {
            "brain1": cls.BRAIN1_GROUPS,
            "brain2": cls.BRAIN2_GROUPS,
            "brain3": cls.BRAIN3_GROUPS,
            "brain4": cls.BRAIN4_GROUPS,
            "k2_hub": cls.K2_HUB_GROUPS,
            "memory": cls.MEMORY_GROUPS
        }
        return brain_groups.get(brain_id, [])

# Stream configuration constants
STREAM_CONFIG = {
    'maxlen': 10000,  # Maximum stream length
    'approximate': True,  # Use approximate trimming for performance
    'block_timeout': 5000,  # Block timeout in milliseconds
    'count': 10,  # Number of messages to read at once
    'consumer_timeout': 30000  # Consumer timeout in milliseconds
}
