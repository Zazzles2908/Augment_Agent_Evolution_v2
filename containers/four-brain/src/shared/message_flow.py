#!/usr/bin/env python3
"""
Message Flow Orchestrator for Four-Brain Architecture
Implements complete message flow between brains using Redis Streams

This module provides message flow orchestration for the Four-Brain System,
enabling seamless communication between all brains with proper task routing,
error handling, and performance monitoring.

Zero Fabrication Policy: ENFORCED
All message flows use real Redis Streams with proper error handling.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import structlog

from .streams import (
    StreamNames, StreamMessage, MessageType, DoclingRequest, EmbeddingRequest,
    RerankRequest, AgenticTask, MemoryUpdate
)
from .redis_client import RedisStreamsClient
from .memory_store import MemoryStore, create_task_score
from .self_grading import SelfGradingSystem, PerformanceScore
from .self_improvement import SelfImprovementEngine

logger = structlog.get_logger(__name__)

class FlowState(Enum):
    """States of message flow"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class FlowContext:
    """Context for tracking message flow"""
    flow_id: str
    task_id: str
    brain_id: str
    operation: str
    start_time: float
    state: FlowState
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    performance_metrics: Optional[PerformanceScore] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'flow_id': self.flow_id,
            'task_id': self.task_id,
            'brain_id': self.brain_id,
            'operation': self.operation,
            'start_time': self.start_time,
            'state': self.state.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics.__dict__ if self.performance_metrics else None
        }

class MessageFlowOrchestrator:
    """Orchestrates message flow between brains"""
    
    def __init__(self, redis_client: RedisStreamsClient,
                 memory_store: Optional[MemoryStore] = None,
                 grading_engine: Optional[SelfGradingSystem] = None,
                 improvement_engine: Optional[SelfImprovementEngine] = None):
        """Initialize message flow orchestrator"""
        self.redis_client = redis_client
        self.memory_store = memory_store
        self.grading_engine = grading_engine
        self.improvement_engine = improvement_engine
        
        # Flow tracking
        self.active_flows: Dict[str, FlowContext] = {}
        self.completed_flows: List[FlowContext] = []
        
        # Flow timeouts (in seconds)
        self.flow_timeouts = {
            "docling": 300,      # 5 minutes for document processing
            "embedding": 60,     # 1 minute for embeddings
            "rerank": 30,        # 30 seconds for reranking
            "agentic": 600       # 10 minutes for agentic tasks
        }
        
        # Statistics
        self.flows_initiated = 0
        self.flows_completed = 0
        self.flows_failed = 0
        self.flows_timeout = 0
        
        # Message handlers
        self._setup_message_handlers()
        
        logger.info("Message flow orchestrator initialized")
    
    def _setup_message_handlers(self):
        """Setup message handlers for each stream"""
        
        # Register handlers for result streams
        self.redis_client.register_handler(
            StreamNames.DOCLING_RESULTS, self._handle_docling_result
        )
        self.redis_client.register_handler(
            StreamNames.EMBEDDING_RESULTS, self._handle_embedding_result
        )
        self.redis_client.register_handler(
            StreamNames.RERANK_RESULTS, self._handle_rerank_result
        )
        self.redis_client.register_handler(
            StreamNames.AGENTIC_RESULTS, self._handle_agentic_result
        )
        
        # Register handler for memory updates
        self.redis_client.register_handler(
            StreamNames.MEMORY_UPDATES, self._handle_memory_update
        )
    
    async def initiate_document_processing_flow(self, document_path: str,
                                              document_type: str,
                                              processing_options: Dict[str, Any] = None) -> str:
        """Initiate document processing flow"""
        flow_id = f"doc_flow_{int(time.time() * 1000)}"
        
        # Create docling request
        request = DoclingRequest(document_path, document_type, processing_options)
        
        # Track flow
        flow_context = FlowContext(
            flow_id=flow_id,
            task_id=request.task_id,
            brain_id="brain4",
            operation="document_process",
            start_time=time.time(),
            state=FlowState.PENDING,
            inputs={
                'document_path': document_path,
                'document_type': document_type,
                'processing_options': processing_options or {}
            }
        )
        
        self.active_flows[flow_id] = flow_context
        self.flows_initiated += 1
        
        # Send request
        await self.redis_client.send_message(StreamNames.DOCLING_REQUESTS, request)
        
        logger.info("Document processing flow initiated", 
                   flow_id=flow_id, task_id=request.task_id)
        
        return flow_id
    
    async def initiate_embedding_flow(self, text: str, 
                                    embedding_type: str = "default") -> str:
        """Initiate embedding generation flow"""
        flow_id = f"emb_flow_{int(time.time() * 1000)}"
        
        # Create embedding request
        request = EmbeddingRequest(text, embedding_type)
        
        # Track flow
        flow_context = FlowContext(
            flow_id=flow_id,
            task_id=request.task_id,
            brain_id="brain1",
            operation="embedding",
            start_time=time.time(),
            state=FlowState.PENDING,
            inputs={
                'text': text,
                'embedding_type': embedding_type
            }
        )
        
        self.active_flows[flow_id] = flow_context
        self.flows_initiated += 1
        
        # Send request
        await self.redis_client.send_message(StreamNames.EMBEDDING_REQUESTS, request)
        
        logger.info("Embedding flow initiated", 
                   flow_id=flow_id, task_id=request.task_id)
        
        return flow_id
    
    async def initiate_rerank_flow(self, query: str, 
                                 documents: List[Dict[str, Any]]) -> str:
        """Initiate reranking flow"""
        flow_id = f"rerank_flow_{int(time.time() * 1000)}"
        
        # Create rerank request
        request = RerankRequest(query, documents)
        
        # Track flow
        flow_context = FlowContext(
            flow_id=flow_id,
            task_id=request.task_id,
            brain_id="brain2",
            operation="rerank",
            start_time=time.time(),
            state=FlowState.PENDING,
            inputs={
                'query': query,
                'documents': documents
            }
        )
        
        self.active_flows[flow_id] = flow_context
        self.flows_initiated += 1
        
        # Send request
        await self.redis_client.send_message(StreamNames.RERANK_REQUESTS, request)
        
        logger.info("Rerank flow initiated", 
                   flow_id=flow_id, task_id=request.task_id)
        
        return flow_id
    
    async def initiate_agentic_flow(self, task_description: str,
                                  context: Dict[str, Any] = None) -> str:
        """Initiate agentic task flow"""
        flow_id = f"agent_flow_{int(time.time() * 1000)}"
        
        # Create agentic task
        task = AgenticTask(task_description, context)
        
        # Track flow
        flow_context = FlowContext(
            flow_id=flow_id,
            task_id=task.task_id,
            brain_id="brain3",
            operation="agentic_task",
            start_time=time.time(),
            state=FlowState.PENDING,
            inputs={
                'task_description': task_description,
                'context': context or {}
            }
        )
        
        self.active_flows[flow_id] = flow_context
        self.flows_initiated += 1
        
        # Send request
        await self.redis_client.send_message(StreamNames.AGENTIC_TASKS, task)
        
        logger.info("Agentic flow initiated", 
                   flow_id=flow_id, task_id=task.task_id)
        
        return flow_id
    
    async def _handle_docling_result(self, message: StreamMessage):
        """Handle docling processing result"""
        await self._handle_result_message(message, "docling")
    
    async def _handle_embedding_result(self, message: StreamMessage):
        """Handle embedding generation result"""
        await self._handle_result_message(message, "embedding")
    
    async def _handle_rerank_result(self, message: StreamMessage):
        """Handle reranking result"""
        await self._handle_result_message(message, "rerank")
    
    async def _handle_agentic_result(self, message: StreamMessage):
        """Handle agentic task result"""
        await self._handle_result_message(message, "agentic")
    
    async def _handle_result_message(self, message: StreamMessage, operation_type: str):
        """Handle result message from any brain"""
        try:
            # Find corresponding flow
            flow_context = None
            for flow in self.active_flows.values():
                if flow.task_id == message.task_id:
                    flow_context = flow
                    break
            
            if not flow_context:
                logger.warning("No active flow found for result", 
                             task_id=message.task_id, operation=operation_type)
                return
            
            # Update flow context
            execution_time = time.time() - flow_context.start_time
            flow_context.outputs = message.data
            flow_context.state = FlowState.COMPLETED
            
            # Create performance metrics
            success = message.data.get('success', True)
            error_count = 0 if success else 1
            
            # Calculate performance score based on success and execution time
            score = 1.0 if success else 0.0
            if execution_time > 0:
                score *= max(0.1, 1.0 - (execution_time / 30.0))  # Penalize slow execution

            flow_context.performance_metrics = PerformanceScore(
                task_id=flow_context.task_id,
                brain_id=message.brain_id,
                operation=flow_context.current_step,
                score=score,
                timestamp=time.time(),
                details={
                    'execution_time': execution_time,
                    'error_count': error_count,
                    'success_rate': 1.0 if success else 0.0,
                    'output_quality': message.data.get('quality_score'),
                    'custom_metrics': message.data.get('metrics', {})
                }
            )
            
            # Perform grading if engine available
            if self.grading_engine:
                scoring_result = await self.grading_engine.evaluate_performance(
                    flow_context.task_id,
                    flow_context.brain_id,
                    flow_context.operation,
                    flow_context.inputs,
                    flow_context.outputs,
                    flow_context.performance_metrics
                )
                
                # Send memory update
                if self.memory_store:
                    memory_update = MemoryUpdate(
                        operation=flow_context.operation,
                        score=scoring_result.overall_score,
                        metadata={
                            'flow_id': flow_context.flow_id,
                            'execution_time': execution_time,
                            'category_scores': {cat.value: score for cat, score in scoring_result.category_scores.items()},
                            'improvement_suggestions': scoring_result.improvement_suggestions
                        },
                        brain_id=flow_context.brain_id,
                        task_id=flow_context.task_id
                    )
                    
                    await self.redis_client.send_message(StreamNames.MEMORY_UPDATES, memory_update)
            
            # Move to completed flows
            self.completed_flows.append(flow_context)
            del self.active_flows[flow_context.flow_id]
            self.flows_completed += 1
            
            logger.info("Flow completed", 
                       flow_id=flow_context.flow_id,
                       task_id=message.task_id,
                       operation=operation_type,
                       execution_time=execution_time,
                       success=success)
            
        except Exception as e:
            logger.error("Failed to handle result message", 
                        task_id=message.task_id, operation=operation_type, error=str(e))
    
    async def _handle_memory_update(self, message: StreamMessage):
        """Handle memory update message"""
        try:
            if self.memory_store:
                # Extract memory update data
                operation = message.data.get('operation')
                score = message.data.get('score')
                metadata = message.data.get('metadata', {})
                
                # Create task score
                task_score = create_task_score(
                    message.task_id,
                    message.brain_id,
                    operation,
                    score,
                    metadata.get('inputs', {}),
                    metadata
                )
                
                # Store in memory
                await self.memory_store.store_score(task_score)
                
                logger.debug("Memory update processed", 
                           task_id=message.task_id, operation=operation, score=score)
            
        except Exception as e:
            logger.error("Failed to handle memory update", 
                        task_id=message.task_id, error=str(e))
    
    async def check_flow_timeouts(self):
        """Check for and handle flow timeouts"""
        current_time = time.time()
        timed_out_flows = []
        
        for flow_id, flow_context in self.active_flows.items():
            operation_timeout = self.flow_timeouts.get(
                flow_context.operation.split('_')[0], 300  # Default 5 minutes
            )
            
            if (current_time - flow_context.start_time) > operation_timeout:
                flow_context.state = FlowState.TIMEOUT
                flow_context.error_message = f"Flow timed out after {operation_timeout} seconds"
                timed_out_flows.append(flow_context)
        
        # Move timed out flows to completed
        for flow_context in timed_out_flows:
            self.completed_flows.append(flow_context)
            del self.active_flows[flow_context.flow_id]
            self.flows_timeout += 1
            
            logger.warning("Flow timed out", 
                          flow_id=flow_context.flow_id,
                          operation=flow_context.operation,
                          elapsed_time=current_time - flow_context.start_time)
    
    async def get_flow_status(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific flow"""
        # Check active flows
        if flow_id in self.active_flows:
            return self.active_flows[flow_id].to_dict()
        
        # Check completed flows
        for flow in self.completed_flows:
            if flow.flow_id == flow_id:
                return flow.to_dict()
        
        return None
    
    async def wait_for_flow_completion(self, flow_id: str, 
                                     timeout: float = 300.0) -> Optional[FlowContext]:
        """Wait for a flow to complete"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Check if flow is completed
            for flow in self.completed_flows:
                if flow.flow_id == flow_id:
                    return flow
            
            # Check if flow is still active
            if flow_id not in self.active_flows:
                return None  # Flow not found
            
            await asyncio.sleep(0.5)  # Check every 500ms
        
        return None  # Timeout
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message flow statistics"""
        return {
            "flows_initiated": self.flows_initiated,
            "flows_completed": self.flows_completed,
            "flows_failed": self.flows_failed,
            "flows_timeout": self.flows_timeout,
            "active_flows": len(self.active_flows),
            "completion_rate": self.flows_completed / max(self.flows_initiated, 1),
            "timeout_rate": self.flows_timeout / max(self.flows_initiated, 1),
            "average_completion_time": self._calculate_average_completion_time(),
            "flow_timeouts": self.flow_timeouts
        }
    
    def _calculate_average_completion_time(self) -> float:
        """Calculate average completion time for completed flows"""
        if not self.completed_flows:
            return 0.0
        
        completed_flows = [f for f in self.completed_flows if f.state == FlowState.COMPLETED]
        if not completed_flows:
            return 0.0
        
        total_time = 0.0
        for flow in completed_flows:
            if flow.performance_metrics:
                total_time += flow.performance_metrics.execution_time
        
        return total_time / len(completed_flows)
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        # Start timeout checker
        asyncio.create_task(self._timeout_monitor())
        
        # Start performance reflection (if improvement engine available)
        if self.improvement_engine:
            asyncio.create_task(self._reflection_monitor())
    
    async def _timeout_monitor(self):
        """Background task to monitor flow timeouts"""
        while True:
            try:
                await self.check_flow_timeouts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Timeout monitor error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _reflection_monitor(self):
        """Background task to trigger performance reflections"""
        while True:
            try:
                # Import ReflectionLevel here to avoid circular imports
                from .self_improvement import ReflectionLevel

                # Check if reflection should be performed for each brain
                for brain_id in ["brain1", "brain2", "brain3", "brain4"]:
                    # Check for medium-level reflection (every 6 hours)
                    if await self.improvement_engine.should_perform_reflection(brain_id, ReflectionLevel.MEDIUM):
                        reflection_report = await self.improvement_engine.perform_reflection(brain_id, ReflectionLevel.MEDIUM)

                        # Log reflection results
                        logger.info("Reflection completed",
                                   brain_id=brain_id,
                                   suggestions=len(reflection_report.improvement_suggestions),
                                   confidence=reflection_report.confidence_score)

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error("Reflection monitor error", error=str(e))
                await asyncio.sleep(3600)  # Wait longer on error

# Global message flow orchestrator instance
_flow_orchestrator: Optional[MessageFlowOrchestrator] = None

def get_flow_orchestrator(redis_client: RedisStreamsClient,
                         memory_store: Optional[MemoryStore] = None,
                         grading_engine: Optional[SelfGradingSystem] = None,
                         improvement_engine: Optional[SelfImprovementEngine] = None) -> MessageFlowOrchestrator:
    """Get or create the global message flow orchestrator instance"""
    global _flow_orchestrator
    if _flow_orchestrator is None:
        _flow_orchestrator = MessageFlowOrchestrator(
            redis_client, memory_store, grading_engine, improvement_engine
        )
    return _flow_orchestrator
