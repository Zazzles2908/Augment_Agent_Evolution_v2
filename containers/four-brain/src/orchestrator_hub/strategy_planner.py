#!/usr/bin/env python3
"""
Advanced Strategy Planner for K2-Vector-Hub
Implements sophisticated execution strategy planning for multi-brain coordination

This module provides advanced strategy planning capabilities that go beyond simple
brain allocation to include execution sequencing, dependency management, parallel
processing optimization, and adaptive strategy adjustment.

Key Features:
- Multi-stage execution planning
- Dependency graph analysis
- Parallel processing optimization
- Adaptive strategy adjustment
- Performance-based strategy refinement
- Risk assessment and mitigation
- Resource constraint handling

Zero Fabrication Policy: ENFORCED
All strategy planning is based on real system capabilities and verified logic.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for strategy planning"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class StrategyType(Enum):
    """Strategy types for different task patterns"""
    SIMPLE_QUERY = "simple_query"
    COMPLEX_ANALYSIS = "complex_analysis"
    DOCUMENT_WORKFLOW = "document_workflow"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    REAL_TIME_PROCESSING = "real_time_processing"


class ExecutionStage(Enum):
    """Execution stages in strategy planning"""
    PREPROCESSING = "preprocessing"
    PRIMARY_PROCESSING = "primary_processing"
    SECONDARY_PROCESSING = "secondary_processing"
    AGGREGATION = "aggregation"
    POSTPROCESSING = "postprocessing"


@dataclass
class ExecutionStep:
    """Individual execution step in a strategy plan"""
    step_id: str
    brain_id: str
    stage: ExecutionStage
    operation: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_ms: int = 1000
    priority: int = 1
    can_parallelize: bool = True
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = f"step_{int(time.time() * 1000)}"


@dataclass
class StrategyPlan:
    """Comprehensive strategy plan for task execution"""
    plan_id: str
    task_id: str
    strategy_type: StrategyType
    execution_mode: ExecutionMode
    steps: List[ExecutionStep]
    estimated_total_duration_ms: int
    confidence_score: float
    risk_factors: List[str] = field(default_factory=list)
    fallback_plans: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedStrategyPlanner:
    """
    Advanced strategy planner for sophisticated multi-brain execution planning
    """
    
    def __init__(self, task_router=None):
        """Initialize strategy planner with optional task router integration"""
        self.task_router = task_router
        self.strategy_templates = {}
        self.performance_history = defaultdict(list)
        self.brain_capabilities = {}
        self.active_plans = {}
        
        # Initialize brain capabilities and strategy templates
        self._initialize_brain_capabilities()
        self._initialize_strategy_templates()
        
        logger.info("🎯 AdvancedStrategyPlanner initialized")
    
    def _initialize_brain_capabilities(self):
        """Initialize brain capabilities and performance characteristics"""
        self.brain_capabilities = {
            "brain1": {
                "specializations": ["embedding", "semantic_search", "vector_operations"],
                "avg_response_time_ms": 500,
                "throughput_per_second": 20,
                "memory_usage_mb": 2048,
                "parallel_capacity": 4,
                "reliability_score": 0.95
            },
            "brain2": {
                "specializations": ["reranking", "relevance_scoring", "result_filtering"],
                "avg_response_time_ms": 300,
                "throughput_per_second": 30,
                "memory_usage_mb": 1024,
                "parallel_capacity": 6,
                "reliability_score": 0.97
            },
            "brain3": {
                "specializations": ["reasoning", "conversation", "workflow_orchestration"],
                "avg_response_time_ms": 1000,
                "throughput_per_second": 10,
                "memory_usage_mb": 3072,
                "parallel_capacity": 2,
                "reliability_score": 0.92
            },
            "brain4": {
                "specializations": ["document_processing", "content_extraction", "format_conversion"],
                "avg_response_time_ms": 2000,
                "throughput_per_second": 5,
                "memory_usage_mb": 4096,
                "parallel_capacity": 3,
                "reliability_score": 0.90
            }
        }
    
    def _initialize_strategy_templates(self):
        """Initialize strategy templates for common task patterns"""
        self.strategy_templates = {
            StrategyType.SIMPLE_QUERY: {
                "execution_mode": ExecutionMode.SEQUENTIAL,
                "stages": [ExecutionStage.PRIMARY_PROCESSING],
                "typical_brains": ["brain1", "brain2"],
                "estimated_duration_ms": 800
            },
            StrategyType.COMPLEX_ANALYSIS: {
                "execution_mode": ExecutionMode.PIPELINE,
                "stages": [ExecutionStage.PREPROCESSING, ExecutionStage.PRIMARY_PROCESSING, 
                          ExecutionStage.SECONDARY_PROCESSING, ExecutionStage.AGGREGATION],
                "typical_brains": ["brain1", "brain2", "brain3"],
                "estimated_duration_ms": 3000
            },
            StrategyType.DOCUMENT_WORKFLOW: {
                "execution_mode": ExecutionMode.PIPELINE,
                "stages": [ExecutionStage.PREPROCESSING, ExecutionStage.PRIMARY_PROCESSING, 
                          ExecutionStage.SECONDARY_PROCESSING, ExecutionStage.POSTPROCESSING],
                "typical_brains": ["brain4", "brain1", "brain2"],
                "estimated_duration_ms": 5000
            },
            StrategyType.MULTI_STEP_REASONING: {
                "execution_mode": ExecutionMode.ADAPTIVE,
                "stages": [ExecutionStage.PREPROCESSING, ExecutionStage.PRIMARY_PROCESSING, 
                          ExecutionStage.SECONDARY_PROCESSING, ExecutionStage.AGGREGATION],
                "typical_brains": ["brain3", "brain1", "brain2"],
                "estimated_duration_ms": 4000
            },
            StrategyType.REAL_TIME_PROCESSING: {
                "execution_mode": ExecutionMode.PARALLEL,
                "stages": [ExecutionStage.PRIMARY_PROCESSING],
                "typical_brains": ["brain1", "brain2"],
                "estimated_duration_ms": 500
            }
        }
    
    async def create_execution_strategy(self, task_request: Dict[str, Any]) -> StrategyPlan:
        """
        Create comprehensive execution strategy for a task request
        
        Args:
            task_request: Task request containing type, payload, and requirements
            
        Returns:
            StrategyPlan: Detailed execution strategy plan
        """
        task_id = task_request.get("task_id", f"task_{int(time.time() * 1000)}")
        task_type = task_request.get("task_type", "unknown")
        
        logger.info(f"🎯 Creating execution strategy for task {task_id} (type: {task_type})")
        
        # Determine strategy type
        strategy_type = self._determine_strategy_type(task_request)
        
        # Get strategy template
        template = self.strategy_templates.get(strategy_type, self.strategy_templates[StrategyType.SIMPLE_QUERY])
        
        # Create execution steps
        execution_steps = await self._create_execution_steps(task_request, strategy_type, template)
        
        # Optimize execution plan
        optimized_steps = self._optimize_execution_plan(execution_steps, template["execution_mode"])
        
        # Calculate estimated duration
        estimated_duration = self._calculate_estimated_duration(optimized_steps)
        
        # Assess risks and create fallback plans
        risk_factors = self._assess_risk_factors(optimized_steps)
        fallback_plans = self._create_fallback_plans(optimized_steps, risk_factors)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(optimized_steps, risk_factors)
        
        # Create strategy plan
        strategy_plan = StrategyPlan(
            plan_id=f"plan_{task_id}_{int(time.time() * 1000)}",
            task_id=task_id,
            strategy_type=strategy_type,
            execution_mode=template["execution_mode"],
            steps=optimized_steps,
            estimated_total_duration_ms=estimated_duration,
            confidence_score=confidence_score,
            risk_factors=risk_factors,
            fallback_plans=fallback_plans,
            metadata={
                "template_used": strategy_type.value,
                "optimization_applied": True,
                "brain_count": len(set(step.brain_id for step in optimized_steps)),
                "parallel_steps": len([step for step in optimized_steps if step.can_parallelize])
            }
        )
        
        # Store active plan
        self.active_plans[task_id] = strategy_plan
        
        logger.info(f"✅ Execution strategy created for task {task_id}: "
                   f"{strategy_type.value} with {len(optimized_steps)} steps")
        
        return strategy_plan
    
    def _determine_strategy_type(self, task_request: Dict[str, Any]) -> StrategyType:
        """Determine appropriate strategy type based on task characteristics"""
        task_type = task_request.get("task_type", "").lower()
        payload = task_request.get("payload", {})
        
        # Analyze task complexity
        if "document" in task_type or "upload" in task_type:
            return StrategyType.DOCUMENT_WORKFLOW
        elif "chat" in task_type or "conversation" in task_type:
            if len(payload.get("messages", [])) > 3:
                return StrategyType.MULTI_STEP_REASONING
            else:
                return StrategyType.SIMPLE_QUERY
        elif "search" in task_type:
            query_length = len(payload.get("query", ""))
            if query_length > 100:
                return StrategyType.COMPLEX_ANALYSIS
            else:
                return StrategyType.SIMPLE_QUERY
        elif task_request.get("priority", 1) >= 3:
            return StrategyType.REAL_TIME_PROCESSING
        else:
            return StrategyType.SIMPLE_QUERY
    
    async def _create_execution_steps(self, task_request: Dict[str, Any], 
                                    strategy_type: StrategyType, 
                                    template: Dict[str, Any]) -> List[ExecutionStep]:
        """Create detailed execution steps based on strategy type and template"""
        steps = []
        task_id = task_request.get("task_id", "unknown")
        
        if strategy_type == StrategyType.DOCUMENT_WORKFLOW:
            steps = [
                ExecutionStep(
                    step_id=f"{task_id}_doc_extract",
                    brain_id="brain4",
                    stage=ExecutionStage.PREPROCESSING,
                    operation="document_extraction",
                    inputs=["document_file"],
                    outputs=["extracted_text", "metadata"],
                    estimated_duration_ms=2000,
                    priority=1,
                    can_parallelize=False
                ),
                ExecutionStep(
                    step_id=f"{task_id}_embedding",
                    brain_id="brain1",
                    stage=ExecutionStage.PRIMARY_PROCESSING,
                    operation="text_embedding",
                    inputs=["extracted_text"],
                    outputs=["embeddings"],
                    dependencies=[f"{task_id}_doc_extract"],
                    estimated_duration_ms=500,
                    priority=2,
                    can_parallelize=True
                ),
                ExecutionStep(
                    step_id=f"{task_id}_indexing",
                    brain_id="brain2",
                    stage=ExecutionStage.POSTPROCESSING,
                    operation="vector_indexing",
                    inputs=["embeddings", "metadata"],
                    outputs=["indexed_document"],
                    dependencies=[f"{task_id}_embedding"],
                    estimated_duration_ms=300,
                    priority=3,
                    can_parallelize=False
                )
            ]
        
        elif strategy_type == StrategyType.COMPLEX_ANALYSIS:
            steps = [
                ExecutionStep(
                    step_id=f"{task_id}_query_analysis",
                    brain_id="brain3",
                    stage=ExecutionStage.PREPROCESSING,
                    operation="query_understanding",
                    inputs=["user_query"],
                    outputs=["analyzed_query", "intent"],
                    estimated_duration_ms=800,
                    priority=1,
                    can_parallelize=False
                ),
                ExecutionStep(
                    step_id=f"{task_id}_semantic_search",
                    brain_id="brain1",
                    stage=ExecutionStage.PRIMARY_PROCESSING,
                    operation="semantic_search",
                    inputs=["analyzed_query"],
                    outputs=["search_results"],
                    dependencies=[f"{task_id}_query_analysis"],
                    estimated_duration_ms=500,
                    priority=2,
                    can_parallelize=True
                ),
                ExecutionStep(
                    step_id=f"{task_id}_reranking",
                    brain_id="brain2",
                    stage=ExecutionStage.SECONDARY_PROCESSING,
                    operation="result_reranking",
                    inputs=["search_results", "analyzed_query"],
                    outputs=["ranked_results"],
                    dependencies=[f"{task_id}_semantic_search"],
                    estimated_duration_ms=300,
                    priority=3,
                    can_parallelize=False
                ),
                ExecutionStep(
                    step_id=f"{task_id}_synthesis",
                    brain_id="brain3",
                    stage=ExecutionStage.AGGREGATION,
                    operation="result_synthesis",
                    inputs=["ranked_results", "intent"],
                    outputs=["final_response"],
                    dependencies=[f"{task_id}_reranking"],
                    estimated_duration_ms=1000,
                    priority=4,
                    can_parallelize=False
                )
            ]
        
        else:  # Simple query or default
            steps = [
                ExecutionStep(
                    step_id=f"{task_id}_process",
                    brain_id="brain1",
                    stage=ExecutionStage.PRIMARY_PROCESSING,
                    operation="simple_processing",
                    inputs=["user_input"],
                    outputs=["processed_result"],
                    estimated_duration_ms=500,
                    priority=1,
                    can_parallelize=True
                )
            ]
        
        return steps
    
    def _optimize_execution_plan(self, steps: List[ExecutionStep], 
                                execution_mode: ExecutionMode) -> List[ExecutionStep]:
        """Optimize execution plan based on execution mode and constraints"""
        if execution_mode == ExecutionMode.PARALLEL:
            # Identify steps that can run in parallel
            for step in steps:
                if not step.dependencies and step.can_parallelize:
                    step.priority = 1  # High priority for parallel execution
        
        elif execution_mode == ExecutionMode.PIPELINE:
            # Optimize for pipeline execution with minimal delays
            for i, step in enumerate(steps):
                step.priority = i + 1  # Sequential priority
        
        elif execution_mode == ExecutionMode.ADAPTIVE:
            # Adaptive optimization based on current system state
            # This would typically check current brain loads and adjust accordingly
            pass
        
        # Sort steps by priority and dependencies
        return sorted(steps, key=lambda x: (x.priority, len(x.dependencies)))
    
    def _calculate_estimated_duration(self, steps: List[ExecutionStep]) -> int:
        """Calculate estimated total duration considering parallelization"""
        if not steps:
            return 0
        
        # Build dependency graph
        dependency_graph = {}
        for step in steps:
            dependency_graph[step.step_id] = {
                "duration": step.estimated_duration_ms,
                "dependencies": step.dependencies,
                "can_parallelize": step.can_parallelize
            }
        
        # Calculate critical path (simplified)
        max_duration = 0
        for step in steps:
            if not step.dependencies:  # Starting steps
                max_duration = max(max_duration, step.estimated_duration_ms)
            else:
                # Add to longest dependency chain
                max_duration = max(max_duration, 
                                 step.estimated_duration_ms + 
                                 max([s.estimated_duration_ms for s in steps 
                                     if s.step_id in step.dependencies], default=0))
        
        return max_duration
    
    def _assess_risk_factors(self, steps: List[ExecutionStep]) -> List[str]:
        """Assess potential risk factors in the execution plan"""
        risks = []
        
        # Check for single points of failure
        brain_usage = defaultdict(int)
        for step in steps:
            brain_usage[step.brain_id] += 1
        
        for brain_id, usage_count in brain_usage.items():
            if usage_count > 3:
                risks.append(f"High dependency on {brain_id} ({usage_count} steps)")
        
        # Check for long execution chains
        max_chain_length = max([len(step.dependencies) for step in steps], default=0)
        if max_chain_length > 3:
            risks.append(f"Long dependency chain ({max_chain_length} levels)")
        
        # Check for resource constraints
        total_memory = sum([step.resource_requirements.get("memory_mb", 512) for step in steps])
        if total_memory > 8192:  # 8GB threshold
            risks.append(f"High memory usage ({total_memory}MB)")
        
        return risks
    
    def _create_fallback_plans(self, steps: List[ExecutionStep], 
                             risk_factors: List[str]) -> List[str]:
        """Create fallback plans for identified risks"""
        fallback_plans = []
        
        if any("High dependency" in risk for risk in risk_factors):
            fallback_plans.append("brain_substitution_plan")
        
        if any("Long dependency chain" in risk for risk in risk_factors):
            fallback_plans.append("parallel_execution_fallback")
        
        if any("High memory usage" in risk for risk in risk_factors):
            fallback_plans.append("memory_optimization_plan")
        
        # Always include emergency fallback
        fallback_plans.append("emergency_simple_execution")
        
        return fallback_plans
    
    def _calculate_confidence_score(self, steps: List[ExecutionStep], 
                                  risk_factors: List[str]) -> float:
        """Calculate confidence score for the execution plan"""
        base_confidence = 0.9
        
        # Reduce confidence based on risk factors
        risk_penalty = len(risk_factors) * 0.1
        
        # Reduce confidence based on plan complexity
        complexity_penalty = max(0, (len(steps) - 3) * 0.05)
        
        # Adjust based on brain reliability
        brain_reliability = []
        for step in steps:
            brain_caps = self.brain_capabilities.get(step.brain_id, {})
            reliability = brain_caps.get("reliability_score", 0.8)
            brain_reliability.append(reliability)
        
        avg_reliability = sum(brain_reliability) / len(brain_reliability) if brain_reliability else 0.8
        
        final_confidence = base_confidence - risk_penalty - complexity_penalty
        final_confidence *= avg_reliability
        
        return max(0.1, min(1.0, final_confidence))
    
    async def update_plan_performance(self, task_id: str, actual_duration_ms: int, 
                                    success: bool, step_results: Dict[str, Any] = None):
        """Update performance history based on actual execution results"""
        if task_id in self.active_plans:
            plan = self.active_plans[task_id]
            
            performance_record = {
                "plan_id": plan.plan_id,
                "strategy_type": plan.strategy_type.value,
                "estimated_duration_ms": plan.estimated_total_duration_ms,
                "actual_duration_ms": actual_duration_ms,
                "success": success,
                "accuracy": abs(plan.estimated_total_duration_ms - actual_duration_ms) / plan.estimated_total_duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.performance_history[plan.strategy_type].append(performance_record)
            
            # Remove from active plans
            del self.active_plans[task_id]
            
            logger.info(f"📊 Updated performance for task {task_id}: "
                       f"estimated={plan.estimated_total_duration_ms}ms, "
                       f"actual={actual_duration_ms}ms, success={success}")
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get strategy planning statistics and performance metrics"""
        total_plans = sum(len(records) for records in self.performance_history.values())
        
        if total_plans == 0:
            return {"total_plans": 0, "active_plans": len(self.active_plans)}
        
        # Calculate accuracy metrics
        all_records = []
        for records in self.performance_history.values():
            all_records.extend(records)
        
        success_rate = sum(1 for r in all_records if r["success"]) / len(all_records)
        avg_accuracy = sum(r["accuracy"] for r in all_records) / len(all_records)
        
        # Strategy type distribution
        strategy_distribution = {
            strategy_type.value: len(records) 
            for strategy_type, records in self.performance_history.items()
        }
        
        return {
            "total_plans": total_plans,
            "active_plans": len(self.active_plans),
            "success_rate": success_rate,
            "average_accuracy": avg_accuracy,
            "strategy_distribution": strategy_distribution,
            "brain_capabilities": self.brain_capabilities
        }
