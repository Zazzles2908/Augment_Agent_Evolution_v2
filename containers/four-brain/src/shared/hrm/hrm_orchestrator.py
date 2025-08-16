#!/usr/bin/env python3
"""
HRM Orchestrator - Hierarchical Reasoning Module Coordinator
Implements the hierarchical convergence pattern between H-Module and L-Module
as specified in the HRM high & low module system.

This orchestrator:
- Coordinates H-Module strategic planning with L-Module execution
- Implements iterative refinement until convergence
- Manages resource allocation and model loading
- Provides unified interface for HRM processing
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .hrm_module import (
    HRMHModule, HRMLModule, HRMRequest, HRMResponse, 
    HRMTaskType, HRMConvergenceState, HRMModuleType
)

logger = logging.getLogger(__name__)


class HRMOrchestrator:
    """
    HRM Orchestrator - Coordinates H-Module and L-Module processing
    Implements the hierarchical convergence pattern for optimal results
    """
    
    def __init__(self, triton_client, resource_manager=None, blackwell_optimizations: bool = True):
        self.triton_client = triton_client
        self.resource_manager = resource_manager
        self.blackwell_optimizations = blackwell_optimizations
        
        # Initialize H-Module and L-Module
        self.h_module = HRMHModule(triton_client, blackwell_optimizations)
        self.l_module = HRMLModule(triton_client, blackwell_optimizations)
        
        # Orchestrator state
        self.initialized = False
        self.active_tasks = {}
        self.convergence_states = {}
        
        # Performance tracking
        self.orchestrator_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "convergence_rate": 0.0,
            "average_iterations": 0.0,
            "total_processing_time": 0.0
        }
        
        logger.info("🎭 HRM Orchestrator initialized with Blackwell optimizations")
    
    async def initialize(self) -> bool:
        """Initialize HRM Orchestrator and load always-on models"""
        try:
            logger.info("🚀 Initializing HRM Orchestrator...")
            
            # Ensure H-Module and L-Module are always loaded
            h_loaded = await self.h_module.load_model()
            l_loaded = await self.l_module.load_model()
            
            if not h_loaded or not l_loaded:
                logger.error("❌ Failed to load HRM modules")
                return False
            
            # Update resource manager to mark HRM models as always-on
            if self.resource_manager:
                self.resource_manager.ensure_loaded(["hrm_h_trt", "hrm_l_trt"])
                logger.info("✅ HRM models marked as always-loaded in ResourceManager")
            
            self.initialized = True
            logger.info("✅ HRM Orchestrator initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ HRM Orchestrator initialization failed: {e}")
            return False
    
    async def process_task(
        self, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None,
        max_iterations: int = 10,
        convergence_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Process task through HRM hierarchical convergence pattern
        
        Args:
            input_data: Input data for processing
            context: Additional context information
            constraints: Processing constraints (latency, quality, etc.)
            max_iterations: Maximum refinement iterations
            convergence_threshold: Convergence threshold (0.0-1.0)
            
        Returns:
            Dict containing final result and convergence information
        """
        if not self.initialized:
            await self.initialize()
        
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting HRM task processing: {task_id}")
            
            # Initialize convergence state
            convergence_state = HRMConvergenceState(
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold
            )
            self.convergence_states[task_id] = convergence_state
            self.active_tasks[task_id] = {"start_time": start_time, "status": "processing"}
            
            # Step 1: H-Module Strategic Planning
            h_result = await self._h_module_planning(task_id, input_data, context, constraints)
            
            if not h_result or not h_result.result.get("success"):
                raise Exception("H-Module strategic planning failed")
            
            # Step 2: Iterative H-L convergence loop
            final_result = await self._convergence_loop(
                task_id, h_result, input_data, context, constraints, convergence_state
            )
            
            # Calculate final metrics
            total_time = time.time() - start_time
            self._update_orchestrator_metrics(task_id, convergence_state, total_time, True)
            
            # Cleanup
            self.active_tasks.pop(task_id, None)
            self.convergence_states.pop(task_id, None)
            
            logger.info(f"✅ HRM task {task_id} completed in {total_time:.2f}s with {convergence_state.iteration} iterations")
            
            return {
                "task_id": task_id,
                "success": True,
                "result": final_result,
                "convergence_info": {
                    "converged": convergence_state.converged,
                    "iterations": convergence_state.iteration,
                    "final_score": convergence_state.convergence_score,
                    "h_confidence": convergence_state.h_module_confidence,
                    "l_confidence": convergence_state.l_module_confidence
                },
                "processing_time_s": total_time,
                "blackwell_optimized": self.blackwell_optimizations
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            self._update_orchestrator_metrics(task_id, convergence_state, total_time, False)
            
            # Cleanup
            self.active_tasks.pop(task_id, None)
            self.convergence_states.pop(task_id, None)
            
            logger.error(f"❌ HRM task {task_id} failed: {e}")
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "processing_time_s": total_time
            }
    
    async def _h_module_planning(
        self, 
        task_id: str, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> HRMResponse:
        """H-Module strategic planning phase"""
        logger.info(f"🧠 H-Module strategic planning for task {task_id}")
        
        h_request = HRMRequest(
            task_id=task_id,
            task_type=HRMTaskType.STRATEGIC_PLANNING,
            input_data=input_data,
            context=context or {},
            constraints=constraints or {}
        )
        
        h_result = await self.h_module.process_request(h_request)
        
        # Update convergence state
        if task_id in self.convergence_states:
            self.convergence_states[task_id].h_module_confidence = h_result.confidence
        
        return h_result
    
    async def _l_module_execution(
        self, 
        task_id: str, 
        h_guidance: Dict[str, Any], 
        input_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> HRMResponse:
        """L-Module execution phase based on H-Module guidance"""
        logger.info(f"⚡ L-Module execution for task {task_id}")
        
        # Combine original input with H-Module guidance
        l_input_data = {
            **input_data,
            "h_module_guidance": h_guidance,
            "strategic_plan": h_guidance.get("result", {})
        }
        
        l_request = HRMRequest(
            task_id=task_id,
            task_type=HRMTaskType.EXECUTION,
            input_data=l_input_data,
            context=context or {}
        )
        
        l_result = await self.l_module.process_request(l_request)
        
        # Update convergence state
        if task_id in self.convergence_states:
            self.convergence_states[task_id].l_module_confidence = l_result.confidence
        
        return l_result
    
    async def _convergence_loop(
        self,
        task_id: str,
        initial_h_result: HRMResponse,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        convergence_state: HRMConvergenceState
    ) -> Dict[str, Any]:
        """Iterative convergence loop between H-Module and L-Module"""
        
        current_h_result = initial_h_result
        final_result = None
        
        while (convergence_state.iteration < convergence_state.max_iterations and 
               not convergence_state.converged):
            
            convergence_state.iteration += 1
            logger.info(f"🔄 Convergence iteration {convergence_state.iteration} for task {task_id}")
            
            # L-Module execution based on current H-Module guidance
            l_result = await self._l_module_execution(
                task_id, current_h_result.result, input_data, context
            )
            
            # Calculate convergence score
            convergence_score = self._calculate_convergence_score(current_h_result, l_result)
            convergence_state.convergence_score = convergence_score
            
            logger.info(f"📊 Iteration {convergence_state.iteration} convergence score: {convergence_score:.3f}")
            
            # Check convergence
            if convergence_score >= convergence_state.convergence_threshold:
                convergence_state.converged = True
                final_result = self._synthesize_final_result(current_h_result, l_result)
                logger.info(f"✅ Convergence achieved for task {task_id} at iteration {convergence_state.iteration}")
                break
            
            # If not converged, refine H-Module planning based on L-Module feedback
            if convergence_state.iteration < convergence_state.max_iterations:
                refined_context = {
                    **context,
                    "l_module_feedback": l_result.result,
                    "previous_iteration": convergence_state.iteration - 1,
                    "convergence_score": convergence_score
                }
                
                current_h_result = await self._h_module_planning(
                    task_id, input_data, refined_context, constraints
                )
            else:
                # Max iterations reached, use best available result
                final_result = self._synthesize_final_result(current_h_result, l_result)
                logger.warning(f"⚠️ Max iterations reached for task {task_id} without full convergence")
        
        return final_result
    
    def _calculate_convergence_score(self, h_result: HRMResponse, l_result: HRMResponse) -> float:
        """Calculate convergence score between H-Module and L-Module results"""
        try:
            # Weighted combination of confidence scores and result consistency
            h_confidence = h_result.confidence
            l_confidence = l_result.confidence
            
            # Base score from confidence levels
            confidence_score = (h_confidence + l_confidence) / 2
            
            # Consistency bonus (simplified - in real implementation would compare semantic similarity)
            consistency_bonus = 0.1 if h_confidence > 0.8 and l_confidence > 0.8 else 0.0
            
            # Convergence score from individual module responses
            h_convergence = h_result.convergence_score
            l_convergence = l_result.convergence_score
            module_convergence = (h_convergence + l_convergence) / 2
            
            # Final weighted score
            final_score = (confidence_score * 0.4 + module_convergence * 0.5 + consistency_bonus * 0.1)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating convergence score: {e}")
            return 0.5  # Default moderate score
    
    def _synthesize_final_result(self, h_result: HRMResponse, l_result: HRMResponse) -> Dict[str, Any]:
        """Synthesize final result from H-Module and L-Module outputs"""
        return {
            "strategic_plan": h_result.result,
            "execution_result": l_result.result,
            "synthesis": {
                "h_module_confidence": h_result.confidence,
                "l_module_confidence": l_result.confidence,
                "combined_confidence": (h_result.confidence + l_result.confidence) / 2,
                "processing_times": {
                    "h_module_ms": h_result.processing_time_ms,
                    "l_module_ms": l_result.processing_time_ms
                }
            },
            "success": True
        }
    
    def _update_orchestrator_metrics(
        self, 
        task_id: str, 
        convergence_state: HRMConvergenceState, 
        total_time: float, 
        success: bool
    ):
        """Update orchestrator performance metrics"""
        self.orchestrator_metrics["total_tasks"] += 1
        self.orchestrator_metrics["total_processing_time"] += total_time
        
        if success:
            self.orchestrator_metrics["successful_tasks"] += 1
            
            if convergence_state.converged:
                # Update convergence rate
                total_converged = self.orchestrator_metrics["convergence_rate"] * (self.orchestrator_metrics["total_tasks"] - 1)
                total_converged += 1
                self.orchestrator_metrics["convergence_rate"] = total_converged / self.orchestrator_metrics["total_tasks"]
            
            # Update average iterations
            total_iterations = self.orchestrator_metrics["average_iterations"] * (self.orchestrator_metrics["successful_tasks"] - 1)
            total_iterations += convergence_state.iteration
            self.orchestrator_metrics["average_iterations"] = total_iterations / self.orchestrator_metrics["successful_tasks"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics"""
        return {
            "initialized": self.initialized,
            "blackwell_optimizations": self.blackwell_optimizations,
            "active_tasks": len(self.active_tasks),
            "h_module_status": self.h_module.get_status(),
            "l_module_status": self.l_module.get_status(),
            "orchestrator_metrics": self.orchestrator_metrics,
            "active_task_ids": list(self.active_tasks.keys())
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("🛑 Shutting down HRM Orchestrator...")
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            logger.info(f"⏳ Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.sleep(2)  # Brief grace period
        
        self.active_tasks.clear()
        self.convergence_states.clear()
        self.initialized = False
        
        logger.info("✅ HRM Orchestrator shutdown completed")
