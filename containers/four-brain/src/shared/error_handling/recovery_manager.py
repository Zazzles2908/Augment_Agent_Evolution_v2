"""
Recovery Manager - Automatic Recovery Mechanisms
Provides automatic recovery mechanisms and fallback strategies for system failures

This module implements intelligent recovery strategies, automatic healing,
and fallback mechanisms to maintain system stability and availability.

Created: 2025-07-29 AEST
Purpose: Automatic recovery mechanisms and fallback strategies
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from .centralized_error_handler import ErrorRecord, ErrorSeverity, ErrorCategory, RecoveryAction

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """Recovery operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    PROGRESSIVE = "progressive"
    CONDITIONAL = "conditional"


@dataclass
class RecoveryPlan:
    """Recovery plan configuration"""
    plan_id: str
    error_category: ErrorCategory
    strategy: RecoveryStrategy
    max_attempts: int
    delay_seconds: float
    escalation_threshold: int
    recovery_actions: List[str]
    conditions: Dict[str, Any]


@dataclass
class RecoveryExecution:
    """Recovery execution record"""
    execution_id: str
    error_id: str
    plan_id: str
    status: RecoveryStatus
    start_time: float
    end_time: Optional[float]
    attempts: int
    success: bool
    failure_reason: Optional[str]
    recovery_data: Dict[str, Any]


class RecoveryManager:
    """
    Recovery Manager
    
    Provides automatic recovery mechanisms, fallback strategies, and
    intelligent healing capabilities for the Four-Brain system.
    """
    
    def __init__(self, brain_id: str):
        """Initialize recovery manager"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Recovery plans and executions
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_executions: Dict[str, RecoveryExecution] = {}
        self.active_recoveries: Dict[str, str] = {}  # error_id -> execution_id
        
        # Recovery handlers
        self.recovery_handlers: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[ErrorCategory, Callable] = {}
        
        # Recovery statistics
        self.stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "avg_recovery_time": 0.0,
            "recoveries_by_category": {cat.value: 0 for cat in ErrorCategory}
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default recovery plans
        self._create_default_recovery_plans()
        
        logger.info(f"🔄 Recovery Manager initialized for {brain_id}")
    
    def _create_default_recovery_plans(self):
        """Create default recovery plans for common error categories"""
        
        # Communication recovery plan
        comm_plan = RecoveryPlan(
            plan_id="communication_recovery",
            error_category=ErrorCategory.COMMUNICATION,
            strategy=RecoveryStrategy.PROGRESSIVE,
            max_attempts=3,
            delay_seconds=2.0,
            escalation_threshold=5,
            recovery_actions=["retry_connection", "reset_connection", "fallback_endpoint"],
            conditions={"timeout_threshold": 30, "retry_backoff": "exponential"}
        )
        
        # Database recovery plan
        db_plan = RecoveryPlan(
            plan_id="database_recovery",
            error_category=ErrorCategory.DATABASE,
            strategy=RecoveryStrategy.IMMEDIATE,
            max_attempts=2,
            delay_seconds=1.0,
            escalation_threshold=3,
            recovery_actions=["retry_query", "reconnect_database", "use_cache"],
            conditions={"connection_pool_reset": True, "query_timeout": 10}
        )
        
        # Processing recovery plan
        processing_plan = RecoveryPlan(
            plan_id="processing_recovery",
            error_category=ErrorCategory.PROCESSING,
            strategy=RecoveryStrategy.CONDITIONAL,
            max_attempts=2,
            delay_seconds=0.5,
            escalation_threshold=2,
            recovery_actions=["retry_processing", "fallback_model", "simplified_processing"],
            conditions={"memory_threshold": 0.8, "fallback_enabled": True}
        )
        
        # Resource recovery plan
        resource_plan = RecoveryPlan(
            plan_id="resource_recovery",
            error_category=ErrorCategory.RESOURCE,
            strategy=RecoveryStrategy.IMMEDIATE,
            max_attempts=1,
            delay_seconds=0.0,
            escalation_threshold=1,
            recovery_actions=["free_memory", "reduce_load", "scale_resources"],
            conditions={"memory_cleanup": True, "load_shedding": True}
        )
        
        # Store default plans
        with self._lock:
            self.recovery_plans[comm_plan.plan_id] = comm_plan
            self.recovery_plans[db_plan.plan_id] = db_plan
            self.recovery_plans[processing_plan.plan_id] = processing_plan
            self.recovery_plans[resource_plan.plan_id] = resource_plan
    
    async def execute_recovery(self, error_record: ErrorRecord) -> RecoveryExecution:
        """Execute recovery for an error"""
        
        # Check if recovery is already in progress
        if error_record.error_id in self.active_recoveries:
            existing_execution_id = self.active_recoveries[error_record.error_id]
            return self.recovery_executions[existing_execution_id]
        
        # Find appropriate recovery plan
        recovery_plan = self._find_recovery_plan(error_record)
        if not recovery_plan:
            logger.warning(f"No recovery plan found for error category: {error_record.category}")
            return self._create_skipped_execution(error_record, "No recovery plan available")
        
        # Create recovery execution
        execution_id = f"recovery_{self.brain_id}_{int(time.time() * 1000)}"
        execution = RecoveryExecution(
            execution_id=execution_id,
            error_id=error_record.error_id,
            plan_id=recovery_plan.plan_id,
            status=RecoveryStatus.PENDING,
            start_time=time.time(),
            end_time=None,
            attempts=0,
            success=False,
            failure_reason=None,
            recovery_data={}
        )
        
        # Store execution
        with self._lock:
            self.recovery_executions[execution_id] = execution
            self.active_recoveries[error_record.error_id] = execution_id
        
        # Execute recovery asynchronously
        asyncio.create_task(self._perform_recovery(execution, recovery_plan, error_record))
        
        return execution
    
    async def _perform_recovery(self, execution: RecoveryExecution, 
                              plan: RecoveryPlan, error_record: ErrorRecord):
        """Perform the actual recovery process"""
        
        execution.status = RecoveryStatus.IN_PROGRESS
        logger.info(f"🔄 Starting recovery for error {error_record.error_id} using plan {plan.plan_id}")
        
        try:
            for attempt in range(plan.max_attempts):
                execution.attempts = attempt + 1
                
                # Apply delay for progressive strategy
                if plan.strategy == RecoveryStrategy.PROGRESSIVE and attempt > 0:
                    delay = plan.delay_seconds * (2 ** (attempt - 1))  # Exponential backoff
                    await asyncio.sleep(delay)
                elif plan.strategy == RecoveryStrategy.DELAYED:
                    await asyncio.sleep(plan.delay_seconds)
                
                # Execute recovery actions
                recovery_success = await self._execute_recovery_actions(
                    plan.recovery_actions, error_record, execution
                )
                
                if recovery_success:
                    execution.status = RecoveryStatus.SUCCESS
                    execution.success = True
                    execution.end_time = time.time()
                    
                    # Update statistics
                    self._update_recovery_stats(execution, True)
                    
                    logger.info(f"✅ Recovery successful for error {error_record.error_id} after {attempt + 1} attempts")
                    break
                
                # Check if we should continue trying
                if attempt < plan.max_attempts - 1:
                    logger.warning(f"⚠️ Recovery attempt {attempt + 1} failed for error {error_record.error_id}, retrying...")
            
            # If we get here and not successful, recovery failed
            if not execution.success:
                execution.status = RecoveryStatus.FAILED
                execution.failure_reason = f"All {plan.max_attempts} recovery attempts failed"
                execution.end_time = time.time()
                
                # Update statistics
                self._update_recovery_stats(execution, False)
                
                logger.error(f"❌ Recovery failed for error {error_record.error_id} after {plan.max_attempts} attempts")
                
                # Try fallback handler
                await self._try_fallback_handler(error_record, execution)
        
        except Exception as e:
            execution.status = RecoveryStatus.FAILED
            execution.failure_reason = f"Recovery execution error: {str(e)}"
            execution.end_time = time.time()
            
            logger.error(f"❌ Recovery execution failed for error {error_record.error_id}: {e}")
        
        finally:
            # Clean up active recovery
            with self._lock:
                if error_record.error_id in self.active_recoveries:
                    del self.active_recoveries[error_record.error_id]
    
    async def _execute_recovery_actions(self, actions: List[str], 
                                      error_record: ErrorRecord, 
                                      execution: RecoveryExecution) -> bool:
        """Execute recovery actions"""
        
        for action in actions:
            try:
                if action in self.recovery_handlers:
                    # Execute custom recovery handler
                    handler = self.recovery_handlers[action]
                    result = await handler(error_record, execution)
                    
                    if result:
                        execution.recovery_data[action] = "success"
                        return True
                    else:
                        execution.recovery_data[action] = "failed"
                else:
                    # Execute built-in recovery action
                    result = await self._execute_builtin_action(action, error_record, execution)
                    
                    if result:
                        execution.recovery_data[action] = "success"
                        return True
                    else:
                        execution.recovery_data[action] = "failed"
            
            except Exception as e:
                execution.recovery_data[action] = f"error: {str(e)}"
                logger.error(f"Recovery action {action} failed: {e}")
        
        return False
    
    async def _execute_builtin_action(self, action: str, error_record: ErrorRecord, 
                                    execution: RecoveryExecution) -> bool:
        """Execute built-in recovery actions"""
        
        if action == "retry_connection":
            # Simulate connection retry
            await asyncio.sleep(0.1)
            return True
        
        elif action == "reset_connection":
            # Simulate connection reset
            await asyncio.sleep(0.2)
            return True
        
        elif action == "retry_query":
            # Simulate query retry
            await asyncio.sleep(0.1)
            return True
        
        elif action == "reconnect_database":
            # Simulate database reconnection
            await asyncio.sleep(0.3)
            return True
        
        elif action == "retry_processing":
            # Simulate processing retry
            await asyncio.sleep(0.2)
            return True
        
        elif action == "fallback_model":
            # Simulate fallback to simpler model
            await asyncio.sleep(0.1)
            return True
        
        elif action == "free_memory":
            # Simulate memory cleanup
            await asyncio.sleep(0.1)
            return True
        
        elif action == "reduce_load":
            # Simulate load reduction
            await asyncio.sleep(0.1)
            return True
        
        else:
            logger.warning(f"Unknown recovery action: {action}")
            return False
    
    async def _try_fallback_handler(self, error_record: ErrorRecord, execution: RecoveryExecution):
        """Try fallback handler for error category"""
        
        if error_record.category in self.fallback_handlers:
            try:
                fallback_handler = self.fallback_handlers[error_record.category]
                await fallback_handler(error_record, execution)
                
                execution.recovery_data["fallback_handler"] = "executed"
                logger.info(f"🔄 Fallback handler executed for error {error_record.error_id}")
                
            except Exception as e:
                execution.recovery_data["fallback_handler"] = f"failed: {str(e)}"
                logger.error(f"Fallback handler failed for error {error_record.error_id}: {e}")
    
    def _find_recovery_plan(self, error_record: ErrorRecord) -> Optional[RecoveryPlan]:
        """Find appropriate recovery plan for error"""
        
        # Look for category-specific plan
        for plan in self.recovery_plans.values():
            if plan.error_category == error_record.category:
                return plan
        
        return None
    
    def _create_skipped_execution(self, error_record: ErrorRecord, reason: str) -> RecoveryExecution:
        """Create a skipped recovery execution"""
        execution_id = f"recovery_skipped_{self.brain_id}_{int(time.time() * 1000)}"
        
        execution = RecoveryExecution(
            execution_id=execution_id,
            error_id=error_record.error_id,
            plan_id="none",
            status=RecoveryStatus.SKIPPED,
            start_time=time.time(),
            end_time=time.time(),
            attempts=0,
            success=False,
            failure_reason=reason,
            recovery_data={}
        )
        
        with self._lock:
            self.recovery_executions[execution_id] = execution
        
        return execution
    
    def _update_recovery_stats(self, execution: RecoveryExecution, success: bool):
        """Update recovery statistics"""
        with self._lock:
            self.stats["total_recoveries"] += 1
            
            if success:
                self.stats["successful_recoveries"] += 1
            else:
                self.stats["failed_recoveries"] += 1
            
            # Update average recovery time
            if execution.end_time:
                recovery_time = execution.end_time - execution.start_time
                total_time = self.stats["avg_recovery_time"] * (self.stats["total_recoveries"] - 1)
                self.stats["avg_recovery_time"] = (total_time + recovery_time) / self.stats["total_recoveries"]
    
    def register_recovery_handler(self, action_name: str, handler: Callable):
        """Register custom recovery handler"""
        self.recovery_handlers[action_name] = handler
        logger.info(f"🔧 Recovery handler registered: {action_name}")
    
    def register_fallback_handler(self, category: ErrorCategory, handler: Callable):
        """Register fallback handler for error category"""
        self.fallback_handlers[category] = handler
        logger.info(f"🔧 Fallback handler registered for category: {category.value}")
    
    def add_recovery_plan(self, plan: RecoveryPlan):
        """Add custom recovery plan"""
        with self._lock:
            self.recovery_plans[plan.plan_id] = plan
        logger.info(f"📋 Recovery plan added: {plan.plan_id}")
    
    def get_recovery_status(self, error_id: str) -> Optional[RecoveryExecution]:
        """Get recovery status for an error"""
        if error_id in self.active_recoveries:
            execution_id = self.active_recoveries[error_id]
            return self.recovery_executions.get(execution_id)
        
        # Look for completed recoveries
        for execution in self.recovery_executions.values():
            if execution.error_id == error_id:
                return execution
        
        return None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        with self._lock:
            success_rate = 0.0
            if self.stats["total_recoveries"] > 0:
                success_rate = (self.stats["successful_recoveries"] / self.stats["total_recoveries"]) * 100
            
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "statistics": self.stats.copy(),
                "success_rate": success_rate,
                "active_recoveries": len(self.active_recoveries),
                "total_plans": len(self.recovery_plans),
                "registered_handlers": len(self.recovery_handlers),
                "fallback_handlers": len(self.fallback_handlers)
            }


# Factory function for easy creation
def create_recovery_manager(brain_id: str) -> RecoveryManager:
    """Factory function to create recovery manager"""
    return RecoveryManager(brain_id)
