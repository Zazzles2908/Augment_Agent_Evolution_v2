"""
Retry Engine - Intelligent Retry Mechanisms
Provides intelligent retry mechanisms with backoff strategies for failed operations

This module implements sophisticated retry logic with various backoff strategies,
jitter, and conditional retry policies for the Four-Brain system.

Created: 2025-07-29 AEST
Purpose: Intelligent retry mechanisms with backoff strategies
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
import random
from typing import Dict, Any, Optional, List, Callable, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategy types"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


class RetryCondition(Enum):
    """Retry condition types"""
    ALWAYS = "always"
    ON_EXCEPTION = "on_exception"
    ON_RESULT = "on_result"
    CONDITIONAL = "conditional"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retry_on_exceptions: List[Type[Exception]] = None
    stop_on_exceptions: List[Type[Exception]] = None
    retry_condition: RetryCondition = RetryCondition.ON_EXCEPTION
    condition_func: Optional[Callable] = None


@dataclass
class RetryAttempt:
    """Retry attempt record"""
    attempt_number: int
    start_time: float
    end_time: Optional[float]
    delay_before: float
    success: bool
    exception: Optional[Exception]
    result: Any


@dataclass
class RetryExecution:
    """Retry execution record"""
    execution_id: str
    function_name: str
    config: RetryConfig
    attempts: List[RetryAttempt]
    total_duration: float
    final_success: bool
    final_result: Any
    final_exception: Optional[Exception]


class RetryEngine:
    """
    Intelligent Retry Engine
    
    Provides sophisticated retry mechanisms with various backoff strategies,
    jitter, and conditional retry policies for robust error handling.
    """
    
    def __init__(self, brain_id: str):
        """Initialize retry engine"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Retry executions
        self.retry_executions: Dict[str, RetryExecution] = {}
        self.max_executions = 1000  # Limit memory usage
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_attempts": 0,
            "avg_attempts_per_execution": 0.0,
            "avg_execution_time": 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🔄 Retry Engine initialized for {brain_id}")
    
    def retry(self, config: RetryConfig = None):
        """Decorator for adding retry logic to functions"""
        if config is None:
            config = RetryConfig()
        
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    return await self.execute_async(func, config, *args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    return self.execute(func, config, *args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def execute(self, func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
        """Execute function with retry logic (sync)"""
        execution_id = f"retry_{self.brain_id}_{int(time.time() * 1000000)}"
        execution_start = time.time()
        
        execution = RetryExecution(
            execution_id=execution_id,
            function_name=func.__name__,
            config=config,
            attempts=[],
            total_duration=0.0,
            final_success=False,
            final_result=None,
            final_exception=None
        )
        
        for attempt_num in range(1, config.max_attempts + 1):
            # Calculate delay before attempt (except first)
            delay_before = 0.0
            if attempt_num > 1:
                delay_before = self._calculate_delay(attempt_num - 1, config)
                time.sleep(delay_before)
            
            # Execute attempt
            attempt_start = time.time()
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                start_time=attempt_start,
                end_time=None,
                delay_before=delay_before,
                success=False,
                exception=None,
                result=None
            )
            
            try:
                result = func(*args, **kwargs)
                attempt.end_time = time.time()
                attempt.success = True
                attempt.result = result
                
                # Check if result meets retry condition
                if self._should_retry_on_result(result, config):
                    execution.attempts.append(attempt)
                    continue
                
                # Success - no retry needed
                execution.attempts.append(attempt)
                execution.final_success = True
                execution.final_result = result
                break
                
            except Exception as e:
                attempt.end_time = time.time()
                attempt.exception = e
                execution.attempts.append(attempt)
                
                # Check if we should retry on this exception
                if not self._should_retry_on_exception(e, config):
                    execution.final_exception = e
                    break
                
                # If this is the last attempt, don't retry
                if attempt_num == config.max_attempts:
                    execution.final_exception = e
                    break
        
        # Finalize execution
        execution.total_duration = time.time() - execution_start
        
        # Store execution
        self._store_execution(execution)
        
        # Update statistics
        self._update_statistics(execution)
        
        # Log execution
        self._log_execution(execution)
        
        # Return result or raise exception
        if execution.final_success:
            return execution.final_result
        else:
            raise execution.final_exception
    
    async def execute_async(self, func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
        """Execute async function with retry logic"""
        execution_id = f"retry_async_{self.brain_id}_{int(time.time() * 1000000)}"
        execution_start = time.time()
        
        execution = RetryExecution(
            execution_id=execution_id,
            function_name=func.__name__,
            config=config,
            attempts=[],
            total_duration=0.0,
            final_success=False,
            final_result=None,
            final_exception=None
        )
        
        for attempt_num in range(1, config.max_attempts + 1):
            # Calculate delay before attempt (except first)
            delay_before = 0.0
            if attempt_num > 1:
                delay_before = self._calculate_delay(attempt_num - 1, config)
                await asyncio.sleep(delay_before)
            
            # Execute attempt
            attempt_start = time.time()
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                start_time=attempt_start,
                end_time=None,
                delay_before=delay_before,
                success=False,
                exception=None,
                result=None
            )
            
            try:
                result = await func(*args, **kwargs)
                attempt.end_time = time.time()
                attempt.success = True
                attempt.result = result
                
                # Check if result meets retry condition
                if self._should_retry_on_result(result, config):
                    execution.attempts.append(attempt)
                    continue
                
                # Success - no retry needed
                execution.attempts.append(attempt)
                execution.final_success = True
                execution.final_result = result
                break
                
            except Exception as e:
                attempt.end_time = time.time()
                attempt.exception = e
                execution.attempts.append(attempt)
                
                # Check if we should retry on this exception
                if not self._should_retry_on_exception(e, config):
                    execution.final_exception = e
                    break
                
                # If this is the last attempt, don't retry
                if attempt_num == config.max_attempts:
                    execution.final_exception = e
                    break
        
        # Finalize execution
        execution.total_duration = time.time() - execution_start
        
        # Store execution
        self._store_execution(execution)
        
        # Update statistics
        self._update_statistics(execution)
        
        # Log execution
        self._log_execution(execution)
        
        # Return result or raise exception
        if execution.final_success:
            return execution.final_result
        else:
            raise execution.final_exception
    
    def _calculate_delay(self, attempt_num: int, config: RetryConfig) -> float:
        """Calculate delay before retry attempt"""
        
        if config.backoff_strategy == BackoffStrategy.FIXED:
            delay = config.base_delay
        
        elif config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = config.base_delay * attempt_num
        
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.backoff_multiplier ** (attempt_num - 1))
        
        elif config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = config.base_delay * self._fibonacci(attempt_num)
        
        else:  # CUSTOM or fallback
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    def _should_retry_on_exception(self, exception: Exception, config: RetryConfig) -> bool:
        """Check if should retry based on exception"""
        
        # Check stop conditions first
        if config.stop_on_exceptions:
            for stop_exception in config.stop_on_exceptions:
                if isinstance(exception, stop_exception):
                    return False
        
        # Check retry conditions
        if config.retry_condition == RetryCondition.ALWAYS:
            return True
        
        elif config.retry_condition == RetryCondition.ON_EXCEPTION:
            if config.retry_on_exceptions:
                for retry_exception in config.retry_on_exceptions:
                    if isinstance(exception, retry_exception):
                        return True
                return False
            else:
                # Retry on any exception if no specific exceptions specified
                return True
        
        elif config.retry_condition == RetryCondition.CONDITIONAL:
            if config.condition_func:
                return config.condition_func(exception=exception)
        
        return False
    
    def _should_retry_on_result(self, result: Any, config: RetryConfig) -> bool:
        """Check if should retry based on result"""
        
        if config.retry_condition == RetryCondition.ON_RESULT:
            if config.condition_func:
                return config.condition_func(result=result)
        
        elif config.retry_condition == RetryCondition.CONDITIONAL:
            if config.condition_func:
                return config.condition_func(result=result)
        
        return False
    
    def _store_execution(self, execution: RetryExecution):
        """Store retry execution"""
        with self._lock:
            self.retry_executions[execution.execution_id] = execution
            
            # Limit memory usage
            if len(self.retry_executions) > self.max_executions:
                # Remove oldest execution
                oldest_id = min(self.retry_executions.keys())
                del self.retry_executions[oldest_id]
    
    def _update_statistics(self, execution: RetryExecution):
        """Update retry statistics"""
        with self._lock:
            self.stats["total_executions"] += 1
            self.stats["total_attempts"] += len(execution.attempts)
            
            if execution.final_success:
                self.stats["successful_executions"] += 1
            else:
                self.stats["failed_executions"] += 1
            
            # Update averages
            self.stats["avg_attempts_per_execution"] = (
                self.stats["total_attempts"] / self.stats["total_executions"]
            )
            
            # Update average execution time
            total_time = (
                self.stats["avg_execution_time"] * (self.stats["total_executions"] - 1) +
                execution.total_duration
            )
            self.stats["avg_execution_time"] = total_time / self.stats["total_executions"]
    
    def _log_execution(self, execution: RetryExecution):
        """Log retry execution"""
        if execution.final_success:
            if len(execution.attempts) > 1:
                logger.info(
                    f"🔄 Retry successful: {execution.function_name} "
                    f"succeeded after {len(execution.attempts)} attempts "
                    f"({execution.total_duration:.2f}s)"
                )
        else:
            logger.warning(
                f"❌ Retry failed: {execution.function_name} "
                f"failed after {len(execution.attempts)} attempts "
                f"({execution.total_duration:.2f}s) - {execution.final_exception}"
            )
    
    def get_execution(self, execution_id: str) -> Optional[RetryExecution]:
        """Get retry execution by ID"""
        return self.retry_executions.get(execution_id)
    
    def get_recent_executions(self, limit: int = 50) -> List[RetryExecution]:
        """Get recent retry executions"""
        with self._lock:
            executions = list(self.retry_executions.values())
            executions.sort(key=lambda x: x.attempts[0].start_time if x.attempts else 0, reverse=True)
            return executions[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry engine statistics"""
        with self._lock:
            success_rate = 0.0
            if self.stats["total_executions"] > 0:
                success_rate = (self.stats["successful_executions"] / self.stats["total_executions"]) * 100
            
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "statistics": self.stats.copy(),
                "success_rate": success_rate,
                "stored_executions": len(self.retry_executions)
            }


# Factory function for easy creation
def create_retry_engine(brain_id: str) -> RetryEngine:
    """Factory function to create retry engine"""
    return RetryEngine(brain_id)


# Convenience functions for common retry patterns
def exponential_backoff(max_attempts: int = 3, base_delay: float = 1.0, 
                       max_delay: float = 60.0) -> RetryConfig:
    """Create exponential backoff retry configuration"""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_multiplier=2.0,
        jitter=True
    )


def linear_backoff(max_attempts: int = 3, base_delay: float = 1.0, 
                  max_delay: float = 30.0) -> RetryConfig:
    """Create linear backoff retry configuration"""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_strategy=BackoffStrategy.LINEAR,
        jitter=True
    )


def fixed_delay(max_attempts: int = 3, delay: float = 1.0) -> RetryConfig:
    """Create fixed delay retry configuration"""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        max_delay=delay,
        backoff_strategy=BackoffStrategy.FIXED,
        jitter=False
    )
