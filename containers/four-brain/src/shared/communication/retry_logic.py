"""
Retry Logic Module - Intelligent Retry Mechanisms
Provides automatic retry with exponential backoff and jitter

This module implements sophisticated retry logic for inter-brain
communication, ensuring reliable message delivery with smart backoff.

Created: 2025-07-29 AEST
Purpose: Implement intelligent retry mechanisms
Module Size: 150 lines (modular design)
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class RetryResult(Enum):
    """Retry operation results"""
    SUCCESS = "success"
    FAILED_RETRIES_EXHAUSTED = "failed_retries_exhausted"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_CIRCUIT_BREAKER = "failed_circuit_breaker"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    timeout: Optional[float] = None  # total timeout for all retries
    backoff_multiplier: float = 2.0


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay_before_attempt: float
    timestamp: float
    success: bool
    error: Optional[str] = None
    duration: Optional[float] = None


class RetryManager:
    """
    Intelligent Retry Manager
    
    Provides sophisticated retry mechanisms with various backoff strategies,
    jitter, circuit breaker integration, and comprehensive tracking.
    """
    
    def __init__(self, brain_id: str, default_config: Optional[RetryConfig] = None):
        """Initialize retry manager"""
        self.brain_id = brain_id
        self.default_config = default_config or RetryConfig()
        
        # Retry tracking
        self.retry_history = {}  # operation_id -> List[RetryAttempt]
        self.active_retries = {}  # operation_id -> retry_info
        self.total_retries = 0
        self.successful_retries = 0
        
        # Fibonacci sequence cache for fibonacci backoff
        self._fibonacci_cache = [1, 1]
        
        logger.info(f"🔄 Retry Manager initialized for {brain_id}")
    
    async def retry_operation(self, operation: Callable[[], Awaitable[Any]], 
                             operation_id: str, config: Optional[RetryConfig] = None,
                             circuit_breaker_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """
        Execute operation with intelligent retry logic
        
        Returns a dict with 'result', 'success', 'attempts', and 'total_time'
        """
        retry_config = config or self.default_config
        start_time = time.time()
        attempts = []
        
        # Initialize retry tracking
        self.active_retries[operation_id] = {
            "start_time": start_time,
            "config": retry_config,
            "attempts": 0
        }
        
        for attempt in range(retry_config.max_retries + 1):  # +1 for initial attempt
            attempt_start = time.time()
            
            # Check circuit breaker before attempt
            if circuit_breaker_check and circuit_breaker_check():
                logger.warning(f"🔴 Circuit breaker open for {operation_id}, aborting retry")
                return self._create_retry_result(
                    RetryResult.FAILED_CIRCUIT_BREAKER, None, attempts, start_time
                )
            
            # Check total timeout
            if retry_config.timeout and (time.time() - start_time) > retry_config.timeout:
                logger.warning(f"⏰ Total timeout exceeded for {operation_id}")
                return self._create_retry_result(
                    RetryResult.FAILED_TIMEOUT, None, attempts, start_time
                )
            
            try:
                # Calculate delay for this attempt (0 for first attempt)
                delay = self._calculate_delay(attempt, retry_config) if attempt > 0 else 0
                
                # Apply delay if needed
                if delay > 0:
                    logger.info(f"⏳ Retrying {operation_id} in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                # Execute the operation
                logger.debug(f"🔄 Executing {operation_id} (attempt {attempt + 1})")
                result = await operation()
                
                # Success!
                attempt_duration = time.time() - attempt_start
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    delay_before_attempt=delay,
                    timestamp=attempt_start,
                    success=True,
                    duration=attempt_duration
                )
                attempts.append(attempt_info)
                
                # Update statistics
                self.total_retries += attempt
                if attempt > 0:
                    self.successful_retries += 1
                
                # Store in history
                self._store_retry_history(operation_id, attempts)
                
                # Clean up active retry tracking
                del self.active_retries[operation_id]
                
                logger.info(f"✅ Operation {operation_id} succeeded after {attempt + 1} attempts")
                return self._create_retry_result(RetryResult.SUCCESS, result, attempts, start_time)
                
            except Exception as e:
                attempt_duration = time.time() - attempt_start
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    delay_before_attempt=delay if attempt > 0 else 0,
                    timestamp=attempt_start,
                    success=False,
                    error=str(e),
                    duration=attempt_duration
                )
                attempts.append(attempt_info)
                
                logger.warning(f"❌ Attempt {attempt + 1} failed for {operation_id}: {e}")
                
                # If this was the last attempt, fail
                if attempt >= retry_config.max_retries:
                    self.total_retries += attempt + 1
                    self._store_retry_history(operation_id, attempts)
                    del self.active_retries[operation_id]
                    
                    logger.error(f"💥 Operation {operation_id} failed after {attempt + 1} attempts")
                    return self._create_retry_result(
                        RetryResult.FAILED_RETRIES_EXHAUSTED, None, attempts, start_time
                    )
        
        # Should never reach here, but just in case
        return self._create_retry_result(
            RetryResult.FAILED_RETRIES_EXHAUSTED, None, attempts, start_time
        )
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on strategy"""
        if attempt <= 0:
            return 0
        
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = config.base_delay * self._get_fibonacci(attempt)
        else:
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached for performance)"""
        while len(self._fibonacci_cache) <= n:
            next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            self._fibonacci_cache.append(next_fib)
        
        return self._fibonacci_cache[n]
    
    def _create_retry_result(self, result_type: RetryResult, data: Any, 
                           attempts: list, start_time: float) -> Dict[str, Any]:
        """Create standardized retry result"""
        total_time = time.time() - start_time
        
        return {
            "result": data,
            "success": result_type == RetryResult.SUCCESS,
            "result_type": result_type.value,
            "attempts": len(attempts),
            "total_time": total_time,
            "attempt_details": [
                {
                    "attempt": attempt.attempt_number,
                    "delay": attempt.delay_before_attempt,
                    "success": attempt.success,
                    "error": attempt.error,
                    "duration": attempt.duration
                }
                for attempt in attempts
            ]
        }
    
    def _store_retry_history(self, operation_id: str, attempts: list):
        """Store retry history for analysis"""
        self.retry_history[operation_id] = attempts
        
        # Limit history size to prevent memory issues
        if len(self.retry_history) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.retry_history.keys())[:100]
            for key in oldest_keys:
                del self.retry_history[key]
    
    async def retry_with_circuit_breaker(self, operation: Callable[[], Awaitable[Any]], 
                                       operation_id: str, target_brain: str,
                                       error_handler, config: Optional[RetryConfig] = None) -> Dict[str, Any]:
        """Retry operation with circuit breaker integration"""
        
        def circuit_breaker_check():
            return error_handler.is_circuit_breaker_open(target_brain)
        
        result = await self.retry_operation(operation, operation_id, config, circuit_breaker_check)
        
        # Update circuit breaker state based on result
        if result["success"]:
            error_handler.record_success(target_brain)
        
        return result
    
    def create_retry_config(self, max_retries: int = 3, base_delay: float = 1.0,
                           strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                           max_delay: float = 60.0, jitter: bool = True,
                           timeout: Optional[float] = None) -> RetryConfig:
        """Create custom retry configuration"""
        return RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            jitter=jitter,
            timeout=timeout
        )
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics"""
        active_count = len(self.active_retries)
        history_count = len(self.retry_history)
        
        # Calculate success rate
        success_rate = (
            self.successful_retries / max(self.total_retries, 1) * 100
            if self.total_retries > 0 else 100
        )
        
        return {
            "brain_id": self.brain_id,
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "success_rate_percent": round(success_rate, 2),
            "active_retries": active_count,
            "history_entries": history_count,
            "default_config": {
                "max_retries": self.default_config.max_retries,
                "base_delay": self.default_config.base_delay,
                "strategy": self.default_config.strategy.value
            }
        }
    
    def get_operation_history(self, operation_id: str) -> Optional[List[RetryAttempt]]:
        """Get retry history for specific operation"""
        return self.retry_history.get(operation_id)
    
    def clear_history(self):
        """Clear retry history (for maintenance)"""
        self.retry_history.clear()
        logger.info(f"🧹 Retry history cleared for {self.brain_id}")


# Factory function for easy creation
def create_retry_manager(brain_id: str, config: Optional[RetryConfig] = None) -> RetryManager:
    """Factory function to create retry manager"""
    return RetryManager(brain_id, config)


# Convenience function for simple retries
async def simple_retry(operation: Callable[[], Awaitable[Any]], max_retries: int = 3,
                      base_delay: float = 1.0) -> Any:
    """Simple retry function for basic use cases"""
    config = RetryConfig(max_retries=max_retries, base_delay=base_delay)
    manager = RetryManager("simple", config)
    result = await manager.retry_operation(operation, "simple_operation", config)
    
    if result["success"]:
        return result["result"]
    else:
        raise Exception(f"Operation failed after {result['attempts']} attempts")
