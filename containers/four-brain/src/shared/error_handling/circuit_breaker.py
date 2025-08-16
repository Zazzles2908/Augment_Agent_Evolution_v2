"""
Circuit Breaker - Cascade Failure Prevention
Provides circuit breaker pattern implementation to prevent cascade failures

This module implements the circuit breaker pattern to protect the Four-Brain
system from cascade failures by monitoring service health and automatically
opening circuits when failures exceed thresholds.

Created: 2025-07-29 AEST
Purpose: Cascade failure prevention and system protection
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

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3,
                 timeout: float = 30.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    circuit_opens: int
    circuit_closes: int
    current_state: CircuitState
    last_failure_time: Optional[float]
    last_success_time: Optional[float]


class CircuitBreakerError(Exception):
    """Circuit breaker specific exception"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker Implementation
    
    Implements the circuit breaker pattern to prevent cascade failures
    by monitoring service health and failing fast when thresholds are exceeded.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """Initialize circuit breaker"""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.next_attempt_time = None
        
        # Statistics
        self.stats = CircuitBreakerStats(
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            circuit_opens=0,
            circuit_closes=0,
            current_state=self.state,
            last_failure_time=None,
            last_success_time=None
        )
        
        # Callbacks
        self.on_open_callbacks: List[Callable] = []
        self.on_close_callbacks: List[Callable] = []
        self.on_half_open_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🔌 Circuit Breaker initialized: {name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection (sync)"""
        with self._lock:
            self.stats.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.failed_calls += 1
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
            
            # Execute function
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Check for timeout
                if execution_time > self.config.timeout:
                    raise TimeoutError(f"Function execution exceeded timeout: {execution_time:.2f}s")
                
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        with self._lock:
            self.stats.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.failed_calls += 1
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        # Execute function
        try:
            start_time = time.time()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            execution_time = time.time() - start_time
            self._on_success()
            return result
            
        except (self.config.expected_exception, asyncio.TimeoutError) as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        if self.next_attempt_time is None:
            return True
        
        return time.time() >= self.next_attempt_time
    
    def _on_success(self):
        """Handle successful function execution"""
        with self._lock:
            self.stats.successful_calls += 1
            self.last_success_time = time.time()
            self.stats.last_success_time = self.last_success_time
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed function execution"""
        with self._lock:
            self.stats.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.stats.last_failure_time = self.last_failure_time
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.state = CircuitState.OPEN
        self.stats.current_state = self.state
        self.stats.circuit_opens += 1
        self.next_attempt_time = time.time() + self.config.recovery_timeout
        
        logger.warning(f"🔴 Circuit breaker {self.name} OPENED after {self.failure_count} failures")
        
        # Execute callbacks
        for callback in self.on_open_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Circuit breaker open callback failed: {e}")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.stats.current_state = self.state
        self.success_count = 0
        
        logger.info(f"🟡 Circuit breaker {self.name} HALF-OPEN, testing recovery")
        
        # Execute callbacks
        for callback in self.on_half_open_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Circuit breaker half-open callback failed: {e}")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.stats.current_state = self.state
        self.stats.circuit_closes += 1
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        
        logger.info(f"🟢 Circuit breaker {self.name} CLOSED, service recovered")
        
        # Execute callbacks
        for callback in self.on_close_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Circuit breaker close callback failed: {e}")
    
    def force_open(self):
        """Manually force circuit to OPEN state"""
        with self._lock:
            self._transition_to_open()
        logger.warning(f"🔴 Circuit breaker {self.name} manually OPENED")
    
    def force_close(self):
        """Manually force circuit to CLOSED state"""
        with self._lock:
            self._transition_to_closed()
        logger.info(f"🟢 Circuit breaker {self.name} manually CLOSED")
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.next_attempt_time = None
            
            # Reset statistics
            self.stats = CircuitBreakerStats(
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                circuit_opens=0,
                circuit_closes=0,
                current_state=self.state,
                last_failure_time=None,
                last_success_time=None
            )
        
        logger.info(f"🔄 Circuit breaker {self.name} reset to initial state")
    
    def add_on_open_callback(self, callback: Callable):
        """Add callback for circuit open event"""
        self.on_open_callbacks.append(callback)
    
    def add_on_close_callback(self, callback: Callable):
        """Add callback for circuit close event"""
        self.on_close_callbacks.append(callback)
    
    def add_on_half_open_callback(self, callback: Callable):
        """Add callback for circuit half-open event"""
        self.on_half_open_callbacks.append(callback)
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            success_rate = 0.0
            if self.stats.total_calls > 0:
                success_rate = (self.stats.successful_calls / self.stats.total_calls) * 100
            
            return {
                "name": self.name,
                "state": self.state.value,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                },
                "statistics": asdict(self.stats),
                "success_rate": success_rate,
                "current_failure_count": self.failure_count,
                "current_success_count": self.success_count,
                "next_attempt_time": self.next_attempt_time
            }


class CircuitBreakerManager:
    """
    Circuit Breaker Manager
    
    Manages multiple circuit breakers and provides centralized monitoring.
    """
    
    def __init__(self, brain_id: str):
        """Initialize circuit breaker manager"""
        self.brain_id = brain_id
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        
        logger.info(f"🔌 Circuit Breaker Manager initialized for {brain_id}")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a new circuit breaker"""
        with self._lock:
            if name in self.circuit_breakers:
                logger.warning(f"Circuit breaker {name} already exists, returning existing instance")
                return self.circuit_breakers[name]
            
            circuit_breaker = CircuitBreaker(name, config)
            self.circuit_breakers[name] = circuit_breaker
            
            logger.info(f"🔌 Circuit breaker created: {name}")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove circuit breaker"""
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                logger.info(f"🔌 Circuit breaker removed: {name}")
                return True
            return False
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            stats = {}
            for name, cb in self.circuit_breakers.items():
                stats[name] = cb.get_statistics()
            
            # Overall statistics
            total_breakers = len(self.circuit_breakers)
            open_breakers = len([cb for cb in self.circuit_breakers.values() if cb.get_state() == CircuitState.OPEN])
            half_open_breakers = len([cb for cb in self.circuit_breakers.values() if cb.get_state() == CircuitState.HALF_OPEN])
            
            return {
                "brain_id": self.brain_id,
                "total_circuit_breakers": total_breakers,
                "open_circuit_breakers": open_breakers,
                "half_open_circuit_breakers": half_open_breakers,
                "healthy_circuit_breakers": total_breakers - open_breakers - half_open_breakers,
                "circuit_breakers": stats
            }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
        logger.info(f"🔄 All circuit breakers reset for {self.brain_id}")


# Factory functions for easy creation
def create_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Factory function to create circuit breaker"""
    return CircuitBreaker(name, config)


def create_circuit_breaker_manager(brain_id: str) -> CircuitBreakerManager:
    """Factory function to create circuit breaker manager"""
    return CircuitBreakerManager(brain_id)
