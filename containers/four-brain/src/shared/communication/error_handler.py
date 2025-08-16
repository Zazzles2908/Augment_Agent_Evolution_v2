"""
Communication Error Handler Module - Robust Error Handling
Handles communication failures gracefully across all brains

This module provides comprehensive error handling for inter-brain
communication, ensuring system resilience and proper error recovery.

Created: 2025-07-29 AEST
Purpose: Handle communication errors gracefully
Module Size: 150 lines (modular design)
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for communication issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of communication errors"""
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"
    MESSAGE_INVALID = "message_invalid"
    SERIALIZATION_ERROR = "serialization_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TARGET_UNAVAILABLE = "target_unavailable"
    REDIS_ERROR = "redis_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class CommunicationError:
    """Structured communication error information"""
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    brain_id: str
    target_brain: Optional[str]
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False


class CommunicationErrorHandler:
    """
    Communication Error Handler
    
    Provides comprehensive error handling, logging, and recovery
    mechanisms for inter-brain communication failures.
    """
    
    def __init__(self, brain_id: str):
        """Initialize error handler for specific brain"""
        self.brain_id = brain_id
        self.error_history = []
        self.max_error_history = 1000
        self.error_callbacks = {}
        
        # Error statistics
        self.total_errors = 0
        self.errors_by_type = {}
        self.errors_by_severity = {}
        self.last_error_time = None
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_window = 60  # seconds
        self.circuit_breaker_states = {}  # target_brain -> state
        
        logger.info(f"🛡️ Communication Error Handler initialized for {brain_id}")
    
    async def handle_error(self, error_type: ErrorType, message: str, 
                          target_brain: Optional[str] = None, 
                          context: Dict[str, Any] = None,
                          exception: Optional[Exception] = None) -> CommunicationError:
        """Handle a communication error with appropriate response"""
        
        # Determine severity
        severity = self._determine_severity(error_type, context)
        
        # Create error record
        error_id = f"err_{self.brain_id}_{int(time.time() * 1000)}"
        error_record = CommunicationError(
            error_id=error_id,
            error_type=error_type,
            severity=severity,
            message=message,
            brain_id=self.brain_id,
            target_brain=target_brain,
            timestamp=time.time(),
            context=context or {},
            stack_trace=traceback.format_exc() if exception else None
        )
        
        # Update statistics
        self._update_error_stats(error_record)
        
        # Store in history
        self._store_error(error_record)
        
        # Log error appropriately
        self._log_error(error_record)
        
        # Handle circuit breaker
        if target_brain:
            self._update_circuit_breaker(target_brain, error_record)
        
        # Execute error callbacks
        await self._execute_error_callbacks(error_record)
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error_record)
        if recovery_action:
            await self._execute_recovery_action(recovery_action, error_record)
        
        return error_record
    
    def _determine_severity(self, error_type: ErrorType, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on type and context"""
        
        # Critical errors
        if error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.REDIS_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in [ErrorType.CONNECTION_FAILED, ErrorType.TARGET_UNAVAILABLE]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in [ErrorType.TIMEOUT, ErrorType.RATE_LIMIT_EXCEEDED]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if error_type in [ErrorType.MESSAGE_INVALID, ErrorType.SERIALIZATION_ERROR]:
            return ErrorSeverity.LOW
        
        # Default to medium for unknown errors
        return ErrorSeverity.MEDIUM
    
    def _update_error_stats(self, error: CommunicationError):
        """Update error statistics"""
        self.total_errors += 1
        self.last_error_time = error.timestamp
        
        # Update by type
        error_type_str = error.error_type.value
        self.errors_by_type[error_type_str] = self.errors_by_type.get(error_type_str, 0) + 1
        
        # Update by severity
        severity_str = error.severity.value
        self.errors_by_severity[severity_str] = self.errors_by_severity.get(severity_str, 0) + 1
    
    def _store_error(self, error: CommunicationError):
        """Store error in history with size limit"""
        self.error_history.append(error)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _log_error(self, error: CommunicationError):
        """Log error with appropriate level"""
        log_message = f"Communication Error [{error.error_id}]: {error.message}"
        
        if error.target_brain:
            log_message += f" (Target: {error.target_brain})"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log stack trace for critical errors
        if error.severity == ErrorSeverity.CRITICAL and error.stack_trace:
            logger.critical(f"Stack trace for {error.error_id}:\n{error.stack_trace}")
    
    def _update_circuit_breaker(self, target_brain: str, error: CommunicationError):
        """Update circuit breaker state for target brain"""
        current_time = time.time()
        
        # Initialize circuit breaker state if needed
        if target_brain not in self.circuit_breaker_states:
            self.circuit_breaker_states[target_brain] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure_time": None,
                "next_attempt_time": None
            }
        
        cb_state = self.circuit_breaker_states[target_brain]
        
        # Update failure count and time
        cb_state["failure_count"] += 1
        cb_state["last_failure_time"] = current_time
        
        # Check if we should open the circuit breaker
        if (cb_state["failure_count"] >= self.circuit_breaker_threshold and 
            cb_state["state"] == "closed"):
            
            cb_state["state"] = "open"
            cb_state["next_attempt_time"] = current_time + self.circuit_breaker_window
            
            logger.warning(f"🔴 Circuit breaker OPENED for {target_brain} after {cb_state['failure_count']} failures")
    
    def is_circuit_breaker_open(self, target_brain: str) -> bool:
        """Check if circuit breaker is open for target brain"""
        if target_brain not in self.circuit_breaker_states:
            return False
        
        cb_state = self.circuit_breaker_states[target_brain]
        current_time = time.time()
        
        if cb_state["state"] == "open":
            # Check if we should transition to half-open
            if current_time >= cb_state.get("next_attempt_time", 0):
                cb_state["state"] = "half_open"
                logger.info(f"🟡 Circuit breaker HALF-OPEN for {target_brain}")
                return False
            return True
        
        return False
    
    def record_success(self, target_brain: str):
        """Record successful communication to reset circuit breaker"""
        if target_brain in self.circuit_breaker_states:
            cb_state = self.circuit_breaker_states[target_brain]
            
            if cb_state["state"] in ["half_open", "open"]:
                cb_state["state"] = "closed"
                cb_state["failure_count"] = 0
                cb_state["last_failure_time"] = None
                cb_state["next_attempt_time"] = None
                
                logger.info(f"🟢 Circuit breaker CLOSED for {target_brain} - communication restored")
    
    async def _execute_error_callbacks(self, error: CommunicationError):
        """Execute registered error callbacks"""
        error_type_str = error.error_type.value
        
        if error_type_str in self.error_callbacks:
            try:
                callback = self.error_callbacks[error_type_str]
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"❌ Error callback failed: {e}")
    
    def _determine_recovery_action(self, error: CommunicationError) -> Optional[str]:
        """Determine appropriate recovery action"""
        if error.error_type == ErrorType.CONNECTION_FAILED:
            return "reconnect"
        elif error.error_type == ErrorType.TIMEOUT:
            return "retry_with_backoff"
        elif error.error_type == ErrorType.RATE_LIMIT_EXCEEDED:
            return "delay_and_retry"
        elif error.error_type == ErrorType.MESSAGE_INVALID:
            return "validate_and_reformat"
        
        return None
    
    async def _execute_recovery_action(self, action: str, error: CommunicationError):
        """Execute recovery action"""
        logger.info(f"🔧 Executing recovery action: {action} for error {error.error_id}")
        
        if action == "reconnect":
            # Signal need for reconnection
            error.context["recovery_action"] = "reconnect_required"
        elif action == "retry_with_backoff":
            # Calculate backoff delay
            delay = min(2 ** error.retry_count, 60)  # Exponential backoff, max 60s
            error.context["retry_delay"] = delay
        elif action == "delay_and_retry":
            # Fixed delay for rate limiting
            error.context["retry_delay"] = 30
        elif action == "validate_and_reformat":
            # Mark for message validation
            error.context["needs_validation"] = True
    
    def register_error_callback(self, error_type: ErrorType, callback: Callable):
        """Register callback for specific error type"""
        self.error_callbacks[error_type.value] = callback
        logger.info(f"📝 Registered error callback for {error_type.value}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "brain_id": self.brain_id,
            "total_errors": self.total_errors,
            "errors_by_type": self.errors_by_type,
            "errors_by_severity": self.errors_by_severity,
            "last_error_time": self.last_error_time,
            "circuit_breaker_states": self.circuit_breaker_states,
            "error_history_size": len(self.error_history)
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[CommunicationError]:
        """Get recent errors"""
        return self.error_history[-limit:] if self.error_history else []
    
    def clear_error_history(self):
        """Clear error history (for maintenance)"""
        self.error_history.clear()
        logger.info(f"🧹 Error history cleared for {self.brain_id}")


# Factory function for easy creation
def create_error_handler(brain_id: str) -> CommunicationErrorHandler:
    """Factory function to create communication error handler"""
    return CommunicationErrorHandler(brain_id)
