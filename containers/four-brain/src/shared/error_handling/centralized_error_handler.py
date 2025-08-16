"""
Centralized Error Handler - Unified Error Handling System
Provides centralized error handling and response management for all Four-Brain components

This module creates a unified error handling system that replaces scattered error
handling with a centralized, consistent, and comprehensive error management solution.

Created: 2025-07-29 AEST
Purpose: Centralized error handling for all Four-Brain components
Module Size: 150 lines (modular design)
"""

import time
import logging
import traceback
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    COMMUNICATION = "communication"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    PROCESSING = "processing"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery action types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    RESTART = "restart"
    DEGRADE = "degrade"


@dataclass
class ErrorContext:
    """Error context information"""
    component: str
    operation: str
    brain_id: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ErrorRecord:
    """Comprehensive error record"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    context: ErrorContext
    stack_trace: Optional[str]
    recovery_action: Optional[RecoveryAction]
    resolved: bool
    resolution_time: Optional[float]
    occurrence_count: int


class CentralizedErrorHandler:
    """
    Centralized Error Handler
    
    Provides unified error handling, classification, and response management
    for all Four-Brain components with consistent error reporting and recovery.
    """
    
    def __init__(self, brain_id: str):
        """Initialize centralized error handler"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Error storage
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_history: List[str] = []  # Error IDs in chronological order
        self.max_history_size = 10000
        
        # Error classification patterns
        self.classification_patterns = {
            ErrorCategory.COMMUNICATION: [
                r"connection.*timeout", r"connection.*refused", r"network.*error",
                r"timeout.*error", r"socket.*error", r"http.*error"
            ],
            ErrorCategory.DATABASE: [
                r"database.*error", r"sql.*error", r"connection.*pool",
                r"postgresql.*error", r"supabase.*error", r"query.*failed"
            ],
            ErrorCategory.AUTHENTICATION: [
                r"authentication.*failed", r"auth.*error", r"permission.*denied",
                r"access.*denied", r"unauthorized", r"forbidden"
            ],
            ErrorCategory.PROCESSING: [
                r"processing.*failed", r"model.*error", r"inference.*error",
                r"embedding.*error", r"ai.*error", r"brain.*error"
            ],
            ErrorCategory.RESOURCE: [
                r"out of memory", r"memory.*error", r"disk.*full",
                r"resource.*exhausted", r"quota.*exceeded", r"oom"
            ],
            ErrorCategory.CONFIGURATION: [
                r"config.*error", r"configuration.*invalid", r"setting.*error",
                r"environment.*error", r"missing.*config"
            ]
        }
        
        # Error callbacks
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.severity_callbacks: Dict[ErrorSeverity, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "errors_by_category": {cat.value: 0 for cat in ErrorCategory},
            "errors_by_severity": {sev.value: 0 for sev in ErrorSeverity},
            "resolved_errors": 0,
            "avg_resolution_time": 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🛡️ Centralized Error Handler initialized for {brain_id}")
    
    def handle_error(self, 
                    exception: Union[Exception, str],
                    context: ErrorContext,
                    severity: Optional[ErrorSeverity] = None,
                    category: Optional[ErrorCategory] = None,
                    recovery_action: Optional[RecoveryAction] = None) -> ErrorRecord:
        """Handle an error with comprehensive processing"""
        
        # Generate unique error ID
        error_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Extract error information
        if isinstance(exception, Exception):
            message = str(exception)
            exception_type = exception.__class__.__name__
            stack_trace = traceback.format_exc()
        else:
            message = str(exception)
            exception_type = "StringError"
            stack_trace = None
        
        # Auto-classify if not provided
        if category is None:
            category = self._classify_error(message, exception_type)
        
        if severity is None:
            severity = self._determine_severity(category, message, exception_type)
        
        if recovery_action is None:
            recovery_action = self._determine_recovery_action(category, severity)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=timestamp,
            severity=severity,
            category=category,
            message=message,
            exception_type=exception_type,
            context=context,
            stack_trace=stack_trace,
            recovery_action=recovery_action,
            resolved=False,
            resolution_time=None,
            occurrence_count=1
        )
        
        # Check for duplicate errors
        duplicate_id = self._find_duplicate_error(error_record)
        if duplicate_id:
            # Update existing error
            existing_error = self.error_records[duplicate_id]
            existing_error.occurrence_count += 1
            existing_error.timestamp = timestamp  # Update to latest occurrence
            error_record = existing_error
        else:
            # Store new error
            with self._lock:
                self.error_records[error_id] = error_record
                self.error_history.append(error_id)
                
                # Maintain history size limit
                if len(self.error_history) > self.max_history_size:
                    old_error_id = self.error_history.pop(0)
                    if old_error_id in self.error_records:
                        del self.error_records[old_error_id]
        
        # Update statistics
        self._update_statistics(error_record)
        
        # Log error appropriately
        self._log_error(error_record)
        
        # Execute callbacks
        self._execute_callbacks(error_record)
        
        return error_record
    
    def _classify_error(self, message: str, exception_type: str) -> ErrorCategory:
        """Automatically classify error based on message and type"""
        message_lower = message.lower()
        
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                import re
                if re.search(pattern, message_lower):
                    return category
        
        # Classification by exception type
        if "Connection" in exception_type or "Network" in exception_type:
            return ErrorCategory.COMMUNICATION
        elif "SQL" in exception_type or "Database" in exception_type:
            return ErrorCategory.DATABASE
        elif "Auth" in exception_type or "Permission" in exception_type:
            return ErrorCategory.AUTHENTICATION
        elif "Memory" in exception_type or "Resource" in exception_type:
            return ErrorCategory.RESOURCE
        elif "Config" in exception_type:
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, category: ErrorCategory, message: str, exception_type: str) -> ErrorSeverity:
        """Determine error severity based on category and content"""
        message_lower = message.lower()
        
        # Critical severity indicators
        if any(word in message_lower for word in ["critical", "fatal", "crash", "shutdown", "corrupt"]):
            return ErrorSeverity.CRITICAL
        
        # High severity by category
        if category in [ErrorCategory.DATABASE, ErrorCategory.AUTHENTICATION, ErrorCategory.SYSTEM]:
            return ErrorSeverity.HIGH
        
        # High severity by exception type
        if exception_type in ["SystemExit", "KeyboardInterrupt", "MemoryError", "OSError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity indicators
        if any(word in message_lower for word in ["error", "failed", "exception", "timeout"]):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _determine_recovery_action(self, category: ErrorCategory, severity: ErrorSeverity) -> RecoveryAction:
        """Determine appropriate recovery action"""
        
        # Critical errors need escalation
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ESCALATE
        
        # Category-specific recovery actions
        if category == ErrorCategory.COMMUNICATION:
            return RecoveryAction.RETRY
        elif category == ErrorCategory.DATABASE:
            return RecoveryAction.RETRY
        elif category == ErrorCategory.AUTHENTICATION:
            return RecoveryAction.ESCALATE
        elif category == ErrorCategory.PROCESSING:
            return RecoveryAction.FALLBACK
        elif category == ErrorCategory.RESOURCE:
            return RecoveryAction.DEGRADE
        elif category == ErrorCategory.NETWORK:
            return RecoveryAction.CIRCUIT_BREAK
        
        return RecoveryAction.RETRY
    
    def _find_duplicate_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Find if this error is a duplicate of a recent error"""
        cutoff_time = time.time() - 300  # 5 minutes
        
        with self._lock:
            for existing_id, existing_error in self.error_records.items():
                if (existing_error.timestamp >= cutoff_time and
                    existing_error.message == error_record.message and
                    existing_error.exception_type == error_record.exception_type and
                    existing_error.context.component == error_record.context.component):
                    return existing_id
        
        return None
    
    def _update_statistics(self, error_record: ErrorRecord):
        """Update error statistics"""
        with self._lock:
            if error_record.occurrence_count == 1:  # Only count new errors
                self.stats["total_errors"] += 1
                self.stats["errors_by_category"][error_record.category.value] += 1
                self.stats["errors_by_severity"][error_record.severity.value] += 1
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = f"[{error_record.category.value.upper()}] {error_record.message}"
        
        if error_record.occurrence_count > 1:
            log_message += f" (occurred {error_record.occurrence_count} times)"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _execute_callbacks(self, error_record: ErrorRecord):
        """Execute registered callbacks for error"""
        # Category-specific callbacks
        if error_record.category in self.error_callbacks:
            for callback in self.error_callbacks[error_record.category]:
                try:
                    callback(error_record)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
        
        # Severity-specific callbacks
        if error_record.severity in self.severity_callbacks:
            for callback in self.severity_callbacks[error_record.severity]:
                try:
                    callback(error_record)
                except Exception as e:
                    logger.error(f"Severity callback failed: {e}")
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable[[ErrorRecord], None]):
        """Register callback for specific error category"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
    
    def register_severity_callback(self, severity: ErrorSeverity, callback: Callable[[ErrorRecord], None]):
        """Register callback for specific error severity"""
        if severity not in self.severity_callbacks:
            self.severity_callbacks[severity] = []
        self.severity_callbacks[severity].append(callback)
    
    def resolve_error(self, error_id: str, resolution_notes: str = None):
        """Mark error as resolved"""
        if error_id in self.error_records:
            error_record = self.error_records[error_id]
            error_record.resolved = True
            error_record.resolution_time = time.time()
            
            with self._lock:
                self.stats["resolved_errors"] += 1
                
                # Update average resolution time
                total_resolution_time = sum(
                    (err.resolution_time - err.timestamp) 
                    for err in self.error_records.values() 
                    if err.resolved and err.resolution_time
                )
                self.stats["avg_resolution_time"] = total_resolution_time / max(self.stats["resolved_errors"], 1)
            
            logger.info(f"✅ Error {error_id} resolved: {resolution_notes}")
    
    def get_recent_errors(self, limit: int = 50, severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None) -> List[ErrorRecord]:
        """Get recent errors with optional filtering"""
        with self._lock:
            recent_error_ids = self.error_history[-limit:]
            errors = [self.error_records[eid] for eid in recent_error_ids if eid in self.error_records]
        
        # Apply filters
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        if category:
            errors = [e for e in errors if e.category == category]
        
        return sorted(errors, key=lambda x: x.timestamp, reverse=True)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "statistics": self.stats.copy(),
                "active_errors": len([e for e in self.error_records.values() if not e.resolved]),
                "total_error_records": len(self.error_records),
                "callbacks_registered": {
                    "category_callbacks": len(self.error_callbacks),
                    "severity_callbacks": len(self.severity_callbacks)
                }
            }


# Factory function for easy creation
def create_centralized_error_handler(brain_id: str) -> CentralizedErrorHandler:
    """Factory function to create centralized error handler"""
    return CentralizedErrorHandler(brain_id)
