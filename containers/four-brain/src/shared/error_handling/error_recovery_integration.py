"""
Error Recovery Integration - Unified Error Handling System
Integrates all error handling components into a cohesive system

This module provides a unified interface to all error handling components,
creating a comprehensive error handling and recovery system for the Four-Brain architecture.

Created: 2025-07-29 AEST
Purpose: Unify all error handling components into integrated system
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
import threading

from .centralized_error_handler import (
    CentralizedErrorHandler, ErrorSeverity, ErrorCategory, RecoveryAction, 
    ErrorContext, ErrorRecord, create_centralized_error_handler
)
from .recovery_manager import RecoveryManager, create_recovery_manager
from .circuit_breaker import CircuitBreakerManager, create_circuit_breaker_manager
from .retry_engine import RetryEngine, RetryConfig, create_retry_engine
from .fallback_manager import FallbackManager, create_fallback_manager
from .health_monitor import HealthMonitor, HealthStatus, create_health_monitor

logger = logging.getLogger(__name__)


class ErrorRecoveryIntegration:
    """
    Unified Error Recovery Integration System
    
    Provides a single interface to all error handling components, creating
    a comprehensive error handling and recovery system for the Four-Brain architecture.
    """
    
    def __init__(self, brain_id: str):
        """Initialize error recovery integration"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Initialize all error handling components
        self.error_handler = create_centralized_error_handler(brain_id)
        self.recovery_manager = create_recovery_manager(brain_id)
        self.circuit_breaker_manager = create_circuit_breaker_manager(brain_id)
        self.retry_engine = create_retry_engine(brain_id)
        self.fallback_manager = create_fallback_manager(brain_id)
        self.health_monitor = create_health_monitor(brain_id)
        
        # Integration state
        self.initialized = False
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🛡️ Error Recovery Integration initialized for {brain_id}")
    
    async def initialize(self) -> bool:
        """Initialize the integrated error handling system"""
        try:
            with self._lock:
                if self.initialized:
                    return True
                
                # Start health monitoring
                await self.health_monitor.start_monitoring()
                
                # Register error callbacks
                self._register_error_callbacks()
                
                # Register health callbacks
                self._register_health_callbacks()
                
                self.initialized = True
                
                # Log initialization event
                self.error_handler.handle_error(
                    "Error Recovery Integration initialized successfully",
                    ErrorContext(
                        component="error_recovery_integration",
                        operation="initialization",
                        brain_id=self.brain_id
                    ),
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.SYSTEM
                )
                
                logger.info(f"✅ Error Recovery Integration initialized successfully for {self.brain_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error Recovery Integration initialization failed: {e}")
            return False
    
    def _register_error_callbacks(self):
        """Register error callbacks for integration"""
        
        # Register recovery callback for high severity errors
        def trigger_recovery(error_record: ErrorRecord):
            if error_record.recovery_action in [RecoveryAction.RETRY, RecoveryAction.FALLBACK]:
                asyncio.create_task(self.recovery_manager.execute_recovery(error_record))
        
        self.error_handler.register_severity_callback(ErrorSeverity.HIGH, trigger_recovery)
        self.error_handler.register_severity_callback(ErrorSeverity.CRITICAL, trigger_recovery)
        
        # Register circuit breaker callback for communication errors
        def trigger_circuit_breaker(error_record: ErrorRecord):
            if error_record.category == ErrorCategory.COMMUNICATION:
                cb = self.circuit_breaker_manager.get_circuit_breaker(error_record.context.component)
                if cb:
                    cb.force_open()
        
        self.error_handler.register_category_callback(ErrorCategory.COMMUNICATION, trigger_circuit_breaker)
    
    def _register_health_callbacks(self):
        """Register health monitoring callbacks"""
        
        def on_critical_health(health_result):
            # Trigger recovery for critical health issues
            error_record = self.error_handler.handle_error(
                f"Critical health issue: {health_result.message}",
                ErrorContext(
                    component=health_result.component_name,
                    operation="health_check",
                    brain_id=self.brain_id
                ),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.RESOURCE,
                recovery_action=RecoveryAction.DEGRADE
            )
            
            asyncio.create_task(self.recovery_manager.execute_recovery(error_record))
        
        self.health_monitor.add_critical_callback(on_critical_health)
    
    async def handle_error_with_recovery(self, exception: Union[Exception, str], 
                                       context: ErrorContext,
                                       auto_recover: bool = True) -> ErrorRecord:
        """Handle error with automatic recovery"""
        
        # Handle error through centralized handler
        error_record = self.error_handler.handle_error(exception, context)
        
        # Trigger automatic recovery if enabled
        if auto_recover and error_record.recovery_action != RecoveryAction.IGNORE:
            try:
                recovery_execution = await self.recovery_manager.execute_recovery(error_record)
                
                if recovery_execution.success:
                    # Mark error as resolved
                    self.error_handler.resolve_error(
                        error_record.error_id, 
                        f"Automatically recovered using {recovery_execution.plan_id}"
                    )
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed for error {error_record.error_id}: {recovery_error}")
        
        return error_record
    
    def create_resilient_function(self, func: Callable, 
                                retry_config: RetryConfig = None,
                                circuit_breaker_name: str = None,
                                fallback_strategies: List = None) -> Callable:
        """Create a resilient version of a function with all error handling"""
        
        # Apply retry logic
        if retry_config:
            func = self.retry_engine.retry(retry_config)(func)
        
        # Apply circuit breaker
        if circuit_breaker_name:
            cb = self.circuit_breaker_manager.create_circuit_breaker(circuit_breaker_name)
            func = cb(func)
        
        # Register fallback strategies
        if fallback_strategies:
            for strategy in fallback_strategies:
                self.fallback_manager.register_fallback(func.__name__, strategy)
        
        # Wrap with error handling
        if asyncio.iscoroutinefunction(func):
            async def async_resilient_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component=func.__module__ or "unknown",
                        operation=func.__name__,
                        brain_id=self.brain_id
                    )
                    
                    # Try fallback if available
                    try:
                        return await self.fallback_manager.execute_fallback(func, e, *args, **kwargs)
                    except Exception:
                        # Handle error and re-raise
                        await self.handle_error_with_recovery(e, context)
                        raise
            
            return async_resilient_wrapper
        
        else:
            def sync_resilient_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component=func.__module__ or "unknown",
                        operation=func.__name__,
                        brain_id=self.brain_id
                    )
                    
                    # Handle error and re-raise
                    asyncio.create_task(self.handle_error_with_recovery(e, context))
                    raise
            
            return sync_resilient_wrapper
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all error handling components"""
        try:
            error_stats = self.error_handler.get_error_statistics()
            recovery_stats = self.recovery_manager.get_recovery_statistics()
            circuit_stats = self.circuit_breaker_manager.get_all_statistics()
            retry_stats = self.retry_engine.get_statistics()
            fallback_stats = self.fallback_manager.get_fallback_statistics()
            health_stats = self.health_monitor.get_health_statistics()
            
            # Determine overall health
            overall_status = "healthy"
            issues = []
            
            if error_stats["statistics"]["errors_by_severity"]["critical"] > 0:
                overall_status = "critical"
                issues.append(f"Critical errors: {error_stats['statistics']['errors_by_severity']['critical']}")
            
            if health_stats["overall_status"] == "critical":
                overall_status = "critical"
                issues.append("Critical health issues detected")
            elif health_stats["overall_status"] == "warning" and overall_status == "healthy":
                overall_status = "warning"
                issues.append("Health warnings detected")
            
            if circuit_stats["open_circuit_breakers"] > 0:
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append(f"Open circuit breakers: {circuit_stats['open_circuit_breakers']}")
            
            return {
                "brain_id": self.brain_id,
                "overall_status": overall_status,
                "issues": issues,
                "uptime_seconds": time.time() - self.start_time,
                "enabled": self.enabled,
                "components": {
                    "error_handler": {
                        "active_errors": error_stats["active_errors"],
                        "total_errors": error_stats["statistics"]["total_errors"]
                    },
                    "recovery_manager": {
                        "success_rate": recovery_stats["success_rate"],
                        "active_recoveries": recovery_stats["active_recoveries"]
                    },
                    "circuit_breakers": {
                        "total": circuit_stats["total_circuit_breakers"],
                        "open": circuit_stats["open_circuit_breakers"],
                        "healthy": circuit_stats["healthy_circuit_breakers"]
                    },
                    "retry_engine": {
                        "success_rate": retry_stats["success_rate"],
                        "total_executions": retry_stats["statistics"]["total_executions"]
                    },
                    "fallback_manager": {
                        "success_rate": fallback_stats["success_rate"],
                        "configured_functions": fallback_stats["configured_functions"]
                    },
                    "health_monitor": {
                        "overall_status": health_stats["overall_status"],
                        "monitoring_active": health_stats["monitoring_active"],
                        "registered_checks": health_stats["registered_checks"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return {
                "brain_id": self.brain_id,
                "overall_status": "error",
                "error": str(e)
            }
    
    def export_comprehensive_data(self, format: str = "json") -> str:
        """Export comprehensive error handling data"""
        try:
            data = {
                "brain_id": self.brain_id,
                "export_timestamp": time.time(),
                "status": self.get_comprehensive_status(),
                "error_statistics": self.error_handler.get_error_statistics(),
                "recovery_statistics": self.recovery_manager.get_recovery_statistics(),
                "circuit_breaker_statistics": self.circuit_breaker_manager.get_all_statistics(),
                "retry_statistics": self.retry_engine.get_statistics(),
                "fallback_statistics": self.fallback_manager.get_fallback_statistics(),
                "health_statistics": self.health_monitor.get_health_statistics()
            }
            
            if format == "json":
                import json
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"❌ Comprehensive data export failed: {e}")
            return f'{{"error": "{str(e)}"}}'
    
    def enable_error_handling(self):
        """Enable all error handling components"""
        self.enabled = True
        self.error_handler.enabled = True
        self.recovery_manager.enabled = True
        self.retry_engine.enabled = True
        self.fallback_manager.enabled = True
        self.health_monitor.enabled = True
        logger.info(f"🛡️ All error handling enabled for {self.brain_id}")
    
    def disable_error_handling(self):
        """Disable all error handling components"""
        self.enabled = False
        self.error_handler.enabled = False
        self.recovery_manager.enabled = False
        self.retry_engine.enabled = False
        self.fallback_manager.enabled = False
        self.health_monitor.enabled = False
        logger.info(f"🛡️ All error handling disabled for {self.brain_id}")
    
    async def cleanup(self):
        """Clean up error handling resources"""
        try:
            await self.health_monitor.stop_monitoring()
            logger.info(f"🧹 Error Recovery Integration cleaned up for {self.brain_id}")
        except Exception as e:
            logger.error(f"❌ Error handling cleanup failed: {e}")


# Factory function for easy creation
def create_error_recovery_integration(brain_id: str) -> ErrorRecoveryIntegration:
    """Factory function to create error recovery integration"""
    return ErrorRecoveryIntegration(brain_id)


# Global error handling instance
_global_error_handling: Optional[ErrorRecoveryIntegration] = None


def get_global_error_handling(brain_id: str = None) -> ErrorRecoveryIntegration:
    """Get or create global error handling instance"""
    global _global_error_handling
    
    if _global_error_handling is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_error_handling = ErrorRecoveryIntegration(brain_id)
    
    return _global_error_handling
