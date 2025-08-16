"""
Flow Monitor Core - Simplified Monitoring System
Provides essential monitoring capabilities without complex dependencies

This module creates a simplified FlowMonitor that fixes import issues
while providing essential monitoring functionality for the Four-Brain system.

Created: 2025-07-29 AEST
Purpose: Fix FlowMonitor import failures and restore observability
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import json

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class MonitoringEvent(Enum):
    """Types of monitoring events"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    DATABASE_OPERATION = "database_operation"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class FlowEvent:
    """Represents a monitoring event"""
    event_id: str
    event_type: MonitoringEvent
    timestamp: float
    brain_id: str
    details: Dict[str, Any]
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class FlowMonitorCore:
    """
    Core Flow Monitor - Simplified Monitoring System
    
    Provides essential monitoring capabilities without complex dependencies
    to fix import issues while maintaining system observability.
    """
    
    def __init__(self, brain_id: str, monitoring_level: MonitoringLevel = MonitoringLevel.BASIC):
        """Initialize flow monitor core"""
        self.brain_id = brain_id
        self.monitoring_level = monitoring_level
        self.enabled = True
        
        # Event storage
        self.events: List[FlowEvent] = []
        self.max_events = 1000  # Limit memory usage
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
            "errors": 0,
            "start_time": time.time(),
            "last_activity": time.time()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Active operations tracking
        self.active_operations = {}
        
        logger.info(f"🔍 FlowMonitorCore initialized for {brain_id} (level: {monitoring_level.value})")
    
    def record_event(self, event_type: MonitoringEvent, details: Dict[str, Any], 
                    duration: Optional[float] = None, success: bool = True, 
                    error_message: Optional[str] = None) -> str:
        """Record a monitoring event"""
        if not self.enabled:
            return ""
        
        event_id = f"{self.brain_id}_{int(time.time() * 1000000)}"
        
        event = FlowEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            brain_id=self.brain_id,
            details=details,
            duration=duration,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            # Add event
            self.events.append(event)
            
            # Maintain size limit
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
            
            # Update statistics
            self.stats["total_events"] += 1
            self.stats["last_activity"] = time.time()
            
            event_type_str = event_type.value
            self.stats["events_by_type"][event_type_str] = (
                self.stats["events_by_type"].get(event_type_str, 0) + 1
            )
            
            if not success:
                self.stats["errors"] += 1
        
        if self.monitoring_level in [MonitoringLevel.DETAILED, MonitoringLevel.COMPREHENSIVE]:
            logger.debug(f"📊 Event recorded: {event_type.value} ({event_id})")
        
        return event_id
    
    def start_operation(self, operation_name: str, details: Dict[str, Any] = None) -> str:
        """Start tracking an operation"""
        operation_id = f"op_{self.brain_id}_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                "name": operation_name,
                "start_time": time.time(),
                "details": details or {}
            }
        
        self.record_event(
            MonitoringEvent.TOOL_CALL_START,
            {"operation_name": operation_name, "operation_id": operation_id, **(details or {})}
        )
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error_message: Optional[str] = None, result_details: Dict[str, Any] = None):
        """End tracking an operation"""
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"⚠️ Unknown operation ID: {operation_id}")
                return
            
            operation = self.active_operations.pop(operation_id)
            duration = time.time() - operation["start_time"]
        
        self.record_event(
            MonitoringEvent.TOOL_CALL_END,
            {
                "operation_name": operation["name"],
                "operation_id": operation_id,
                "duration": duration,
                **(result_details or {})
            },
            duration=duration,
            success=success,
            error_message=error_message
        )
    
    def record_message_flow(self, source_brain: str, target_brain: str, 
                           message_type: str, message_size: int = 0):
        """Record inter-brain message flow"""
        self.record_event(
            MonitoringEvent.MESSAGE_SENT,
            {
                "source_brain": source_brain,
                "target_brain": target_brain,
                "message_type": message_type,
                "message_size_bytes": message_size
            }
        )
    
    def record_database_operation(self, operation_type: str, table_name: str, 
                                 duration: float, success: bool = True, 
                                 error_message: Optional[str] = None):
        """Record database operation"""
        self.record_event(
            MonitoringEvent.DATABASE_OPERATION,
            {
                "operation_type": operation_type,
                "table_name": table_name
            },
            duration=duration,
            success=success,
            error_message=error_message
        )
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record an error event"""
        self.record_event(
            MonitoringEvent.ERROR_OCCURRED,
            {
                "error_type": error_type,
                "context": context or {}
            },
            success=False,
            error_message=error_message
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        with self._lock:
            uptime = time.time() - self.stats["start_time"]
            
            return {
                "brain_id": self.brain_id,
                "monitoring_level": self.monitoring_level.value,
                "enabled": self.enabled,
                "uptime_seconds": uptime,
                "total_events": self.stats["total_events"],
                "events_by_type": self.stats["events_by_type"].copy(),
                "error_count": self.stats["errors"],
                "active_operations": len(self.active_operations),
                "last_activity": self.stats["last_activity"],
                "events_stored": len(self.events)
            }
    
    def get_recent_events(self, limit: int = 10, event_type: Optional[MonitoringEvent] = None) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self._lock:
            events = self.events.copy()
        
        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Get most recent events
        recent_events = events[-limit:] if events else []
        
        # Convert to dict format
        return [asdict(event) for event in recent_events]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        with self._lock:
            current_time = time.time()
            time_since_activity = current_time - self.stats["last_activity"]
            
            # Determine health status
            if not self.enabled:
                status = "disabled"
            elif time_since_activity > 300:  # 5 minutes
                status = "stale"
            elif self.stats["errors"] > self.stats["total_events"] * 0.1:  # >10% errors
                status = "degraded"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "enabled": self.enabled,
                "time_since_activity": time_since_activity,
                "error_rate": (self.stats["errors"] / max(self.stats["total_events"], 1)) * 100,
                "active_operations": len(self.active_operations)
            }
    
    def enable(self):
        """Enable monitoring"""
        self.enabled = True
        logger.info(f"🔍 Flow monitoring enabled for {self.brain_id}")
    
    def disable(self):
        """Disable monitoring"""
        self.enabled = False
        logger.info(f"🔍 Flow monitoring disabled for {self.brain_id}")
    
    def clear_events(self):
        """Clear stored events (for maintenance)"""
        with self._lock:
            self.events.clear()
            logger.info(f"🧹 Events cleared for {self.brain_id}")
    
    def export_events(self, format: str = "json") -> str:
        """Export events for analysis"""
        with self._lock:
            events_data = [asdict(event) for event in self.events]
        
        if format == "json":
            return json.dumps(events_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitor instance
_global_monitor: Optional[FlowMonitorCore] = None


def create_flow_monitor(brain_id: str, monitoring_level: MonitoringLevel = MonitoringLevel.BASIC) -> FlowMonitorCore:
    """Factory function to create flow monitor"""
    return FlowMonitorCore(brain_id, monitoring_level)


def get_global_monitor(brain_id: str = None) -> FlowMonitorCore:
    """Get or create global monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_monitor = FlowMonitorCore(brain_id)
    
    return _global_monitor


# Convenience functions for easy integration
def record_operation(operation_name: str, details: Dict[str, Any] = None) -> str:
    """Convenience function to record operation start"""
    monitor = get_global_monitor()
    return monitor.start_operation(operation_name, details)


def complete_operation(operation_id: str, success: bool = True, error_message: Optional[str] = None):
    """Convenience function to complete operation"""
    monitor = get_global_monitor()
    monitor.end_operation(operation_id, success, error_message)
