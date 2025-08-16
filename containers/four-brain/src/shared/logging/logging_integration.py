"""
Logging Integration - Unified Logging System
Integrates all logging components into a cohesive system

This module provides a unified interface to all logging components,
creating a comprehensive logging system for the Four-Brain architecture.

Created: 2025-07-29 AEST
Purpose: Unify all logging components into integrated system
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

from .centralized_logger import CentralizedLogger, LogLevel, create_centralized_logger
from .log_aggregator import LogAggregator, LogEntry, create_log_aggregator
from .log_analyzer import LogAnalyzer, create_log_analyzer
from .log_dashboard import LogDashboardManager, create_log_dashboard_manager
from .log_retention import LogRetentionManager, create_log_retention_manager

logger = logging.getLogger(__name__)


class LoggingIntegration:
    """
    Unified Logging Integration System
    
    Provides a single interface to all logging components, creating
    a comprehensive logging system for the Four-Brain architecture.
    """
    
    def __init__(self, brain_id: str, log_level: LogLevel = LogLevel.INFO):
        """Initialize logging integration"""
        self.brain_id = brain_id
        self.log_level = log_level
        self.enabled = True
        
        # Initialize all logging components
        self.centralized_logger = create_centralized_logger(brain_id, log_level)
        self.log_aggregator = create_log_aggregator(f"{brain_id}_aggregator")
        self.log_analyzer = create_log_analyzer(f"{brain_id}_analyzer")
        self.dashboard_manager = create_log_dashboard_manager(f"{brain_id}_dashboard")
        self.retention_manager = create_log_retention_manager(f"{brain_id}_retention")
        
        # Integration state
        self.initialized = False
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🔗 Logging Integration initialized for {brain_id}")
    
    async def initialize(self) -> bool:
        """Initialize the integrated logging system"""
        try:
            with self._lock:
                if self.initialized:
                    return True
                
                # Connect data sources
                self.dashboard_manager.set_data_sources(self.log_aggregator, self.log_analyzer)
                
                # Add log sources for aggregation
                log_files = self.centralized_logger.get_log_files()
                if log_files:
                    self.log_aggregator.add_log_source(self.brain_id, log_files)
                
                # Start monitoring and retention
                self.log_aggregator.start_monitoring()
                self.retention_manager.start_auto_cleanup()
                
                # Register custom log processors
                self._register_log_processors()
                
                self.initialized = True
                
                # Log initialization event
                self.centralized_logger.log_system_event(
                    "logging_integration_initialized",
                    f"Logging integration initialized for {self.brain_id}",
                    LogLevel.INFO,
                    {"components": ["centralized_logger", "aggregator", "analyzer", "dashboard", "retention"]}
                )
                
                logger.info(f"✅ Logging Integration initialized successfully for {self.brain_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Logging Integration initialization failed: {e}")
            return False
    
    def _register_log_processors(self):
        """Register custom log processors"""
        
        # Processor to extract operation information
        def extract_operation_info(entry: LogEntry) -> LogEntry:
            if "operation" in entry.message.lower():
                # Extract operation details from message
                if "started:" in entry.message:
                    entry.metadata["operation_status"] = "started"
                elif "completed:" in entry.message or "✅" in entry.message:
                    entry.metadata["operation_status"] = "completed"
                elif "failed:" in entry.message or "❌" in entry.message:
                    entry.metadata["operation_status"] = "failed"
            
            return entry
        
        # Processor to categorize log entries
        def categorize_entry(entry: LogEntry) -> LogEntry:
            message_lower = entry.message.lower()
            
            if any(word in message_lower for word in ["error", "failed", "exception", "❌"]):
                entry.metadata["category"] = "error"
            elif any(word in message_lower for word in ["warning", "warn", "⚠️"]):
                entry.metadata["category"] = "warning"
            elif any(word in message_lower for word in ["performance", "slow", "latency"]):
                entry.metadata["category"] = "performance"
            elif any(word in message_lower for word in ["communication", "message", "📤", "📥"]):
                entry.metadata["category"] = "communication"
            elif any(word in message_lower for word in ["database", "sql", "🗄️"]):
                entry.metadata["category"] = "database"
            else:
                entry.metadata["category"] = "general"
            
            return entry
        
        # Add processors to aggregator
        self.log_aggregator.add_processor(extract_operation_info)
        self.log_aggregator.add_processor(categorize_entry)
        
        # Add filters to reduce noise
        def filter_debug_spam(entry: LogEntry) -> bool:
            # Filter out excessive debug messages
            if entry.level == "DEBUG" and "heartbeat" in entry.message.lower():
                return False
            return True
        
        self.log_aggregator.add_filter(filter_debug_spam)
    
    def log_operation_start(self, operation_name: str, details: Dict[str, Any] = None) -> str:
        """Log operation start across all logging systems"""
        if not self.enabled:
            return ""
        
        # Log in centralized logger
        operation_id = self.centralized_logger.log_operation_start(operation_name, details)
        
        # Create log entry for aggregator
        log_entry = LogEntry(
            timestamp=time.time(),
            brain_id=self.brain_id,
            level="INFO",
            logger_name="operations",
            message=f"🚀 Operation started: {operation_name}",
            metadata={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "operation_status": "started",
                "details": details or {}
            }
        )
        
        self.log_aggregator.add_log_entry_direct(log_entry)
        
        return operation_id
    
    def log_operation_end(self, operation_id: str, operation_name: str, 
                         duration: float, success: bool = True, 
                         error_message: str = None, result: Dict[str, Any] = None):
        """Log operation completion across all logging systems"""
        if not self.enabled:
            return
        
        # Log in centralized logger
        self.centralized_logger.log_operation_end(
            operation_id, operation_name, duration, success, error_message, result
        )
        
        # Create log entry for aggregator
        status_emoji = "✅" if success else "❌"
        message = f"{status_emoji} Operation {'completed' if success else 'failed'}: {operation_name} ({duration:.2f}s)"
        if error_message:
            message += f" - {error_message}"
        
        log_entry = LogEntry(
            timestamp=time.time(),
            brain_id=self.brain_id,
            level="INFO" if success else "ERROR",
            logger_name="operations",
            message=message,
            metadata={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "operation_status": "completed" if success else "failed",
                "duration_seconds": duration,
                "success": success,
                "error_message": error_message,
                "result": result or {}
            }
        )
        
        self.log_aggregator.add_log_entry_direct(log_entry)
    
    def log_inter_brain_communication(self, source_brain: str, target_brain: str, 
                                    message_type: str, message_id: str, 
                                    success: bool = True, error_message: str = None):
        """Log inter-brain communication across all logging systems"""
        if not self.enabled:
            return
        
        # Log in centralized logger
        self.centralized_logger.log_inter_brain_communication(
            source_brain, target_brain, message_type, message_id, success, error_message
        )
        
        # Create log entry for aggregator
        status_emoji = "📤" if success else "❌"
        message = f"{status_emoji} Message {'sent' if success else 'failed'}: {source_brain} → {target_brain} ({message_type})"
        if error_message:
            message += f" - {error_message}"
        
        log_entry = LogEntry(
            timestamp=time.time(),
            brain_id=self.brain_id,
            level="INFO" if success else "ERROR",
            logger_name="communication",
            message=message,
            metadata={
                "message_id": message_id,
                "source_brain": source_brain,
                "target_brain": target_brain,
                "message_type": message_type,
                "communication_status": "sent" if success else "failed",
                "success": success,
                "error_message": error_message
            }
        )
        
        self.log_aggregator.add_log_entry_direct(log_entry)
    
    def log_database_operation(self, operation_type: str, table_name: str, 
                             duration: float, success: bool = True, 
                             error_message: str = None, affected_rows: int = None):
        """Log database operations across all logging systems"""
        if not self.enabled:
            return
        
        # Log in centralized logger
        self.centralized_logger.log_database_operation(
            operation_type, table_name, duration, success, error_message, affected_rows
        )
        
        # Create log entry for aggregator
        status_emoji = "🗄️" if success else "❌"
        message = f"{status_emoji} Database {operation_type}: {table_name} ({duration:.3f}s)"
        if error_message:
            message += f" - {error_message}"
        
        log_entry = LogEntry(
            timestamp=time.time(),
            brain_id=self.brain_id,
            level="INFO" if success else "ERROR",
            logger_name="database",
            message=message,
            metadata={
                "operation_type": operation_type,
                "table_name": table_name,
                "duration_seconds": duration,
                "success": success,
                "error_message": error_message,
                "affected_rows": affected_rows
            }
        )
        
        self.log_aggregator.add_log_entry_direct(log_entry)
    
    def analyze_recent_logs(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze recent logs and generate insights"""
        if not self.enabled:
            return {"error": "Logging integration disabled"}
        
        # Get recent logs
        since = time.time() - (time_window_minutes * 60)
        recent_logs = self.log_aggregator.get_recent_logs(limit=1000, since=since)
        
        # Analyze logs
        analysis_result = self.log_analyzer.analyze_logs(recent_logs)
        
        return analysis_result
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status from all logging components"""
        try:
            aggregator_stats = self.log_aggregator.get_aggregator_stats()
            analyzer_summary = self.log_analyzer.get_analysis_summary()
            dashboard_summary = self.dashboard_manager.get_dashboard_summary()
            retention_summary = self.retention_manager.get_retention_summary()
            logger_stats = self.centralized_logger.get_logger_stats()
            
            # Determine overall health
            overall_status = "healthy"
            issues = []
            
            if not aggregator_stats["monitoring_active"]:
                overall_status = "degraded"
                issues.append("Log aggregation monitoring not active")
            
            if analyzer_summary["high_severity_insights"] > 0:
                overall_status = "degraded"
                issues.append(f"High severity insights: {analyzer_summary['high_severity_insights']}")
            
            if not retention_summary["auto_cleanup_enabled"]:
                issues.append("Auto cleanup disabled")
            
            return {
                "brain_id": self.brain_id,
                "overall_status": overall_status,
                "issues": issues,
                "uptime_seconds": time.time() - self.start_time,
                "enabled": self.enabled,
                "components": {
                    "centralized_logger": {
                        "active_loggers": logger_stats["active_loggers"],
                        "log_level": logger_stats["log_level"]
                    },
                    "log_aggregator": {
                        "monitoring_active": aggregator_stats["monitoring_active"],
                        "total_entries": aggregator_stats["total_entries"]
                    },
                    "log_analyzer": {
                        "total_patterns": analyzer_summary["total_patterns"],
                        "total_anomalies": analyzer_summary["total_anomalies"],
                        "total_insights": analyzer_summary["total_insights"]
                    },
                    "dashboard_manager": {
                        "total_dashboards": dashboard_summary["total_dashboards"],
                        "data_sources_connected": dashboard_summary["data_sources_connected"]
                    },
                    "retention_manager": {
                        "auto_cleanup_enabled": retention_summary["auto_cleanup_enabled"],
                        "cleanup_running": retention_summary["cleanup_running"]
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
    
    def export_comprehensive_logs(self, format: str = "json", time_window_hours: int = 24) -> str:
        """Export comprehensive logging data"""
        try:
            since = time.time() - (time_window_hours * 3600)
            
            data = {
                "brain_id": self.brain_id,
                "export_timestamp": time.time(),
                "time_window_hours": time_window_hours,
                "status": self.get_comprehensive_status(),
                "recent_logs": self.log_aggregator.get_recent_logs(limit=1000, since=since),
                "log_analysis": self.analyze_recent_logs(time_window_hours * 60),
                "aggregator_stats": self.log_aggregator.get_aggregator_stats(),
                "retention_summary": self.retention_manager.get_retention_summary()
            }
            
            if format == "json":
                import json
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"❌ Comprehensive log export failed: {e}")
            return f'{{"error": "{str(e)}"}}'
    
    def enable_logging(self):
        """Enable all logging components"""
        self.enabled = True
        self.log_aggregator.enabled = True
        self.log_analyzer.enabled = True
        self.dashboard_manager.enabled = True
        self.retention_manager.enabled = True
        logger.info(f"🔍 All logging enabled for {self.brain_id}")
    
    def disable_logging(self):
        """Disable all logging components"""
        self.enabled = False
        self.log_aggregator.enabled = False
        self.log_analyzer.enabled = False
        self.dashboard_manager.enabled = False
        self.retention_manager.enabled = False
        logger.info(f"🔍 All logging disabled for {self.brain_id}")
    
    async def cleanup(self):
        """Clean up logging resources"""
        try:
            self.log_aggregator.stop_monitoring()
            self.retention_manager.stop_auto_cleanup()
            logger.info(f"🧹 Logging Integration cleaned up for {self.brain_id}")
        except Exception as e:
            logger.error(f"❌ Logging cleanup failed: {e}")


# Factory function for easy creation
def create_logging_integration(brain_id: str, log_level: LogLevel = LogLevel.INFO) -> LoggingIntegration:
    """Factory function to create logging integration"""
    return LoggingIntegration(brain_id, log_level)


# Global logging instance
_global_logging: Optional[LoggingIntegration] = None


def get_global_logging(brain_id: str = None) -> LoggingIntegration:
    """Get or create global logging instance"""
    global _global_logging
    
    if _global_logging is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_logging = LoggingIntegration(brain_id)
    
    return _global_logging
