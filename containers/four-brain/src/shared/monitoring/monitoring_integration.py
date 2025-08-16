"""
Monitoring Integration - Unified Monitoring System
Integrates all monitoring components into a cohesive system

This module provides a unified interface to all monitoring components,
creating a comprehensive observability system for the Four-Brain architecture.

Created: 2025-07-29 AEST
Purpose: Unify all monitoring components into integrated system
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

from .flow_monitor import FlowMonitorCore, MonitoringLevel, create_flow_monitor
from .metrics_collector import MetricsCollector, create_metrics_collector
from .performance_tracker import PerformanceTracker, create_performance_tracker
from .observability_dashboard import ObservabilityDashboard, create_observability_dashboard

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """
    Unified Monitoring Integration System
    
    Provides a single interface to all monitoring components, creating
    a comprehensive observability system for the Four-Brain architecture.
    """
    
    def __init__(self, brain_id: str, monitoring_level: MonitoringLevel = MonitoringLevel.BASIC):
        """Initialize monitoring integration"""
        self.brain_id = brain_id
        self.monitoring_level = monitoring_level
        self.enabled = True
        
        # Initialize all monitoring components
        self.flow_monitor = create_flow_monitor(brain_id, monitoring_level)
        self.metrics_collector = create_metrics_collector(brain_id)
        self.performance_tracker = create_performance_tracker(brain_id)
        self.dashboard = create_observability_dashboard(brain_id)
        
        # Integration state
        self.initialized = False
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🔗 Monitoring Integration initialized for {brain_id}")
    
    async def initialize(self) -> bool:
        """Initialize the integrated monitoring system"""
        try:
            with self._lock:
                if self.initialized:
                    return True
                
                # Register data sources with dashboard
                self.dashboard.register_data_source("flow_monitor", self.flow_monitor)
                self.dashboard.register_data_source("system_metrics", self.metrics_collector)
                self.dashboard.register_data_source("application_metrics", self.metrics_collector)
                self.dashboard.register_data_source("performance_metrics", self.performance_tracker)
                self.dashboard.register_data_source("performance_analysis", self.performance_tracker)
                
                # Start metrics collection
                self.metrics_collector.start_collection()
                
                # Register custom metrics callbacks
                self._register_custom_metrics()
                
                self.initialized = True
                
                # Record initialization event
                self.flow_monitor.record_event(
                    self.flow_monitor.MonitoringEvent.SYSTEM_START,
                    {"component": "monitoring_integration", "monitoring_level": self.monitoring_level.value}
                )
                
                logger.info(f"✅ Monitoring Integration initialized successfully for {self.brain_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Monitoring Integration initialization failed: {e}")
            return False
    
    def _register_custom_metrics(self):
        """Register custom metrics with the metrics collector"""
        
        # Flow monitor metrics
        self.metrics_collector.register_metric_callback(
            "flow_monitor_events",
            lambda: self.flow_monitor.get_statistics()["total_events"]
        )
        
        self.metrics_collector.register_metric_callback(
            "flow_monitor_errors",
            lambda: self.flow_monitor.get_statistics()["error_count"]
        )
        
        # Performance tracker metrics
        self.metrics_collector.register_metric_callback(
            "performance_operations",
            lambda: len(self.performance_tracker.operation_metrics)
        )
        
        # Dashboard metrics
        self.metrics_collector.register_metric_callback(
            "dashboard_count",
            lambda: len(self.dashboard.dashboards)
        )
    
    def record_operation_start(self, operation_name: str, details: Dict[str, Any] = None) -> str:
        """Record the start of an operation across all monitoring systems"""
        if not self.enabled:
            return ""
        
        # Record in flow monitor
        operation_id = self.flow_monitor.start_operation(operation_name, details)
        
        # Record request in metrics collector
        self.metrics_collector.record_connection_change(1)  # Increment active operations
        
        return operation_id
    
    def record_operation_end(self, operation_id: str, duration: float, 
                           success: bool = True, error_message: Optional[str] = None,
                           result_details: Dict[str, Any] = None):
        """Record the end of an operation across all monitoring systems"""
        if not self.enabled:
            return
        
        # Record in flow monitor
        self.flow_monitor.end_operation(operation_id, success, error_message, result_details)
        
        # Record in performance tracker
        operation_name = result_details.get("operation_name", "unknown") if result_details else "unknown"
        self.performance_tracker.record_operation(operation_name, duration, success, result_details)
        
        # Record in metrics collector
        self.metrics_collector.record_request(duration, success)
        self.metrics_collector.record_connection_change(-1)  # Decrement active operations
    
    def record_inter_brain_communication(self, source_brain: str, target_brain: str, 
                                       message_type: str, message_size: int = 0,
                                       duration: float = 0, success: bool = True):
        """Record inter-brain communication across monitoring systems"""
        if not self.enabled:
            return
        
        # Record message flow
        self.flow_monitor.record_message_flow(source_brain, target_brain, message_type, message_size)
        
        # Record as operation if duration provided
        if duration > 0:
            self.performance_tracker.record_operation(
                f"communication_{message_type}",
                duration,
                success,
                {"source": source_brain, "target": target_brain, "size": message_size}
            )
    
    def record_database_operation(self, operation_type: str, table_name: str, 
                                duration: float, success: bool = True, 
                                error_message: Optional[str] = None):
        """Record database operation across monitoring systems"""
        if not self.enabled:
            return
        
        # Record in flow monitor
        self.flow_monitor.record_database_operation(operation_type, table_name, duration, success, error_message)
        
        # Record in performance tracker
        self.performance_tracker.record_operation(
            f"db_{operation_type}",
            duration,
            success,
            {"table": table_name, "operation": operation_type}
        )
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record error across monitoring systems"""
        if not self.enabled:
            return
        
        # Record in flow monitor
        self.flow_monitor.record_error(error_type, error_message, context)
        
        # Record in metrics collector
        self.metrics_collector.record_request(0, False)  # Record as failed request
    
    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status from all monitoring components"""
        try:
            flow_health = self.flow_monitor.get_health_status()
            metrics_summary = self.metrics_collector.get_metrics_summary()
            performance_summary = self.performance_tracker.get_performance_summary()
            dashboard_summary = self.dashboard.get_dashboard_summary()
            
            # Determine overall health
            overall_status = "healthy"
            issues = []
            
            if flow_health["status"] != "healthy":
                overall_status = "degraded"
                issues.append(f"Flow monitor: {flow_health['status']}")
            
            if not metrics_summary["collection_enabled"]:
                overall_status = "degraded"
                issues.append("Metrics collection disabled")
            
            if performance_summary["overall_success_rate"] < 95.0:
                overall_status = "degraded"
                issues.append(f"Low success rate: {performance_summary['overall_success_rate']:.1f}%")
            
            return {
                "brain_id": self.brain_id,
                "overall_status": overall_status,
                "issues": issues,
                "uptime_seconds": time.time() - self.start_time,
                "monitoring_level": self.monitoring_level.value,
                "enabled": self.enabled,
                "components": {
                    "flow_monitor": flow_health,
                    "metrics_collector": {
                        "enabled": metrics_summary["collection_enabled"],
                        "metrics_count": metrics_summary["metrics_collected"]
                    },
                    "performance_tracker": {
                        "operations_tracked": performance_summary["unique_operations"],
                        "overall_success_rate": performance_summary["overall_success_rate"]
                    },
                    "dashboard": {
                        "enabled": dashboard_summary["enabled"],
                        "dashboards_count": dashboard_summary["total_dashboards"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Health status check failed: {e}")
            return {
                "brain_id": self.brain_id,
                "overall_status": "error",
                "error": str(e)
            }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        try:
            # Get top bottlenecks
            bottlenecks = self.performance_tracker.get_top_bottlenecks(5)
            
            # Get slowest operations
            slow_operations = self.performance_tracker.get_slowest_operations(5)
            
            # Get recent errors
            recent_errors = self.flow_monitor.get_recent_events(10, self.flow_monitor.MonitoringEvent.ERROR_OCCURRED)
            
            # Generate insights
            insights = []
            
            if bottlenecks:
                top_bottleneck = bottlenecks[0]
                insights.append(f"Top bottleneck: {top_bottleneck.operation_name} (score: {top_bottleneck.bottleneck_score:.2f})")
            
            if slow_operations:
                slowest = slow_operations[0]
                insights.append(f"Slowest operation: {slowest.operation_name} (avg: {slowest.avg_duration:.2f}s)")
            
            if recent_errors:
                insights.append(f"Recent errors: {len(recent_errors)} in last 10 events")
            
            return {
                "brain_id": self.brain_id,
                "insights": insights,
                "top_bottlenecks": [{"name": b.operation_name, "score": b.bottleneck_score} for b in bottlenecks[:3]],
                "slowest_operations": [{"name": s.operation_name, "duration": s.avg_duration} for s in slow_operations[:3]],
                "recent_error_count": len(recent_errors),
                "generated_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Performance insights generation failed: {e}")
            return {"error": str(e)}
    
    def export_monitoring_data(self, format: str = "json") -> str:
        """Export comprehensive monitoring data"""
        try:
            data = {
                "brain_id": self.brain_id,
                "export_timestamp": time.time(),
                "health_status": self.get_comprehensive_health_status(),
                "performance_insights": self.get_performance_insights(),
                "flow_events": self.flow_monitor.get_recent_events(100),
                "metrics_summary": self.metrics_collector.get_metrics_summary(),
                "performance_analysis": self.performance_tracker.export_analysis(),
                "dashboard_summary": self.dashboard.get_dashboard_summary()
            }
            
            if format == "json":
                import json
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"❌ Monitoring data export failed: {e}")
            return f'{{"error": "{str(e)}"}}'
    
    def enable_monitoring(self):
        """Enable all monitoring components"""
        self.enabled = True
        self.flow_monitor.enable()
        self.metrics_collector.enabled = True
        self.performance_tracker.enabled = True
        self.dashboard.enabled = True
        logger.info(f"🔍 All monitoring enabled for {self.brain_id}")
    
    def disable_monitoring(self):
        """Disable all monitoring components"""
        self.enabled = False
        self.flow_monitor.disable()
        self.metrics_collector.enabled = False
        self.performance_tracker.enabled = False
        self.dashboard.enabled = False
        logger.info(f"🔍 All monitoring disabled for {self.brain_id}")
    
    async def cleanup(self):
        """Clean up monitoring resources"""
        try:
            self.metrics_collector.stop_collection()
            logger.info(f"🧹 Monitoring Integration cleaned up for {self.brain_id}")
        except Exception as e:
            logger.error(f"❌ Monitoring cleanup failed: {e}")


# Factory function for easy creation
def create_monitoring_integration(brain_id: str, monitoring_level: MonitoringLevel = MonitoringLevel.BASIC) -> MonitoringIntegration:
    """Factory function to create monitoring integration"""
    return MonitoringIntegration(brain_id, monitoring_level)


# Global monitoring instance
_global_monitoring: Optional[MonitoringIntegration] = None


def get_global_monitoring(brain_id: str = None) -> MonitoringIntegration:
    """Get or create global monitoring instance"""
    global _global_monitoring
    
    if _global_monitoring is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_monitoring = MonitoringIntegration(brain_id)
    
    return _global_monitoring
