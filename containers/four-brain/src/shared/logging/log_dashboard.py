"""
Log Dashboard - Log Visualization and Monitoring
Provides comprehensive log visualization and monitoring dashboards

This module creates interactive dashboards for log visualization, real-time
monitoring, and comprehensive log analysis for the Four-Brain system.

Created: 2025-07-29 AEST
Purpose: Log visualization and monitoring dashboards
Module Size: 150 lines (modular design)
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class LogWidget:
    """Log dashboard widget configuration"""
    widget_id: str
    widget_type: str  # chart, table, metric, alert, timeline
    title: str
    query_filter: Dict[str, Any]
    refresh_interval: int
    display_config: Dict[str, Any]


@dataclass
class LogDashboard:
    """Log dashboard configuration"""
    dashboard_id: str
    title: str
    description: str
    widgets: List[LogWidget]
    layout_config: Dict[str, Any]
    created_at: float
    updated_at: float


class LogDashboardManager:
    """
    Log Dashboard Manager
    
    Provides comprehensive log visualization and monitoring dashboards
    with real-time updates and interactive analysis capabilities.
    """
    
    def __init__(self, dashboard_manager_id: str = "log_dashboard_manager"):
        """Initialize log dashboard manager"""
        self.manager_id = dashboard_manager_id
        self.enabled = True
        
        # Dashboard storage
        self.dashboards: Dict[str, LogDashboard] = {}
        self.widget_data_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, float] = {}
        
        # Data sources
        self.log_aggregator = None
        self.log_analyzer = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default dashboards
        self._create_default_dashboards()
        
        logger.info(f"📊 Log Dashboard Manager initialized: {dashboard_manager_id}")
    
    def set_data_sources(self, log_aggregator, log_analyzer):
        """Set data sources for dashboard widgets"""
        self.log_aggregator = log_aggregator
        self.log_analyzer = log_analyzer
        logger.info("🔗 Data sources connected to log dashboard")
    
    def _create_default_dashboards(self):
        """Create default log monitoring dashboards"""
        
        # System Overview Dashboard
        system_widgets = [
            LogWidget(
                widget_id="log_volume",
                widget_type="chart",
                title="Log Volume Over Time",
                query_filter={"time_range": "1h", "group_by": "timestamp"},
                refresh_interval=30,
                display_config={"chart_type": "line", "time_series": True}
            ),
            LogWidget(
                widget_id="log_levels",
                widget_type="chart",
                title="Log Level Distribution",
                query_filter={"time_range": "1h", "group_by": "level"},
                refresh_interval=30,
                display_config={"chart_type": "pie"}
            ),
            LogWidget(
                widget_id="brain_activity",
                widget_type="chart",
                title="Brain Activity",
                query_filter={"time_range": "1h", "group_by": "brain_id"},
                refresh_interval=30,
                display_config={"chart_type": "bar"}
            ),
            LogWidget(
                widget_id="error_count",
                widget_type="metric",
                title="Error Count",
                query_filter={"time_range": "1h", "level": ["ERROR", "CRITICAL"]},
                refresh_interval=10,
                display_config={"format": "number", "alert_threshold": 10}
            )
        ]
        
        system_dashboard = LogDashboard(
            dashboard_id="system_logs",
            title="System Log Overview",
            description="Overall system log monitoring and analysis",
            widgets=system_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Error Analysis Dashboard
        error_widgets = [
            LogWidget(
                widget_id="error_timeline",
                widget_type="timeline",
                title="Error Timeline",
                query_filter={"time_range": "4h", "level": ["ERROR", "CRITICAL"]},
                refresh_interval=60,
                display_config={"show_details": True, "group_similar": True}
            ),
            LogWidget(
                widget_id="error_patterns",
                widget_type="table",
                title="Error Patterns",
                query_filter={"time_range": "1h", "analysis_type": "patterns"},
                refresh_interval=120,
                display_config={"columns": ["pattern", "count", "affected_brains"], "limit": 10}
            ),
            LogWidget(
                widget_id="error_by_brain",
                widget_type="chart",
                title="Errors by Brain",
                query_filter={"time_range": "1h", "level": ["ERROR", "CRITICAL"], "group_by": "brain_id"},
                refresh_interval=60,
                display_config={"chart_type": "bar", "sort": "desc"}
            ),
            LogWidget(
                widget_id="recent_errors",
                widget_type="table",
                title="Recent Errors",
                query_filter={"time_range": "30m", "level": ["ERROR", "CRITICAL"]},
                refresh_interval=30,
                display_config={"columns": ["timestamp", "brain_id", "message"], "limit": 20}
            )
        ]
        
        error_dashboard = LogDashboard(
            dashboard_id="error_analysis",
            title="Error Analysis",
            description="Detailed error monitoring and pattern analysis",
            widgets=error_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Performance Dashboard
        performance_widgets = [
            LogWidget(
                widget_id="performance_logs",
                widget_type="chart",
                title="Performance-Related Logs",
                query_filter={"time_range": "2h", "pattern": "slow|performance|latency"},
                refresh_interval=60,
                display_config={"chart_type": "line", "time_series": True}
            ),
            LogWidget(
                widget_id="operation_duration",
                widget_type="chart",
                title="Operation Duration Trends",
                query_filter={"time_range": "1h", "logger_name": "operations"},
                refresh_interval=60,
                display_config={"chart_type": "line", "extract_duration": True}
            ),
            LogWidget(
                widget_id="slow_operations",
                widget_type="table",
                title="Slow Operations",
                query_filter={"time_range": "1h", "pattern": "slow|took.*seconds"},
                refresh_interval=120,
                display_config={"columns": ["timestamp", "brain_id", "operation", "duration"], "limit": 15}
            )
        ]
        
        performance_dashboard = LogDashboard(
            dashboard_id="performance_logs",
            title="Performance Log Analysis",
            description="Performance-related log monitoring and analysis",
            widgets=performance_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Store dashboards
        with self._lock:
            self.dashboards["system_logs"] = system_dashboard
            self.dashboards["error_analysis"] = error_dashboard
            self.dashboards["performance_logs"] = performance_dashboard
    
    def create_dashboard(self, dashboard_id: str, title: str, description: str,
                        widgets: List[LogWidget], layout_config: Dict[str, Any] = None) -> bool:
        """Create a custom log dashboard"""
        try:
            dashboard = LogDashboard(
                dashboard_id=dashboard_id,
                title=title,
                description=description,
                widgets=widgets,
                layout_config=layout_config or {"columns": 2, "auto_refresh": True},
                created_at=time.time(),
                updated_at=time.time()
            )
            
            with self._lock:
                self.dashboards[dashboard_id] = dashboard
            
            logger.info(f"📊 Log dashboard created: {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create log dashboard {dashboard_id}: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[LogDashboard]:
        """Get dashboard configuration"""
        with self._lock:
            return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available log dashboards"""
        with self._lock:
            return [
                {
                    "dashboard_id": dashboard.dashboard_id,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "widget_count": len(dashboard.widgets),
                    "created_at": dashboard.created_at,
                    "updated_at": dashboard.updated_at
                }
                for dashboard in self.dashboards.values()
            ]
    
    def get_widget_data(self, widget: LogWidget, force_refresh: bool = False) -> Dict[str, Any]:
        """Get data for a specific widget"""
        cache_key = f"{widget.widget_id}_{hash(str(widget.query_filter))}"
        
        # Check cache first
        if not force_refresh and cache_key in self.cache_expiry:
            if time.time() < self.cache_expiry[cache_key]:
                return self.widget_data_cache.get(cache_key, {})
        
        try:
            # Get data based on widget type
            if widget.widget_type == "chart":
                data = self._get_chart_data(widget)
            elif widget.widget_type == "table":
                data = self._get_table_data(widget)
            elif widget.widget_type == "metric":
                data = self._get_metric_data(widget)
            elif widget.widget_type == "timeline":
                data = self._get_timeline_data(widget)
            elif widget.widget_type == "alert":
                data = self._get_alert_data(widget)
            else:
                data = {"error": f"Unknown widget type: {widget.widget_type}"}
            
            # Cache the data
            self.widget_data_cache[cache_key] = data
            self.cache_expiry[cache_key] = time.time() + widget.refresh_interval
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to get widget data for {widget.widget_id}: {e}")
            return {"error": str(e)}
    
    def _get_chart_data(self, widget: LogWidget) -> Dict[str, Any]:
        """Get data for chart widgets"""
        if not self.log_aggregator:
            return {"error": "Log aggregator not available"}
        
        query_filter = widget.query_filter
        time_range = query_filter.get("time_range", "1h")
        
        # Convert time range to seconds
        time_seconds = self._parse_time_range(time_range)
        since = time.time() - time_seconds
        
        # Get log entries
        logs = self.log_aggregator.get_recent_logs(
            limit=1000,
            brain_id=query_filter.get("brain_id"),
            level=query_filter.get("level"),
            since=since
        )
        
        # Group data based on query
        group_by = query_filter.get("group_by", "timestamp")
        
        if group_by == "timestamp":
            # Time series data
            time_buckets = defaultdict(int)
            bucket_size = max(time_seconds // 20, 60)  # 20 buckets minimum 1 minute
            
            for log in logs:
                bucket = int(log["timestamp"] // bucket_size) * bucket_size
                time_buckets[bucket] += 1
            
            return {
                "chart_type": widget.display_config.get("chart_type", "line"),
                "data": [{"x": bucket, "y": count} for bucket, count in sorted(time_buckets.items())],
                "timestamp": time.time()
            }
        
        elif group_by in ["level", "brain_id"]:
            # Categorical data
            counts = Counter(log.get(group_by, "unknown") for log in logs)
            
            return {
                "chart_type": widget.display_config.get("chart_type", "bar"),
                "data": [{"label": label, "value": count} for label, count in counts.most_common()],
                "timestamp": time.time()
            }
        
        return {"error": f"Unsupported group_by: {group_by}"}
    
    def _get_table_data(self, widget: LogWidget) -> Dict[str, Any]:
        """Get data for table widgets"""
        if not self.log_aggregator:
            return {"error": "Log aggregator not available"}
        
        query_filter = widget.query_filter
        time_range = query_filter.get("time_range", "1h")
        time_seconds = self._parse_time_range(time_range)
        since = time.time() - time_seconds
        
        # Get log entries
        if "pattern" in query_filter:
            logs = self.log_aggregator.get_logs_by_pattern(
                query_filter["pattern"],
                limit=widget.display_config.get("limit", 100)
            )
        else:
            logs = self.log_aggregator.get_recent_logs(
                limit=widget.display_config.get("limit", 100),
                brain_id=query_filter.get("brain_id"),
                level=query_filter.get("level"),
                since=since
            )
        
        # Format for table display
        columns = widget.display_config.get("columns", ["timestamp", "brain_id", "level", "message"])
        
        rows = []
        for log in logs:
            row = {}
            for col in columns:
                if col == "timestamp":
                    row[col] = datetime.fromtimestamp(log.get("timestamp", 0)).strftime("%H:%M:%S")
                else:
                    row[col] = log.get(col, "")
            rows.append(row)
        
        return {
            "columns": columns,
            "rows": rows,
            "timestamp": time.time()
        }
    
    def _get_metric_data(self, widget: LogWidget) -> Dict[str, Any]:
        """Get data for metric widgets"""
        if not self.log_aggregator:
            return {"error": "Log aggregator not available"}
        
        query_filter = widget.query_filter
        time_range = query_filter.get("time_range", "1h")
        time_seconds = self._parse_time_range(time_range)
        since = time.time() - time_seconds
        
        # Get log entries
        logs = self.log_aggregator.get_recent_logs(
            limit=None,
            brain_id=query_filter.get("brain_id"),
            level=query_filter.get("level"),
            since=since
        )
        
        value = len(logs)
        alert_threshold = widget.display_config.get("alert_threshold", 0)
        
        return {
            "value": value,
            "format": widget.display_config.get("format", "number"),
            "alert": value > alert_threshold if alert_threshold > 0 else False,
            "timestamp": time.time()
        }
    
    def _get_timeline_data(self, widget: LogWidget) -> Dict[str, Any]:
        """Get data for timeline widgets"""
        if not self.log_aggregator:
            return {"error": "Log aggregator not available"}
        
        query_filter = widget.query_filter
        time_range = query_filter.get("time_range", "4h")
        time_seconds = self._parse_time_range(time_range)
        since = time.time() - time_seconds
        
        # Get log entries
        logs = self.log_aggregator.get_recent_logs(
            limit=500,
            brain_id=query_filter.get("brain_id"),
            level=query_filter.get("level"),
            since=since
        )
        
        # Format for timeline
        events = []
        for log in logs:
            events.append({
                "timestamp": log.get("timestamp", 0),
                "title": f"{log.get('brain_id', 'unknown')}: {log.get('level', 'INFO')}",
                "description": log.get("message", "")[:100],
                "severity": log.get("level", "INFO").lower()
            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "events": events,
            "timestamp": time.time()
        }
    
    def _get_alert_data(self, widget: LogWidget) -> Dict[str, Any]:
        """Get data for alert widgets"""
        if not self.log_analyzer:
            return {"error": "Log analyzer not available"}
        
        # Get recent anomalies and insights
        analysis_summary = self.log_analyzer.get_analysis_summary()
        
        alerts = []
        
        # Add high severity insights as alerts
        for insight in self.log_analyzer.generated_insights[-10:]:  # Last 10 insights
            if insight.severity in ["high", "critical"]:
                alerts.append({
                    "id": insight.insight_id,
                    "title": insight.title,
                    "description": insight.description,
                    "severity": insight.severity,
                    "timestamp": insight.generated_at
                })
        
        return {
            "alerts": alerts,
            "timestamp": time.time()
        }
    
    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to seconds"""
        if time_range.endswith("m"):
            return int(time_range[:-1]) * 60
        elif time_range.endswith("h"):
            return int(time_range[:-1]) * 3600
        elif time_range.endswith("d"):
            return int(time_range[:-1]) * 86400
        else:
            return 3600  # Default 1 hour
    
    def render_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Render complete dashboard with all widget data"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": f"Dashboard '{dashboard_id}' not found"}
        
        try:
            widget_data = {}
            for widget in dashboard.widgets:
                widget_data[widget.widget_id] = self.get_widget_data(widget)
            
            return {
                "dashboard": asdict(dashboard),
                "widget_data": widget_data,
                "rendered_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to render dashboard {dashboard_id}: {e}")
            return {"error": str(e)}
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary of dashboard manager"""
        with self._lock:
            return {
                "manager_id": self.manager_id,
                "enabled": self.enabled,
                "total_dashboards": len(self.dashboards),
                "cache_entries": len(self.widget_data_cache),
                "data_sources_connected": {
                    "log_aggregator": self.log_aggregator is not None,
                    "log_analyzer": self.log_analyzer is not None
                },
                "dashboards": self.list_dashboards()
            }


# Factory function for easy creation
def create_log_dashboard_manager(manager_id: str = "log_dashboard_manager") -> LogDashboardManager:
    """Factory function to create log dashboard manager"""
    return LogDashboardManager(manager_id)
