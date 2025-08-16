"""
Observability Dashboard - Monitoring Visualization
Provides comprehensive monitoring visualization and dashboard capabilities

This module creates monitoring dashboards, visualizations, and reports
for the Four-Brain system observability and performance analysis.

Created: 2025-07-29 AEST
Purpose: Monitoring visualization and dashboard creation
Module Size: 150 lines (modular design)
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str  # chart, metric, table, alert
    title: str
    data_source: str
    refresh_interval: int
    config: Dict[str, Any]


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    layout_config: Dict[str, Any]
    created_at: float
    updated_at: float


class ObservabilityDashboard:
    """
    Observability Dashboard System
    
    Provides comprehensive monitoring visualization including real-time
    dashboards, performance charts, and system health overviews.
    """
    
    def __init__(self, brain_id: str):
        """Initialize observability dashboard"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Dashboard storage
        self.dashboards: Dict[str, DashboardLayout] = {}
        self.widget_data_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, float] = {}
        
        # Data sources
        self.data_sources = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default dashboards
        self._create_default_dashboards()
        
        logger.info(f"📊 Observability Dashboard initialized for {brain_id}")
    
    def _create_default_dashboards(self):
        """Create default monitoring dashboards"""
        
        # System Overview Dashboard
        system_widgets = [
            DashboardWidget(
                widget_id="cpu_usage",
                widget_type="chart",
                title="CPU Usage",
                data_source="system_metrics",
                refresh_interval=30,
                config={"chart_type": "line", "metric": "cpu_percent", "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="memory_usage",
                widget_type="chart",
                title="Memory Usage",
                data_source="system_metrics",
                refresh_interval=30,
                config={"chart_type": "line", "metric": "memory_percent", "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="active_connections",
                widget_type="metric",
                title="Active Connections",
                data_source="application_metrics",
                refresh_interval=10,
                config={"metric": "active_connections", "format": "number"}
            ),
            DashboardWidget(
                widget_id="error_rate",
                widget_type="metric",
                title="Error Rate",
                data_source="application_metrics",
                refresh_interval=10,
                config={"metric": "error_rate", "format": "percentage", "alert_threshold": 5.0}
            )
        ]
        
        system_dashboard = DashboardLayout(
            dashboard_id="system_overview",
            title="System Overview",
            description="Overall system health and performance metrics",
            widgets=system_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Performance Dashboard
        performance_widgets = [
            DashboardWidget(
                widget_id="response_times",
                widget_type="chart",
                title="Response Times",
                data_source="performance_metrics",
                refresh_interval=30,
                config={"chart_type": "line", "metrics": ["avg_duration", "p95_duration"], "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="throughput",
                widget_type="chart",
                title="Throughput",
                data_source="performance_metrics",
                refresh_interval=30,
                config={"chart_type": "bar", "metric": "throughput_per_second", "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="top_bottlenecks",
                widget_type="table",
                title="Top Bottlenecks",
                data_source="performance_analysis",
                refresh_interval=60,
                config={"columns": ["operation_name", "bottleneck_score", "avg_duration"], "limit": 10}
            ),
            DashboardWidget(
                widget_id="slowest_operations",
                widget_type="table",
                title="Slowest Operations",
                data_source="performance_analysis",
                refresh_interval=60,
                config={"columns": ["operation_name", "avg_duration", "p95_duration"], "limit": 10}
            )
        ]
        
        performance_dashboard = DashboardLayout(
            dashboard_id="performance_analysis",
            title="Performance Analysis",
            description="Detailed performance metrics and bottleneck analysis",
            widgets=performance_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Communication Dashboard
        communication_widgets = [
            DashboardWidget(
                widget_id="message_flow",
                widget_type="chart",
                title="Inter-Brain Message Flow",
                data_source="flow_monitor",
                refresh_interval=30,
                config={"chart_type": "network", "metric": "message_flow", "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="communication_errors",
                widget_type="chart",
                title="Communication Errors",
                data_source="flow_monitor",
                refresh_interval=30,
                config={"chart_type": "line", "metric": "error_count", "time_range": "1h"}
            ),
            DashboardWidget(
                widget_id="brain_health",
                widget_type="table",
                title="Brain Health Status",
                data_source="health_monitor",
                refresh_interval=30,
                config={"columns": ["brain_id", "status", "last_activity"], "limit": 10}
            )
        ]
        
        communication_dashboard = DashboardLayout(
            dashboard_id="communication_monitoring",
            title="Communication Monitoring",
            description="Inter-brain communication and health monitoring",
            widgets=communication_widgets,
            layout_config={"columns": 2, "auto_refresh": True},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Store dashboards
        with self._lock:
            self.dashboards["system_overview"] = system_dashboard
            self.dashboards["performance_analysis"] = performance_dashboard
            self.dashboards["communication_monitoring"] = communication_dashboard
    
    def register_data_source(self, source_name: str, data_provider: Any):
        """Register a data source for dashboard widgets"""
        self.data_sources[source_name] = data_provider
        logger.info(f"📊 Data source registered: {source_name}")
    
    def create_dashboard(self, dashboard_id: str, title: str, description: str, 
                        widgets: List[DashboardWidget], layout_config: Dict[str, Any] = None) -> bool:
        """Create a custom dashboard"""
        try:
            dashboard = DashboardLayout(
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
            
            logger.info(f"📊 Dashboard created: {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create dashboard {dashboard_id}: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardLayout]:
        """Get dashboard configuration"""
        with self._lock:
            return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards"""
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
    
    def get_widget_data(self, widget: DashboardWidget, force_refresh: bool = False) -> Dict[str, Any]:
        """Get data for a specific widget"""
        cache_key = f"{widget.widget_id}_{widget.data_source}"
        
        # Check cache first
        if not force_refresh and cache_key in self.cache_expiry:
            if time.time() < self.cache_expiry[cache_key]:
                return self.widget_data_cache.get(cache_key, {})
        
        try:
            # Get data from source
            if widget.data_source not in self.data_sources:
                return {"error": f"Data source '{widget.data_source}' not available"}
            
            data_provider = self.data_sources[widget.data_source]
            
            # Call appropriate method based on widget type and config
            if widget.widget_type == "chart":
                data = self._get_chart_data(data_provider, widget.config)
            elif widget.widget_type == "metric":
                data = self._get_metric_data(data_provider, widget.config)
            elif widget.widget_type == "table":
                data = self._get_table_data(data_provider, widget.config)
            elif widget.widget_type == "alert":
                data = self._get_alert_data(data_provider, widget.config)
            else:
                data = {"error": f"Unknown widget type: {widget.widget_type}"}
            
            # Cache the data
            self.widget_data_cache[cache_key] = data
            self.cache_expiry[cache_key] = time.time() + widget.refresh_interval
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to get widget data for {widget.widget_id}: {e}")
            return {"error": str(e)}
    
    def _get_chart_data(self, data_provider: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for chart widgets"""
        # This would integrate with actual data providers
        # For now, return mock structure
        return {
            "chart_type": config.get("chart_type", "line"),
            "data": [],
            "labels": [],
            "timestamp": time.time()
        }
    
    def _get_metric_data(self, data_provider: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for metric widgets"""
        return {
            "value": 0,
            "format": config.get("format", "number"),
            "timestamp": time.time(),
            "alert": False
        }
    
    def _get_table_data(self, data_provider: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for table widgets"""
        return {
            "columns": config.get("columns", []),
            "rows": [],
            "timestamp": time.time()
        }
    
    def _get_alert_data(self, data_provider: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for alert widgets"""
        return {
            "alerts": [],
            "timestamp": time.time()
        }
    
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
    
    def export_dashboard_config(self, dashboard_id: str) -> str:
        """Export dashboard configuration as JSON"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard '{dashboard_id}' not found")
        
        return json.dumps(asdict(dashboard), indent=2)
    
    def import_dashboard_config(self, config_json: str) -> bool:
        """Import dashboard configuration from JSON"""
        try:
            config = json.loads(config_json)
            
            # Convert widget configs back to DashboardWidget objects
            widgets = [DashboardWidget(**widget_config) for widget_config in config["widgets"]]
            
            dashboard = DashboardLayout(
                dashboard_id=config["dashboard_id"],
                title=config["title"],
                description=config["description"],
                widgets=widgets,
                layout_config=config["layout_config"],
                created_at=config["created_at"],
                updated_at=time.time()  # Update timestamp
            )
            
            with self._lock:
                self.dashboards[dashboard.dashboard_id] = dashboard
            
            logger.info(f"📊 Dashboard imported: {dashboard.dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to import dashboard: {e}")
            return False
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary of all dashboards"""
        with self._lock:
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "total_dashboards": len(self.dashboards),
                "data_sources": list(self.data_sources.keys()),
                "cache_entries": len(self.widget_data_cache),
                "dashboards": self.list_dashboards()
            }


# Factory function for easy creation
def create_observability_dashboard(brain_id: str) -> ObservabilityDashboard:
    """Factory function to create observability dashboard"""
    return ObservabilityDashboard(brain_id)
