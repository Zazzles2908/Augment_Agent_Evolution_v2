#!/usr/bin/env python3
"""
Four-Brain AI System Flow Monitoring Utility
Comprehensive monitoring for inter-brain communication, tool interactions, and system flow

Created: 2025-07-13 20:50 AEST
Project: Augment Agent Evolution - Phase 7 Four-Brain Production Ready
Purpose: Complete flow monitoring and observability for Four-Brain AI System
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus client imports (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available - metrics will be disabled")
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    def start_http_server(*args, **kwargs): pass

import psutil

class BrainType(Enum):
    """Four-Brain system brain types"""
    BRAIN1_EMBEDDING = "brain1_embedding"
    BRAIN2_RERANKER = "brain2_reranker" 
    BRAIN3_AUGMENT = "brain3_augment"
    BRAIN4_DOCLING = "brain4_docling"

class ToolType(Enum):
    """Types of tools used in Four-Brain system"""
    MCP_TOOL = "mcp_tool"
    MODEL_TOOL = "model_tool"
    DATABASE_TOOL = "database_tool"
    EXTERNAL_API = "external_api"

class DatabaseType(Enum):
    """Database types in Four-Brain system"""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    VECTOR_DB = "vector_db"

@dataclass
class FlowMetrics:
    """Container for flow monitoring metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    processing_time: float = 0.0
    tool_calls: int = 0
    database_operations: int = 0
    data_size_bytes: int = 0

class FourBrainFlowMonitor:
    """
    Comprehensive flow monitoring for Four-Brain AI System
    
    Provides Prometheus metrics for:
    - Inter-brain communication tracking
    - Tool interaction monitoring  
    - Database operation tracking
    - Processing pipeline flow
    - Data transformation monitoring
    """
    
    def __init__(self, brain_id: str, enable_http_server: bool = False, port: int = 8000):
        self.brain_id = brain_id
        self.start_time = time.time()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Start HTTP server for metrics if requested
        if enable_http_server:
            start_http_server(port)
            logger.info(f"🚀 Flow monitoring HTTP server started on port {port}")
        
        logger.info(f"🧠 FourBrainFlowMonitor initialized for {brain_id}")
    
    def _init_prometheus_metrics(self):
        """Initialize all Prometheus metrics for comprehensive monitoring"""
        
        # Brain Message Tracking
        self.brain_messages_sent_total = Counter(
            'brain_messages_sent_total',
            'Total messages sent by brain',
            ['source_brain', 'target_brain', 'message_type', 'status']
        )
        
        self.brain_messages_received_total = Counter(
            'brain_messages_received_total', 
            'Total messages received by brain',
            ['source_brain', 'target_brain', 'message_type', 'status']
        )
        
        self.brain_message_processing_duration_seconds = Histogram(
            'brain_message_processing_duration_seconds',
            'Time spent processing inter-brain messages',
            ['source_brain', 'target_brain', 'message_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Data Transformation Tracking
        self.data_transformation_size_bytes = Histogram(
            'data_transformation_size_bytes',
            'Size of data being transformed between brains',
            ['source_brain', 'target_brain', 'transformation_type'],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600]  # 1KB to 100MB
        )
        
        # Tool Interaction Metrics
        self.brain_tool_calls_total = Counter(
            'brain_tool_calls_total',
            'Total tool calls made by brain',
            ['brain_id', 'tool_type', 'tool_name', 'status']
        )
        
        self.brain_tool_response_duration_seconds = Histogram(
            'brain_tool_response_duration_seconds',
            'Tool response time duration',
            ['brain_id', 'tool_type', 'tool_name', 'status'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # Database Operation Tracking
        self.brain_database_operations_total = Counter(
            'brain_database_operations_total',
            'Total database operations performed',
            ['brain_id', 'database_type', 'operation_type', 'status']
        )
        
        self.brain_database_query_duration_seconds = Histogram(
            'brain_database_query_duration_seconds',
            'Database query execution time',
            ['brain_id', 'database_type', 'operation_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Processing Pipeline Flow
        self.brain_processing_duration_seconds = Histogram(
            'brain_processing_duration_seconds',
            'Total processing time for brain operations',
            ['brain_id', 'operation_type', 'status'],
            buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.brain_pipeline_flow_status = Gauge(
            'brain_pipeline_flow_status',
            'Current status of brain processing pipeline (1=active, 0=idle)',
            ['brain_id', 'pipeline_stage']
        )
        
        # System Resource Metrics
        self.brain_memory_usage_bytes = Gauge(
            'brain_memory_usage_bytes',
            'Memory usage by brain process',
            ['brain_id', 'memory_type']
        )
        
        self.brain_cpu_usage_percent = Gauge(
            'brain_cpu_usage_percent',
            'CPU usage percentage by brain',
            ['brain_id']
        )
        
        # Queue and Communication Metrics
        self.brain_queue_size = Gauge(
            'brain_queue_size',
            'Number of items in brain processing queue',
            ['brain_id', 'queue_type']
        )
        
        self.brain_connection_status = Gauge(
            'brain_connection_status',
            'Connection status to external services (1=connected, 0=disconnected)',
            ['brain_id', 'service_type', 'service_name']
        )
        
        # Flow Summary Metrics
        self.brain_flow_summary = Info(
            'brain_flow_summary',
            'Summary information about brain flow monitoring',
            ['brain_id']
        )
        
        # Set initial brain info
        self.brain_flow_summary.labels(brain_id=self.brain_id).info({
            'brain_type': self.brain_id,
            'monitoring_version': '1.0.0',
            'start_time': str(self.start_time)
        })
    
    # Context Managers for Automatic Tracking
    @asynccontextmanager
    async def track_message_processing(self, source_brain: str, target_brain: str, message_type: str):
        """Context manager for tracking message processing time"""
        start_time = time.time()
        status = "success"
        
        try:
            # Set pipeline status to active
            self.brain_pipeline_flow_status.labels(
                brain_id=self.brain_id, 
                pipeline_stage="message_processing"
            ).set(1)
            
            yield
            
        except Exception as e:
            status = "error"
            logger.error(f"❌ Message processing error: {e}")
            raise
        finally:
            # Record processing duration
            duration = time.time() - start_time
            self.brain_message_processing_duration_seconds.labels(
                source_brain=source_brain,
                target_brain=target_brain,
                message_type=message_type
            ).observe(duration)
            
            # Set pipeline status to idle
            self.brain_pipeline_flow_status.labels(
                brain_id=self.brain_id,
                pipeline_stage="message_processing"
            ).set(0)
            
            logger.debug(f"📊 Message processing completed in {duration:.3f}s ({status})")
    
    @asynccontextmanager
    async def track_tool_call(self, tool_type: ToolType, tool_name: str):
        """Context manager for tracking tool call duration and status"""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            logger.error(f"❌ Tool call error for {tool_name}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            
            # Record tool call
            self.brain_tool_calls_total.labels(
                brain_id=self.brain_id,
                tool_type=tool_type.value,
                tool_name=tool_name,
                status=status
            ).inc()
            
            # Record response duration
            self.brain_tool_response_duration_seconds.labels(
                brain_id=self.brain_id,
                tool_type=tool_type.value,
                tool_name=tool_name,
                status=status
            ).observe(duration)
            
            logger.debug(f"🛠️ Tool call {tool_name} completed in {duration:.3f}s ({status})")
    
    @asynccontextmanager
    async def track_database_operation(self, db_type: DatabaseType, operation_type: str):
        """Context manager for tracking database operations"""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            logger.error(f"❌ Database operation error: {e}")
            raise
        finally:
            duration = time.time() - start_time
            
            # Record database operation
            self.brain_database_operations_total.labels(
                brain_id=self.brain_id,
                database_type=db_type.value,
                operation_type=operation_type,
                status=status
            ).inc()
            
            # Record query duration
            self.brain_database_query_duration_seconds.labels(
                brain_id=self.brain_id,
                database_type=db_type.value,
                operation_type=operation_type
            ).observe(duration)
            
            logger.debug(f"💾 Database {operation_type} completed in {duration:.3f}s ({status})")

    # Direct Tracking Methods
    def record_message_sent(self, target_brain: str, message_type: str, data_size: int = 0, status: str = "success"):
        """Record a message sent to another brain"""
        self.brain_messages_sent_total.labels(
            source_brain=self.brain_id,
            target_brain=target_brain,
            message_type=message_type,
            status=status
        ).inc()

        if data_size > 0:
            self.data_transformation_size_bytes.labels(
                source_brain=self.brain_id,
                target_brain=target_brain,
                transformation_type=message_type
            ).observe(data_size)

        logger.debug(f"📤 Message sent to {target_brain}: {message_type} ({data_size} bytes)")

    def record_message_received(self, source_brain: str, message_type: str, data_size: int = 0, status: str = "success"):
        """Record a message received from another brain"""
        self.brain_messages_received_total.labels(
            source_brain=source_brain,
            target_brain=self.brain_id,
            message_type=message_type,
            status=status
        ).inc()

        if data_size > 0:
            self.data_transformation_size_bytes.labels(
                source_brain=source_brain,
                target_brain=self.brain_id,
                transformation_type=message_type
            ).observe(data_size)

        logger.debug(f"📥 Message received from {source_brain}: {message_type} ({data_size} bytes)")

    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size metric"""
        self.brain_queue_size.labels(
            brain_id=self.brain_id,
            queue_type=queue_type
        ).set(size)

    def update_connection_status(self, service_type: str, service_name: str, connected: bool):
        """Update connection status to external services"""
        self.brain_connection_status.labels(
            brain_id=self.brain_id,
            service_type=service_type,
            service_name=service_name
        ).set(1 if connected else 0)

        status_text = "connected" if connected else "disconnected"
        logger.debug(f"🔗 {service_name} ({service_type}): {status_text}")

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # Memory metrics
            memory_info = psutil.virtual_memory()
            self.brain_memory_usage_bytes.labels(
                brain_id=self.brain_id,
                memory_type="virtual"
            ).set(memory_info.used)

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.brain_cpu_usage_percent.labels(brain_id=self.brain_id).set(cpu_percent)

        except Exception as e:
            logger.error(f"❌ Error updating system metrics: {e}")

    def get_flow_summary(self) -> Dict[str, Any]:
        """Get comprehensive flow monitoring summary"""
        uptime = time.time() - self.start_time

        return {
            "brain_id": self.brain_id,
            "uptime_seconds": uptime,
            "monitoring_active": True,
            "metrics_collected": [
                "brain_messages_sent_total",
                "brain_messages_received_total",
                "brain_message_processing_duration_seconds",
                "data_transformation_size_bytes",
                "brain_tool_calls_total",
                "brain_tool_response_duration_seconds",
                "brain_database_operations_total",
                "brain_database_query_duration_seconds",
                "brain_processing_duration_seconds",
                "brain_pipeline_flow_status",
                "brain_memory_usage_bytes",
                "brain_cpu_usage_percent",
                "brain_queue_size",
                "brain_connection_status"
            ]
        }

# Global flow monitor instance
_global_flow_monitor: Optional[FourBrainFlowMonitor] = None

def get_flow_monitor(brain_id: str = None) -> FourBrainFlowMonitor:
    """Get or create global flow monitor instance"""
    global _global_flow_monitor

    if _global_flow_monitor is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_flow_monitor = FourBrainFlowMonitor(brain_id)

    return _global_flow_monitor

def initialize_flow_monitoring(brain_id: str, enable_http_server: bool = False, port: int = 8000) -> FourBrainFlowMonitor:
    """Initialize flow monitoring for a brain"""
    global _global_flow_monitor
    _global_flow_monitor = FourBrainFlowMonitor(brain_id, enable_http_server, port)
    return _global_flow_monitor

# Convenience functions for easy integration
async def track_message_flow(source_brain: str, target_brain: str, message_type: str, operation_func):
    """Convenience function to track message flow"""
    monitor = get_flow_monitor()
    async with monitor.track_message_processing(source_brain, target_brain, message_type):
        return await operation_func()

async def track_tool_usage(tool_type: ToolType, tool_name: str, operation_func):
    """Convenience function to track tool usage"""
    monitor = get_flow_monitor()
    async with monitor.track_tool_call(tool_type, tool_name):
        return await operation_func()

async def track_db_operation(db_type: DatabaseType, operation_type: str, operation_func):
    """Convenience function to track database operations"""
    monitor = get_flow_monitor()
    async with monitor.track_database_operation(db_type, operation_type):
        return await operation_func()

if __name__ == "__main__":
    # Example usage and testing
    async def test_flow_monitoring():
        """Test the flow monitoring system"""
        print("🧪 Testing Four-Brain Flow Monitoring System...")

        # Initialize monitoring
        monitor = initialize_flow_monitoring("test_brain", enable_http_server=True, port=8001)

        # Test message tracking
        monitor.record_message_sent("brain2_reranker", "embedding_request", 1024)
        monitor.record_message_received("brain1_embedding", "embedding_response", 2048)

        # Test tool tracking
        async with monitor.track_tool_call(ToolType.MCP_TOOL, "prometheus_query"):
            await asyncio.sleep(0.1)  # Simulate tool call

        # Test database tracking
        async with monitor.track_database_operation(DatabaseType.REDIS, "set"):
            await asyncio.sleep(0.05)  # Simulate database operation

        # Update system metrics
        monitor.update_system_metrics()
        monitor.update_connection_status("database", "redis", True)
        monitor.update_queue_size("processing", 5)

        # Get summary
        summary = monitor.get_flow_summary()
        print(f"📊 Flow monitoring summary: {json.dumps(summary, indent=2)}")

        print("✅ Flow monitoring test completed!")
        print("🌐 Metrics available at: http://localhost:8001/metrics")

    # Run test
    asyncio.run(test_flow_monitoring())
