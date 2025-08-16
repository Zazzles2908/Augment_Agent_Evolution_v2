"""
Health Monitor - Component Health Tracking
Provides proactive health monitoring and recovery triggers for system components

This module implements comprehensive health monitoring to detect issues before
they become failures and trigger appropriate recovery mechanisms.

Created: 2025-07-29 AEST
Purpose: Component health tracking and recovery triggers
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration"""
    check_id: str
    component_name: str
    check_function: Callable
    interval_seconds: float
    timeout_seconds: float
    warning_threshold: float
    critical_threshold: float
    enabled: bool


@dataclass
class HealthResult:
    """Health check result"""
    check_id: str
    component_name: str
    status: HealthStatus
    value: float
    message: str
    timestamp: float
    duration: float


class HealthMonitor:
    """
    Health Monitor
    
    Provides proactive health monitoring for system components with
    configurable thresholds and automatic recovery triggering.
    """
    
    def __init__(self, brain_id: str):
        """Initialize health monitor"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, List[HealthResult]] = {}
        self.max_results_per_check = 100
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Health callbacks
        self.status_change_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        self.critical_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "healthy_checks": 0,
            "warning_checks": 0,
            "critical_checks": 0,
            "failed_checks": 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        logger.info(f"🏥 Health Monitor initialized for {brain_id}")
    
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        
        async def memory_usage_check():
            """Check memory usage"""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent
            except ImportError:
                return 0.0
        
        async def cpu_usage_check():
            """Check CPU usage"""
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=1)
                return cpu
            except ImportError:
                return 0.0
        
        async def disk_usage_check():
            """Check disk usage"""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100
            except ImportError:
                return 0.0
        
        # Register default checks
        self.register_health_check(HealthCheck(
            check_id="memory_usage",
            component_name="system",
            check_function=memory_usage_check,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            warning_threshold=80.0,
            critical_threshold=95.0,
            enabled=True
        ))
        
        self.register_health_check(HealthCheck(
            check_id="cpu_usage",
            component_name="system",
            check_function=cpu_usage_check,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            warning_threshold=80.0,
            critical_threshold=95.0,
            enabled=True
        ))
        
        self.register_health_check(HealthCheck(
            check_id="disk_usage",
            component_name="system",
            check_function=disk_usage_check,
            interval_seconds=60.0,
            timeout_seconds=5.0,
            warning_threshold=85.0,
            critical_threshold=95.0,
            enabled=True
        ))
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        with self._lock:
            self.health_checks[health_check.check_id] = health_check
            self.health_results[health_check.check_id] = []
        
        logger.info(f"🏥 Health check registered: {health_check.check_id}")
    
    def remove_health_check(self, check_id: str):
        """Remove a health check"""
        with self._lock:
            if check_id in self.health_checks:
                del self.health_checks[check_id]
                del self.health_results[check_id]
                logger.info(f"🏥 Health check removed: {check_id}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🏥 Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🏥 Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        last_check_times = {check_id: 0.0 for check_id in self.health_checks.keys()}
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check which health checks need to run
                for check_id, health_check in list(self.health_checks.items()):
                    if not health_check.enabled:
                        continue
                    
                    if current_time - last_check_times.get(check_id, 0) >= health_check.interval_seconds:
                        # Run health check
                        asyncio.create_task(self._run_health_check(health_check))
                        last_check_times[check_id] = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check"""
        start_time = time.time()
        
        try:
            # Execute health check with timeout
            value = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Determine status based on thresholds
            if value >= health_check.critical_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical: {value:.1f} >= {health_check.critical_threshold}"
            elif value >= health_check.warning_threshold:
                status = HealthStatus.WARNING
                message = f"Warning: {value:.1f} >= {health_check.warning_threshold}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Healthy: {value:.1f}"
            
            # Create result
            result = HealthResult(
                check_id=health_check.check_id,
                component_name=health_check.component_name,
                status=status,
                value=value,
                message=message,
                timestamp=time.time(),
                duration=duration
            )
            
            # Store result
            self._store_result(result)
            
            # Update statistics
            self._update_statistics(result)
            
            # Execute callbacks
            self._execute_callbacks(result)
            
        except asyncio.TimeoutError:
            result = HealthResult(
                check_id=health_check.check_id,
                component_name=health_check.component_name,
                status=HealthStatus.UNKNOWN,
                value=0.0,
                message="Health check timed out",
                timestamp=time.time(),
                duration=health_check.timeout_seconds
            )
            
            self._store_result(result)
            logger.warning(f"Health check {health_check.check_id} timed out")
            
        except Exception as e:
            result = HealthResult(
                check_id=health_check.check_id,
                component_name=health_check.component_name,
                status=HealthStatus.UNKNOWN,
                value=0.0,
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                duration=time.time() - start_time
            )
            
            self._store_result(result)
            logger.error(f"Health check {health_check.check_id} failed: {e}")
    
    def _store_result(self, result: HealthResult):
        """Store health check result"""
        with self._lock:
            if result.check_id not in self.health_results:
                self.health_results[result.check_id] = []
            
            self.health_results[result.check_id].append(result)
            
            # Limit results per check
            if len(self.health_results[result.check_id]) > self.max_results_per_check:
                self.health_results[result.check_id].pop(0)
    
    def _update_statistics(self, result: HealthResult):
        """Update health statistics"""
        with self._lock:
            self.stats["total_checks"] += 1
            
            if result.status == HealthStatus.HEALTHY:
                self.stats["healthy_checks"] += 1
            elif result.status == HealthStatus.WARNING:
                self.stats["warning_checks"] += 1
            elif result.status == HealthStatus.CRITICAL:
                self.stats["critical_checks"] += 1
            else:
                self.stats["failed_checks"] += 1
    
    def _execute_callbacks(self, result: HealthResult):
        """Execute health status callbacks"""
        try:
            # Status change callbacks
            for callback in self.status_change_callbacks:
                callback(result)
            
            # Specific status callbacks
            if result.status == HealthStatus.WARNING:
                for callback in self.warning_callbacks:
                    callback(result)
            elif result.status == HealthStatus.CRITICAL:
                for callback in self.critical_callbacks:
                    callback(result)
                    
        except Exception as e:
            logger.error(f"Health callback execution failed: {e}")
    
    def add_status_change_callback(self, callback: Callable):
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)
    
    def add_warning_callback(self, callback: Callable):
        """Add callback for warning status"""
        self.warning_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable):
        """Add callback for critical status"""
        self.critical_callbacks.append(callback)
    
    def get_current_health(self) -> Dict[str, HealthResult]:
        """Get current health status for all checks"""
        current_health = {}
        
        with self._lock:
            for check_id, results in self.health_results.items():
                if results:
                    current_health[check_id] = results[-1]  # Latest result
        
        return current_health
    
    def get_component_health(self, component_name: str) -> List[HealthResult]:
        """Get health results for a specific component"""
        component_results = []
        
        with self._lock:
            for check_id, results in self.health_results.items():
                if results and results[-1].component_name == component_name:
                    component_results.append(results[-1])
        
        return component_results
    
    def get_health_statistics(self) -> Dict[str, Any]:
        """Get health monitoring statistics"""
        with self._lock:
            current_health = self.get_current_health()
            
            overall_status = HealthStatus.HEALTHY
            if any(r.status == HealthStatus.CRITICAL for r in current_health.values()):
                overall_status = HealthStatus.CRITICAL
            elif any(r.status == HealthStatus.WARNING for r in current_health.values()):
                overall_status = HealthStatus.WARNING
            
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "monitoring_active": self.monitoring_active,
                "overall_status": overall_status.value,
                "registered_checks": len(self.health_checks),
                "statistics": self.stats.copy(),
                "current_health": {k: asdict(v) for k, v in current_health.items()}
            }


# Factory function for easy creation
def create_health_monitor(brain_id: str) -> HealthMonitor:
    """Factory function to create health monitor"""
    return HealthMonitor(brain_id)
