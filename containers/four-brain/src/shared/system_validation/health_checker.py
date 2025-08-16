"""
Health Checker for Four-Brain System v2
Comprehensive connection health monitoring and validation

Created: 2025-07-30 AEST
Purpose: Monitor and validate health of all system connections and dependencies
"""

import asyncio
import json
import logging
import time
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import aiohttp
import psycopg2
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Types of system components"""
    DATABASE = "database"
    CACHE = "cache"
    API_SERVICE = "api_service"
    AI_BRAIN = "ai_brain"
    CONTAINER = "container"
    NETWORK = "network"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class HealthCheck:
    """Individual health check result"""
    check_id: str
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: float
    error_message: Optional[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    last_healthy: Optional[datetime]
    consecutive_failures: int

@dataclass
class SystemHealth:
    """Overall system health status"""
    system_id: str
    overall_status: HealthStatus
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    critical_components: int
    checks: List[HealthCheck]
    system_metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class HealthChecker:
    """
    Comprehensive health monitoring system
    
    Features:
    - Multi-component health monitoring
    - Real-time connection testing
    - Performance metrics collection
    - Failure detection and alerting
    - Health trend analysis
    - Automatic recovery recommendations
    - Comprehensive health reporting
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'check_interval_seconds': 30,
            'timeout_seconds': 10,
            'max_consecutive_failures': 3,
            'degraded_threshold_ms': 1000,
            'unhealthy_threshold_ms': 5000,
            'critical_failure_count': 5,
            'health_history_hours': 24
        }
        
        # Component definitions
        self.components = {
            'redis': {
                'type': ComponentType.CACHE,
                'connection_string': 'redis://localhost:6379',
                'health_check': self._check_redis_health
            },
            'postgresql': {
                'type': ComponentType.DATABASE,
                'connection_string': 'postgresql://localhost:5432',
                'health_check': self._check_postgresql_health
            },
            'brain1': {
                'type': ComponentType.AI_BRAIN,
                'endpoint': 'http://localhost:8001/health',
                'health_check': self._check_brain_health
            },
            'brain2': {
                'type': ComponentType.AI_BRAIN,
                'endpoint': 'http://localhost:8002/health',
                'health_check': self._check_brain_health
            },
            'brain3': {
                'type': ComponentType.AI_BRAIN,
                'endpoint': 'http://localhost:8003/health',
                'health_check': self._check_brain_health
            },
            'brain4': {
                'type': ComponentType.AI_BRAIN,
                'endpoint': 'http://localhost:8004/health',
                'health_check': self._check_brain_health
            },
            'k2_hub': {
                'type': ComponentType.API_SERVICE,
                'endpoint': 'http://localhost:9098/health',
                'health_check': self._check_api_health
            },
            'supabase': {
                'type': ComponentType.EXTERNAL_SERVICE,
                'endpoint': 'https://ustcfwmonegxeoqeixgg.supabase.co/rest/v1/',
                'health_check': self._check_supabase_health
            }
        }
        
        # Health state
        self.health_history: Dict[str, List[HealthCheck]] = {}
        self.current_health: Dict[str, HealthCheck] = {}
        self.system_health_history: List[SystemHealth] = []
        
        # Performance metrics
        self.metrics = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'average_response_time': 0.0,
            'uptime_percentage': 0.0,
            'critical_alerts': 0,
            'recovery_events': 0
        }
        
        logger.info("🏥 Health Checker initialized")
    
    async def initialize(self):
        """Initialize Redis connection and health monitoring services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load health history
            await self._load_health_history()
            
            # Start background health monitoring
            asyncio.create_task(self._continuous_health_monitoring())
            asyncio.create_task(self._health_analysis())
            
            logger.info("✅ Health Checker Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Health Checker: {e}")
            raise
    
    async def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        try:
            start_time = time.time()
            
            # Generate system ID
            system_id = f"health_{int(time.time() * 1000)}"
            
            # Perform health checks for all components
            health_checks = []
            
            for component_name, component_config in self.components.items():
                health_check = await self._perform_component_health_check(component_name, component_config)
                health_checks.append(health_check)
                
                # Update current health
                self.current_health[component_name] = health_check
                
                # Update health history
                if component_name not in self.health_history:
                    self.health_history[component_name] = []
                self.health_history[component_name].append(health_check)
                
                # Trim history
                cutoff_time = datetime.now() - timedelta(hours=self.config['health_history_hours'])
                self.health_history[component_name] = [
                    check for check in self.health_history[component_name]
                    if check.timestamp > cutoff_time
                ]
            
            # Calculate overall system health
            overall_status = self._calculate_overall_health_status(health_checks)
            
            # Count components by status
            status_counts = self._count_components_by_status(health_checks)
            
            # Collect system metrics
            system_metrics = await self._collect_system_metrics()
            
            # Generate recommendations
            recommendations = await self._generate_health_recommendations(health_checks)
            
            # Create system health report
            system_health = SystemHealth(
                system_id=system_id,
                overall_status=overall_status,
                total_components=len(health_checks),
                healthy_components=status_counts['healthy'],
                degraded_components=status_counts['degraded'],
                unhealthy_components=status_counts['unhealthy'],
                critical_components=status_counts['critical'],
                checks=health_checks,
                system_metrics=system_metrics,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store system health
            self.system_health_history.append(system_health)
            await self._store_system_health(system_health)
            
            # Update metrics
            self.metrics['total_checks'] += len(health_checks)
            self.metrics['successful_checks'] += sum(1 for check in health_checks if check.status == HealthStatus.HEALTHY)
            self.metrics['failed_checks'] += sum(1 for check in health_checks if check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL])
            self._update_average_response_time(health_checks)
            self._update_uptime_percentage()
            
            processing_time = time.time() - start_time
            logger.info(f"✅ System health check completed in {processing_time:.2f}s: {overall_status.value}")
            
            return system_health
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            raise
    
    async def _perform_component_health_check(self, component_name: str, component_config: Dict[str, Any]) -> HealthCheck:
        """Perform health check for individual component"""
        try:
            start_time = time.time()
            
            # Get previous health check for failure tracking
            previous_check = self.current_health.get(component_name)
            consecutive_failures = previous_check.consecutive_failures if previous_check and previous_check.status != HealthStatus.HEALTHY else 0
            
            # Perform component-specific health check
            health_check_func = component_config['health_check']
            status, error_message, metadata = await health_check_func(component_name, component_config)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine health status based on response time and errors
            if error_message:
                if consecutive_failures >= self.config['critical_failure_count']:
                    status = HealthStatus.CRITICAL
                else:
                    status = HealthStatus.UNHEALTHY
                consecutive_failures += 1
            elif response_time_ms > self.config['unhealthy_threshold_ms']:
                status = HealthStatus.UNHEALTHY
                consecutive_failures += 1
            elif response_time_ms > self.config['degraded_threshold_ms']:
                status = HealthStatus.DEGRADED
                consecutive_failures = 0
            else:
                status = HealthStatus.HEALTHY
                consecutive_failures = 0
            
            # Determine last healthy time
            last_healthy = None
            if status == HealthStatus.HEALTHY:
                last_healthy = datetime.now()
            elif previous_check and previous_check.last_healthy:
                last_healthy = previous_check.last_healthy
            
            health_check = HealthCheck(
                check_id=f"{component_name}_{int(time.time() * 1000)}",
                component_name=component_name,
                component_type=component_config['type'],
                status=status,
                response_time_ms=response_time_ms,
                error_message=error_message,
                metadata=metadata,
                timestamp=datetime.now(),
                last_healthy=last_healthy,
                consecutive_failures=consecutive_failures
            )
            
            return health_check
            
        except Exception as e:
            logger.error(f"❌ Component health check failed for {component_name}: {e}")
            
            # Return critical health check on exception
            return HealthCheck(
                check_id=f"{component_name}_{int(time.time() * 1000)}",
                component_name=component_name,
                component_type=component_config['type'],
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                error_message=str(e),
                metadata={},
                timestamp=datetime.now(),
                last_healthy=None,
                consecutive_failures=consecutive_failures + 1 if previous_check else 1
            )
    
    async def _check_redis_health(self, component_name: str, config: Dict[str, Any]) -> Tuple[HealthStatus, Optional[str], Dict[str, Any]]:
        """Check Redis health"""
        try:
            # Test Redis connection
            redis_client = aioredis.from_url(config['connection_string'])
            
            # Ping test
            await redis_client.ping()
            
            # Memory usage
            info = await redis_client.info('memory')
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            # Connection count
            clients_info = await redis_client.info('clients')
            connected_clients = clients_info.get('connected_clients', 0)
            
            await redis_client.close()
            
            metadata = {
                'used_memory_mb': used_memory / (1024 * 1024),
                'max_memory_mb': max_memory / (1024 * 1024) if max_memory > 0 else 'unlimited',
                'connected_clients': connected_clients,
                'memory_usage_percent': (used_memory / max_memory * 100) if max_memory > 0 else 0
            }
            
            return HealthStatus.HEALTHY, None, metadata
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e), {}
    
    async def _check_postgresql_health(self, component_name: str, config: Dict[str, Any]) -> Tuple[HealthStatus, Optional[str], Dict[str, Any]]:
        """Check PostgreSQL health"""
        try:
            # Test PostgreSQL connection
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='postgres',
                user='postgres',
                password='password',
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            
            # Test query
            cursor.execute('SELECT version();')
            version = cursor.fetchone()[0]
            
            # Get connection count
            cursor.execute('SELECT count(*) FROM pg_stat_activity;')
            connection_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            metadata = {
                'version': version,
                'connection_count': connection_count
            }
            
            return HealthStatus.HEALTHY, None, metadata
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e), {}
    
    async def _check_brain_health(self, component_name: str, config: Dict[str, Any]) -> Tuple[HealthStatus, Optional[str], Dict[str, Any]]:
        """Check AI Brain health"""
        try:
            endpoint = config['endpoint']
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['timeout_seconds'])) as session:
                async with session.get(endpoint) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        metadata = {
                            'status_code': response.status,
                            'health_data': health_data
                        }
                        
                        return HealthStatus.HEALTHY, None, metadata
                    else:
                        return HealthStatus.UNHEALTHY, f"HTTP {response.status}", {'status_code': response.status}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e), {}
    
    async def _check_api_health(self, component_name: str, config: Dict[str, Any]) -> Tuple[HealthStatus, Optional[str], Dict[str, Any]]:
        """Check API service health"""
        try:
            endpoint = config['endpoint']
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['timeout_seconds'])) as session:
                async with session.get(endpoint) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        metadata = {
                            'status_code': response.status,
                            'health_data': health_data
                        }
                        
                        return HealthStatus.HEALTHY, None, metadata
                    else:
                        return HealthStatus.DEGRADED, f"HTTP {response.status}", {'status_code': response.status}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e), {}
    
    async def _check_supabase_health(self, component_name: str, config: Dict[str, Any]) -> Tuple[HealthStatus, Optional[str], Dict[str, Any]]:
        """Check Supabase health"""
        try:
            endpoint = config['endpoint']
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['timeout_seconds'])) as session:
                async with session.get(endpoint) as response:
                    if response.status in [200, 401]:  # 401 is expected without auth
                        metadata = {
                            'status_code': response.status,
                            'service': 'supabase'
                        }
                        
                        return HealthStatus.HEALTHY, None, metadata
                    else:
                        return HealthStatus.DEGRADED, f"HTTP {response.status}", {'status_code': response.status}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e), {}
    
    def _calculate_overall_health_status(self, health_checks: List[HealthCheck]) -> HealthStatus:
        """Calculate overall system health status"""
        if not health_checks:
            return HealthStatus.UNKNOWN
        
        # Count statuses
        critical_count = sum(1 for check in health_checks if check.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for check in health_checks if check.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for check in health_checks if check.status == HealthStatus.DEGRADED)
        healthy_count = sum(1 for check in health_checks if check.status == HealthStatus.HEALTHY)
        
        # Determine overall status
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif unhealthy_count > len(health_checks) * 0.3:  # More than 30% unhealthy
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > len(health_checks) * 0.5:  # Any unhealthy or >50% degraded
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _count_components_by_status(self, health_checks: List[HealthCheck]) -> Dict[str, int]:
        """Count components by health status"""
        counts = {
            'healthy': 0,
            'degraded': 0,
            'unhealthy': 0,
            'critical': 0,
            'unknown': 0
        }
        
        for check in health_checks:
            if check.status == HealthStatus.HEALTHY:
                counts['healthy'] += 1
            elif check.status == HealthStatus.DEGRADED:
                counts['degraded'] += 1
            elif check.status == HealthStatus.UNHEALTHY:
                counts['unhealthy'] += 1
            elif check.status == HealthStatus.CRITICAL:
                counts['critical'] += 1
            else:
                counts['unknown'] += 1
        
        return counts
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
            return {}
    
    async def _generate_health_recommendations(self, health_checks: List[HealthCheck]) -> List[str]:
        """Generate health recommendations based on check results"""
        recommendations = []
        
        for check in health_checks:
            if check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                if check.component_type == ComponentType.AI_BRAIN:
                    recommendations.append(f"Check {check.component_name} container and restart if necessary")
                elif check.component_type == ComponentType.DATABASE:
                    recommendations.append(f"Verify {check.component_name} connection and configuration")
                elif check.component_type == ComponentType.CACHE:
                    recommendations.append(f"Check {check.component_name} memory usage and connections")
                elif check.component_type == ComponentType.API_SERVICE:
                    recommendations.append(f"Restart {check.component_name} service")
                
                if check.consecutive_failures >= self.config['max_consecutive_failures']:
                    recommendations.append(f"URGENT: {check.component_name} has failed {check.consecutive_failures} consecutive times")
            
            elif check.status == HealthStatus.DEGRADED:
                if check.response_time_ms > self.config['degraded_threshold_ms']:
                    recommendations.append(f"Optimize {check.component_name} performance - response time: {check.response_time_ms:.0f}ms")
        
        # Remove duplicates
        return list(dict.fromkeys(recommendations))
    
    async def _continuous_health_monitoring(self):
        """Continuous background health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config['check_interval_seconds'])
                
                # Perform health check
                await self.check_system_health()
                
            except Exception as e:
                logger.error(f"❌ Continuous health monitoring error: {e}")
    
    async def _health_analysis(self):
        """Analyze health trends and patterns"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze health trends
                # Detect patterns
                # Generate alerts
                
            except Exception as e:
                logger.error(f"❌ Health analysis error: {e}")
    
    async def _store_system_health(self, system_health: SystemHealth):
        """Store system health in Redis"""
        if self.redis_client:
            try:
                key = f"system_health:{system_health.system_id}"
                data = json.dumps(asdict(system_health), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store system health: {e}")
    
    async def _load_health_history(self):
        """Load health history from Redis"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("system_health:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        # Would deserialize SystemHealth
                        pass
        except Exception as e:
            logger.error(f"Failed to load health history: {e}")
    
    def _update_average_response_time(self, health_checks: List[HealthCheck]):
        """Update average response time metric"""
        if health_checks:
            avg_response_time = sum(check.response_time_ms for check in health_checks) / len(health_checks)
            
            if self.metrics['total_checks'] == len(health_checks):  # First check
                self.metrics['average_response_time'] = avg_response_time
            else:
                alpha = 0.1
                self.metrics['average_response_time'] = (
                    alpha * avg_response_time + 
                    (1 - alpha) * self.metrics['average_response_time']
                )
    
    def _update_uptime_percentage(self):
        """Update uptime percentage metric"""
        if self.metrics['total_checks'] > 0:
            self.metrics['uptime_percentage'] = (
                self.metrics['successful_checks'] / self.metrics['total_checks']
            ) * 100
    
    async def get_component_health(self, component_name: str) -> Optional[HealthCheck]:
        """Get current health of specific component"""
        return self.current_health.get(component_name)
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics"""
        return {
            'metrics': self.metrics.copy(),
            'current_health': {name: check.status.value for name, check in self.current_health.items()},
            'health_history_size': sum(len(history) for history in self.health_history.values()),
            'system_health_history_size': len(self.system_health_history),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global health checker instance
health_checker = HealthChecker()

async def initialize_health_checker():
    """Initialize the global health checker"""
    await health_checker.initialize()

if __name__ == "__main__":
    # Test the health checker
    async def test_health_checker():
        await initialize_health_checker()
        
        # Perform system health check
        system_health = await health_checker.check_system_health()
        
        print(f"System Health: {system_health.overall_status.value}")
        print(f"Healthy Components: {system_health.healthy_components}/{system_health.total_components}")
        print(f"Recommendations: {len(system_health.recommendations)}")
        
        for check in system_health.checks:
            print(f"- {check.component_name}: {check.status.value} ({check.response_time_ms:.0f}ms)")
        
        # Get metrics
        metrics = await health_checker.get_health_metrics()
        print(f"Health metrics: {metrics}")
    
    asyncio.run(test_health_checker())
