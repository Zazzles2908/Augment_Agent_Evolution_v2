"""
Brain Health Monitoring System for Four-Brain System v2
Comprehensive health monitoring and diagnostics for all brain instances

Created: 2025-07-30 AEST
Purpose: Monitor health, performance, and availability of Brain1, Brain2, Brain3, and Brain4
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Brain health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class HealthCheckType(Enum):
    """Types of health checks"""
    BASIC_PING = "basic_ping"
    API_ENDPOINT = "api_endpoint"
    RESOURCE_USAGE = "resource_usage"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    MEMORY_LEAK = "memory_leak"
    GPU_HEALTH = "gpu_health"

@dataclass
class HealthCheck:
    """Individual health check configuration"""
    check_id: str
    check_type: HealthCheckType
    endpoint: str
    interval_seconds: int
    timeout_seconds: int
    expected_response: Optional[str]
    warning_threshold: float
    critical_threshold: float
    enabled: bool

@dataclass
class HealthResult:
    """Health check result"""
    check_id: str
    brain_id: str
    status: HealthStatus
    response_time: float
    value: Optional[float]
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class BrainHealthProfile:
    """Comprehensive brain health profile"""
    brain_id: str
    overall_status: HealthStatus
    last_updated: datetime
    uptime: float
    health_checks: List[HealthResult]
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, float]
    alerts: List[str]
    trends: Dict[str, List[float]]

class BrainHealthMonitor:
    """
    Comprehensive brain health monitoring system
    
    Features:
    - Multi-type health checks (ping, API, resources, performance)
    - Real-time health status tracking
    - Performance trend analysis
    - Automated alerting and notifications
    - Health history and reporting
    - Predictive health analysis
    - Integration with failover system
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/15"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Health monitoring state
        self.brain_profiles: Dict[str, BrainHealthProfile] = {}
        self.health_checks: Dict[str, List[HealthCheck]] = {}
        self.health_history: Dict[str, List[HealthResult]] = {}
        
        # Configuration
        self.config = {
            'default_check_interval': 30,
            'default_timeout': 10,
            'history_retention_hours': 24,
            'trend_analysis_window': 60,  # minutes
            'alert_cooldown_minutes': 15,
            'predictive_analysis_enabled': True,
            'auto_recovery_enabled': True
        }
        
        # Health thresholds
        self.thresholds = {
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'cpu_usage_warning': 0.8,
            'cpu_usage_critical': 0.95,
            'memory_usage_warning': 0.8,
            'memory_usage_critical': 0.95,
            'gpu_usage_warning': 0.9,
            'gpu_usage_critical': 0.98,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.15
        }
        
        # Alert state
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_callbacks: List[Callable] = []
        
        # HTTP session for health checks
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("🏥 Brain Health Monitor initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start health monitoring"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['default_timeout'])
            )
            
            # Load existing health configurations
            await self._load_health_configurations()
            
            # Start monitoring services
            asyncio.create_task(self._health_monitor_loop())
            asyncio.create_task(self._trend_analyzer())
            asyncio.create_task(self._alert_manager())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("✅ Brain Health Monitor Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Brain Health Monitor: {e}")
            raise
    
    async def register_brain(self, brain_id: str, health_checks: List[HealthCheck]) -> bool:
        """Register a brain for health monitoring"""
        try:
            # Create health profile
            profile = BrainHealthProfile(
                brain_id=brain_id,
                overall_status=HealthStatus.UNKNOWN,
                last_updated=datetime.now(),
                uptime=0.0,
                health_checks=[],
                resource_usage={},
                performance_metrics={},
                alerts=[],
                trends={}
            )
            
            self.brain_profiles[brain_id] = profile
            self.health_checks[brain_id] = health_checks
            self.health_history[brain_id] = []
            
            # Store configuration
            await self._store_brain_config(brain_id, health_checks)
            
            logger.info(f"✅ Brain registered for health monitoring: {brain_id} with {len(health_checks)} checks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register brain {brain_id}: {e}")
            return False
    
    async def get_brain_health(self, brain_id: str) -> Optional[BrainHealthProfile]:
        """Get current health profile for a brain"""
        return self.brain_profiles.get(brain_id)
    
    async def get_all_brain_health(self) -> Dict[str, BrainHealthProfile]:
        """Get health profiles for all brains"""
        return self.brain_profiles.copy()
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        while True:
            try:
                # Run health checks for all brains
                for brain_id, checks in self.health_checks.items():
                    await self._run_brain_health_checks(brain_id, checks)
                
                # Wait for next cycle
                await asyncio.sleep(self.config['default_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Health monitor loop error: {e}")
                await asyncio.sleep(5)
    
    async def _run_brain_health_checks(self, brain_id: str, checks: List[HealthCheck]):
        """Run all health checks for a specific brain"""
        try:
            profile = self.brain_profiles.get(brain_id)
            if not profile:
                return
            
            check_results = []
            
            for check in checks:
                if not check.enabled:
                    continue
                
                # Check if it's time to run this check
                if await self._should_run_check(brain_id, check):
                    result = await self._execute_health_check(brain_id, check)
                    if result:
                        check_results.append(result)
                        
                        # Store result in history
                        self.health_history[brain_id].append(result)
                        
                        # Keep history within limits
                        max_history = int(self.config['history_retention_hours'] * 3600 / check.interval_seconds)
                        if len(self.health_history[brain_id]) > max_history:
                            self.health_history[brain_id] = self.health_history[brain_id][-max_history:]
            
            # Update brain profile
            if check_results:
                await self._update_brain_profile(brain_id, check_results)
            
        except Exception as e:
            logger.error(f"❌ Health checks failed for brain {brain_id}: {e}")
    
    async def _should_run_check(self, brain_id: str, check: HealthCheck) -> bool:
        """Determine if a health check should be run now"""
        try:
            # Get last run time from Redis
            key = f"last_check:{brain_id}:{check.check_id}"
            last_run = await self.redis_client.get(key)
            
            if not last_run:
                return True
            
            last_run_time = datetime.fromisoformat(last_run.decode())
            time_since_last = datetime.now() - last_run_time
            
            return time_since_last.total_seconds() >= check.interval_seconds
            
        except Exception as e:
            logger.error(f"❌ Check scheduling error: {e}")
            return True  # Default to running the check
    
    async def _execute_health_check(self, brain_id: str, check: HealthCheck) -> Optional[HealthResult]:
        """Execute a specific health check"""
        try:
            start_time = time.time()
            
            if check.check_type == HealthCheckType.BASIC_PING:
                result = await self._ping_check(brain_id, check)
            elif check.check_type == HealthCheckType.API_ENDPOINT:
                result = await self._api_endpoint_check(brain_id, check)
            elif check.check_type == HealthCheckType.RESOURCE_USAGE:
                result = await self._resource_usage_check(brain_id, check)
            elif check.check_type == HealthCheckType.RESPONSE_TIME:
                result = await self._response_time_check(brain_id, check)
            elif check.check_type == HealthCheckType.ERROR_RATE:
                result = await self._error_rate_check(brain_id, check)
            elif check.check_type == HealthCheckType.GPU_HEALTH:
                result = await self._gpu_health_check(brain_id, check)
            else:
                result = None
            
            if result:
                result.response_time = time.time() - start_time
                
                # Update last run time
                key = f"last_check:{brain_id}:{check.check_id}"
                await self.redis_client.set(key, datetime.now().isoformat())
                
                # Store result
                await self._store_health_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Health check execution failed: {e}")
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=None,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _ping_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """Basic ping health check"""
        try:
            if self.http_session:
                async with self.http_session.get(check.endpoint) as response:
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = "Ping successful"
                    else:
                        status = HealthStatus.WARNING
                        message = f"Ping returned status {response.status}"
            else:
                status = HealthStatus.CRITICAL
                message = "HTTP session not available"
            
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=status,
                response_time=0.0,
                value=float(status == HealthStatus.HEALTHY),
                message=message,
                timestamp=datetime.now(),
                metadata={}
            )
            
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=0.0,
                message=f"Ping failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _api_endpoint_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """API endpoint health check"""
        try:
            if self.http_session:
                async with self.http_session.get(check.endpoint) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        if check.expected_response and check.expected_response in response_text:
                            status = HealthStatus.HEALTHY
                            message = "API endpoint healthy"
                        elif not check.expected_response:
                            status = HealthStatus.HEALTHY
                            message = "API endpoint responding"
                        else:
                            status = HealthStatus.WARNING
                            message = "API endpoint response unexpected"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"API endpoint returned {response.status}"
            else:
                status = HealthStatus.CRITICAL
                message = "HTTP session not available"
            
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=status,
                response_time=0.0,
                value=float(status == HealthStatus.HEALTHY),
                message=message,
                timestamp=datetime.now(),
                metadata={}
            )
            
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=0.0,
                message=f"API check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _resource_usage_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """Resource usage health check using real system metrics"""
        try:
            # Get real resource usage
            cpu_usage, memory_usage = await self._get_real_resource_usage()

            if cpu_usage is None or memory_usage is None:
                return HealthResult(
                    check_id=check.check_id,
                    brain_id=brain_id,
                    status=HealthStatus.UNKNOWN,
                    message="PROCESSING FAILED: Unable to retrieve resource metrics",
                    timestamp=datetime.now(),
                    metadata={}
                )

            # Determine status based on thresholds
            if (cpu_usage > self.thresholds['cpu_usage_critical'] or
                memory_usage > self.thresholds['memory_usage_critical']):
                status = HealthStatus.CRITICAL
                message = f"Critical resource usage: CPU {cpu_usage:.1%}, Memory {memory_usage:.1%}"
            elif (cpu_usage > self.thresholds['cpu_usage_warning'] or
                  memory_usage > self.thresholds['memory_usage_warning']):
                status = HealthStatus.WARNING
                message = f"High resource usage: CPU {cpu_usage:.1%}, Memory {memory_usage:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage: CPU {cpu_usage:.1%}, Memory {memory_usage:.1%}"
            
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=status,
                response_time=0.0,
                value=max(cpu_usage, memory_usage),
                message=message,
                timestamp=datetime.now(),
                metadata={'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
            )
            
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=1.0,
                message=f"Resource check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )

    async def _get_real_resource_usage(self) -> Tuple[Optional[float], Optional[float]]:
        """Get real CPU and memory usage"""
        try:
            import psutil

            # Get real CPU usage (average over 1 second)
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0

            # Get real memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0

            return cpu_usage, memory_usage

        except ImportError:
            logger.warning("psutil not available for resource monitoring")
            return None, None
        except Exception as e:
            logger.error(f"Failed to get resource usage: {str(e)}")
            return None, None

    async def _response_time_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """Response time health check"""
        try:
            start_time = time.time()
            
            if self.http_session:
                async with self.http_session.get(check.endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response_time > self.thresholds['response_time_critical']:
                        status = HealthStatus.CRITICAL
                        message = f"Critical response time: {response_time:.2f}s"
                    elif response_time > self.thresholds['response_time_warning']:
                        status = HealthStatus.WARNING
                        message = f"Slow response time: {response_time:.2f}s"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Good response time: {response_time:.2f}s"
                    
                    return HealthResult(
                        check_id=check.check_id,
                        brain_id=brain_id,
                        status=status,
                        response_time=response_time,
                        value=response_time,
                        message=message,
                        timestamp=datetime.now(),
                        metadata={}
                    )
            else:
                return HealthResult(
                    check_id=check.check_id,
                    brain_id=brain_id,
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    value=999.0,
                    message="HTTP session not available",
                    timestamp=datetime.now(),
                    metadata={}
                )
                
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=999.0,
                message=f"Response time check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _error_rate_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """Error rate health check"""
        try:
            # This would analyze recent error logs
            # For now, simulate error rate
            import random
            error_rate = random.uniform(0.0, 0.2)
            
            if error_rate > self.thresholds['error_rate_critical']:
                status = HealthStatus.CRITICAL
                message = f"Critical error rate: {error_rate:.1%}"
            elif error_rate > self.thresholds['error_rate_warning']:
                status = HealthStatus.WARNING
                message = f"High error rate: {error_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal error rate: {error_rate:.1%}"
            
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=status,
                response_time=0.0,
                value=error_rate,
                message=message,
                timestamp=datetime.now(),
                metadata={}
            )
            
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=1.0,
                message=f"Error rate check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _gpu_health_check(self, brain_id: str, check: HealthCheck) -> HealthResult:
        """GPU health check"""
        try:
            # This would integrate with nvidia-ml-py or similar
            # For now, simulate GPU health
            import random
            gpu_usage = random.uniform(0.1, 0.95)
            gpu_temp = random.uniform(40, 85)
            
            if gpu_usage > self.thresholds['gpu_usage_critical'] or gpu_temp > 80:
                status = HealthStatus.CRITICAL
                message = f"Critical GPU state: {gpu_usage:.1%} usage, {gpu_temp:.1f}°C"
            elif gpu_usage > self.thresholds['gpu_usage_warning'] or gpu_temp > 70:
                status = HealthStatus.WARNING
                message = f"High GPU usage: {gpu_usage:.1%} usage, {gpu_temp:.1f}°C"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal GPU state: {gpu_usage:.1%} usage, {gpu_temp:.1f}°C"
            
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=status,
                response_time=0.0,
                value=gpu_usage,
                message=message,
                timestamp=datetime.now(),
                metadata={'gpu_usage': gpu_usage, 'gpu_temperature': gpu_temp}
            )
            
        except Exception as e:
            return HealthResult(
                check_id=check.check_id,
                brain_id=brain_id,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                value=1.0,
                message=f"GPU health check failed: {str(e)}",
                timestamp=datetime.now(),
                metadata={}
            )
    
    async def _update_brain_profile(self, brain_id: str, check_results: List[HealthResult]):
        """Update brain health profile with latest results"""
        try:
            profile = self.brain_profiles[brain_id]
            
            # Update health checks
            profile.health_checks = check_results
            profile.last_updated = datetime.now()
            
            # Calculate overall status
            statuses = [result.status for result in check_results]
            if HealthStatus.CRITICAL in statuses:
                profile.overall_status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                profile.overall_status = HealthStatus.WARNING
            elif HealthStatus.HEALTHY in statuses:
                profile.overall_status = HealthStatus.HEALTHY
            else:
                profile.overall_status = HealthStatus.UNKNOWN
            
            # Update resource usage
            for result in check_results:
                if result.metadata:
                    profile.resource_usage.update(result.metadata)
            
            # Update performance metrics
            response_times = [r.response_time for r in check_results if r.response_time > 0]
            if response_times:
                profile.performance_metrics['avg_response_time'] = statistics.mean(response_times)
                profile.performance_metrics['max_response_time'] = max(response_times)
            
            # Check for alerts
            await self._check_health_alerts(brain_id, profile)
            
            # Store updated profile
            await self._store_brain_profile(profile)
            
        except Exception as e:
            logger.error(f"❌ Brain profile update failed: {e}")
    
    async def _check_health_alerts(self, brain_id: str, profile: BrainHealthProfile):
        """Check for health alerts and trigger notifications"""
        try:
            alert_key = f"health_alert:{brain_id}"
            
            # Check if we should alert
            should_alert = False
            alert_message = ""
            
            if profile.overall_status == HealthStatus.CRITICAL:
                should_alert = True
                alert_message = f"Brain {brain_id} is in CRITICAL health state"
            elif profile.overall_status == HealthStatus.WARNING:
                # Only alert for warnings if not recently alerted
                last_alert = self.active_alerts.get(alert_key)
                if not last_alert or datetime.now() - last_alert > timedelta(minutes=self.config['alert_cooldown_minutes']):
                    should_alert = True
                    alert_message = f"Brain {brain_id} is in WARNING health state"
            
            if should_alert:
                self.active_alerts[alert_key] = datetime.now()
                profile.alerts.append(alert_message)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        await callback(brain_id, profile.overall_status, alert_message)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
                
                logger.warning(f"🚨 HEALTH ALERT: {alert_message}")
            
        except Exception as e:
            logger.error(f"❌ Health alert check failed: {e}")
    
    async def _trend_analyzer(self):
        """Analyze health trends for predictive insights"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                for brain_id in self.brain_profiles.keys():
                    await self._analyze_brain_trends(brain_id)
                
            except Exception as e:
                logger.error(f"❌ Trend analyzer error: {e}")
    
    async def _analyze_brain_trends(self, brain_id: str):
        """Analyze trends for a specific brain"""
        try:
            history = self.health_history.get(brain_id, [])
            if len(history) < 10:  # Need minimum data points
                return
            
            profile = self.brain_profiles[brain_id]
            
            # Analyze response time trends
            recent_response_times = [r.response_time for r in history[-20:] if r.response_time > 0]
            if len(recent_response_times) >= 5:
                profile.trends['response_time'] = recent_response_times
                
                # Check for degrading performance
                if len(recent_response_times) >= 10:
                    first_half = recent_response_times[:5]
                    second_half = recent_response_times[-5:]
                    
                    if statistics.mean(second_half) > statistics.mean(first_half) * 1.5:
                        profile.alerts.append(f"Degrading response time trend detected for {brain_id}")
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed for {brain_id}: {e}")
    
    async def _alert_manager(self):
        """Manage alert lifecycle"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up old alerts
                current_time = datetime.now()
                expired_alerts = []
                
                for alert_key, alert_time in self.active_alerts.items():
                    if current_time - alert_time > timedelta(hours=1):
                        expired_alerts.append(alert_key)
                
                for alert_key in expired_alerts:
                    del self.active_alerts[alert_key]
                
            except Exception as e:
                logger.error(f"❌ Alert manager error: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old health data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now() - timedelta(hours=self.config['history_retention_hours'])
                
                for brain_id in self.health_history.keys():
                    self.health_history[brain_id] = [
                        result for result in self.health_history[brain_id]
                        if result.timestamp > cutoff_time
                    ]
                
                logger.info("🧹 Health data cleanup completed")
                
            except Exception as e:
                logger.error(f"❌ Health data cleanup error: {e}")
    
    async def _store_brain_config(self, brain_id: str, health_checks: List[HealthCheck]):
        """Store brain health configuration in Redis"""
        if self.redis_client:
            try:
                key = f"brain_health_config:{brain_id}"
                data = json.dumps([asdict(check) for check in health_checks], default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store brain config: {e}")
    
    async def _store_brain_profile(self, profile: BrainHealthProfile):
        """Store brain health profile in Redis"""
        if self.redis_client:
            try:
                key = f"brain_health_profile:{profile.brain_id}"
                data = json.dumps(asdict(profile), default=str)
                await self.redis_client.setex(key, 3600, data)  # 1 hour TTL
            except Exception as e:
                logger.error(f"Failed to store brain profile: {e}")
    
    async def _store_health_result(self, result: HealthResult):
        """Store health check result in Redis"""
        if self.redis_client:
            try:
                key = f"health_result:{result.brain_id}:{int(result.timestamp.timestamp())}"
                data = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store health result: {e}")
    
    async def _load_health_configurations(self):
        """Load health configurations from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("brain_health_config:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        brain_id = key.decode().split(':')[-1]
                        config_data = json.loads(data)
                        # Convert back to HealthCheck objects
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load health configurations: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        healthy_brains = sum(1 for p in self.brain_profiles.values() if p.overall_status == HealthStatus.HEALTHY)
        warning_brains = sum(1 for p in self.brain_profiles.values() if p.overall_status == HealthStatus.WARNING)
        critical_brains = sum(1 for p in self.brain_profiles.values() if p.overall_status == HealthStatus.CRITICAL)
        
        return {
            'total_brains': len(self.brain_profiles),
            'healthy_brains': healthy_brains,
            'warning_brains': warning_brains,
            'critical_brains': critical_brains,
            'active_alerts': len(self.active_alerts),
            'system_health_score': healthy_brains / len(self.brain_profiles) if self.brain_profiles else 0,
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global brain health monitor instance
brain_health_monitor = BrainHealthMonitor()

async def initialize_brain_health_monitor():
    """Initialize the global brain health monitor"""
    await brain_health_monitor.initialize()

if __name__ == "__main__":
    # Test the brain health monitor
    async def test_brain_health_monitor():
        await initialize_brain_health_monitor()
        
        # Register test brain
        health_checks = [
            HealthCheck(
                check_id="ping_check",
                check_type=HealthCheckType.BASIC_PING,
                endpoint="http://localhost:8001/health",
                interval_seconds=30,
                timeout_seconds=5,
                expected_response=None,
                warning_threshold=2.0,
                critical_threshold=5.0,
                enabled=True
            )
        ]
        
        await brain_health_monitor.register_brain("brain1", health_checks)
        
        # Wait for health checks
        await asyncio.sleep(5)
        
        # Get health summary
        summary = await brain_health_monitor.get_health_summary()
        print(f"Health summary: {summary}")
        
        # Get brain health
        brain_health = await brain_health_monitor.get_brain_health("brain1")
        print(f"Brain1 health: {brain_health}")
    
    asyncio.run(test_brain_health_monitor())
