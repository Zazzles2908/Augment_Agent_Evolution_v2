"""
Coordination Metrics System for Four-Brain System v2
Comprehensive metrics collection and analysis for brain coordination

Created: 2025-07-30 AEST
Purpose: Monitor and analyze coordination performance across all brain instances
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of coordination metrics"""
    PERFORMANCE = "performance"
    UTILIZATION = "utilization"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    EFFICIENCY = "efficiency"

class AggregationType(Enum):
    """Metric aggregation types"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    RATE = "rate"

@dataclass
class MetricPoint:
    """Individual metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    brain_id: Optional[str]
    task_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class AggregatedMetric:
    """Aggregated metric over time period"""
    metric_name: str
    aggregation_type: AggregationType
    value: float
    start_time: datetime
    end_time: datetime
    sample_count: int
    metadata: Dict[str, Any]

@dataclass
class CoordinationReport:
    """Comprehensive coordination performance report"""
    report_id: str
    generated_at: datetime
    time_period: timedelta
    brain_metrics: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    recommendations: List[str]
    alerts: List[str]

class CoordinationMetrics:
    """
    Comprehensive coordination metrics system
    
    Features:
    - Real-time metrics collection and aggregation
    - Performance trend analysis
    - Brain utilization monitoring
    - Coordination efficiency tracking
    - Automated alerting and recommendations
    - Historical data analysis
    - Custom metric definitions
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/14"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Metrics storage
        self.metric_points: deque = deque(maxlen=10000)  # Keep last 10k points
        self.aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        
        # Real-time metrics
        self.current_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'collection_interval_seconds': 10,
            'aggregation_intervals': [60, 300, 3600],  # 1min, 5min, 1hour
            'retention_days': 30,
            'alert_thresholds': {
                'high_latency': 5.0,
                'high_error_rate': 0.1,
                'low_availability': 0.95,
                'high_utilization': 0.9
            },
            'performance_targets': {
                'average_response_time': 1.0,
                'throughput_per_brain': 100.0,
                'system_availability': 0.99,
                'coordination_efficiency': 0.85
            }
        }
        
        # Metric definitions
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        
        # Alert state
        self.active_alerts: Set[str] = set()
        
        logger.info("📊 Coordination Metrics initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start metrics collection"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load historical metrics
            await self._load_historical_metrics()
            
            # Start background services
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._metrics_aggregator())
            asyncio.create_task(self._alert_monitor())
            asyncio.create_task(self._baseline_updater())
            
            logger.info("✅ Coordination Metrics Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Coordination Metrics: {e}")
            raise
    
    async def record_metric(self, metric_name: str, metric_type: MetricType, value: float,
                          brain_id: Optional[str] = None, task_id: Optional[str] = None,
                          metadata: Dict[str, Any] = None):
        """Record a new metric point"""
        try:
            metric_point = MetricPoint(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                brain_id=brain_id,
                task_id=task_id,
                metadata=metadata or {}
            )
            
            # Add to collection
            self.metric_points.append(metric_point)
            
            # Update current metrics
            self._update_current_metrics(metric_point)
            
            # Store in Redis
            await self._store_metric_point(metric_point)
            
        except Exception as e:
            logger.error(f"❌ Failed to record metric: {e}")
    
    async def get_brain_metrics(self, brain_id: str, 
                              time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get metrics for a specific brain"""
        try:
            time_range = time_range or timedelta(hours=1)
            cutoff_time = datetime.now() - time_range
            
            brain_metrics = {
                'brain_id': brain_id,
                'time_range': time_range.total_seconds(),
                'metrics': {}
            }
            
            # Filter metrics for this brain
            brain_points = [
                point for point in self.metric_points
                if point.brain_id == brain_id and point.timestamp > cutoff_time
            ]
            
            if not brain_points:
                return brain_metrics
            
            # Calculate aggregated metrics
            metrics_by_name = defaultdict(list)
            for point in brain_points:
                metrics_by_name[point.metric_name].append(point.value)
            
            for metric_name, values in metrics_by_name.items():
                brain_metrics['metrics'][metric_name] = {
                    'count': len(values),
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0
                }
                
                if len(values) > 1:
                    brain_metrics['metrics'][metric_name]['std_dev'] = statistics.stdev(values)
            
            return brain_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get brain metrics: {e}")
            return {}
    
    async def get_system_metrics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get system-wide coordination metrics"""
        try:
            time_range = time_range or timedelta(hours=1)
            cutoff_time = datetime.now() - time_range
            
            # Filter recent metrics
            recent_points = [
                point for point in self.metric_points
                if point.timestamp > cutoff_time
            ]
            
            system_metrics = {
                'time_range': time_range.total_seconds(),
                'total_metrics': len(recent_points),
                'brain_count': len(set(p.brain_id for p in recent_points if p.brain_id)),
                'metrics': {}
            }
            
            # Group by metric type
            metrics_by_type = defaultdict(list)
            for point in recent_points:
                metrics_by_type[point.metric_type].append(point.value)
            
            # Calculate system-wide aggregations
            for metric_type, values in metrics_by_type.items():
                if values:
                    system_metrics['metrics'][metric_type.value] = {
                        'count': len(values),
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'total': sum(values)
                    }
            
            # Calculate coordination efficiency
            system_metrics['coordination_efficiency'] = await self._calculate_coordination_efficiency()
            
            # Calculate system availability
            system_metrics['system_availability'] = await self._calculate_system_availability()
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {}
    
    async def generate_coordination_report(self, time_period: timedelta) -> CoordinationReport:
        """Generate comprehensive coordination performance report"""
        try:
            report_id = f"coord_report_{int(time.time())}"
            end_time = datetime.now()
            start_time = end_time - time_period
            
            # Get metrics for all brains
            brain_metrics = {}
            brain_ids = set(p.brain_id for p in self.metric_points if p.brain_id)
            
            for brain_id in brain_ids:
                brain_metrics[brain_id] = await self.get_brain_metrics(brain_id, time_period)
            
            # Get system metrics
            system_metrics = await self.get_system_metrics(time_period)
            
            # Generate performance summary
            performance_summary = await self._generate_performance_summary(
                brain_metrics, system_metrics, time_period
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                brain_metrics, system_metrics, performance_summary
            )
            
            # Check for alerts
            alerts = await self._check_performance_alerts(system_metrics)
            
            report = CoordinationReport(
                report_id=report_id,
                generated_at=datetime.now(),
                time_period=time_period,
                brain_metrics=brain_metrics,
                system_metrics=system_metrics,
                performance_summary=performance_summary,
                recommendations=recommendations,
                alerts=alerts
            )
            
            # Store report
            await self._store_coordination_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate coordination report: {e}")
            raise
    
    async def _calculate_coordination_efficiency(self) -> float:
        """Calculate overall coordination efficiency"""
        try:
            # Get recent task completion metrics
            recent_points = [
                p for p in self.metric_points
                if p.timestamp > datetime.now() - timedelta(minutes=30)
                and p.metric_name in ['task_completion_time', 'task_success_rate']
            ]
            
            if not recent_points:
                return 0.5  # Default neutral efficiency
            
            # Calculate efficiency based on completion time and success rate
            completion_times = [p.value for p in recent_points if p.metric_name == 'task_completion_time']
            success_rates = [p.value for p in recent_points if p.metric_name == 'task_success_rate']
            
            efficiency_score = 0.5  # Base score
            
            if completion_times:
                avg_completion_time = statistics.mean(completion_times)
                target_time = self.config['performance_targets']['average_response_time']
                time_efficiency = max(0, min(1, target_time / avg_completion_time))
                efficiency_score = efficiency_score * 0.5 + time_efficiency * 0.5
            
            if success_rates:
                avg_success_rate = statistics.mean(success_rates)
                efficiency_score = efficiency_score * 0.7 + avg_success_rate * 0.3
            
            return min(1.0, max(0.0, efficiency_score))
            
        except Exception as e:
            logger.error(f"❌ Coordination efficiency calculation failed: {e}")
            return 0.5
    
    async def _calculate_system_availability(self) -> float:
        """Calculate system availability"""
        try:
            # Get brain health metrics from last hour
            recent_points = [
                p for p in self.metric_points
                if p.timestamp > datetime.now() - timedelta(hours=1)
                and p.metric_name == 'brain_health'
            ]
            
            if not recent_points:
                return 1.0  # Assume healthy if no data
            
            # Calculate availability as percentage of healthy checks
            healthy_checks = sum(1 for p in recent_points if p.value > 0.5)
            total_checks = len(recent_points)
            
            return healthy_checks / total_checks if total_checks > 0 else 1.0
            
        except Exception as e:
            logger.error(f"❌ System availability calculation failed: {e}")
            return 1.0
    
    async def _generate_performance_summary(self, brain_metrics: Dict[str, Any],
                                          system_metrics: Dict[str, Any],
                                          time_period: timedelta) -> Dict[str, Any]:
        """Generate performance summary"""
        try:
            summary = {
                'time_period_hours': time_period.total_seconds() / 3600,
                'total_brains': len(brain_metrics),
                'active_brains': len([b for b in brain_metrics.values() 
                                    if b.get('metrics', {}).get('brain_health', {}).get('latest', 0) > 0.5]),
                'overall_performance': 'good',  # Will be calculated
                'key_metrics': {}
            }
            
            # Calculate key performance indicators
            if system_metrics.get('coordination_efficiency'):
                summary['key_metrics']['coordination_efficiency'] = system_metrics['coordination_efficiency']
            
            if system_metrics.get('system_availability'):
                summary['key_metrics']['system_availability'] = system_metrics['system_availability']
            
            # Determine overall performance rating
            efficiency = summary['key_metrics'].get('coordination_efficiency', 0.5)
            availability = summary['key_metrics'].get('system_availability', 1.0)
            
            overall_score = (efficiency + availability) / 2
            
            if overall_score >= 0.9:
                summary['overall_performance'] = 'excellent'
            elif overall_score >= 0.8:
                summary['overall_performance'] = 'good'
            elif overall_score >= 0.6:
                summary['overall_performance'] = 'fair'
            else:
                summary['overall_performance'] = 'poor'
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Performance summary generation failed: {e}")
            return {}
    
    async def _generate_recommendations(self, brain_metrics: Dict[str, Any],
                                      system_metrics: Dict[str, Any],
                                      performance_summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            # Check coordination efficiency
            efficiency = performance_summary.get('key_metrics', {}).get('coordination_efficiency', 0.5)
            if efficiency < 0.7:
                recommendations.append("Consider optimizing task routing algorithms to improve coordination efficiency")
            
            # Check system availability
            availability = performance_summary.get('key_metrics', {}).get('system_availability', 1.0)
            if availability < 0.95:
                recommendations.append("Investigate brain health issues to improve system availability")
            
            # Check brain utilization
            for brain_id, metrics in brain_metrics.items():
                brain_metrics_data = metrics.get('metrics', {})
                
                # Check for high utilization
                if 'cpu_usage' in brain_metrics_data:
                    cpu_usage = brain_metrics_data['cpu_usage'].get('average', 0)
                    if cpu_usage > 0.9:
                        recommendations.append(f"Brain {brain_id} has high CPU utilization - consider load balancing")
                
                # Check for high error rates
                if 'error_rate' in brain_metrics_data:
                    error_rate = brain_metrics_data['error_rate'].get('average', 0)
                    if error_rate > 0.1:
                        recommendations.append(f"Brain {brain_id} has high error rate - investigate and fix issues")
            
            # Check overall performance
            if performance_summary.get('overall_performance') == 'poor':
                recommendations.append("Overall system performance is poor - consider system optimization or scaling")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendations generation failed: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    async def _check_performance_alerts(self, system_metrics: Dict[str, Any]) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        try:
            thresholds = self.config['alert_thresholds']
            
            # Check system availability
            availability = system_metrics.get('system_availability', 1.0)
            if availability < thresholds['low_availability']:
                alerts.append(f"CRITICAL: System availability ({availability:.2%}) below threshold ({thresholds['low_availability']:.2%})")
            
            # Check coordination efficiency
            efficiency = system_metrics.get('coordination_efficiency', 0.5)
            if efficiency < 0.5:
                alerts.append(f"WARNING: Coordination efficiency ({efficiency:.2%}) is low")
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Performance alerts check failed: {e}")
            return ["Unable to check performance alerts due to error"]
    
    def _update_current_metrics(self, metric_point: MetricPoint):
        """Update current metrics cache"""
        try:
            key = f"{metric_point.brain_id or 'system'}:{metric_point.metric_name}"
            
            if key not in self.current_metrics:
                self.current_metrics[key] = {
                    'latest_value': metric_point.value,
                    'latest_timestamp': metric_point.timestamp,
                    'metric_type': metric_point.metric_type.value
                }
            else:
                self.current_metrics[key]['latest_value'] = metric_point.value
                self.current_metrics[key]['latest_timestamp'] = metric_point.timestamp
            
        except Exception as e:
            logger.error(f"❌ Current metrics update failed: {e}")
    
    async def _metrics_collector(self):
        """Background metrics collection"""
        while True:
            try:
                await asyncio.sleep(self.config['collection_interval_seconds'])
                
                # Collect system metrics
                await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # Record current timestamp
            now = datetime.now()
            
            # Record active metrics count
            await self.record_metric(
                'active_metrics_count',
                MetricType.PERFORMANCE,
                len(self.metric_points)
            )
            
            # Record coordination efficiency
            efficiency = await self._calculate_coordination_efficiency()
            await self.record_metric(
                'coordination_efficiency',
                MetricType.EFFICIENCY,
                efficiency
            )
            
            # Record system availability
            availability = await self._calculate_system_availability()
            await self.record_metric(
                'system_availability',
                MetricType.AVAILABILITY,
                availability
            )
            
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
    
    async def _metrics_aggregator(self):
        """Background metrics aggregation"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for interval in self.config['aggregation_intervals']:
                    await self._aggregate_metrics(interval)
                
            except Exception as e:
                logger.error(f"❌ Metrics aggregation error: {e}")
    
    async def _aggregate_metrics(self, interval_seconds: int):
        """Aggregate metrics for a specific time interval"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=interval_seconds)
            
            # Filter metrics for this interval
            interval_points = [
                point for point in self.metric_points
                if start_time <= point.timestamp <= end_time
            ]
            
            if not interval_points:
                return
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for point in interval_points:
                metrics_by_name[point.metric_name].append(point.value)
            
            # Create aggregated metrics
            for metric_name, values in metrics_by_name.items():
                if values:
                    # Create average aggregation
                    avg_metric = AggregatedMetric(
                        metric_name=metric_name,
                        aggregation_type=AggregationType.AVERAGE,
                        value=statistics.mean(values),
                        start_time=start_time,
                        end_time=end_time,
                        sample_count=len(values),
                        metadata={'interval_seconds': interval_seconds}
                    )
                    
                    # Store aggregated metric
                    self.aggregated_metrics[f"{metric_name}_{interval_seconds}"].append(avg_metric)
                    
                    # Keep only recent aggregations
                    max_aggregations = 1440  # 24 hours worth of minute aggregations
                    if len(self.aggregated_metrics[f"{metric_name}_{interval_seconds}"]) > max_aggregations:
                        self.aggregated_metrics[f"{metric_name}_{interval_seconds}"] = \
                            self.aggregated_metrics[f"{metric_name}_{interval_seconds}"][-max_aggregations:]
            
        except Exception as e:
            logger.error(f"❌ Metrics aggregation failed for interval {interval_seconds}: {e}")
    
    async def _alert_monitor(self):
        """Monitor for alert conditions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check current metrics against thresholds
                await self._check_alert_conditions()
                
            except Exception as e:
                logger.error(f"❌ Alert monitor error: {e}")
    
    async def _check_alert_conditions(self):
        """Check current metrics against alert thresholds"""
        try:
            thresholds = self.config['alert_thresholds']
            
            for key, metric_data in self.current_metrics.items():
                metric_name = key.split(':')[-1]
                value = metric_data['latest_value']
                
                # Check specific alert conditions
                alert_key = None
                
                if metric_name == 'response_time' and value > thresholds['high_latency']:
                    alert_key = f"high_latency_{key}"
                elif metric_name == 'error_rate' and value > thresholds['high_error_rate']:
                    alert_key = f"high_error_rate_{key}"
                elif metric_name == 'system_availability' and value < thresholds['low_availability']:
                    alert_key = f"low_availability_{key}"
                elif metric_name in ['cpu_usage', 'memory_usage', 'gpu_usage'] and value > thresholds['high_utilization']:
                    alert_key = f"high_utilization_{key}"
                
                if alert_key:
                    if alert_key not in self.active_alerts:
                        self.active_alerts.add(alert_key)
                        logger.warning(f"🚨 ALERT: {alert_key} - Value: {value}")
                else:
                    # Clear alert if condition no longer met
                    alerts_to_clear = [a for a in self.active_alerts if a.endswith(key)]
                    for alert in alerts_to_clear:
                        self.active_alerts.remove(alert)
                        logger.info(f"✅ ALERT CLEARED: {alert}")
            
        except Exception as e:
            logger.error(f"❌ Alert condition check failed: {e}")
    
    async def _baseline_updater(self):
        """Update performance baselines"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                
                await self._update_performance_baselines()
                
            except Exception as e:
                logger.error(f"❌ Baseline updater error: {e}")
    
    async def _update_performance_baselines(self):
        """Update performance baselines based on historical data"""
        try:
            # Calculate baselines from last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_points = [p for p in self.metric_points if p.timestamp > cutoff_time]
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for point in recent_points:
                metrics_by_name[point.metric_name].append(point.value)
            
            # Calculate baselines
            for metric_name, values in metrics_by_name.items():
                if len(values) >= 10:  # Need minimum data points
                    self.baselines[metric_name] = statistics.median(values)
            
        except Exception as e:
            logger.error(f"❌ Baseline update failed: {e}")
    
    def _initialize_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metric definitions"""
        return {
            'task_completion_time': {
                'type': MetricType.LATENCY,
                'unit': 'seconds',
                'description': 'Time taken to complete a task'
            },
            'task_success_rate': {
                'type': MetricType.PERFORMANCE,
                'unit': 'percentage',
                'description': 'Percentage of successfully completed tasks'
            },
            'brain_health': {
                'type': MetricType.AVAILABILITY,
                'unit': 'score',
                'description': 'Health score of brain instance'
            },
            'cpu_usage': {
                'type': MetricType.UTILIZATION,
                'unit': 'percentage',
                'description': 'CPU utilization percentage'
            },
            'memory_usage': {
                'type': MetricType.UTILIZATION,
                'unit': 'percentage',
                'description': 'Memory utilization percentage'
            },
            'gpu_usage': {
                'type': MetricType.UTILIZATION,
                'unit': 'percentage',
                'description': 'GPU utilization percentage'
            }
        }
    
    async def _store_metric_point(self, metric_point: MetricPoint):
        """Store metric point in Redis"""
        if self.redis_client:
            try:
                key = f"metric_point:{int(metric_point.timestamp.timestamp())}"
                data = json.dumps(asdict(metric_point), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store metric point: {e}")
    
    async def _store_coordination_report(self, report: CoordinationReport):
        """Store coordination report in Redis"""
        if self.redis_client:
            try:
                key = f"coordination_report:{report.report_id}"
                data = json.dumps(asdict(report), default=str)
                await self.redis_client.setex(key, 86400 * 7, data)  # 7 days retention
            except Exception as e:
                logger.error(f"Failed to store coordination report: {e}")
    
    async def _load_historical_metrics(self):
        """Load historical metrics from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("metric_point:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        metric_data = json.loads(data)
                        # Convert back to MetricPoint object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load historical metrics: {e}")
    
    async def get_coordination_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive coordination metrics summary"""
        return {
            'current_metrics_count': len(self.metric_points),
            'aggregated_metrics_count': sum(len(metrics) for metrics in self.aggregated_metrics.values()),
            'active_alerts': list(self.active_alerts),
            'baselines': self.baselines.copy(),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global coordination metrics instance
coordination_metrics = CoordinationMetrics()

async def initialize_coordination_metrics():
    """Initialize the global coordination metrics"""
    await coordination_metrics.initialize()

if __name__ == "__main__":
    # Test the coordination metrics
    async def test_coordination_metrics():
        await initialize_coordination_metrics()
        
        # Record test metrics
        await coordination_metrics.record_metric(
            'task_completion_time', MetricType.LATENCY, 0.5, 'brain1', 'task123'
        )
        await coordination_metrics.record_metric(
            'cpu_usage', MetricType.UTILIZATION, 0.7, 'brain1'
        )
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get brain metrics
        brain_metrics = await coordination_metrics.get_brain_metrics('brain1')
        print(f"Brain metrics: {brain_metrics}")
        
        # Get system metrics
        system_metrics = await coordination_metrics.get_system_metrics()
        print(f"System metrics: {system_metrics}")
        
        # Generate report
        report = await coordination_metrics.generate_coordination_report(timedelta(hours=1))
        print(f"Coordination report: {report.report_id}")
    
    asyncio.run(test_coordination_metrics())
