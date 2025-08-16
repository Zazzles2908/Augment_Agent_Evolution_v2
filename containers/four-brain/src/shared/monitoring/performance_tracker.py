"""
Performance Tracker - Advanced Performance Monitoring
Tracks performance metrics, bottlenecks, and optimization opportunities

This module provides advanced performance tracking including response times,
throughput analysis, bottleneck detection, and optimization recommendations.

Created: 2025-07-29 AEST
Purpose: Advanced performance monitoring and optimization
Module Size: 150 lines (modular design)
"""

import time
import logging
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: float
    operation_name: str
    duration: float
    success: bool
    metadata: Dict[str, Any]


@dataclass
class PerformanceAnalysis:
    """Performance analysis results"""
    operation_name: str
    sample_count: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    success_rate: float
    throughput_per_second: float
    bottleneck_score: float
    recommendations: List[str]


class PerformanceTracker:
    """
    Advanced Performance Tracker
    
    Provides comprehensive performance monitoring with bottleneck detection,
    trend analysis, and optimization recommendations for the Four-Brain system.
    """
    
    def __init__(self, brain_id: str, analysis_window_minutes: int = 60):
        """Initialize performance tracker"""
        self.brain_id = brain_id
        self.analysis_window = analysis_window_minutes * 60  # Convert to seconds
        self.enabled = True
        
        # Performance data storage
        self.metrics: deque = deque(maxlen=10000)  # Limit memory usage
        self.operation_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Analysis cache
        self.analysis_cache: Dict[str, PerformanceAnalysis] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance thresholds
        self.thresholds = {
            "slow_operation_seconds": 5.0,
            "very_slow_operation_seconds": 10.0,
            "low_success_rate_percent": 95.0,
            "high_bottleneck_score": 0.8
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"⚡ Performance Tracker initialized for {brain_id}")
    
    def record_operation(self, operation_name: str, duration: float, 
                        success: bool = True, metadata: Dict[str, Any] = None) -> str:
        """Record a performance metric"""
        if not self.enabled:
            return ""
        
        metric = PerformanceMetric(
            timestamp=time.time(),
            operation_name=operation_name,
            duration=duration,
            success=success,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Add to general metrics
            self.metrics.append(metric)
            
            # Add to operation-specific metrics
            self.operation_metrics[operation_name].append(metric)
            
            # Invalidate cache for this operation
            if operation_name in self.cache_expiry:
                del self.cache_expiry[operation_name]
                del self.analysis_cache[operation_name]
        
        # Log slow operations
        if duration > self.thresholds["slow_operation_seconds"]:
            severity = "⚠️" if duration < self.thresholds["very_slow_operation_seconds"] else "🔥"
            logger.warning(f"{severity} Slow operation: {operation_name} took {duration:.2f}s")
        
        return f"perf_{int(time.time() * 1000000)}"
    
    def analyze_operation(self, operation_name: str, force_refresh: bool = False) -> Optional[PerformanceAnalysis]:
        """Analyze performance for a specific operation"""
        
        # Check cache first
        if not force_refresh and operation_name in self.cache_expiry:
            if time.time() < self.cache_expiry[operation_name]:
                return self.analysis_cache[operation_name]
        
        with self._lock:
            if operation_name not in self.operation_metrics:
                return None
            
            # Get metrics within analysis window
            cutoff_time = time.time() - self.analysis_window
            recent_metrics = [
                m for m in self.operation_metrics[operation_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            # Calculate statistics
            durations = [m.duration for m in recent_metrics]
            successes = [m.success for m in recent_metrics]
            
            sample_count = len(durations)
            avg_duration = statistics.mean(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # Percentiles
            sorted_durations = sorted(durations)
            p50_duration = statistics.median(sorted_durations)
            p95_duration = sorted_durations[int(0.95 * len(sorted_durations))] if len(sorted_durations) > 1 else sorted_durations[0]
            p99_duration = sorted_durations[int(0.99 * len(sorted_durations))] if len(sorted_durations) > 1 else sorted_durations[0]
            
            # Success rate
            success_rate = (sum(successes) / len(successes)) * 100
            
            # Throughput (operations per second)
            time_span = max(recent_metrics, key=lambda x: x.timestamp).timestamp - min(recent_metrics, key=lambda x: x.timestamp).timestamp
            throughput_per_second = sample_count / max(time_span, 1)
            
            # Bottleneck score (0-1, higher = more bottleneck)
            bottleneck_score = self._calculate_bottleneck_score(durations, avg_duration)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                operation_name, avg_duration, success_rate, bottleneck_score, p95_duration
            )
            
            analysis = PerformanceAnalysis(
                operation_name=operation_name,
                sample_count=sample_count,
                avg_duration=avg_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                p50_duration=p50_duration,
                p95_duration=p95_duration,
                p99_duration=p99_duration,
                success_rate=success_rate,
                throughput_per_second=throughput_per_second,
                bottleneck_score=bottleneck_score,
                recommendations=recommendations
            )
            
            # Cache the analysis
            self.analysis_cache[operation_name] = analysis
            self.cache_expiry[operation_name] = time.time() + self.cache_ttl
            
            return analysis
    
    def _calculate_bottleneck_score(self, durations: List[float], avg_duration: float) -> float:
        """Calculate bottleneck score based on duration variance and average"""
        if len(durations) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        std_dev = statistics.stdev(durations)
        cv = std_dev / avg_duration if avg_duration > 0 else 0
        
        # Normalize to 0-1 scale (higher variance = higher bottleneck potential)
        bottleneck_score = min(cv / 2.0, 1.0)  # Assume CV > 2.0 is maximum bottleneck
        
        # Factor in absolute duration
        if avg_duration > self.thresholds["slow_operation_seconds"]:
            bottleneck_score = min(bottleneck_score + 0.3, 1.0)
        
        return bottleneck_score
    
    def _generate_recommendations(self, operation_name: str, avg_duration: float, 
                                success_rate: float, bottleneck_score: float, 
                                p95_duration: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Duration-based recommendations
        if avg_duration > self.thresholds["very_slow_operation_seconds"]:
            recommendations.append(f"CRITICAL: {operation_name} is very slow (avg: {avg_duration:.2f}s). Consider optimization or caching.")
        elif avg_duration > self.thresholds["slow_operation_seconds"]:
            recommendations.append(f"WARNING: {operation_name} is slow (avg: {avg_duration:.2f}s). Review implementation for optimization opportunities.")
        
        # Success rate recommendations
        if success_rate < self.thresholds["low_success_rate_percent"]:
            recommendations.append(f"ERROR RATE: {operation_name} has low success rate ({success_rate:.1f}%). Investigate error causes.")
        
        # Bottleneck recommendations
        if bottleneck_score > self.thresholds["high_bottleneck_score"]:
            recommendations.append(f"BOTTLENECK: {operation_name} shows high variance (score: {bottleneck_score:.2f}). Check for resource contention.")
        
        # P95 vs average recommendations
        if p95_duration > avg_duration * 2:
            recommendations.append(f"OUTLIERS: {operation_name} has significant outliers (P95: {p95_duration:.2f}s vs avg: {avg_duration:.2f}s). Investigate edge cases.")
        
        # General recommendations
        if not recommendations:
            if avg_duration < 1.0 and success_rate > 99.0:
                recommendations.append(f"EXCELLENT: {operation_name} performance is optimal.")
            else:
                recommendations.append(f"GOOD: {operation_name} performance is acceptable but could be optimized.")
        
        return recommendations
    
    def get_top_bottlenecks(self, limit: int = 5) -> List[PerformanceAnalysis]:
        """Get operations with highest bottleneck scores"""
        analyses = []
        
        with self._lock:
            for operation_name in self.operation_metrics.keys():
                analysis = self.analyze_operation(operation_name)
                if analysis:
                    analyses.append(analysis)
        
        # Sort by bottleneck score (descending)
        analyses.sort(key=lambda x: x.bottleneck_score, reverse=True)
        
        return analyses[:limit]
    
    def get_slowest_operations(self, limit: int = 5) -> List[PerformanceAnalysis]:
        """Get slowest operations by average duration"""
        analyses = []
        
        with self._lock:
            for operation_name in self.operation_metrics.keys():
                analysis = self.analyze_operation(operation_name)
                if analysis:
                    analyses.append(analysis)
        
        # Sort by average duration (descending)
        analyses.sort(key=lambda x: x.avg_duration, reverse=True)
        
        return analyses[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self._lock:
            total_operations = len(self.metrics)
            
            if total_operations == 0:
                return {
                    "brain_id": self.brain_id,
                    "total_operations": 0,
                    "analysis_window_minutes": self.analysis_window / 60,
                    "message": "No performance data available"
                }
            
            # Calculate overall statistics
            recent_cutoff = time.time() - self.analysis_window
            recent_metrics = [m for m in self.metrics if m.timestamp >= recent_cutoff]
            
            if recent_metrics:
                durations = [m.duration for m in recent_metrics]
                successes = [m.success for m in recent_metrics]
                
                overall_avg = statistics.mean(durations)
                overall_success_rate = (sum(successes) / len(successes)) * 100
                total_recent_operations = len(recent_metrics)
            else:
                overall_avg = 0.0
                overall_success_rate = 0.0
                total_recent_operations = 0
            
            return {
                "brain_id": self.brain_id,
                "analysis_window_minutes": self.analysis_window / 60,
                "total_operations": total_operations,
                "recent_operations": total_recent_operations,
                "overall_avg_duration": overall_avg,
                "overall_success_rate": overall_success_rate,
                "unique_operations": len(self.operation_metrics),
                "thresholds": self.thresholds,
                "enabled": self.enabled
            }
    
    def export_analysis(self, format: str = "json") -> str:
        """Export performance analysis"""
        analyses = {}
        
        with self._lock:
            for operation_name in self.operation_metrics.keys():
                analysis = self.analyze_operation(operation_name)
                if analysis:
                    analyses[operation_name] = asdict(analysis)
        
        data = {
            "summary": self.get_performance_summary(),
            "analyses": analyses,
            "top_bottlenecks": [asdict(a) for a in self.get_top_bottlenecks()],
            "slowest_operations": [asdict(a) for a in self.get_slowest_operations()]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self):
        """Clear performance metrics"""
        with self._lock:
            self.metrics.clear()
            self.operation_metrics.clear()
            self.analysis_cache.clear()
            self.cache_expiry.clear()
            logger.info(f"🧹 Performance metrics cleared for {self.brain_id}")
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set performance threshold"""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            logger.info(f"⚙️ Threshold updated: {threshold_name} = {value}")
        else:
            raise ValueError(f"Unknown threshold: {threshold_name}")


# Factory function for easy creation
def create_performance_tracker(brain_id: str, analysis_window_minutes: int = 60) -> PerformanceTracker:
    """Factory function to create performance tracker"""
    return PerformanceTracker(brain_id, analysis_window_minutes)
