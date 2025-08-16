"""
Log Analyzer - Advanced Log Analysis and Pattern Detection
Analyzes logs for patterns, anomalies, and insights

This module provides advanced log analysis capabilities including pattern
detection, anomaly identification, trend analysis, and automated insights.

Created: 2025-07-29 AEST
Purpose: Advanced log analysis and pattern detection for Four-Brain system
Module Size: 150 lines (modular design)
"""

import re
import time
import logging
import statistics
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class LogPattern:
    """Detected log pattern"""
    pattern_id: str
    pattern_regex: str
    pattern_description: str
    occurrences: int
    first_seen: float
    last_seen: float
    affected_brains: Set[str]
    severity: str
    examples: List[str]


@dataclass
class LogAnomaly:
    """Detected log anomaly"""
    anomaly_id: str
    anomaly_type: str
    description: str
    severity: str
    timestamp: float
    affected_brain: str
    context: Dict[str, Any]
    confidence: float


@dataclass
class LogInsight:
    """Generated log insight"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    severity: str
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: float


class LogAnalyzer:
    """
    Advanced Log Analyzer
    
    Provides comprehensive log analysis including pattern detection,
    anomaly identification, trend analysis, and automated insights.
    """
    
    def __init__(self, analyzer_id: str = "log_analyzer"):
        """Initialize log analyzer"""
        self.analyzer_id = analyzer_id
        self.enabled = True
        
        # Analysis storage
        self.detected_patterns: Dict[str, LogPattern] = {}
        self.detected_anomalies: List[LogAnomaly] = []
        self.generated_insights: List[LogInsight] = []
        
        # Pattern definitions
        self.error_patterns = {
            "connection_timeout": r"(?i)connection.*timeout|timeout.*connection",
            "authentication_failed": r"(?i)authentication.*failed|auth.*error|login.*failed",
            "database_error": r"(?i)database.*error|sql.*error|connection.*refused",
            "memory_error": r"(?i)out of memory|memory.*error|oom",
            "permission_denied": r"(?i)permission.*denied|access.*denied|forbidden",
            "file_not_found": r"(?i)file.*not.*found|no such file|path.*not.*exist",
            "network_error": r"(?i)network.*error|connection.*reset|host.*unreachable",
            "api_error": r"(?i)api.*error|http.*error|status.*[45]\d\d"
        }
        
        self.performance_patterns = {
            "slow_operation": r"(?i)slow|took.*\d+.*seconds|duration.*\d+",
            "high_cpu": r"(?i)cpu.*high|cpu.*\d{2,3}%",
            "high_memory": r"(?i)memory.*high|memory.*\d{2,3}%",
            "bottleneck": r"(?i)bottleneck|performance.*issue|latency"
        }
        
        # Analysis configuration
        self.analysis_window = 3600  # 1 hour
        self.anomaly_threshold = 0.7  # Confidence threshold for anomalies
        self.pattern_min_occurrences = 3
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🔍 Log Analyzer initialized: {analyzer_id}")
    
    def analyze_logs(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive log analysis"""
        analysis_start = time.time()
        
        # Pattern detection
        patterns = self._detect_patterns(log_entries)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(log_entries)
        
        # Trend analysis
        trends = self._analyze_trends(log_entries)
        
        # Generate insights
        insights = self._generate_insights(log_entries, patterns, anomalies, trends)
        
        analysis_duration = time.time() - analysis_start
        
        return {
            "analysis_id": f"analysis_{int(time.time())}",
            "analyzer_id": self.analyzer_id,
            "analysis_timestamp": analysis_start,
            "analysis_duration": analysis_duration,
            "log_entries_analyzed": len(log_entries),
            "patterns_detected": len(patterns),
            "anomalies_detected": len(anomalies),
            "insights_generated": len(insights),
            "patterns": [asdict(p) for p in patterns],
            "anomalies": [asdict(a) for a in anomalies],
            "trends": trends,
            "insights": [asdict(i) for i in insights]
        }
    
    def _detect_patterns(self, log_entries: List[Dict[str, Any]]) -> List[LogPattern]:
        """Detect patterns in log entries"""
        patterns = []
        
        # Combine all pattern definitions
        all_patterns = {**self.error_patterns, **self.performance_patterns}
        
        for pattern_name, pattern_regex in all_patterns.items():
            matches = []
            affected_brains = set()
            
            for entry in log_entries:
                message = entry.get('message', '')
                if re.search(pattern_regex, message):
                    matches.append(entry)
                    affected_brains.add(entry.get('brain_id', 'unknown'))
            
            if len(matches) >= self.pattern_min_occurrences:
                # Determine severity
                severity = "high" if pattern_name in self.error_patterns else "medium"
                
                pattern = LogPattern(
                    pattern_id=f"pattern_{pattern_name}_{int(time.time())}",
                    pattern_regex=pattern_regex,
                    pattern_description=f"Detected pattern: {pattern_name}",
                    occurrences=len(matches),
                    first_seen=min(m.get('timestamp', 0) for m in matches),
                    last_seen=max(m.get('timestamp', 0) for m in matches),
                    affected_brains=affected_brains,
                    severity=severity,
                    examples=[m.get('message', '')[:100] for m in matches[:3]]
                )
                
                patterns.append(pattern)
                
                with self._lock:
                    self.detected_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _detect_anomalies(self, log_entries: List[Dict[str, Any]]) -> List[LogAnomaly]:
        """Detect anomalies in log entries"""
        anomalies = []
        
        # Group entries by brain and time windows
        brain_activity = defaultdict(list)
        for entry in log_entries:
            brain_id = entry.get('brain_id', 'unknown')
            timestamp = entry.get('timestamp', 0)
            brain_activity[brain_id].append(timestamp)
        
        # Detect unusual activity patterns
        for brain_id, timestamps in brain_activity.items():
            if len(timestamps) < 10:  # Need sufficient data
                continue
            
            # Check for activity spikes
            spike_anomaly = self._detect_activity_spike(brain_id, timestamps)
            if spike_anomaly:
                anomalies.append(spike_anomaly)
            
            # Check for silence periods
            silence_anomaly = self._detect_silence_period(brain_id, timestamps)
            if silence_anomaly:
                anomalies.append(silence_anomaly)
        
        # Detect error rate anomalies
        error_anomaly = self._detect_error_rate_anomaly(log_entries)
        if error_anomaly:
            anomalies.append(error_anomaly)
        
        with self._lock:
            self.detected_anomalies.extend(anomalies)
        
        return anomalies
    
    def _detect_activity_spike(self, brain_id: str, timestamps: List[float]) -> Optional[LogAnomaly]:
        """Detect unusual activity spikes"""
        if len(timestamps) < 20:
            return None
        
        # Calculate activity rate in 5-minute windows
        window_size = 300  # 5 minutes
        current_time = max(timestamps)
        window_start = current_time - window_size
        
        recent_activity = len([t for t in timestamps if t >= window_start])
        historical_activity = len([t for t in timestamps if t < window_start]) / ((window_start - min(timestamps)) / window_size)
        
        # Detect spike (activity > 3x historical average)
        if recent_activity > historical_activity * 3 and recent_activity > 10:
            return LogAnomaly(
                anomaly_id=f"spike_{brain_id}_{int(time.time())}",
                anomaly_type="activity_spike",
                description=f"Unusual activity spike detected in {brain_id}",
                severity="medium",
                timestamp=current_time,
                affected_brain=brain_id,
                context={
                    "recent_activity": recent_activity,
                    "historical_average": historical_activity,
                    "spike_ratio": recent_activity / max(historical_activity, 1)
                },
                confidence=0.8
            )
        
        return None
    
    def _detect_silence_period(self, brain_id: str, timestamps: List[float]) -> Optional[LogAnomaly]:
        """Detect unusual silence periods"""
        if len(timestamps) < 10:
            return None
        
        # Check for gaps in activity
        sorted_timestamps = sorted(timestamps)
        gaps = []
        
        for i in range(1, len(sorted_timestamps)):
            gap = sorted_timestamps[i] - sorted_timestamps[i-1]
            gaps.append(gap)
        
        if gaps:
            avg_gap = statistics.mean(gaps)
            max_gap = max(gaps)
            
            # Detect unusual silence (gap > 5x average and > 10 minutes)
            if max_gap > avg_gap * 5 and max_gap > 600:
                return LogAnomaly(
                    anomaly_id=f"silence_{brain_id}_{int(time.time())}",
                    anomaly_type="silence_period",
                    description=f"Unusual silence period detected in {brain_id}",
                    severity="medium",
                    timestamp=sorted_timestamps[-1],
                    affected_brain=brain_id,
                    context={
                        "silence_duration": max_gap,
                        "average_gap": avg_gap,
                        "silence_ratio": max_gap / avg_gap
                    },
                    confidence=0.7
                )
        
        return None
    
    def _detect_error_rate_anomaly(self, log_entries: List[Dict[str, Any]]) -> Optional[LogAnomaly]:
        """Detect unusual error rates"""
        error_levels = ['ERROR', 'CRITICAL']
        
        total_entries = len(log_entries)
        error_entries = len([e for e in log_entries if e.get('level') in error_levels])
        
        if total_entries == 0:
            return None
        
        error_rate = error_entries / total_entries
        
        # Detect high error rate (>10%)
        if error_rate > 0.1 and error_entries > 5:
            return LogAnomaly(
                anomaly_id=f"error_rate_{int(time.time())}",
                anomaly_type="high_error_rate",
                description=f"High error rate detected: {error_rate:.1%}",
                severity="high",
                timestamp=time.time(),
                affected_brain="system",
                context={
                    "error_rate": error_rate,
                    "error_count": error_entries,
                    "total_entries": total_entries
                },
                confidence=0.9
            )
        
        return None
    
    def _analyze_trends(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in log data"""
        if not log_entries:
            return {}
        
        # Time-based analysis
        timestamps = [e.get('timestamp', 0) for e in log_entries]
        time_span = max(timestamps) - min(timestamps)
        
        # Activity trend
        activity_rate = len(log_entries) / max(time_span / 3600, 1)  # entries per hour
        
        # Error trend
        error_entries = [e for e in log_entries if e.get('level') in ['ERROR', 'CRITICAL']]
        error_rate = len(error_entries) / max(len(log_entries), 1)
        
        # Brain activity distribution
        brain_activity = Counter(e.get('brain_id', 'unknown') for e in log_entries)
        
        # Level distribution
        level_distribution = Counter(e.get('level', 'UNKNOWN') for e in log_entries)
        
        return {
            "time_span_hours": time_span / 3600,
            "activity_rate_per_hour": activity_rate,
            "error_rate": error_rate,
            "brain_activity_distribution": dict(brain_activity),
            "level_distribution": dict(level_distribution),
            "most_active_brain": brain_activity.most_common(1)[0] if brain_activity else None,
            "dominant_log_level": level_distribution.most_common(1)[0] if level_distribution else None
        }
    
    def _generate_insights(self, log_entries: List[Dict[str, Any]], patterns: List[LogPattern], 
                          anomalies: List[LogAnomaly], trends: Dict[str, Any]) -> List[LogInsight]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # Pattern-based insights
        for pattern in patterns:
            if pattern.severity == "high":
                insight = LogInsight(
                    insight_id=f"insight_pattern_{int(time.time())}",
                    insight_type="pattern_analysis",
                    title=f"Critical Pattern Detected: {pattern.pattern_description}",
                    description=f"Pattern detected {pattern.occurrences} times across {len(pattern.affected_brains)} brains",
                    severity="high",
                    recommendations=[
                        f"Investigate root cause of {pattern.pattern_description}",
                        f"Monitor affected brains: {', '.join(pattern.affected_brains)}",
                        "Consider implementing preventive measures"
                    ],
                    supporting_data=asdict(pattern),
                    generated_at=time.time()
                )
                insights.append(insight)
        
        # Anomaly-based insights
        for anomaly in anomalies:
            if anomaly.confidence >= self.anomaly_threshold:
                insight = LogInsight(
                    insight_id=f"insight_anomaly_{int(time.time())}",
                    insight_type="anomaly_analysis",
                    title=f"Anomaly Detected: {anomaly.description}",
                    description=f"Anomaly in {anomaly.affected_brain} with {anomaly.confidence:.1%} confidence",
                    severity=anomaly.severity,
                    recommendations=[
                        f"Investigate {anomaly.anomaly_type} in {anomaly.affected_brain}",
                        "Check system resources and performance",
                        "Review recent changes or deployments"
                    ],
                    supporting_data=asdict(anomaly),
                    generated_at=time.time()
                )
                insights.append(insight)
        
        # Trend-based insights
        if trends.get('error_rate', 0) > 0.05:  # >5% error rate
            insight = LogInsight(
                insight_id=f"insight_trend_{int(time.time())}",
                insight_type="trend_analysis",
                title="Elevated Error Rate Detected",
                description=f"System error rate is {trends['error_rate']:.1%}, above normal threshold",
                severity="medium",
                recommendations=[
                    "Review error logs for common patterns",
                    "Check system health and resources",
                    "Consider scaling or optimization"
                ],
                supporting_data=trends,
                generated_at=time.time()
            )
            insights.append(insight)
        
        with self._lock:
            self.generated_insights.extend(insights)
        
        return insights
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results"""
        with self._lock:
            return {
                "analyzer_id": self.analyzer_id,
                "enabled": self.enabled,
                "total_patterns": len(self.detected_patterns),
                "total_anomalies": len(self.detected_anomalies),
                "total_insights": len(self.generated_insights),
                "recent_patterns": len([p for p in self.detected_patterns.values() 
                                     if time.time() - p.last_seen < 3600]),
                "recent_anomalies": len([a for a in self.detected_anomalies 
                                       if time.time() - a.timestamp < 3600]),
                "high_severity_insights": len([i for i in self.generated_insights 
                                             if i.severity == "high"])
            }


# Factory function for easy creation
def create_log_analyzer(analyzer_id: str = "log_analyzer") -> LogAnalyzer:
    """Factory function to create log analyzer"""
    return LogAnalyzer(analyzer_id)
