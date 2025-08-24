#!/usr/bin/env python3
"""
Redis Performance Monitor for Four-Brain System
Real-time monitoring and optimization for Redis Streams communication

Created: 2025-07-27 AEST
Author: Zazzles's Agent - Redis Performance Optimization
"""

import redis
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StreamMetrics:
    name: str
    length: int
    last_entry_id: Optional[str]
    consumer_groups: List[str]
    pending_messages: int
    messages_per_second: float
    memory_usage: int

@dataclass
class RedisPerformanceSnapshot:
    timestamp: float
    memory_used_mb: float
    memory_peak_mb: float
    memory_fragmentation_ratio: float
    connected_clients: int
    commands_processed: int
    keyspace_hits: int
    keyspace_misses: int
    hit_rate_percentage: float
    streams: Dict[str, StreamMetrics]
    total_stream_messages: int

class RedisPerformanceMonitor:
    """Monitor and optimize Redis performance for Four-Brain system"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True
        )
        
        self.four_brain_streams = [
            "embedding_requests", "embedding_results",
            "rerank_requests", "rerank_results", 
            "docling_requests", "docling_results",
            "agentic_tasks", "agentic_results",
            "memory_updates"
        ]
        
        self.previous_snapshot: Optional[RedisPerformanceSnapshot] = None
        
    def get_stream_metrics(self, stream_name: str) -> StreamMetrics:
        """Get detailed metrics for a specific stream"""
        try:
            # Basic stream info
            length = self.redis_client.xlen(stream_name)
            
            # Get last entry ID
            last_entry_id = None
            if length > 0:
                entries = self.redis_client.xrevrange(stream_name, count=1)
                if entries:
                    last_entry_id = entries[0][0]
            
            # Get consumer groups
            try:
                groups_info = self.redis_client.xinfo_groups(stream_name)
                consumer_groups = [group['name'] for group in groups_info]
                
                # Calculate pending messages across all groups
                pending_messages = sum(group['pending'] for group in groups_info)
            except:
                consumer_groups = []
                pending_messages = 0
            
            # Calculate messages per second (requires previous snapshot)
            messages_per_second = 0.0
            if self.previous_snapshot and stream_name in self.previous_snapshot.streams:
                time_diff = time.time() - self.previous_snapshot.timestamp
                if time_diff > 0:
                    prev_length = self.previous_snapshot.streams[stream_name].length
                    messages_per_second = (length - prev_length) / time_diff
            
            # Get memory usage for this stream (approximate)
            try:
                memory_usage = self.redis_client.memory_usage(stream_name) or 0
            except:
                memory_usage = 0
                
            return StreamMetrics(
                name=stream_name,
                length=length,
                last_entry_id=last_entry_id,
                consumer_groups=consumer_groups,
                pending_messages=pending_messages,
                messages_per_second=messages_per_second,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Error getting metrics for stream {stream_name}: {e}")
            return StreamMetrics(
                name=stream_name,
                length=0,
                last_entry_id=None,
                consumer_groups=[],
                pending_messages=0,
                messages_per_second=0.0,
                memory_usage=0
            )
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            info = self.redis_client.info()
            return info
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}
    
    def take_performance_snapshot(self) -> RedisPerformanceSnapshot:
        """Take a comprehensive performance snapshot"""
        logger.info("📊 Taking Redis performance snapshot...")
        
        # Get Redis server info
        redis_info = self.get_redis_info()
        
        # Extract key metrics
        memory_used_mb = redis_info.get('used_memory', 0) / (1024 * 1024)
        memory_peak_mb = redis_info.get('used_memory_peak', 0) / (1024 * 1024)
        memory_fragmentation_ratio = redis_info.get('mem_fragmentation_ratio', 1.0)
        connected_clients = redis_info.get('connected_clients', 0)
        commands_processed = redis_info.get('total_commands_processed', 0)
        keyspace_hits = redis_info.get('keyspace_hits', 0)
        keyspace_misses = redis_info.get('keyspace_misses', 0)
        
        # Calculate hit rate
        total_requests = keyspace_hits + keyspace_misses
        hit_rate_percentage = (keyspace_hits / total_requests * 100) if total_requests > 0 else 100.0
        
        # Get stream metrics
        streams = {}
        total_stream_messages = 0
        
        for stream_name in self.four_brain_streams:
            stream_metrics = self.get_stream_metrics(stream_name)
            streams[stream_name] = stream_metrics
            total_stream_messages += stream_metrics.length
        
        snapshot = RedisPerformanceSnapshot(
            timestamp=time.time(),
            memory_used_mb=memory_used_mb,
            memory_peak_mb=memory_peak_mb,
            memory_fragmentation_ratio=memory_fragmentation_ratio,
            connected_clients=connected_clients,
            commands_processed=commands_processed,
            keyspace_hits=keyspace_hits,
            keyspace_misses=keyspace_misses,
            hit_rate_percentage=hit_rate_percentage,
            streams=streams,
            total_stream_messages=total_stream_messages
        )
        
        self.previous_snapshot = snapshot
        return snapshot
    
    def analyze_performance(self, snapshot: RedisPerformanceSnapshot) -> Dict[str, Any]:
        """Analyze performance and provide recommendations"""
        analysis = {
            "timestamp": snapshot.timestamp,
            "overall_health": "good",
            "recommendations": [],
            "alerts": [],
            "metrics_summary": {}
        }
        
        # Memory analysis
        memory_usage_percentage = (snapshot.memory_used_mb / 2048) * 100  # 2GB max
        analysis["metrics_summary"]["memory_usage_percentage"] = memory_usage_percentage
        
        if memory_usage_percentage > 80:
            analysis["alerts"].append("High memory usage detected")
            analysis["recommendations"].append("Consider increasing Redis memory limit")
            analysis["overall_health"] = "warning"
        elif memory_usage_percentage > 95:
            analysis["alerts"].append("Critical memory usage - near limit")
            analysis["overall_health"] = "critical"
        
        # Fragmentation analysis
        if snapshot.memory_fragmentation_ratio > 2.0:
            analysis["alerts"].append("High memory fragmentation detected")
            analysis["recommendations"].append("Consider Redis restart to defragment memory")
        
        # Stream analysis
        active_streams = sum(1 for stream in snapshot.streams.values() if stream.length > 0)
        analysis["metrics_summary"]["active_streams"] = active_streams
        analysis["metrics_summary"]["total_messages"] = snapshot.total_stream_messages
        
        # Check for stream imbalances
        stream_lengths = [stream.length for stream in snapshot.streams.values()]
        if stream_lengths:
            max_length = max(stream_lengths)
            min_length = min(stream_lengths)
            if max_length > 0 and (max_length - min_length) / max_length > 0.5:
                analysis["alerts"].append("Stream length imbalance detected")
                analysis["recommendations"].append("Check for processing bottlenecks in specific brains")
        
        # Performance recommendations
        if snapshot.total_stream_messages > 1000:
            analysis["recommendations"].append("Consider implementing stream trimming (MAXLEN)")
        
        if snapshot.hit_rate_percentage < 90:
            analysis["recommendations"].append("Low cache hit rate - review key access patterns")
        
        # Communication health
        if snapshot.total_stream_messages > 0:
            analysis["metrics_summary"]["ai_communication_status"] = "active"
        else:
            analysis["metrics_summary"]["ai_communication_status"] = "inactive"
            analysis["alerts"].append("No AI communication detected")
        
        return analysis
    
    def format_performance_report(self, snapshot: RedisPerformanceSnapshot, analysis: Dict[str, Any]) -> str:
        """Format performance data as readable report"""
        report = []
        report.append("🔄 REDIS PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {datetime.fromtimestamp(snapshot.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Overall Health: {analysis['overall_health'].upper()}")
        report.append("")
        
        # Memory metrics
        report.append("💾 MEMORY METRICS:")
        report.append(f"Used Memory: {snapshot.memory_used_mb:.2f} MB")
        report.append(f"Peak Memory: {snapshot.memory_peak_mb:.2f} MB")
        report.append(f"Memory Usage: {analysis['metrics_summary']['memory_usage_percentage']:.1f}%")
        report.append(f"Fragmentation Ratio: {snapshot.memory_fragmentation_ratio:.2f}")
        report.append("")
        
        # Connection metrics
        report.append("🔗 CONNECTION METRICS:")
        report.append(f"Connected Clients: {snapshot.connected_clients}")
        report.append(f"Commands Processed: {snapshot.commands_processed:,}")
        report.append(f"Cache Hit Rate: {snapshot.hit_rate_percentage:.1f}%")
        report.append("")
        
        # Stream metrics
        report.append("📊 STREAM METRICS:")
        report.append(f"Total Messages: {snapshot.total_stream_messages:,}")
        report.append(f"Active Streams: {analysis['metrics_summary']['active_streams']}/{len(snapshot.streams)}")
        report.append("")
        
        for stream_name, stream in snapshot.streams.items():
            mps = f" ({stream.messages_per_second:.1f} msg/s)" if stream.messages_per_second > 0 else ""
            pending = f" | {stream.pending_messages} pending" if stream.pending_messages > 0 else ""
            memory_kb = stream.memory_usage / 1024 if stream.memory_usage > 0 else 0
            memory_info = f" | {memory_kb:.1f} KB" if memory_kb > 0 else ""
            report.append(f"  {stream_name}: {stream.length:,} messages{mps}{pending}{memory_info}")
        
        # Alerts and recommendations
        if analysis["alerts"]:
            report.append("")
            report.append("⚠️ ALERTS:")
            for alert in analysis["alerts"]:
                report.append(f"  • {alert}")
        
        if analysis["recommendations"]:
            report.append("")
            report.append("💡 RECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                report.append(f"  • {rec}")
        
        return "\n".join(report)
    
    def optimize_streams(self) -> Dict[str, Any]:
        """Apply performance optimizations to streams"""
        optimizations = {
            "applied": [],
            "skipped": [],
            "errors": []
        }
        
        try:
            # Trim streams if they're too long (keep last 1000 messages)
            for stream_name in self.four_brain_streams:
                try:
                    length = self.redis_client.xlen(stream_name)
                    if length > 1000:
                        # Trim to keep last 1000 messages
                        self.redis_client.xtrim(stream_name, maxlen=1000, approximate=True)
                        optimizations["applied"].append(f"Trimmed {stream_name} from {length} to ~1000 messages")
                    else:
                        optimizations["skipped"].append(f"{stream_name} length OK ({length} messages)")
                except Exception as e:
                    optimizations["errors"].append(f"Failed to trim {stream_name}: {e}")
            
            logger.info(f"Stream optimization complete: {len(optimizations['applied'])} applied")
            
        except Exception as e:
            optimizations["errors"].append(f"General optimization error: {e}")
        
        return optimizations

def main():
    """Main monitoring execution"""
    monitor = RedisPerformanceMonitor()
    
    # Take performance snapshot
    snapshot = monitor.take_performance_snapshot()
    
    # Analyze performance
    analysis = monitor.analyze_performance(snapshot)
    
    # Generate report
    report = monitor.format_performance_report(snapshot, analysis)
    print(report)
    
    # Save detailed results
    results = {
        "snapshot": asdict(snapshot),
        "analysis": analysis
    }
    
    with open("redis_performance_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Apply optimizations if needed
    if snapshot.total_stream_messages > 500:
        print("\n🔧 APPLYING OPTIMIZATIONS...")
        optimizations = monitor.optimize_streams()
        
        if optimizations["applied"]:
            print("Applied optimizations:")
            for opt in optimizations["applied"]:
                print(f"  ✅ {opt}")
        
        if optimizations["errors"]:
            print("Optimization errors:")
            for error in optimizations["errors"]:
                print(f"  ❌ {error}")
    
    logger.info("Redis performance monitoring complete")

if __name__ == "__main__":
    main()
