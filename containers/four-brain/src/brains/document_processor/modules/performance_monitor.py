"""
Performance Monitor Module for Brain-4
Handles performance tracking, metrics collection, and optimization insights

Extracted from brain4_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance Monitor for Brain-4
    Tracks and analyzes performance metrics for document processing workloads
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize Performance Monitor"""
        self.max_history = max_history
        
        # Performance tracking
        self.request_times = deque(maxlen=max_history)
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        
        # Document processing specific metrics
        self.processing_times = deque(maxlen=max_history)
        self.file_sizes = deque(maxlen=max_history)
        self.chunk_counts = deque(maxlen=max_history)
        self.conversion_times = deque(maxlen=max_history)
        self.docling_load_time = 0.0
        self.startup_time = time.time()
        
        # Performance thresholds
        self.slow_request_threshold = 30.0  # seconds (document processing is slower)
        self.error_rate_threshold = 0.10   # 10% (higher tolerance for document processing)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("🔧 Performance Monitor (Brain-4) initialized")
    
    async def start(self):
        """Start performance monitoring"""
        try:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("✅ Performance monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start performance monitoring: {e}")
    
    async def stop(self):
        """Stop performance monitoring"""
        try:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("✅ Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping performance monitoring: {e}")
    
    async def record_document_processing(self, processing_time: float, 
                                       file_size_mb: float, chunk_count: int,
                                       conversion_time: float = 0.0,
                                       success: bool = True, error: str = None):
        """Record a document processing request for performance tracking"""
        try:
            self.total_requests += 1
            self.request_times.append(processing_time)
            self.processing_times.append(processing_time)
            self.file_sizes.append(file_size_mb)
            self.chunk_counts.append(chunk_count)
            
            if conversion_time > 0:
                self.conversion_times.append(conversion_time)
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                if error:
                    logger.debug(f"📊 Request error recorded: {error}")
            
            # Log slow requests
            if processing_time > self.slow_request_threshold:
                logger.warning(f"⚠️ Slow document processing: {processing_time:.2f}s for {file_size_mb:.1f}MB file")
            
            # Log efficiency metrics
            if file_size_mb > 0:
                mb_per_second = file_size_mb / processing_time
                logger.debug(f"📊 Processing efficiency: {mb_per_second:.2f} MB/sec")
            
            if chunk_count > 0:
                chunks_per_second = chunk_count / processing_time
                logger.debug(f"📊 Chunking efficiency: {chunks_per_second:.1f} chunks/sec")
            
        except Exception as e:
            logger.error(f"❌ Error recording request: {e}")
    
    async def record_docling_load_time(self, load_time: float):
        """Record Docling converter loading time"""
        self.docling_load_time = load_time
        logger.info(f"📊 Docling load time recorded: {load_time:.2f}s")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            if not self.request_times:
                return {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "error_rate": 0.0,
                    "average_response_time": 0.0
                }
            
            # Calculate basic metrics
            avg_response_time = statistics.mean(self.request_times)
            success_rate = self.success_count / self.total_requests if self.total_requests > 0 else 0
            error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
            
            # Calculate percentiles
            sorted_times = sorted(self.request_times)
            p50 = statistics.median(sorted_times)
            p95 = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 0 else 0
            p99 = sorted_times[int(0.99 * len(sorted_times))] if len(sorted_times) > 0 else 0
            
            # Calculate throughput (requests per hour for document processing)
            uptime_hours = (time.time() - self.startup_time) / 3600
            throughput = self.total_requests / uptime_hours if uptime_hours > 0 else 0
            
            # Document processing specific metrics
            total_file_size = sum(self.file_sizes) if self.file_sizes else 0
            avg_file_size = statistics.mean(self.file_sizes) if self.file_sizes else 0
            avg_chunk_count = statistics.mean(self.chunk_counts) if self.chunk_counts else 0
            avg_conversion_time = statistics.mean(self.conversion_times) if self.conversion_times else 0
            
            # Processing efficiency
            total_processing_time = sum(self.processing_times) if self.processing_times else 0
            mb_per_second = total_file_size / total_processing_time if total_processing_time > 0 else 0
            
            return {
                "total_requests": self.total_requests,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": success_rate,
                "error_rate": error_rate,
                "average_response_time": avg_response_time,
                "median_response_time": p50,
                "p95_response_time": p95,
                "p99_response_time": p99,
                "min_response_time": min(self.request_times),
                "max_response_time": max(self.request_times),
                "throughput_per_hour": throughput,
                "docling_load_time": self.docling_load_time,
                "uptime_seconds": time.time() - self.startup_time,
                "slow_requests": sum(1 for t in self.request_times if t > self.slow_request_threshold),
                # Document processing specific metrics
                "total_file_size_mb": total_file_size,
                "average_file_size_mb": avg_file_size,
                "average_chunk_count": avg_chunk_count,
                "average_conversion_time": avg_conversion_time,
                "processing_efficiency_mb_per_sec": mb_per_second
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform performance health check"""
        try:
            if self.total_requests == 0:
                return {
                    "healthy": True,
                    "status": "no_requests_yet",
                    "message": "No requests processed yet"
                }
            
            # Calculate current metrics
            error_rate = self.error_count / self.total_requests
            avg_response_time = statistics.mean(self.request_times) if self.request_times else 0
            
            # Determine health status
            healthy = True
            issues = []
            
            if error_rate > self.error_rate_threshold:
                healthy = False
                issues.append(f"High error rate: {error_rate:.2%}")
            
            if avg_response_time > self.slow_request_threshold:
                healthy = False
                issues.append(f"Slow average response time: {avg_response_time:.2f}s")
            
            # Check document processing efficiency
            if self.file_sizes and self.processing_times:
                avg_file_size = statistics.mean(self.file_sizes)
                avg_processing_time = statistics.mean(self.processing_times)
                if avg_file_size > 0:
                    efficiency = avg_file_size / avg_processing_time
                    if efficiency < 0.1:  # Less than 0.1 MB/sec
                        issues.append(f"Low processing efficiency: {efficiency:.3f} MB/sec")
            
            status = "healthy" if healthy else "performance_issues"
            
            return {
                "healthy": healthy,
                "status": status,
                "error_rate": error_rate,
                "average_response_time": avg_response_time,
                "issues": issues,
                "total_requests": self.total_requests
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform periodic performance analysis
                stats = await self.get_performance_stats()
                
                # Log performance summary every 10 minutes
                if self.total_requests > 0 and self.total_requests % 10 == 0:
                    logger.info(f"📊 Document Processing Performance Summary: "
                              f"{stats['total_requests']} requests, "
                              f"{stats['success_rate']:.2%} success rate, "
                              f"{stats['average_response_time']:.1f}s avg time, "
                              f"{stats.get('processing_efficiency_mb_per_sec', 0):.2f} MB/sec")
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(300)
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.request_times.clear()
        self.processing_times.clear()
        self.file_sizes.clear()
        self.chunk_counts.clear()
        self.conversion_times.clear()
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        self.startup_time = time.time()
        logger.info("📊 Performance statistics reset")
    
    def get_recent_performance(self, last_n: int = 20) -> Dict[str, Any]:
        """Get performance metrics for the last N requests"""
        try:
            if not self.request_times or len(self.request_times) < last_n:
                recent_times = list(self.request_times)
                recent_sizes = list(self.file_sizes)
                recent_chunks = list(self.chunk_counts)
            else:
                recent_times = list(self.request_times)[-last_n:]
                recent_sizes = list(self.file_sizes)[-last_n:]
                recent_chunks = list(self.chunk_counts)[-last_n:]
            
            if not recent_times:
                return {"message": "No recent requests"}
            
            total_size = sum(recent_sizes) if recent_sizes else 0
            total_time = sum(recent_times)
            
            return {
                "request_count": len(recent_times),
                "average_time": statistics.mean(recent_times),
                "median_time": statistics.median(recent_times),
                "min_time": min(recent_times),
                "max_time": max(recent_times),
                "slow_requests": sum(1 for t in recent_times if t > self.slow_request_threshold),
                "total_file_size_mb": total_size,
                "average_file_size_mb": statistics.mean(recent_sizes) if recent_sizes else 0,
                "average_chunk_count": statistics.mean(recent_chunks) if recent_chunks else 0,
                "processing_efficiency_mb_per_sec": total_size / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting recent performance: {e}")
            return {"error": str(e)}
