"""
Performance Monitor Module for Brain-2
Handles performance tracking, metrics collection, and optimization insights

Extracted from brain2_manager.py for modular architecture.
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
    Performance Monitor for Brain-2
    Tracks and analyzes performance metrics for modular architecture
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize Performance Monitor"""
        self.max_history = max_history
        
        # Performance tracking
        self.request_times = deque(maxlen=max_history)
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        
        # Reranking-specific metrics
        self.reranking_times = deque(maxlen=max_history)
        self.documents_processed = deque(maxlen=max_history)
        self.batch_sizes = deque(maxlen=max_history)
        self.model_load_time = 0.0
        self.startup_time = time.time()
        
        # Performance thresholds
        self.slow_request_threshold = 1.0  # seconds (reranking is faster than embedding)
        self.error_rate_threshold = 0.05   # 5%
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("🔧 Performance Monitor (Brain-2) initialized")
    
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
    
    async def record_reranking_request(self, processing_time: float, 
                                     document_count: int, batch_size: int,
                                     success: bool = True, error: str = None):
        """Record a reranking request for performance tracking"""
        try:
            self.total_requests += 1
            self.request_times.append(processing_time)
            self.reranking_times.append(processing_time)
            self.documents_processed.append(document_count)
            self.batch_sizes.append(batch_size)
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                if error:
                    logger.debug(f"📊 Request error recorded: {error}")
            
            # Log slow requests
            if processing_time > self.slow_request_threshold:
                logger.warning(f"⚠️ Slow reranking request: {processing_time:.2f}s for {document_count} docs")
            
            # Log efficiency metrics
            if document_count > 0:
                docs_per_second = document_count / processing_time
                logger.debug(f"📊 Reranking efficiency: {docs_per_second:.1f} docs/sec")
            
        except Exception as e:
            logger.error(f"❌ Error recording request: {e}")
    
    async def record_model_load_time(self, load_time: float):
        """Record model loading time"""
        self.model_load_time = load_time
        logger.info(f"📊 Model load time recorded: {load_time:.2f}s")
    
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
            
            # Calculate throughput (requests per minute)
            uptime_minutes = (time.time() - self.startup_time) / 60
            throughput = self.total_requests / uptime_minutes if uptime_minutes > 0 else 0
            
            # Reranking-specific metrics
            total_docs = sum(self.documents_processed) if self.documents_processed else 0
            avg_docs_per_request = statistics.mean(self.documents_processed) if self.documents_processed else 0
            avg_batch_size = statistics.mean(self.batch_sizes) if self.batch_sizes else 0
            
            # Document processing efficiency
            total_processing_time = sum(self.reranking_times) if self.reranking_times else 0
            docs_per_second = total_docs / total_processing_time if total_processing_time > 0 else 0
            
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
                "throughput_per_minute": throughput,
                "model_load_time": self.model_load_time,
                "uptime_seconds": time.time() - self.startup_time,
                "slow_requests": sum(1 for t in self.request_times if t > self.slow_request_threshold),
                # Reranking-specific metrics
                "total_documents_processed": total_docs,
                "average_documents_per_request": avg_docs_per_request,
                "average_batch_size": avg_batch_size,
                "documents_per_second": docs_per_second
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
            
            # Check reranking efficiency
            if self.documents_processed:
                avg_docs = statistics.mean(self.documents_processed)
                if avg_docs < 5:
                    issues.append(f"Low document count per request: {avg_docs:.1f}")
            
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
                
                # Log performance summary every 5 minutes
                if self.total_requests > 0 and self.total_requests % 50 == 0:
                    logger.info(f"📊 Reranking Performance Summary: "
                              f"{stats['total_requests']} requests, "
                              f"{stats['success_rate']:.2%} success rate, "
                              f"{stats['average_response_time']:.2f}s avg time, "
                              f"{stats.get('documents_per_second', 0):.1f} docs/sec")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.request_times.clear()
        self.reranking_times.clear()
        self.documents_processed.clear()
        self.batch_sizes.clear()
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        self.startup_time = time.time()
        logger.info("📊 Performance statistics reset")
    
    def get_recent_performance(self, last_n: int = 50) -> Dict[str, Any]:
        """Get performance metrics for the last N requests"""
        try:
            if not self.request_times or len(self.request_times) < last_n:
                recent_times = list(self.request_times)
                recent_docs = list(self.documents_processed)
            else:
                recent_times = list(self.request_times)[-last_n:]
                recent_docs = list(self.documents_processed)[-last_n:]
            
            if not recent_times:
                return {"message": "No recent requests"}
            
            total_docs = sum(recent_docs) if recent_docs else 0
            total_time = sum(recent_times)
            
            return {
                "request_count": len(recent_times),
                "average_time": statistics.mean(recent_times),
                "median_time": statistics.median(recent_times),
                "min_time": min(recent_times),
                "max_time": max(recent_times),
                "slow_requests": sum(1 for t in recent_times if t > self.slow_request_threshold),
                "total_documents": total_docs,
                "documents_per_second": total_docs / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting recent performance: {e}")
            return {"error": str(e)}
