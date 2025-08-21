#!/usr/bin/env python3
"""
Four-Brain System Health Check
Comprehensive health validation for all system components

Created: 2025-07-27 AEST
Author: AugmentAI - System Health Implementation
"""

import asyncio
import aiohttp
import redis
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    response_time_ms: Optional[float]
    details: Dict[str, Any]
    error: Optional[str] = None

class FourBrainHealthChecker:
    """Comprehensive health checker for Four-Brain system"""
    
    def __init__(self):
        self.brain_endpoints = {
            "brain1_embedding": "http://localhost:8001/health",
            "brain2_reranker": "http://localhost:8002/health", 
            "brain3_augment": "http://localhost:8003/health",
            "brain4_docling": "http://localhost:8004/health",
            "k2_vector_hub": "http://localhost:9098/health"
        }
        
        self.infrastructure_endpoints = {
            "grafana": "http://localhost:3000/api/health",
            "prometheus": "http://localhost:9090/-/healthy",
            "loki": "http://localhost:3100/ready"
        }
        
        self.redis_config = {
            "host": "localhost",
            "port": 6379,
            "decode_responses": True
        }
        
        self.expected_streams = [
            "embedding_requests", "embedding_results",
            "rerank_requests", "rerank_results", 
            "docling_requests", "docling_results",
            "agentic_tasks", "agentic_results",
            "memory_updates"
        ]

    async def check_http_endpoint(self, name: str, url: str, timeout: int = 10) -> ComponentHealth:
        """Check HTTP endpoint health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return ComponentHealth(
                                name=name,
                                status=HealthStatus.HEALTHY,
                                response_time_ms=response_time,
                                details={"status_code": response.status, "response": data}
                            )
                        except:
                            return ComponentHealth(
                                name=name,
                                status=HealthStatus.HEALTHY,
                                response_time_ms=response_time,
                                details={"status_code": response.status, "response": "non-json"}
                            )
                    else:
                        return ComponentHealth(
                            name=name,
                            status=HealthStatus.DEGRADED,
                            response_time_ms=response_time,
                            details={"status_code": response.status},
                            error=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                details={},
                error="Timeout"
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                details={},
                error=str(e)
            )

    def check_redis_health(self) -> ComponentHealth:
        """Check Redis connectivity and streams"""
        try:
            r = redis.Redis(**self.redis_config)
            
            # Test basic connectivity
            start_time = time.time()
            r.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Check stream status
            stream_status = {}
            total_messages = 0
            
            for stream in self.expected_streams:
                try:
                    length = r.xlen(stream)
                    stream_status[stream] = length
                    total_messages += length
                except:
                    stream_status[stream] = "error"
            
            # Determine health status
            if total_messages > 0:
                status = HealthStatus.HEALTHY
            elif any(v == "error" for v in stream_status.values()):
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
                
            return ComponentHealth(
                name="redis_streams",
                status=status,
                response_time_ms=response_time,
                details={
                    "streams": stream_status,
                    "total_messages": total_messages,
                    "expected_streams": len(self.expected_streams),
                    "active_streams": len([v for v in stream_status.values() if isinstance(v, int)])
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="redis_streams",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                details={},
                error=str(e)
            )

    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run complete system health check"""
        logger.info("🔍 Starting comprehensive Four-Brain health check...")
        
        results = {
            "timestamp": time.time(),
            "overall_status": HealthStatus.UNKNOWN,
            "components": {},
            "summary": {}
        }
        
        # Check all brain services
        logger.info("Checking brain services...")
        brain_tasks = [
            self.check_http_endpoint(name, url) 
            for name, url in self.brain_endpoints.items()
        ]
        brain_results = await asyncio.gather(*brain_tasks)
        
        for result in brain_results:
            results["components"][result.name] = result
        
        # Check infrastructure services
        logger.info("Checking infrastructure services...")
        infra_tasks = [
            self.check_http_endpoint(name, url)
            for name, url in self.infrastructure_endpoints.items()
        ]
        infra_results = await asyncio.gather(*infra_tasks)
        
        for result in infra_results:
            results["components"][result.name] = result
        
        # Check Redis
        logger.info("Checking Redis streams...")
        redis_result = self.check_redis_health()
        results["components"]["redis_streams"] = redis_result
        
        # Calculate overall status
        component_statuses = [comp.status for comp in results["components"].values()]
        healthy_count = sum(1 for status in component_statuses if status == HealthStatus.HEALTHY)
        total_count = len(component_statuses)
        
        if healthy_count == total_count:
            results["overall_status"] = HealthStatus.HEALTHY
        elif healthy_count >= total_count * 0.7:  # 70% healthy threshold
            results["overall_status"] = HealthStatus.DEGRADED
        else:
            results["overall_status"] = HealthStatus.UNHEALTHY
        
        # Generate summary
        results["summary"] = {
            "total_components": total_count,
            "healthy_components": healthy_count,
            "degraded_components": sum(1 for s in component_statuses if s == HealthStatus.DEGRADED),
            "unhealthy_components": sum(1 for s in component_statuses if s == HealthStatus.UNHEALTHY),
            "health_percentage": (healthy_count / total_count) * 100,
            "ai_communication_active": redis_result.details.get("total_messages", 0) > 0
        }
        
        logger.info(f"✅ Health check complete: {results['overall_status'].value}")
        return results

    def format_health_report(self, results: Dict[str, Any]) -> str:
        """Format health check results as readable report"""
        report = []
        report.append("🧠 FOUR-BRAIN SYSTEM HEALTH REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {time.ctime(results['timestamp'])}")
        report.append(f"Overall Status: {results['overall_status'].value.upper()}")
        report.append(f"Health Percentage: {results['summary']['health_percentage']:.1f}%")
        report.append("")
        
        # Component details
        report.append("📊 COMPONENT STATUS:")
        for name, component in results["components"].items():
            status_icon = "✅" if component.status == HealthStatus.HEALTHY else "⚠️" if component.status == HealthStatus.DEGRADED else "❌"
            response_time = f" ({component.response_time_ms:.1f}ms)" if component.response_time_ms else ""
            error_info = f" - {component.error}" if component.error else ""
            report.append(f"{status_icon} {name}: {component.status.value}{response_time}{error_info}")
        
        report.append("")
        
        # AI Communication Status
        redis_component = results["components"].get("redis_streams")
        if redis_component and redis_component.details:
            report.append("🔄 AI COMMUNICATION STATUS:")
            total_messages = redis_component.details.get("total_messages", 0)
            report.append(f"Total Messages: {total_messages}")
            
            if "streams" in redis_component.details:
                for stream, count in redis_component.details["streams"].items():
                    report.append(f"  {stream}: {count}")
        
        return "\n".join(report)

async def main():
    """Main health check execution"""
    checker = FourBrainHealthChecker()
    results = await checker.run_comprehensive_health_check()
    
    # Print formatted report
    report = checker.format_health_report(results)
    print(report)
    
    # Save results to file
    with open("health_check_results.json", "w") as f:
        # Convert ComponentHealth objects to dicts for JSON serialization
        json_results = {
            "timestamp": results["timestamp"],
            "overall_status": results["overall_status"].value,
            "components": {
                name: {
                    "name": comp.name,
                    "status": comp.status.value,
                    "response_time_ms": comp.response_time_ms,
                    "details": comp.details,
                    "error": comp.error
                }
                for name, comp in results["components"].items()
            },
            "summary": results["summary"]
        }
        json.dump(json_results, f, indent=2)
    
    logger.info("Health check results saved to health_check_results.json")
    
    # Return exit code based on health
    if results["overall_status"] == HealthStatus.HEALTHY:
        return 0
    elif results["overall_status"] == HealthStatus.DEGRADED:
        return 1
    else:
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
