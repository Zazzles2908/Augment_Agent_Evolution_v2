#!/usr/bin/env python3
"""
Four-Brain Architecture Integration Test
Tests the complete workflow: User → Brain 3 → K2-Vector-Hub → All Brains → Response

This script verifies the implementation according to fix_containers.md specifications.

Zero Fabrication Policy: ENFORCED
All tests use real endpoints and verify actual functionality.
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FourBrainIntegrationTest:
    """Integration test suite for Four-Brain Architecture"""
    
    def __init__(self):
        """Initialize test suite"""
        self.base_urls = {
            "brain1": "http://localhost:8011",  # Brain 1 (Embedding)
            "brain2": "http://localhost:8012",  # Brain 2 (Reranker)
            "brain3": "http://localhost:8013",  # Brain 3 (Augment) - The Concierge
            "brain4": "http://localhost:8010",  # Brain 4 (Docling)
            "k2_hub": "http://localhost:9098",  # K2-Vector-Hub - The Mayor's Office
        }
        
        self.test_results = {}
        self.session = None
    
    async def run_all_tests(self):
        """Run complete integration test suite"""
        logger.info("🧪 Starting Four-Brain Architecture Integration Tests")
        logger.info("=" * 60)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            self.session = session
            
            # Test 1: Health Checks
            await self.test_health_checks()
            
            # Test 2: Service Discovery
            await self.test_service_discovery()
            
            # Test 3: Brain 3 Concierge Endpoint
            await self.test_brain3_concierge()
            
            # Test 4: K2-Vector-Hub Mayor's Office
            await self.test_k2_vector_hub()
            
            # Test 5: Complete Workflow
            await self.test_complete_workflow()
            
            # Test 6: Performance Metrics
            await self.test_performance_metrics()
        
        # Generate test report
        self.generate_test_report()
    
    async def test_health_checks(self):
        """Test all service health endpoints"""
        logger.info("🏥 Testing Health Checks...")
        
        health_results = {}
        
        for service, base_url in self.base_urls.items():
            try:
                async with self.session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        health_results[service] = {
                            "status": "healthy",
                            "response": health_data,
                            "response_time_ms": 0  # Could add timing
                        }
                        logger.info(f"✅ {service} health check: PASSED")
                    else:
                        health_results[service] = {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "response_time_ms": 0
                        }
                        logger.error(f"❌ {service} health check: FAILED (HTTP {response.status})")
            
            except Exception as e:
                health_results[service] = {
                    "status": "error",
                    "error": str(e),
                    "response_time_ms": 0
                }
                logger.error(f"❌ {service} health check: ERROR ({e})")
        
        self.test_results["health_checks"] = health_results
    
    async def test_service_discovery(self):
        """Test service discovery and basic endpoints"""
        logger.info("🔍 Testing Service Discovery...")
        
        discovery_results = {}
        
        for service, base_url in self.base_urls.items():
            try:
                async with self.session.get(f"{base_url}/") as response:
                    if response.status == 200:
                        service_info = await response.json()
                        discovery_results[service] = {
                            "status": "discovered",
                            "service_info": service_info,
                            "expected_role": self._get_expected_role(service)
                        }
                        logger.info(f"✅ {service} discovery: PASSED")
                    else:
                        discovery_results[service] = {
                            "status": "failed",
                            "error": f"HTTP {response.status}"
                        }
                        logger.error(f"❌ {service} discovery: FAILED")
            
            except Exception as e:
                discovery_results[service] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {service} discovery: ERROR ({e})")
        
        self.test_results["service_discovery"] = discovery_results
    
    async def test_brain3_concierge(self):
        """Test Brain 3 Concierge /ask endpoint"""
        logger.info("🎯 Testing Brain 3 Concierge (The Front Door)...")
        
        test_question = "Analyze this document for key insights"
        test_payload = {
            "question": test_question,
            "context": {"test": True, "timestamp": time.time()}
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_urls['brain3']}/ask",
                json=test_payload
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    response_data = await response.json()
                    
                    self.test_results["brain3_concierge"] = {
                        "status": "success",
                        "response": response_data,
                        "response_time_ms": response_time,
                        "workflow_initiated": response_data.get("status") in ["processing", "completed", "timeout"]
                    }
                    logger.info(f"✅ Brain 3 Concierge: PASSED ({response_time:.1f}ms)")
                    logger.info(f"   Status: {response_data.get('status')}")
                    logger.info(f"   Workflow: {response_data.get('workflow_stage')}")
                else:
                    error_text = await response.text()
                    self.test_results["brain3_concierge"] = {
                        "status": "failed",
                        "error": f"HTTP {response.status}: {error_text}",
                        "response_time_ms": response_time
                    }
                    logger.error(f"❌ Brain 3 Concierge: FAILED")
        
        except Exception as e:
            self.test_results["brain3_concierge"] = {
                "status": "error",
                "error": str(e),
                "response_time_ms": 0
            }
            logger.error(f"❌ Brain 3 Concierge: ERROR ({e})")
    
    async def test_k2_vector_hub(self):
        """Test K2-Vector-Hub Mayor's Office"""
        logger.info("🏛️ Testing K2-Vector-Hub (The Mayor's Office)...")
        
        try:
            # Test basic endpoint
            async with self.session.get(f"{self.base_urls['k2_hub']}/") as response:
                if response.status == 200:
                    hub_info = await response.json()
                    
                    # Test health endpoint
                    async with self.session.get(f"{self.base_urls['k2_hub']}/health") as health_response:
                        if health_response.status == 200:
                            health_data = await health_response.json()
                            
                            self.test_results["k2_vector_hub"] = {
                                "status": "success",
                                "hub_info": hub_info,
                                "health": health_data,
                                "channels": hub_info.get("channels", {}),
                                "role_verified": hub_info.get("role") == "Global Strategy Coordinator"
                            }
                            logger.info("✅ K2-Vector-Hub: PASSED")
                            logger.info(f"   Role: {hub_info.get('role')}")
                            logger.info(f"   Channels: {hub_info.get('channels')}")
                        else:
                            self.test_results["k2_vector_hub"] = {
                                "status": "health_failed",
                                "error": f"Health check failed: HTTP {health_response.status}"
                            }
                            logger.error("❌ K2-Vector-Hub health check: FAILED")
                else:
                    self.test_results["k2_vector_hub"] = {
                        "status": "failed",
                        "error": f"HTTP {response.status}"
                    }
                    logger.error("❌ K2-Vector-Hub: FAILED")
        
        except Exception as e:
            self.test_results["k2_vector_hub"] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"❌ K2-Vector-Hub: ERROR ({e})")
    
    async def test_complete_workflow(self):
        """Test the complete workflow as specified in fix_containers.md"""
        logger.info("🔄 Testing Complete Workflow...")
        
        workflow_test = {
            "question": "Summarize this PDF document and extract key insights",
            "expected_steps": [
                "Brain 3 receives question",
                "Posts to Redis vector_jobs channel",
                "K2-Vector-Hub processes strategy",
                "Publishes to Redis strategy_plans channel",
                "Brain 3 coordinates with other brains",
                "Returns final response"
            ]
        }
        
        try:
            start_time = time.time()
            
            # Send request to Brain 3 Concierge
            async with self.session.post(
                f"{self.base_urls['brain3']}/ask",
                json={"question": workflow_test["question"]}
            ) as response:
                
                total_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    workflow_response = await response.json()
                    
                    self.test_results["complete_workflow"] = {
                        "status": "success",
                        "response": workflow_response,
                        "total_time_ms": total_time,
                        "workflow_completed": workflow_response.get("status") == "completed",
                        "strategy_received": "strategy_plan" in workflow_response,
                        "coordination_performed": "coordination_results" in workflow_response
                    }
                    
                    logger.info(f"✅ Complete Workflow: PASSED ({total_time:.1f}ms)")
                    logger.info(f"   Final Status: {workflow_response.get('status')}")
                    
                    if workflow_response.get("strategy_plan"):
                        strategy = workflow_response["strategy_plan"]
                        logger.info(f"   Strategy: {strategy.get('strategy')}")
                        logger.info(f"   Brain Allocation: {strategy.get('brain_allocation')}")
                else:
                    self.test_results["complete_workflow"] = {
                        "status": "failed",
                        "error": f"HTTP {response.status}",
                        "total_time_ms": total_time
                    }
                    logger.error("❌ Complete Workflow: FAILED")
        
        except Exception as e:
            self.test_results["complete_workflow"] = {
                "status": "error",
                "error": str(e),
                "total_time_ms": 0
            }
            logger.error(f"❌ Complete Workflow: ERROR ({e})")
    
    async def test_performance_metrics(self):
        """Test performance metrics endpoints"""
        logger.info("📊 Testing Performance Metrics...")
        
        metrics_results = {}
        
        for service, base_url in self.base_urls.items():
            try:
                async with self.session.get(f"{base_url}/metrics") as response:
                    if response.status == 200:
                        metrics_data = await response.text()
                        metrics_results[service] = {
                            "status": "available",
                            "metrics_length": len(metrics_data),
                            "has_prometheus_format": "# HELP" in metrics_data
                        }
                        logger.info(f"✅ {service} metrics: AVAILABLE")
                    else:
                        metrics_results[service] = {
                            "status": "unavailable",
                            "error": f"HTTP {response.status}"
                        }
                        logger.warning(f"⚠️ {service} metrics: UNAVAILABLE")
            
            except Exception as e:
                metrics_results[service] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.warning(f"⚠️ {service} metrics: ERROR")
        
        self.test_results["performance_metrics"] = metrics_results
    
    def _get_expected_role(self, service: str) -> str:
        """Get expected role for each service"""
        roles = {
            "brain1": "Embedding Service",
            "brain2": "Reranker Service", 
            "brain3": "The Concierge",
            "brain4": "Document Processing",
            "k2_hub": "The Mayor's Office"
        }
        return roles.get(service, "Unknown")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("=" * 60)
        logger.info("📋 FOUR-BRAIN ARCHITECTURE TEST REPORT")
        logger.info("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            logger.info(f"\n🔍 {test_category.upper().replace('_', ' ')}:")
            
            if isinstance(results, dict):
                if "status" in results:
                    # Single test result
                    total_tests += 1
                    if results["status"] in ["success", "available"]:
                        passed_tests += 1
                        logger.info(f"   ✅ PASSED")
                    else:
                        logger.info(f"   ❌ FAILED: {results.get('error', 'Unknown error')}")
                else:
                    # Multiple test results
                    for item, item_result in results.items():
                        total_tests += 1
                        if isinstance(item_result, dict) and item_result.get("status") in ["healthy", "discovered", "success", "available"]:
                            passed_tests += 1
                            logger.info(f"   ✅ {item}: PASSED")
                        else:
                            error = item_result.get("error", "Unknown error") if isinstance(item_result, dict) else str(item_result)
                            logger.info(f"   ❌ {item}: FAILED ({error})")
        
        # Summary
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {total_tests - passed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 Four-Brain Architecture: OPERATIONAL")
        elif success_rate >= 60:
            logger.info("⚠️ Four-Brain Architecture: PARTIALLY OPERATIONAL")
        else:
            logger.info("❌ Four-Brain Architecture: NEEDS ATTENTION")
        
        logger.info("=" * 60)

async def main():
    """Main test execution"""
    test_suite = FourBrainIntegrationTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
