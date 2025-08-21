#!/usr/bin/env python3
"""
Deployment and Validation Script for Enhanced Four-Brain System
Deploys the system and runs comprehensive validation tests

This script handles the complete deployment process including:
- Docker container startup
- System health validation
- Component testing
- Performance benchmarking
- Integration validation

Zero Fabrication Policy: ENFORCED
All validations use real system checks and actual performance measurements.
"""

import asyncio
import subprocess
import time
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class SystemDeployer:
    """Handles deployment and validation of the Four-Brain system"""
    
    def __init__(self):
        """Initialize the deployer"""
        self.deployment_start_time = time.time()
        self.validation_results = {}
        self.performance_metrics = {}
        
        # Deployment configuration
        self.container_name = "four-brain-system-v6-fixed"
        self.redis_port = 6379
        self.health_check_timeout = 300  # 5 minutes
        self.validation_timeout = 600    # 10 minutes
        
        print("🚀 Four-Brain System Deployer Initialized")
        print(f"📅 Deployment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def deploy_system(self) -> bool:
        """Deploy the Four-Brain system"""
        try:
            print("\n" + "="*60)
            print("🔧 PHASE 1: SYSTEM DEPLOYMENT")
            print("="*60)
            
            # Step 1: Check Docker availability
            if not await self._check_docker_available():
                print("❌ Docker is not available")
                return False
            
            # Step 2: Start Redis if needed
            if not await self._ensure_redis_running():
                print("❌ Failed to start Redis")
                return False
            
            # Step 3: Start Four-Brain container
            if not await self._start_four_brain_container():
                print("❌ Failed to start Four-Brain container")
                return False
            
            # Step 4: Wait for system initialization
            if not await self._wait_for_system_ready():
                print("❌ System failed to initialize")
                return False
            
            print("✅ System deployment completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    async def validate_system(self) -> bool:
        """Validate the deployed system"""
        try:
            print("\n" + "="*60)
            print("🔍 PHASE 2: SYSTEM VALIDATION")
            print("="*60)
            
            # Component validation
            validation_tasks = [
                ("Redis Connectivity", self._validate_redis),
                ("Container Health", self._validate_container_health),
                ("Python Environment", self._validate_python_environment),
                ("Module Imports", self._validate_module_imports),
                ("Core Components", self._validate_core_components),
                ("Integration Tests", self._run_integration_tests)
            ]
            
            all_passed = True
            for test_name, test_func in validation_tasks:
                print(f"\n🧪 Running: {test_name}")
                try:
                    result = await test_func()
                    if result:
                        print(f"✅ {test_name}: PASSED")
                        self.validation_results[test_name] = "PASSED"
                    else:
                        print(f"❌ {test_name}: FAILED")
                        self.validation_results[test_name] = "FAILED"
                        all_passed = False
                except Exception as e:
                    print(f"❌ {test_name}: ERROR - {e}")
                    self.validation_results[test_name] = f"ERROR: {e}"
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    async def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark system performance"""
        try:
            print("\n" + "="*60)
            print("📊 PHASE 3: PERFORMANCE BENCHMARKING")
            print("="*60)
            
            benchmarks = {}
            
            # Container resource usage
            container_stats = await self._get_container_stats()
            if container_stats:
                benchmarks["container_stats"] = container_stats
                print(f"📈 Container Memory: {container_stats.get('memory_usage', 'N/A')}")
                print(f"📈 Container CPU: {container_stats.get('cpu_usage', 'N/A')}")
            
            # Redis performance
            redis_perf = await self._benchmark_redis()
            if redis_perf:
                benchmarks["redis_performance"] = redis_perf
                print(f"📈 Redis Latency: {redis_perf.get('latency_ms', 'N/A')}ms")
            
            # Python import time
            import_time = await self._benchmark_imports()
            benchmarks["import_time"] = import_time
            print(f"📈 Module Import Time: {import_time:.2f}s")
            
            self.performance_metrics = benchmarks
            return benchmarks
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            return {}
    
    async def generate_report(self) -> str:
        """Generate deployment and validation report"""
        try:
            print("\n" + "="*60)
            print("📋 PHASE 4: GENERATING REPORT")
            print("="*60)
            
            deployment_time = time.time() - self.deployment_start_time
            
            report = {
                "deployment_info": {
                    "timestamp": datetime.now().isoformat(),
                    "deployment_time_seconds": deployment_time,
                    "container_name": self.container_name,
                    "status": "SUCCESS" if all(
                        result == "PASSED" for result in self.validation_results.values()
                    ) else "PARTIAL_SUCCESS"
                },
                "validation_results": self.validation_results,
                "performance_metrics": self.performance_metrics,
                "system_info": await self._get_system_info()
            }
            
            # Save report to file
            report_file = f"deployment_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Report saved to: {report_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("📊 DEPLOYMENT SUMMARY")
            print("="*60)
            print(f"⏱️  Total Time: {deployment_time:.1f} seconds")
            print(f"🧪 Tests Run: {len(self.validation_results)}")
            
            passed_tests = sum(1 for result in self.validation_results.values() if result == "PASSED")
            print(f"✅ Tests Passed: {passed_tests}/{len(self.validation_results)}")
            
            if passed_tests == len(self.validation_results):
                print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
            else:
                print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            
            return report_file
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return ""
    
    # Helper methods
    async def _check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Docker available: {result.stdout.strip()}")
                return True
            return False
        except Exception:
            return False
    
    async def _ensure_redis_running(self) -> bool:
        """Ensure Redis is running"""
        try:
            # Check if Redis is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=redis", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if "redis" in result.stdout:
                print("✅ Redis container already running")
                return True
            
            # Start Redis container
            print("🔄 Starting Redis container...")
            result = subprocess.run([
                "docker", "run", "-d", "--name", "redis",
                "-p", f"{self.redis_port}:6379",
                "redis:7-alpine"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Redis container started")
                await asyncio.sleep(3)  # Wait for Redis to initialize
                return True
            
            print(f"❌ Failed to start Redis: {result.stderr}")
            return False
            
        except Exception as e:
            print(f"❌ Redis setup error: {e}")
            return False
    
    async def _start_four_brain_container(self) -> bool:
        """Start the Four-Brain container"""
        try:
            print("🔄 Starting Four-Brain container...")
            
            # Remove existing container if present
            subprocess.run(
                ["docker", "rm", "-f", "four-brain-test"],
                capture_output=True, timeout=10
            )
            
            # Start new container
            result = subprocess.run([
                "docker", "run", "-d", "--name", "four-brain-test",
                "--link", "redis:redis",
                "-e", "REDIS_URL=redis://redis:6379/0",
                "-e", "PYTHONPATH=/workspace/src",
                self.container_name,
                "tail", "-f", "/dev/null"  # Keep container running
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Four-Brain container started")
                return True
            
            print(f"❌ Failed to start container: {result.stderr}")
            return False
            
        except Exception as e:
            print(f"❌ Container startup error: {e}")
            return False
    
    async def _wait_for_system_ready(self) -> bool:
        """Wait for system to be ready"""
        print("⏳ Waiting for system initialization...")
        
        for i in range(30):  # Wait up to 30 seconds
            try:
                # Check if container is running
                result = subprocess.run([
                    "docker", "exec", "four-brain-test", "python", "-c", 
                    "import sys; print('Python ready')"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "Python ready" in result.stdout:
                    print("✅ System ready")
                    return True
                
            except Exception:
                pass
            
            await asyncio.sleep(1)
            print(f"⏳ Waiting... ({i+1}/30)")
        
        print("❌ System initialization timeout")
        return False
    
    async def _validate_redis(self) -> bool:
        """Validate Redis connectivity"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                "import redis; r = redis.Redis(host='redis', port=6379); r.ping(); print('Redis OK')"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0 and "Redis OK" in result.stdout
        except Exception:
            return False
    
    async def _validate_container_health(self) -> bool:
        """Validate container health"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "--version"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0 and "Python" in result.stdout
        except Exception:
            return False
    
    async def _validate_python_environment(self) -> bool:
        """Validate Python environment"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                "import sys; print(f'Python {sys.version}'); import torch; print(f'PyTorch {torch.__version__}')"
            ], capture_output=True, text=True, timeout=15)
            
            return result.returncode == 0 and "Python" in result.stdout and "PyTorch" in result.stdout
        except Exception:
            return False
    
    async def _validate_module_imports(self) -> bool:
        """Validate module imports"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                """
import sys
sys.path.insert(0, '/workspace/src')
from shared.streams import StreamNames
from shared.redis_client import RedisStreamsClient
from shared.memory_store import MemoryStore
from shared.self_grading import SelfGradingEngine
from shared.self_improvement import SelfImprovementEngine
from shared.cli_executor import CLIExecutor
from shared.message_flow import MessageFlowOrchestrator
print('All modules imported successfully')
                """
            ], capture_output=True, text=True, timeout=20)
            
            return result.returncode == 0 and "All modules imported successfully" in result.stdout
        except Exception:
            return False
    
    async def _validate_core_components(self) -> bool:
        """Validate core components"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                """
import sys
sys.path.insert(0, '/workspace/src')
from shared.cli_executor import CLIExecutor
from shared.self_grading import SelfGradingEngine

# Test CLI executor
cli = CLIExecutor('/tmp')
tools = cli.get_available_tools()
assert len(tools) > 0

# Test grading engine
grader = SelfGradingEngine()
assert grader is not None

print('Core components validated')
                """
            ], capture_output=True, text=True, timeout=20)
            
            return result.returncode == 0 and "Core components validated" in result.stdout
        except Exception:
            return False
    
    async def _run_integration_tests(self) -> bool:
        """Run integration tests"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "/workspace/tests/test_enhanced_system.py"
            ], capture_output=True, text=True, timeout=60)
            
            return result.returncode == 0 and "All available tests completed successfully" in result.stdout
        except Exception:
            return False
    
    async def _get_container_stats(self) -> Optional[Dict[str, Any]]:
        """Get container resource statistics"""
        try:
            result = subprocess.run([
                "docker", "stats", "four-brain-test", "--no-stream", "--format",
                "table {{.MemUsage}}\t{{.CPUPerc}}"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    stats_line = lines[1].split('\t')
                    return {
                        "memory_usage": stats_line[0] if len(stats_line) > 0 else "N/A",
                        "cpu_usage": stats_line[1] if len(stats_line) > 1 else "N/A"
                    }
            return None
        except Exception:
            return None
    
    async def _benchmark_redis(self) -> Optional[Dict[str, Any]]:
        """Benchmark Redis performance"""
        try:
            start_time = time.time()
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                """
import redis
import time
r = redis.Redis(host='redis', port=6379)
start = time.time()
for i in range(100):
    r.set(f'test_key_{i}', f'test_value_{i}')
    r.get(f'test_key_{i}')
end = time.time()
print(f'Redis benchmark: {(end-start)*1000:.2f}ms for 200 operations')
                """
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "Redis benchmark:" in result.stdout:
                # Extract latency from output
                output = result.stdout.strip()
                latency_str = output.split("Redis benchmark: ")[1].split("ms")[0]
                return {"latency_ms": float(latency_str)}
            return None
        except Exception:
            return None
    
    async def _benchmark_imports(self) -> float:
        """Benchmark module import time"""
        try:
            start_time = time.time()
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                """
import time
start = time.time()
import sys
sys.path.insert(0, '/workspace/src')
from shared.streams import StreamNames
from shared.redis_client import RedisStreamsClient
from shared.memory_store import MemoryStore
end = time.time()
print(f'Import time: {end-start:.3f}s')
                """
            ], capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0 and "Import time:" in result.stdout:
                output = result.stdout.strip()
                time_str = output.split("Import time: ")[1].split("s")[0]
                return float(time_str)
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            result = subprocess.run([
                "docker", "exec", "four-brain-test", "python", "-c",
                """
import sys, platform, os
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
print(f'Processor: {platform.processor()}')
                """
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return {
                    "python_version": lines[0] if len(lines) > 0 else "Unknown",
                    "platform": lines[1] if len(lines) > 1 else "Unknown",
                    "architecture": lines[2] if len(lines) > 2 else "Unknown",
                    "processor": lines[3] if len(lines) > 3 else "Unknown"
                }
            return {}
        except Exception:
            return {}

async def main():
    """Main deployment and validation function"""
    deployer = SystemDeployer()
    
    try:
        # Phase 1: Deploy
        if not await deployer.deploy_system():
            print("❌ Deployment failed - aborting")
            return False
        
        # Phase 2: Validate
        if not await deployer.validate_system():
            print("⚠️  Validation completed with issues")
        
        # Phase 3: Benchmark
        await deployer.benchmark_performance()
        
        # Phase 4: Report
        report_file = await deployer.generate_report()
        
        print(f"\n🎯 Deployment process completed!")
        print(f"📄 Full report available in: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment process failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
