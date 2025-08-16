"""
Configuration Validator for Four-Brain System v2
Comprehensive environment and configuration validation

Created: 2025-07-30 AEST
Purpose: Validate system configuration, environment, and dependencies for optimal operation
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import redis.asyncio as aioredis
import aiohttp
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"

class ValidationCategory(Enum):
    """Configuration validation categories"""
    ENVIRONMENT = "environment"
    NETWORK = "network"
    DEPENDENCIES = "dependencies"
    RESOURCES = "resources"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    CUDA_COMPATIBILITY = "cuda_compatibility"  # Added for CUDA 13 validation

@dataclass
class ValidationResult:
    """Individual validation result"""
    check_id: str
    category: ValidationCategory
    level: ValidationLevel
    title: str
    description: str
    expected: str
    actual: str
    passed: bool
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ValidationReport:
    """Complete validation report"""
    report_id: str
    system_name: str
    validation_time: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    results: List[ValidationResult]
    overall_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class ConfigValidator:
    """
    Comprehensive configuration and environment validator
    
    Features:
    - Environment variable validation
    - Network connectivity testing
    - Dependency verification
    - Resource availability checking
    - Security configuration validation
    - Performance parameter verification
    - Integration endpoint testing
    - Comprehensive reporting with recommendations
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'min_memory_gb': 8,
            'min_disk_gb': 50,
            'required_ports': [6379, 5432, 8001, 8002, 8003, 8004, 9098],
            'required_env_vars': [
                'REDIS_URL', 'POSTGRES_URL', 'SUPABASE_URL', 'SUPABASE_ANON_KEY'
            ],
            'critical_services': ['redis', 'postgresql', 'docker'],
            'network_endpoints': [
                'http://localhost:8001/health',
                'http://localhost:8002/health', 
                'http://localhost:8003/health',
                'http://localhost:8004/health',
                'http://localhost:9098/health'
            ]
        }
        
        # Validation state
        self.validation_results: List[ValidationResult] = []
        self.validation_history: List[ValidationReport] = []
        
        # Performance metrics
        self.metrics = {
            'total_validations': 0,
            'average_validation_time': 0.0,
            'success_rate': 0.0,
            'critical_issues_found': 0,
            'recommendations_generated': 0
        }
        
        logger.info("🔍 Config Validator initialized")
    
    async def initialize(self):
        """Initialize Redis connection and validation services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load validation history
            await self._load_validation_history()
            
            logger.info("✅ Config Validator Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Config Validator: {e}")
            raise
    
    async def validate_system(self, validation_categories: Optional[List[ValidationCategory]] = None) -> ValidationReport:
        """Perform comprehensive system validation"""
        try:
            start_time = time.time()
            
            # Generate report ID
            report_id = f"validation_{int(time.time() * 1000)}"
            
            # Set default categories
            if not validation_categories:
                validation_categories = list(ValidationCategory)
            
            # Clear previous results
            self.validation_results = []
            
            # Run validation checks by category
            for category in validation_categories:
                await self._run_category_validations(category)
            
            # Generate comprehensive report
            validation_time = time.time() - start_time
            report = await self._generate_validation_report(report_id, validation_time)
            
            # Store report
            self.validation_history.append(report)
            await self._store_validation_report(report)
            
            # Update metrics
            self.metrics['total_validations'] += 1
            self._update_average_validation_time(validation_time)
            self._update_success_rate(report)
            
            logger.info(f"✅ System validation completed: {report.passed_checks}/{report.total_checks} checks passed")
            return report
            
        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            raise
    
    async def _run_category_validations(self, category: ValidationCategory):
        """Run validations for specific category"""
        try:
            if category == ValidationCategory.ENVIRONMENT:
                await self._validate_environment()
            elif category == ValidationCategory.NETWORK:
                await self._validate_network()
            elif category == ValidationCategory.DEPENDENCIES:
                await self._validate_dependencies()
            elif category == ValidationCategory.RESOURCES:
                await self._validate_resources()
            elif category == ValidationCategory.SECURITY:
                await self._validate_security()
            elif category == ValidationCategory.PERFORMANCE:
                await self._validate_performance()
            elif category == ValidationCategory.INTEGRATION:
                await self._validate_integration()
            
        except Exception as e:
            logger.error(f"❌ Category validation failed for {category.value}: {e}")
    
    async def _validate_environment(self):
        """Validate environment variables and configuration"""
        try:
            # Check required environment variables
            for env_var in self.config['required_env_vars']:
                value = os.getenv(env_var)
                
                result = ValidationResult(
                    check_id=f"env_{env_var.lower()}",
                    category=ValidationCategory.ENVIRONMENT,
                    level=ValidationLevel.CRITICAL if not value else ValidationLevel.SUCCESS,
                    title=f"Environment Variable: {env_var}",
                    description=f"Check if {env_var} is properly configured",
                    expected="Valid configuration value",
                    actual=f"{'Set' if value else 'Not set'} ({value[:20] + '...' if value and len(value) > 20 else value})",
                    passed=bool(value),
                    recommendations=[] if value else [f"Set {env_var} environment variable"],
                    metadata={'env_var': env_var, 'value_length': len(value) if value else 0},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
            # Check Python version
            import sys
            python_version = sys.version_info
            min_python = (3, 8)
            
            result = ValidationResult(
                check_id="python_version",
                category=ValidationCategory.ENVIRONMENT,
                level=ValidationLevel.SUCCESS if python_version >= min_python else ValidationLevel.ERROR,
                title="Python Version",
                description="Check Python version compatibility",
                expected=f"Python {min_python[0]}.{min_python[1]}+",
                actual=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                passed=python_version >= min_python,
                recommendations=[] if python_version >= min_python else ["Upgrade to Python 3.8 or higher"],
                metadata={'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}"},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
            # Check working directory
            cwd = os.getcwd()
            expected_path = "Augment_Agent_Evolution"
            
            result = ValidationResult(
                check_id="working_directory",
                category=ValidationCategory.ENVIRONMENT,
                level=ValidationLevel.SUCCESS if expected_path in cwd else ValidationLevel.WARNING,
                title="Working Directory",
                description="Check if running from correct directory",
                expected=f"Path containing '{expected_path}'",
                actual=cwd,
                passed=expected_path in cwd,
                recommendations=[] if expected_path in cwd else [f"Navigate to {expected_path} directory"],
                metadata={'current_dir': cwd},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
    
    async def _validate_network(self):
        """Validate network connectivity and ports"""
        try:
            # Check port availability
            for port in self.config['required_ports']:
                is_available = await self._check_port_availability('localhost', port)
                
                result = ValidationResult(
                    check_id=f"port_{port}",
                    category=ValidationCategory.NETWORK,
                    level=ValidationLevel.SUCCESS if is_available else ValidationLevel.ERROR,
                    title=f"Port {port} Availability",
                    description=f"Check if port {port} is accessible",
                    expected="Port accessible",
                    actual="Accessible" if is_available else "Not accessible",
                    passed=is_available,
                    recommendations=[] if is_available else [f"Ensure service on port {port} is running"],
                    metadata={'port': port, 'host': 'localhost'},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
            # Check network endpoints
            for endpoint in self.config['network_endpoints']:
                is_healthy = await self._check_endpoint_health(endpoint)
                
                result = ValidationResult(
                    check_id=f"endpoint_{endpoint.split('/')[-2]}",
                    category=ValidationCategory.NETWORK,
                    level=ValidationLevel.SUCCESS if is_healthy else ValidationLevel.WARNING,
                    title=f"Endpoint Health: {endpoint}",
                    description=f"Check health of {endpoint}",
                    expected="HTTP 200 response",
                    actual="Healthy" if is_healthy else "Unhealthy",
                    passed=is_healthy,
                    recommendations=[] if is_healthy else [f"Check service at {endpoint}"],
                    metadata={'endpoint': endpoint},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Network validation failed: {e}")
    
    async def _validate_dependencies(self):
        """Validate system dependencies"""
        try:
            # Check Docker
            docker_available = await self._check_command_available('docker')
            
            result = ValidationResult(
                check_id="docker_availability",
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.CRITICAL if not docker_available else ValidationLevel.SUCCESS,
                title="Docker Availability",
                description="Check if Docker is installed and accessible",
                expected="Docker command available",
                actual="Available" if docker_available else "Not available",
                passed=docker_available,
                recommendations=[] if docker_available else ["Install Docker Desktop"],
                metadata={'command': 'docker'},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
            # Check Git
            git_available = await self._check_command_available('git')
            
            result = ValidationResult(
                check_id="git_availability",
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR if not git_available else ValidationLevel.SUCCESS,
                title="Git Availability",
                description="Check if Git is installed and accessible",
                expected="Git command available",
                actual="Available" if git_available else "Not available",
                passed=git_available,
                recommendations=[] if git_available else ["Install Git"],
                metadata={'command': 'git'},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
            # Check Python packages
            required_packages = ['redis', 'psycopg2', 'fastapi', 'uvicorn', 'numpy']
            
            for package in required_packages:
                package_available = await self._check_python_package(package)
                
                result = ValidationResult(
                    check_id=f"package_{package}",
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR if not package_available else ValidationLevel.SUCCESS,
                    title=f"Python Package: {package}",
                    description=f"Check if {package} is installed",
                    expected="Package installed",
                    actual="Installed" if package_available else "Not installed",
                    passed=package_available,
                    recommendations=[] if package_available else [f"Install {package}: pip install {package}"],
                    metadata={'package': package},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Dependencies validation failed: {e}")
    
    async def _validate_resources(self):
        """Validate system resources"""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            min_memory = self.config['min_memory_gb']
            
            result = ValidationResult(
                check_id="system_memory",
                category=ValidationCategory.RESOURCES,
                level=ValidationLevel.SUCCESS if memory_gb >= min_memory else ValidationLevel.WARNING,
                title="System Memory",
                description="Check available system memory",
                expected=f"{min_memory}+ GB",
                actual=f"{memory_gb:.1f} GB",
                passed=memory_gb >= min_memory,
                recommendations=[] if memory_gb >= min_memory else [f"Consider upgrading to {min_memory}+ GB RAM"],
                metadata={'total_gb': memory_gb, 'available_gb': memory.available / (1024**3)},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            min_disk = self.config['min_disk_gb']
            
            result = ValidationResult(
                check_id="disk_space",
                category=ValidationCategory.RESOURCES,
                level=ValidationLevel.SUCCESS if disk_gb >= min_disk else ValidationLevel.WARNING,
                title="Disk Space",
                description="Check available disk space",
                expected=f"{min_disk}+ GB free",
                actual=f"{disk_gb:.1f} GB free",
                passed=disk_gb >= min_disk,
                recommendations=[] if disk_gb >= min_disk else ["Free up disk space"],
                metadata={'total_gb': disk.total / (1024**3), 'free_gb': disk_gb},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            min_cpu = 4
            
            result = ValidationResult(
                check_id="cpu_cores",
                category=ValidationCategory.RESOURCES,
                level=ValidationLevel.SUCCESS if cpu_count >= min_cpu else ValidationLevel.INFO,
                title="CPU Cores",
                description="Check number of CPU cores",
                expected=f"{min_cpu}+ cores",
                actual=f"{cpu_count} cores",
                passed=cpu_count >= min_cpu,
                recommendations=[] if cpu_count >= min_cpu else ["Consider upgrading CPU for better performance"],
                metadata={'cpu_count': cpu_count},
                timestamp=datetime.now()
            )
            
            self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Resources validation failed: {e}")
    
    async def _validate_security(self):
        """Validate security configuration"""
        try:
            # Check file permissions
            sensitive_files = ['.env', 'config.yaml', 'secrets.json']
            
            for file_name in sensitive_files:
                if os.path.exists(file_name):
                    file_stat = os.stat(file_name)
                    permissions = oct(file_stat.st_mode)[-3:]
                    secure_permissions = permissions in ['600', '644']
                    
                    result = ValidationResult(
                        check_id=f"file_permissions_{file_name}",
                        category=ValidationCategory.SECURITY,
                        level=ValidationLevel.SUCCESS if secure_permissions else ValidationLevel.WARNING,
                        title=f"File Permissions: {file_name}",
                        description=f"Check permissions for {file_name}",
                        expected="600 or 644",
                        actual=permissions,
                        passed=secure_permissions,
                        recommendations=[] if secure_permissions else [f"Set secure permissions: chmod 600 {file_name}"],
                        metadata={'file': file_name, 'permissions': permissions},
                        timestamp=datetime.now()
                    )
                    
                    self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Security validation failed: {e}")
    
    async def _validate_performance(self):
        """Validate performance configuration"""
        try:
            # Check Redis performance
            if self.redis_client:
                start_time = time.time()
                await self.redis_client.ping()
                redis_latency = (time.time() - start_time) * 1000
                
                result = ValidationResult(
                    check_id="redis_latency",
                    category=ValidationCategory.PERFORMANCE,
                    level=ValidationLevel.SUCCESS if redis_latency < 10 else ValidationLevel.WARNING,
                    title="Redis Latency",
                    description="Check Redis response time",
                    expected="< 10ms",
                    actual=f"{redis_latency:.2f}ms",
                    passed=redis_latency < 10,
                    recommendations=[] if redis_latency < 10 else ["Check Redis configuration and network"],
                    metadata={'latency_ms': redis_latency},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Performance validation failed: {e}")
    
    async def _validate_integration(self):
        """Validate system integrations"""
        try:
            # Check Four-Brain integration
            brain_endpoints = {
                'brain1': 'http://localhost:8001/health',
                'brain2': 'http://localhost:8002/health',
                'brain3': 'http://localhost:8003/health',
                'brain4': 'http://localhost:8004/health'
            }
            
            for brain_name, endpoint in brain_endpoints.items():
                is_healthy = await self._check_endpoint_health(endpoint)
                
                result = ValidationResult(
                    check_id=f"integration_{brain_name}",
                    category=ValidationCategory.INTEGRATION,
                    level=ValidationLevel.SUCCESS if is_healthy else ValidationLevel.ERROR,
                    title=f"{brain_name.title()} Integration",
                    description=f"Check {brain_name} health and integration",
                    expected="Healthy response",
                    actual="Healthy" if is_healthy else "Unhealthy",
                    passed=is_healthy,
                    recommendations=[] if is_healthy else [f"Check {brain_name} container and configuration"],
                    metadata={'brain': brain_name, 'endpoint': endpoint},
                    timestamp=datetime.now()
                )
                
                self.validation_results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Integration validation failed: {e}")
    
    async def _check_port_availability(self, host: str, port: int) -> bool:
        """Check if port is available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    async def _check_endpoint_health(self, endpoint: str) -> bool:
        """Check endpoint health"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(endpoint) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_command_available(self, command: str) -> bool:
        """Check if command is available"""
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_python_package(self, package: str) -> bool:
        """Check if Python package is installed"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    async def _generate_validation_report(self, report_id: str, validation_time: float) -> ValidationReport:
        """Generate comprehensive validation report"""
        try:
            total_checks = len(self.validation_results)
            passed_checks = sum(1 for r in self.validation_results if r.passed)
            failed_checks = total_checks - passed_checks
            
            # Count by severity
            critical_issues = sum(1 for r in self.validation_results if r.level == ValidationLevel.CRITICAL and not r.passed)
            error_issues = sum(1 for r in self.validation_results if r.level == ValidationLevel.ERROR and not r.passed)
            warning_issues = sum(1 for r in self.validation_results if r.level == ValidationLevel.WARNING and not r.passed)
            
            # Calculate overall score
            overall_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            
            # Generate recommendations
            recommendations = []
            for result in self.validation_results:
                if not result.passed and result.recommendations:
                    recommendations.extend(result.recommendations)
            
            # Remove duplicates
            recommendations = list(dict.fromkeys(recommendations))
            
            report = ValidationReport(
                report_id=report_id,
                system_name="Four-Brain System v2",
                validation_time=datetime.now(),
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                critical_issues=critical_issues,
                error_issues=error_issues,
                warning_issues=warning_issues,
                results=self.validation_results.copy(),
                overall_score=overall_score,
                recommendations=recommendations,
                metadata={
                    'validation_duration_seconds': validation_time,
                    'categories_checked': len(set(r.category for r in self.validation_results)),
                    'system_info': {
                        'platform': os.name,
                        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                        'timestamp': datetime.now().isoformat()
                    }
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise
    
    async def _store_validation_report(self, report: ValidationReport):
        """Store validation report in Redis"""
        if self.redis_client:
            try:
                key = f"validation_report:{report.report_id}"
                data = json.dumps(asdict(report), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store validation report: {e}")
    
    async def _load_validation_history(self):
        """Load validation history from Redis"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("validation_report:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        # Would deserialize ValidationReport
                        pass
        except Exception as e:
            logger.error(f"Failed to load validation history: {e}")
    
    def _update_average_validation_time(self, validation_time: float):
        """Update average validation time metric"""
        if self.metrics['total_validations'] == 1:
            self.metrics['average_validation_time'] = validation_time
        else:
            alpha = 0.1
            self.metrics['average_validation_time'] = (
                alpha * validation_time + 
                (1 - alpha) * self.metrics['average_validation_time']
            )
    
    def _update_success_rate(self, report: ValidationReport):
        """Update success rate metric"""
        success_rate = (report.passed_checks / report.total_checks) * 100 if report.total_checks > 0 else 0
        
        if self.metrics['total_validations'] == 1:
            self.metrics['success_rate'] = success_rate
        else:
            alpha = 0.1
            self.metrics['success_rate'] = (
                alpha * success_rate + 
                (1 - alpha) * self.metrics['success_rate']
            )
        
        self.metrics['critical_issues_found'] += report.critical_issues
        self.metrics['recommendations_generated'] += len(report.recommendations)
    
    async def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics"""
        return {
            'metrics': self.metrics.copy(),
            'validation_history_size': len(self.validation_history),
            'last_validation': self.validation_history[-1].validation_time.isoformat() if self.validation_history else None,
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global config validator instance
config_validator = ConfigValidator()

async def initialize_config_validator():
    """Initialize the global config validator"""
    await config_validator.initialize()

if __name__ == "__main__":
    # Test the config validator
    async def test_config_validator():
        await initialize_config_validator()
        
        # Run comprehensive validation
        report = await config_validator.validate_system()
        
        print(f"Validation Report: {report.report_id}")
        print(f"Overall Score: {report.overall_score:.1f}%")
        print(f"Passed: {report.passed_checks}/{report.total_checks}")
        print(f"Critical Issues: {report.critical_issues}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        # Get metrics
        metrics = await config_validator.get_validation_metrics()
        print(f"Validation metrics: {metrics}")

    asyncio.run(test_config_validator())

# CUDA 13.0 Integration - Consolidated from scripts/cuda13/environment_validator.py
class CUDA13Validator:
    """CUDA 13.0 specific validation integrated into main config validator"""

    def __init__(self):
        self.cuda_version = None
        self.tensorrt_version = None
        self.gpu_compute_capability = None
        self.blackwell_support = False

    async def validate_cuda13_environment(self) -> List[ValidationResult]:
        """Validate CUDA 13.0 environment for RTX 5070 Ti Blackwell"""
        results = []

        # Check CUDA 13.0+ availability
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version and cuda_version.startswith("13."):
                    results.append(ValidationResult(
                        check_id="cuda13_version",
                        category=ValidationCategory.CUDA_COMPATIBILITY,
                        level=ValidationLevel.SUCCESS,
                        title="CUDA Version",
                        description="CUDA 13.x detected",
                        expected="CUDA 13.x",
                        actual=str(cuda_version),
                        passed=True,
                        recommendations=[],
                        metadata={"cuda_version": cuda_version},
                        timestamp=datetime.now()
                    ))
                    self.cuda_version = cuda_version
                else:
                    results.append(ValidationResult(
                        check_id="cuda13_version",
                        category=ValidationCategory.CUDA_COMPATIBILITY,
                        level=ValidationLevel.ERROR,
                        title="CUDA Version",
                        description="CUDA 13.x required",
                        expected="CUDA 13.x",
                        actual=str(cuda_version),
                        passed=False,
                        recommendations=["Install CUDA 13.x or ensure matching torch build"],
                        metadata={"cuda_version": cuda_version},
                        timestamp=datetime.now()
                    ))
            else:
                results.append(ValidationResult(
                    check_id="cuda13_availability",
                    category=ValidationCategory.CUDA_COMPATIBILITY,
                    level=ValidationLevel.CRITICAL,
                    title="CUDA Availability",
                    description="CUDA runtime not available",
                    expected="CUDA available",
                    actual="False",
                    passed=False,
                    recommendations=["Install NVIDIA drivers and enable GPU access"],
                    metadata={},
                    timestamp=datetime.now()
                ))
        except ImportError:
            results.append(ValidationResult(
                check_id="torch_import",
                category=ValidationCategory.CUDA_COMPATIBILITY,
                level=ValidationLevel.CRITICAL,
                title="PyTorch Import",
                description="PyTorch not available for CUDA validation",
                expected="torch import succeeds",
                actual="ImportError",
                passed=False,
                recommendations=["Install torch with correct CUDA support"],
                metadata={},
                timestamp=datetime.now()
            ))

        # Check RTX 5070 Ti Blackwell (sm_120) support
        try:
            import torch
            if torch.cuda.is_available():
                device_cap = torch.cuda.get_device_capability()
                if device_cap >= (12, 0):  # sm_120
                    results.append(ValidationResult(
                        check_id="blackwell_sm120",
                        category=ValidationCategory.CUDA_COMPATIBILITY,
                        level=ValidationLevel.SUCCESS,
                        title="GPU Architecture",
                        description=f"Blackwell GPU detected: sm_{device_cap[0]}{device_cap[1]}",
                        expected="sm_120+",
                        actual=f"sm_{device_cap[0]}{device_cap[1]}",
                        passed=True,
                        recommendations=[],
                        metadata={"compute_capability": device_cap},
                        timestamp=datetime.now()
                    ))
                    self.blackwell_support = True
                    self.gpu_compute_capability = device_cap
                else:
                    results.append(ValidationResult(
                        check_id="blackwell_sm120",
                        category=ValidationCategory.CUDA_COMPATIBILITY,
                        level=ValidationLevel.WARNING,
                        title="GPU Architecture",
                        description=f"Non-Blackwell GPU: sm_{device_cap[0]}{device_cap[1]}",
                        expected="sm_120+",
                        actual=f"sm_{device_cap[0]}{device_cap[1]}",
                        passed=False,
                        recommendations=["Use Blackwell GPU for FP4/FP8 path"],
                        metadata={"compute_capability": device_cap},
                        timestamp=datetime.now()
                    ))
        except Exception as e:
            results.append(ValidationResult(
                check_id="gpu_capability_check",
                category=ValidationCategory.CUDA_COMPATIBILITY,
                level=ValidationLevel.ERROR,
                message=f"Failed to check GPU capability: {e}",
                details={"error": str(e)}
            ))

        # Check TensorRT 10.13.2+ with CUDA 13 support
        try:
            import tensorrt as trt
            trt_version = trt.__version__
            if trt_version.startswith("10.13."):
                results.append(ValidationResult(
                    check_id="tensorrt_cuda13",
                    category=ValidationCategory.CUDA_COMPATIBILITY,
                    level=ValidationLevel.SUCCESS,
                    title="TensorRT Version",
                    description="TensorRT compatible with CUDA 13",
                    expected="TensorRT 10.13.x",
                    actual=trt_version,
                    passed=True,
                    recommendations=[],
                    metadata={"tensorrt_version": trt_version},
                    timestamp=datetime.now()
                ))
                self.tensorrt_version = trt_version
            else:
                results.append(ValidationResult(
                    check_id="tensorrt_cuda13",
                    category=ValidationCategory.CUDA_COMPATIBILITY,
                    level=ValidationLevel.WARNING,
                    title="TensorRT Version",
                    description="TensorRT version may not support CUDA 13",
                    expected="TensorRT 10.13.x",
                    actual=trt_version,
                    passed=False,
                    recommendations=["Use 10.13.x for CUDA 13"],
                    metadata={"tensorrt_version": trt_version},
                    timestamp=datetime.now()
                ))
        except ImportError:
            results.append(ValidationResult(
                check_id="tensorrt_import",
                category=ValidationCategory.CUDA_COMPATIBILITY,
                level=ValidationLevel.ERROR,
                title="TensorRT Import",
                description="TensorRT not available",
                expected="tensorrt import succeeds",
                actual="ImportError",
                passed=False,
                recommendations=["Install TensorRT runtime/libraries"],
                metadata={},
                timestamp=datetime.now()
            ))

        return results
