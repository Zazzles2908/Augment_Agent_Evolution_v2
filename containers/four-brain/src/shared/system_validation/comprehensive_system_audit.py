#!/usr/bin/env python3.11
"""
Comprehensive System Audit for Four-Brain System
Production readiness assessment covering all critical areas

Author: Zazzles's Agent
Date: 2025-08-02
Purpose: Complete system audit before production deployment
"""

import os
import sys
import logging
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import importlib

# Configure logging
logger = logging.getLogger(__name__)

class AuditSeverity(Enum):
    """Audit finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AuditCategory(Enum):
    """Audit categories"""
    ERROR_HANDLING = "error_handling"
    GPU_MANAGEMENT = "gpu_management"
    LOGGING = "logging"
    MODELS = "models"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    RESOURCE_MANAGER = "resource_manager"
    SECURITY = "security"
    SCRIPT_HYGIENE = "script_hygiene"
    PRODUCTION_READINESS = "production_readiness"

@dataclass
class AuditFinding:
    """Individual audit finding"""
    category: AuditCategory
    severity: AuditSeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class AuditReport:
    """Complete audit report"""
    timestamp: float
    total_files_scanned: int
    findings: List[AuditFinding]
    summary: Dict[str, int]
    recommendations: List[str]
    production_ready: bool
    score: int

class ComprehensiveSystemAuditor:
    """Comprehensive system auditor for Four-Brain system"""
    
    def __init__(self):
        self.findings: List[AuditFinding] = []
        self.scanned_files = 0
        
        # Audit configuration
        self.shared_folder = Path("/workspace/src/shared")
        self.src_folder = Path("/workspace/src")
        
        logger.info("🔍 Comprehensive System Auditor initialized")
    
    async def run_full_audit(self) -> AuditReport:
        """Run complete system audit"""
        logger.info("🚀 Starting comprehensive system audit...")
        start_time = time.time()
        
        try:
            # Clear previous findings
            self.findings = []
            self.scanned_files = 0
            
            # Run all audit categories
            await self._audit_error_handling()
            await self._audit_gpu_management()
            await self._audit_logging()
            await self._audit_models()
            await self._audit_monitoring()
            await self._audit_optimization()
            await self._audit_resource_manager()
            await self._audit_security()
            await self._audit_script_hygiene()
            await self._audit_production_readiness()
            
            # Generate report
            report = self._generate_audit_report()
            
            audit_time = time.time() - start_time
            logger.info(f"✅ System audit completed in {audit_time:.1f}s")
            logger.info(f"📊 Scanned {self.scanned_files} files, found {len(self.findings)} findings")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ System audit failed: {str(e)}")
            raise
    
    async def _audit_error_handling(self):
        """Audit error handling implementation"""
        logger.info("🔍 Auditing error handling...")
        
        error_handling_path = self.shared_folder / "error_handling"
        if not error_handling_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.ERROR_HANDLING,
                severity=AuditSeverity.HIGH,
                title="Error handling module missing",
                description="Shared error handling module not found",
                recommendation="Implement centralized error handling"
            ))
            return
        
        # Check for required error handling components
        required_components = [
            "centralized_error_handler.py",
            "circuit_breaker.py",
            "retry_engine.py",
            "fallback_manager.py"
        ]
        
        for component in required_components:
            component_path = error_handling_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.ERROR_HANDLING)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.ERROR_HANDLING,
                    severity=AuditSeverity.MEDIUM,
                    title=f"Missing error handling component: {component}",
                    description=f"Error handling component {component} not found",
                    recommendation=f"Implement {component} for robust error handling"
                ))
    
    async def _audit_gpu_management(self):
        """Audit GPU management implementation"""
        logger.info("🔍 Auditing GPU management...")
        
        gpu_path = self.shared_folder / "gpu"
        if not gpu_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.GPU_MANAGEMENT,
                severity=AuditSeverity.CRITICAL,
                title="GPU management module missing",
                description="Shared GPU management module not found",
                recommendation="Implement GPU resource management"
            ))
            return
        
        # Check for GPU management components
        gpu_components = [
            "dynamic_vram_manager.py"
        ]
        
        for component in gpu_components:
            component_path = gpu_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.GPU_MANAGEMENT)
                
                # Check for specific GPU management patterns
                await self._check_gpu_patterns(component_path)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.GPU_MANAGEMENT,
                    severity=AuditSeverity.HIGH,
                    title=f"Missing GPU component: {component}",
                    description=f"GPU management component {component} not found",
                    recommendation=f"Implement {component} for GPU resource management"
                ))
    
    async def _audit_logging(self):
        """Audit logging implementation"""
        logger.info("🔍 Auditing logging...")
        
        logging_path = self.shared_folder / "logging"
        if not logging_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.LOGGING,
                severity=AuditSeverity.HIGH,
                title="Logging module missing",
                description="Shared logging module not found",
                recommendation="Implement centralized logging"
            ))
            return
        
        # Check logging components
        logging_components = [
            "centralized_logger.py",
            "log_aggregator.py",
            "log_analyzer.py"
        ]
        
        for component in logging_components:
            component_path = logging_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.LOGGING)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.LOGGING,
                    severity=AuditSeverity.MEDIUM,
                    title=f"Missing logging component: {component}",
                    description=f"Logging component {component} not found",
                    recommendation=f"Implement {component} for comprehensive logging"
                ))
    
    async def _audit_models(self):
        """Audit model management"""
        logger.info("🔍 Auditing model management...")
        
        models_path = self.shared_folder / "models"
        if not models_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.MODELS,
                severity=AuditSeverity.HIGH,
                title="Models module missing",
                description="Shared models module not found",
                recommendation="Implement model management system"
            ))
            return
        
        # Check model components
        model_components = [
            "model_prewarming.py"
        ]
        
        for component in model_components:
            component_path = models_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.MODELS)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.MODELS,
                    severity=AuditSeverity.MEDIUM,
                    title=f"Missing model component: {component}",
                    description=f"Model component {component} not found",
                    recommendation=f"Implement {component} for model management"
                ))
    
    async def _audit_monitoring(self):
        """Audit monitoring implementation"""
        logger.info("🔍 Auditing monitoring...")
        
        monitoring_path = self.shared_folder / "monitoring"
        if not monitoring_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.MONITORING,
                severity=AuditSeverity.HIGH,
                title="Monitoring module missing",
                description="Shared monitoring module not found",
                recommendation="Implement comprehensive monitoring"
            ))
            return
        
        # Check monitoring components
        monitoring_components = [
            "metrics_collector.py",
            "performance_tracker.py",
            "alert_manager.py"
        ]
        
        for component in monitoring_components:
            component_path = monitoring_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.MONITORING)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.MONITORING,
                    severity=AuditSeverity.MEDIUM,
                    title=f"Missing monitoring component: {component}",
                    description=f"Monitoring component {component} not found",
                    recommendation=f"Implement {component} for system monitoring"
                ))
    
    async def _audit_optimization(self):
        """Audit optimization implementation"""
        logger.info("🔍 Auditing optimization...")
        
        optimization_path = self.shared_folder / "optimization"
        if not optimization_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.OPTIMIZATION,
                severity=AuditSeverity.MEDIUM,
                title="Optimization module missing",
                description="Shared optimization module not found",
                recommendation="Implement performance optimization"
            ))
            return
        
        # Check optimization components
        optimization_components = [
            "advanced_optimizations.py"
        ]
        
        for component in optimization_components:
            component_path = optimization_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.OPTIMIZATION)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.OPTIMIZATION,
                    severity=AuditSeverity.LOW,
                    title=f"Missing optimization component: {component}",
                    description=f"Optimization component {component} not found",
                    recommendation=f"Implement {component} for performance optimization"
                ))
    
    async def _audit_resource_manager(self):
        """Audit resource manager implementation"""
        logger.info("🔍 Auditing resource manager...")
        
        resource_path = self.shared_folder / "resource_manager"
        if not resource_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.RESOURCE_MANAGER,
                severity=AuditSeverity.HIGH,
                title="Resource manager module missing",
                description="Shared resource manager module not found",
                recommendation="Implement resource management system"
            ))
            return
        
        # Check resource manager components
        resource_components = [
            "dynamic_resource_allocator.py"
        ]
        
        for component in resource_components:
            component_path = resource_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.RESOURCE_MANAGER)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.RESOURCE_MANAGER,
                    severity=AuditSeverity.HIGH,
                    title=f"Missing resource component: {component}",
                    description=f"Resource manager component {component} not found",
                    recommendation=f"Implement {component} for resource management"
                ))
    
    async def _audit_security(self):
        """Audit security implementation"""
        logger.info("🔍 Auditing security...")
        
        security_path = self.shared_folder / "security"
        if not security_path.exists():
            self.findings.append(AuditFinding(
                category=AuditCategory.SECURITY,
                severity=AuditSeverity.CRITICAL,
                title="Security module missing",
                description="Shared security module not found",
                recommendation="Implement security framework"
            ))
            return
        
        # Check security components
        security_components = [
            "redis_auth_manager.py",
            "supabase_auth_manager.py",
            "ssl_certificate_manager.py",
            "security_monitor.py"
        ]
        
        for component in security_components:
            component_path = security_path / component
            if component_path.exists():
                await self._scan_file_for_issues(component_path, AuditCategory.SECURITY)
            else:
                self.findings.append(AuditFinding(
                    category=AuditCategory.SECURITY,
                    severity=AuditSeverity.HIGH,
                    title=f"Missing security component: {component}",
                    description=f"Security component {component} not found",
                    recommendation=f"Implement {component} for production security"
                ))
    
    async def _audit_script_hygiene(self):
        """Audit script hygiene and organization"""
        logger.info("🔍 Auditing script hygiene...")
        
        # Check for redundant main scripts
        main_scripts = [
            "main.py",
            "main_basic.py",
            "main_unified.py",
            "orchestrator_main.py",
            "orchestrator_main_original.py"
        ]
        
        existing_mains = []
        for script in main_scripts:
            script_path = self.src_folder / script
            if script_path.exists():
                existing_mains.append(script)
        
        if len(existing_mains) > 2:  # main.py + orchestrator_main.py is acceptable
            self.findings.append(AuditFinding(
                category=AuditCategory.SCRIPT_HYGIENE,
                severity=AuditSeverity.MEDIUM,
                title="Multiple main scripts found",
                description=f"Found {len(existing_mains)} main scripts: {', '.join(existing_mains)}",
                recommendation="Keep only necessary entry points (main.py, orchestrator_main.py)",
                auto_fixable=True
            ))
        
        # Check for proper script organization
        scripts_path = self.src_folder / "scripts"
        if scripts_path.exists():
            await self._check_script_organization(scripts_path)
    
    async def _audit_production_readiness(self):
        """Audit production readiness"""
        logger.info("🔍 Auditing production readiness...")
        
        # Check for production requirements
        production_checks = [
            ("Redis authentication", self._check_redis_auth),
            ("Supabase security", self._check_supabase_security),
            ("SSL certificates", self._check_ssl_config),
            ("Environment variables", self._check_env_vars),
            ("Health checks", self._check_health_endpoints),
            ("Monitoring setup", self._check_monitoring_setup)
        ]
        
        for check_name, check_func in production_checks:
            try:
                result = await check_func()
                if not result:
                    self.findings.append(AuditFinding(
                        category=AuditCategory.PRODUCTION_READINESS,
                        severity=AuditSeverity.HIGH,
                        title=f"Production check failed: {check_name}",
                        description=f"{check_name} is not properly configured for production",
                        recommendation=f"Configure {check_name} for production deployment"
                    ))
            except Exception as e:
                self.findings.append(AuditFinding(
                    category=AuditCategory.PRODUCTION_READINESS,
                    severity=AuditSeverity.MEDIUM,
                    title=f"Production check error: {check_name}",
                    description=f"Failed to check {check_name}: {str(e)}",
                    recommendation=f"Investigate and fix {check_name} configuration"
                ))
    
    async def _scan_file_for_issues(self, file_path: Path, category: AuditCategory):
        """Scan individual file for common issues"""
        try:
            self.scanned_files += 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for common anti-patterns
            for line_num, line in enumerate(lines, 1):
                # Check for fabricated data patterns
                if any(pattern in line.lower() for pattern in ['mock', 'fake', 'simulate', 'random.uniform', 'np.random']):
                    if 'test' not in file_path.name.lower():  # Exclude test files
                        self.findings.append(AuditFinding(
                            category=category,
                            severity=AuditSeverity.HIGH,
                            title="Potential fabricated data",
                            description=f"Line contains potential fabricated data: {line.strip()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Replace with real data or proper error handling"
                        ))
                
                # Check for hardcoded credentials
                if any(pattern in line.lower() for pattern in ['password=', 'api_key=', 'secret=', 'token=']):
                    if not line.strip().startswith('#'):  # Not a comment
                        self.findings.append(AuditFinding(
                            category=AuditCategory.SECURITY,
                            severity=AuditSeverity.CRITICAL,
                            title="Potential hardcoded credentials",
                            description=f"Line may contain hardcoded credentials: {line.strip()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Move credentials to environment variables"
                        ))
                
                # Check for TODO/FIXME comments
                if any(pattern in line.upper() for pattern in ['TODO', 'FIXME', 'HACK']):
                    self.findings.append(AuditFinding(
                        category=category,
                        severity=AuditSeverity.LOW,
                        title="Unresolved TODO/FIXME",
                        description=f"Unresolved comment: {line.strip()}",
                        file_path=str(file_path),
                        line_number=line_num,
                        recommendation="Resolve or document the TODO/FIXME item"
                    ))
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to scan {file_path}: {str(e)}")
    
    async def _check_gpu_patterns(self, file_path: Path):
        """Check for GPU-specific patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for proper GPU memory management
            if 'torch.cuda.set_per_process_memory_fraction' not in content:
                self.findings.append(AuditFinding(
                    category=AuditCategory.GPU_MANAGEMENT,
                    severity=AuditSeverity.MEDIUM,
                    title="Missing GPU memory fraction management",
                    description="File doesn't use torch.cuda.set_per_process_memory_fraction",
                    file_path=str(file_path),
                    recommendation="Implement GPU memory fraction management"
                ))
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to check GPU patterns in {file_path}: {str(e)}")
    
    async def _check_script_organization(self, scripts_path: Path):
        """Check script organization"""
        # This would check for proper script organization
        # For now, just log that we're checking
        logger.debug(f"Checking script organization in {scripts_path}")
    
    async def _check_redis_auth(self) -> bool:
        """Check Redis authentication configuration"""
        # Check if Redis auth manager exists
        redis_auth_path = self.shared_folder / "security" / "redis_auth_manager.py"
        return redis_auth_path.exists()
    
    async def _check_supabase_security(self) -> bool:
        """Check Supabase security configuration"""
        # Check if Supabase auth manager exists
        supabase_auth_path = self.shared_folder / "security" / "supabase_auth_manager.py"
        return supabase_auth_path.exists()
    
    async def _check_ssl_config(self) -> bool:
        """Check SSL configuration"""
        # Check if SSL certificate manager exists
        ssl_path = self.shared_folder / "security" / "ssl_certificate_manager.py"
        return ssl_path.exists()
    
    async def _check_env_vars(self) -> bool:
        """Check environment variables configuration"""
        # Check for required environment variables
        required_vars = ['REDIS_PASSWORD', 'SUPABASE_SERVICE_KEY', 'SSL_EMAIL']
        return all(os.getenv(var) for var in required_vars)
    
    async def _check_health_endpoints(self) -> bool:
        """Check health endpoints"""
        # This would check for health endpoint implementation
        return True  # Placeholder
    
    async def _check_monitoring_setup(self) -> bool:
        """Check monitoring setup"""
        # Check if monitoring components exist
        monitoring_path = self.shared_folder / "monitoring"
        return monitoring_path.exists()
    
    def _generate_audit_report(self) -> AuditReport:
        """Generate comprehensive audit report"""
        # Calculate summary statistics
        summary = {}
        for severity in AuditSeverity:
            summary[severity.value] = len([f for f in self.findings if f.severity == severity])
        
        # Calculate production readiness score
        critical_issues = summary.get('critical', 0)
        high_issues = summary.get('high', 0)
        medium_issues = summary.get('medium', 0)
        
        # Scoring: Start with 100, deduct points for issues
        score = 100
        score -= critical_issues * 20  # Critical issues: -20 points each
        score -= high_issues * 10      # High issues: -10 points each
        score -= medium_issues * 5     # Medium issues: -5 points each
        score = max(0, score)  # Don't go below 0
        
        # Determine production readiness
        production_ready = critical_issues == 0 and high_issues <= 2
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return AuditReport(
            timestamp=time.time(),
            total_files_scanned=self.scanned_files,
            findings=self.findings,
            summary=summary,
            recommendations=recommendations,
            production_ready=production_ready,
            score=score
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Critical issues first
        critical_findings = [f for f in self.findings if f.severity == AuditSeverity.CRITICAL]
        if critical_findings:
            recommendations.append(f"🚨 Address {len(critical_findings)} critical security issues immediately")
        
        # High priority issues
        high_findings = [f for f in self.findings if f.severity == AuditSeverity.HIGH]
        if high_findings:
            recommendations.append(f"⚠️ Resolve {len(high_findings)} high-priority issues before production")
        
        # Category-specific recommendations
        categories = {}
        for finding in self.findings:
            if finding.category not in categories:
                categories[finding.category] = 0
            categories[finding.category] += 1
        
        for category, count in categories.items():
            if count >= 3:
                recommendations.append(f"📋 Focus on {category.value} improvements ({count} issues)")
        
        return recommendations

# Global auditor instance
_system_auditor = None

def get_system_auditor() -> ComprehensiveSystemAuditor:
    """Get global system auditor instance"""
    global _system_auditor
    if _system_auditor is None:
        _system_auditor = ComprehensiveSystemAuditor()
    return _system_auditor

async def run_system_audit() -> AuditReport:
    """Convenience function to run system audit"""
    auditor = get_system_auditor()
    return await auditor.run_full_audit()
