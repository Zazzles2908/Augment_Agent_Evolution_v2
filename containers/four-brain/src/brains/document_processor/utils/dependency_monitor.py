"""
Centralized Dependency Logging System
Monitors and documents missing dependencies with actionable resolution information
AUTHENTIC IMPLEMENTATION - Zero fabrication policy
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

@dataclass
class DependencyInfo:
    """Information about a dependency requirement"""
    name: str
    category: str  # 'package', 'model_file', 'test_data', 'hardware'
    required: bool  # True for critical, False for optional
    status: str  # 'available', 'missing', 'partial'
    expected_location: Optional[str] = None
    actual_location: Optional[str] = None
    version_required: Optional[str] = None
    version_found: Optional[str] = None
    installation_command: Optional[str] = None
    setup_instructions: Optional[str] = None
    skip_reason: Optional[str] = None
    last_checked: Optional[str] = None

@dataclass
class DependencyReport:
    """Comprehensive dependency status report"""
    timestamp: str
    system_info: Dict[str, Any]
    dependencies: List[DependencyInfo]
    summary: Dict[str, int]
    recommendations: List[str]

class DependencyMonitor:
    """
    Centralized dependency monitoring and logging system
    Provides honest assessment of system requirements vs. availability
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file or "dependency_status.json"
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.skip_reasons: List[str] = []
        
    def check_package_dependency(self, package_name: str, required: bool = True, 
                                import_path: Optional[str] = None) -> DependencyInfo:
        """Check if a Python package is available"""
        try:
            if import_path:
                __import__(import_path)
            else:
                __import__(package_name)
            
            # Try to get version
            version = None
            try:
                import importlib.metadata
                version = importlib.metadata.version(package_name)
            except Exception:
                pass
            
            return DependencyInfo(
                name=package_name,
                category="package",
                required=required,
                status="available",
                version_found=version,
                installation_command=f"pip install {package_name}",
                last_checked=datetime.now().isoformat()
            )
            
        except ImportError as e:
            return DependencyInfo(
                name=package_name,
                category="package",
                required=required,
                status="missing",
                installation_command=f"pip install {package_name}",
                setup_instructions=f"Install {package_name} using pip or conda",
                skip_reason=f"Package '{package_name}' not available: {str(e)}",
                last_checked=datetime.now().isoformat()
            )
    
    def check_file_dependency(self, file_path: str, category: str, required: bool = True,
                            description: Optional[str] = None) -> DependencyInfo:
        """Check if a required file or directory exists"""
        path = Path(file_path)
        
        if path.exists():
            return DependencyInfo(
                name=description or path.name,
                category=category,
                required=required,
                status="available",
                expected_location=file_path,
                actual_location=str(path.absolute()),
                last_checked=datetime.now().isoformat()
            )
        else:
            return DependencyInfo(
                name=description or path.name,
                category=category,
                required=required,
                status="missing",
                expected_location=file_path,
                setup_instructions=f"Create or download required {category} to {file_path}",
                skip_reason=f"Required {category} not found at {file_path}",
                last_checked=datetime.now().isoformat()
            )
    
    def check_hardware_dependency(self, hardware_type: str, check_function, 
                                required: bool = True) -> DependencyInfo:
        """Check hardware availability (GPU, memory, etc.)"""
        try:
            available = check_function()
            status = "available" if available else "missing"
            
            return DependencyInfo(
                name=hardware_type,
                category="hardware",
                required=required,
                status=status,
                setup_instructions=f"Ensure {hardware_type} is properly installed and configured" if not available else None,
                skip_reason=f"{hardware_type} not available" if not available else None,
                last_checked=datetime.now().isoformat()
            )
            
        except Exception as e:
            return DependencyInfo(
                name=hardware_type,
                category="hardware",
                required=required,
                status="missing",
                setup_instructions=f"Install and configure {hardware_type}",
                skip_reason=f"{hardware_type} check failed: {str(e)}",
                last_checked=datetime.now().isoformat()
            )
    
    def register_dependency(self, dep_info: DependencyInfo):
        """Register a dependency in the monitoring system"""
        self.dependencies[dep_info.name] = dep_info
        
        if dep_info.skip_reason:
            self.skip_reasons.append(dep_info.skip_reason)
            self.logger.warning(f"Dependency missing: {dep_info.name} - {dep_info.skip_reason}")
        else:
            self.logger.info(f"Dependency available: {dep_info.name}")
    
    def register_skip_reason(self, test_name: str, reason: str, category: str = "test"):
        """Register a test skip reason for analysis"""
        skip_info = DependencyInfo(
            name=f"test_skip_{test_name}",
            category=category,
            required=False,
            status="missing",
            skip_reason=reason,
            last_checked=datetime.now().isoformat()
        )
        self.register_dependency(skip_info)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "python_path": sys.path[:3],  # First 3 entries
            "environment_variables": {
                key: os.environ.get(key, "Not set") 
                for key in ["PYTHONPATH", "CUDA_VISIBLE_DEVICES", "PATH"]
            }
        }
    
    def generate_report(self) -> DependencyReport:
        """Generate comprehensive dependency status report"""
        # Calculate summary statistics
        total_deps = len(self.dependencies)
        available = sum(1 for dep in self.dependencies.values() if dep.status == "available")
        missing = sum(1 for dep in self.dependencies.values() if dep.status == "missing")
        critical_missing = sum(1 for dep in self.dependencies.values() 
                             if dep.status == "missing" and dep.required)
        
        # Generate recommendations
        recommendations = []
        
        if critical_missing > 0:
            recommendations.append(f"CRITICAL: {critical_missing} required dependencies missing")
        
        # Package installation recommendations
        missing_packages = [dep for dep in self.dependencies.values() 
                          if dep.category == "package" and dep.status == "missing"]
        if missing_packages:
            install_commands = [dep.installation_command for dep in missing_packages 
                              if dep.installation_command]
            if install_commands:
                recommendations.append(f"Install packages: {'; '.join(install_commands)}")
        
        # File/model recommendations
        missing_files = [dep for dep in self.dependencies.values() 
                        if dep.category in ["model_file", "test_data"] and dep.status == "missing"]
        if missing_files:
            recommendations.append(f"Missing {len(missing_files)} files - check setup documentation")
        
        # Hardware recommendations
        missing_hardware = [dep for dep in self.dependencies.values() 
                          if dep.category == "hardware" and dep.status == "missing"]
        if missing_hardware:
            recommendations.append(f"Hardware requirements not met: {[dep.name for dep in missing_hardware]}")
        
        return DependencyReport(
            timestamp=datetime.now().isoformat(),
            system_info=self.get_system_info(),
            dependencies=list(self.dependencies.values()),
            summary={
                "total": total_deps,
                "available": available,
                "missing": missing,
                "critical_missing": critical_missing,
                "availability_rate": round((available / total_deps * 100) if total_deps > 0 else 0, 1)
            },
            recommendations=recommendations
        )
    
    def save_report(self, report: Optional[DependencyReport] = None):
        """Save dependency report to file"""
        if report is None:
            report = self.generate_report()

        try:
            # Convert to dict and handle Path objects
            report_dict = asdict(report)

            # Convert any Path objects to strings
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                elif hasattr(obj, '__fspath__'):  # Path-like object
                    return str(obj)
                else:
                    return obj

            report_dict = convert_paths(report_dict)

            with open(self.log_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
            self.logger.info(f"Dependency report saved to {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save dependency report: {e}")
    
    def load_report(self) -> Optional[DependencyReport]:
        """Load previous dependency report"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            return DependencyReport(**data)
        except Exception as e:
            self.logger.warning(f"Could not load previous report: {e}")
            return None
    
    def print_summary(self):
        """Print human-readable dependency summary"""
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("🔍 DEPENDENCY STATUS REPORT")
        print("="*80)
        print(f"📊 Summary: {report.summary['available']}/{report.summary['total']} dependencies available ({report.summary['availability_rate']}%)")
        print(f"⚠️  Critical missing: {report.summary['critical_missing']}")
        print(f"📅 Generated: {report.timestamp}")
        
        if report.recommendations:
            print("\n🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n📋 DEPENDENCY DETAILS:")
        for category in ["package", "model_file", "test_data", "hardware"]:
            deps_in_category = [dep for dep in report.dependencies if dep.category == category]
            if deps_in_category:
                print(f"\n  {category.upper()}:")
                for dep in deps_in_category:
                    status_icon = "✅" if dep.status == "available" else "❌"
                    req_icon = "🔴" if dep.required else "🟡"
                    print(f"    {status_icon} {req_icon} {dep.name}")
                    if dep.skip_reason:
                        print(f"        Reason: {dep.skip_reason}")

# Global instance for easy access
dependency_monitor = DependencyMonitor()
