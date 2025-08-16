#!/usr/bin/env python3.11
"""
API Transparency Endpoints for Four-Brain System
Honest capability reporting and dependency monitoring for all Brain services

Author: AugmentAI
Date: 2025-08-02
Purpose: Provide transparent API endpoints that honestly report service capabilities and dependencies
"""

import os
import sys
import asyncio
import logging
import time
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Configure logging
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status levels"""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    ERROR = "error"

class DependencyStatus(Enum):
    """Dependency status levels"""
    SATISFIED = "satisfied"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"
    UNKNOWN = "unknown"

@dataclass
class CapabilityFlag:
    """Individual capability flag"""
    name: str
    available: bool
    reason: str
    dependencies: List[str]
    performance_impact: Optional[str] = None
    fallback_available: bool = False

@dataclass
class DependencyInfo:
    """Dependency information"""
    name: str
    required_version: Optional[str]
    installed_version: Optional[str]
    status: DependencyStatus
    import_path: str
    critical: bool
    error_message: Optional[str] = None

@dataclass
class ServiceCapabilities:
    """Complete service capability report"""
    service_name: str
    overall_status: ServiceStatus
    capabilities: Dict[str, CapabilityFlag]
    dependencies: Dict[str, DependencyInfo]
    performance_metrics: Dict[str, Any]
    limitations: List[str]
    recommendations: List[str]
    last_updated: float

class TransparencyEndpoints:
    """API transparency endpoints for honest capability reporting"""
    
    def __init__(self):
        self.service_capabilities: Dict[str, ServiceCapabilities] = {}
        self.dependency_cache: Dict[str, DependencyInfo] = {}
        self.last_scan_time = 0.0
        self.scan_interval = 300.0  # 5 minutes
        
        logger.info("🔍 API Transparency Endpoints initialized")
    
    async def scan_brain1_capabilities(self) -> ServiceCapabilities:
        """Scan Brain-1 (Embedding Service) capabilities"""
        try:
            service_name = "brain1_embedding"
            capabilities = {}
            dependencies = {}
            limitations = []
            recommendations = []
            
            # Check core embedding capability
            try:
                from ...brains.embedding_service.core.brain1_manager import Brain1Manager
                capabilities["embedding_generation"] = CapabilityFlag(
                    name="embedding_generation",
                    available=True,
                    reason="Brain1Manager available",
                    dependencies=["transformers", "torch", "sentence-transformers"]
                )
            except ImportError as e:
                capabilities["embedding_generation"] = CapabilityFlag(
                    name="embedding_generation",
                    available=False,
                    reason=f"Brain1Manager import failed: {str(e)}",
                    dependencies=["transformers", "torch", "sentence-transformers"]
                )
                limitations.append("Core embedding functionality unavailable")
            
            # Check TensorRT acceleration
            try:
                from ...brains.embedding_service.core.tensorrt_accelerator import get_brain1_accelerator
                accelerator = get_brain1_accelerator()
                if accelerator and hasattr(accelerator, 'engine') and accelerator.engine is not None:
                    capabilities["tensorrt_acceleration"] = CapabilityFlag(
                        name="tensorrt_acceleration",
                        available=True,
                        reason="TensorRT accelerator with loaded engine",
                        dependencies=["tensorrt", "onnx", "pycuda"],
                        performance_impact="Up to 4x faster inference with FP4 quantization"
                    )
                else:
                    capabilities["tensorrt_acceleration"] = CapabilityFlag(
                        name="tensorrt_acceleration",
                        available=False,
                        reason="TensorRT accelerator available but no engine loaded",
                        dependencies=["tensorrt", "onnx", "pycuda"],
                        fallback_available=True
                    )
                    limitations.append("TensorRT acceleration unavailable - using CPU fallback")
                    recommendations.append("Load TensorRT engine for optimal performance")
            except ImportError as e:
                capabilities["tensorrt_acceleration"] = CapabilityFlag(
                    name="tensorrt_acceleration",
                    available=False,
                    reason=f"TensorRT import failed: {str(e)}",
                    dependencies=["tensorrt", "onnx", "pycuda"],
                    fallback_available=True
                )
                limitations.append("TensorRT dependencies missing")
                recommendations.append("Install TensorRT for GPU acceleration")
            
            # Check model availability
            try:
                # Check if Qwen3-4B model is available
                model_path = Path("/workspace/models/qwen3-4b-embedding")
                if model_path.exists():
                    capabilities["qwen3_model"] = CapabilityFlag(
                        name="qwen3_model",
                        available=True,
                        reason="Qwen3-4B model found on disk",
                        dependencies=["transformers", "torch"]
                    )
                else:
                    capabilities["qwen3_model"] = CapabilityFlag(
                        name="qwen3_model",
                        available=False,
                        reason="Qwen3-4B model not found on disk",
                        dependencies=["transformers", "torch"],
                        fallback_available=True
                    )
                    limitations.append("Local Qwen3-4B model unavailable - will download on first use")
                    recommendations.append("Pre-download Qwen3-4B model for faster startup")
            except Exception as e:
                capabilities["qwen3_model"] = CapabilityFlag(
                    name="qwen3_model",
                    available=False,
                    reason=f"Model check failed: {str(e)}",
                    dependencies=["transformers", "torch"]
                )
            
            # Scan dependencies
            dependencies.update(await self._scan_dependencies([
                ("transformers", "transformers", "4.30.0", True),
                ("torch", "torch", "2.0.0", True),
                ("sentence-transformers", "sentence_transformers", "2.2.0", False),
                ("tensorrt", "tensorrt", "10.13.0", False),
                ("onnx", "onnx", "1.18.0", False),
                ("pycuda", "pycuda", None, False)
            ]))
            
            # Determine overall status
            critical_deps_ok = all(
                dep.status == DependencyStatus.SATISFIED 
                for dep in dependencies.values() 
                if dep.critical
            )
            
            if capabilities["embedding_generation"].available and critical_deps_ok:
                if capabilities["tensorrt_acceleration"].available:
                    overall_status = ServiceStatus.AVAILABLE
                else:
                    overall_status = ServiceStatus.DEGRADED
                    limitations.append("Running without TensorRT acceleration")
            else:
                overall_status = ServiceStatus.UNAVAILABLE
                limitations.append("Critical dependencies missing")
            
            return ServiceCapabilities(
                service_name=service_name,
                overall_status=overall_status,
                capabilities=capabilities,
                dependencies=dependencies,
                performance_metrics={
                    "expected_latency_ms": 50 if capabilities["tensorrt_acceleration"].available else 200,
                    "max_throughput_ops_per_sec": 100 if capabilities["tensorrt_acceleration"].available else 25,
                    "memory_usage_gb": 2.0 if capabilities["tensorrt_acceleration"].available else 4.0
                },
                limitations=limitations,
                recommendations=recommendations,
                last_updated=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Brain-1 capability scan failed: {str(e)}")
            return ServiceCapabilities(
                service_name="brain1_embedding",
                overall_status=ServiceStatus.ERROR,
                capabilities={},
                dependencies={},
                performance_metrics={},
                limitations=[f"Capability scan failed: {str(e)}"],
                recommendations=["Check service configuration and dependencies"],
                last_updated=time.time()
            )
    
    async def scan_brain2_capabilities(self) -> ServiceCapabilities:
        """Scan Brain-2 (Reranker Service) capabilities"""
        try:
            service_name = "brain2_reranker"
            capabilities = {}
            dependencies = {}
            limitations = []
            recommendations = []
            
            # Check core reranking capability
            try:
                # Check if reranker service is available
                capabilities["query_document_reranking"] = CapabilityFlag(
                    name="query_document_reranking",
                    available=False,
                    reason="Brain-2 reranker service not fully implemented",
                    dependencies=["transformers", "torch", "sentence-transformers"]
                )
                limitations.append("Core reranking functionality not implemented")
                recommendations.append("Implement Brain-2 reranker service")
            except Exception as e:
                capabilities["query_document_reranking"] = CapabilityFlag(
                    name="query_document_reranking",
                    available=False,
                    reason=f"Reranker check failed: {str(e)}",
                    dependencies=["transformers", "torch", "sentence-transformers"]
                )
            
            # Check TensorRT acceleration
            try:
                from ...brains.reranker_service.core.tensorrt_accelerator import get_brain2_accelerator
                accelerator = get_brain2_accelerator()
                capabilities["tensorrt_acceleration"] = CapabilityFlag(
                    name="tensorrt_acceleration",
                    available=False,
                    reason="TensorRT accelerator not implemented for Brain-2",
                    dependencies=["tensorrt", "onnx", "pycuda"],
                    fallback_available=False
                )
                limitations.append("TensorRT acceleration not implemented")
                recommendations.append("Implement TensorRT acceleration for Brain-2")
            except ImportError as e:
                capabilities["tensorrt_acceleration"] = CapabilityFlag(
                    name="tensorrt_acceleration",
                    available=False,
                    reason=f"TensorRT import failed: {str(e)}",
                    dependencies=["tensorrt", "onnx", "pycuda"]
                )
            
            # Scan dependencies
            dependencies.update(await self._scan_dependencies([
                ("transformers", "transformers", "4.30.0", True),
                ("torch", "torch", "2.0.0", True),
                ("sentence-transformers", "sentence_transformers", "2.2.0", False),
                ("tensorrt", "tensorrt", "10.13.0", False)
            ]))
            
            overall_status = ServiceStatus.UNAVAILABLE
            limitations.append("Brain-2 service not fully implemented")
            
            return ServiceCapabilities(
                service_name=service_name,
                overall_status=overall_status,
                capabilities=capabilities,
                dependencies=dependencies,
                performance_metrics={
                    "expected_latency_ms": "N/A",
                    "max_throughput_ops_per_sec": "N/A",
                    "memory_usage_gb": "N/A"
                },
                limitations=limitations,
                recommendations=recommendations,
                last_updated=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Brain-2 capability scan failed: {str(e)}")
            return ServiceCapabilities(
                service_name="brain2_reranker",
                overall_status=ServiceStatus.ERROR,
                capabilities={},
                dependencies={},
                performance_metrics={},
                limitations=[f"Capability scan failed: {str(e)}"],
                recommendations=["Check service configuration and dependencies"],
                last_updated=time.time()
            )
    
    async def scan_brain4_capabilities(self) -> ServiceCapabilities:
        """Scan Brain-4 (Document Processing) capabilities"""
        try:
            service_name = "brain4_docling"
            capabilities = {}
            dependencies = {}
            limitations = []
            recommendations = []
            
            # Check Docling capability
            try:
                import docling
                capabilities["document_processing"] = CapabilityFlag(
                    name="document_processing",
                    available=True,
                    reason="Docling library available",
                    dependencies=["docling", "pillow", "pdf2image"]
                )
            except ImportError as e:
                capabilities["document_processing"] = CapabilityFlag(
                    name="document_processing",
                    available=False,
                    reason=f"Docling import failed: {str(e)}",
                    dependencies=["docling", "pillow", "pdf2image"]
                )
                limitations.append("Docling library not available")
                recommendations.append("Install Docling for document processing")
            
            # Check cuDF acceleration
            try:
                import cudf
                capabilities["cudf_acceleration"] = CapabilityFlag(
                    name="cudf_acceleration",
                    available=True,
                    reason="cuDF library available for GPU-accelerated data processing",
                    dependencies=["cudf-cu12", "cupy-cuda12x"],
                    performance_impact="Up to 10x faster data processing on GPU"
                )
                recommendations.append("Use cuDF for large document batch processing")
            except ImportError as e:
                capabilities["cudf_acceleration"] = CapabilityFlag(
                    name="cudf_acceleration",
                    available=False,
                    reason=f"cuDF import failed: {str(e)}",
                    dependencies=["cudf-cu12", "cupy-cuda12x"],
                    fallback_available=True
                )
                limitations.append("cuDF GPU acceleration unavailable - using pandas fallback")
                recommendations.append("Install cuDF for GPU-accelerated document processing")
            
            # Check TensorRT acceleration
            try:
                from ...brains.document_processor.core.tensorrt_accelerator import get_brain4_accelerator
                accelerator = get_brain4_accelerator()
                capabilities["tensorrt_acceleration"] = CapabilityFlag(
                    name="tensorrt_acceleration",
                    available=False,
                    reason="TensorRT accelerator not fully implemented for Brain-4",
                    dependencies=["tensorrt", "onnx", "pycuda"],
                    fallback_available=True
                )
                limitations.append("TensorRT acceleration not fully implemented")
                recommendations.append("Complete TensorRT integration for Brain-4")
            except ImportError as e:
                capabilities["tensorrt_acceleration"] = CapabilityFlag(
                    name="tensorrt_acceleration",
                    available=False,
                    reason=f"TensorRT import failed: {str(e)}",
                    dependencies=["tensorrt", "onnx", "pycuda"]
                )
            
            # Scan dependencies
            dependencies.update(await self._scan_dependencies([
                ("docling", "docling", None, True),
                ("pillow", "PIL", "9.0.0", True),
                ("pdf2image", "pdf2image", None, False),
                ("cudf", "cudf", "25.06", False),
                ("cupy", "cupy", "13.0.0", False),
                ("tensorrt", "tensorrt", "10.13.0", False)
            ]))
            
            # Determine overall status
            critical_deps_ok = all(
                dep.status == DependencyStatus.SATISFIED 
                for dep in dependencies.values() 
                if dep.critical
            )
            
            if capabilities["document_processing"].available and critical_deps_ok:
                if capabilities["cudf_acceleration"].available:
                    overall_status = ServiceStatus.AVAILABLE
                else:
                    overall_status = ServiceStatus.DEGRADED
                    limitations.append("Running without cuDF GPU acceleration")
            else:
                overall_status = ServiceStatus.UNAVAILABLE
                limitations.append("Critical dependencies missing")
            
            return ServiceCapabilities(
                service_name=service_name,
                overall_status=overall_status,
                capabilities=capabilities,
                dependencies=dependencies,
                performance_metrics={
                    "expected_latency_ms": 500 if capabilities["cudf_acceleration"].available else 2000,
                    "max_throughput_docs_per_min": 60 if capabilities["cudf_acceleration"].available else 15,
                    "memory_usage_gb": 3.0 if capabilities["cudf_acceleration"].available else 2.0
                },
                limitations=limitations,
                recommendations=recommendations,
                last_updated=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Brain-4 capability scan failed: {str(e)}")
            return ServiceCapabilities(
                service_name="brain4_docling",
                overall_status=ServiceStatus.ERROR,
                capabilities={},
                dependencies={},
                performance_metrics={},
                limitations=[f"Capability scan failed: {str(e)}"],
                recommendations=["Check service configuration and dependencies"],
                last_updated=time.time()
            )
    
    async def _scan_dependencies(self, dependency_list: List[Tuple[str, str, Optional[str], bool]]) -> Dict[str, DependencyInfo]:
        """Scan dependencies and return status information"""
        dependencies = {}
        
        for dep_name, import_path, required_version, critical in dependency_list:
            try:
                # Try to import the module
                module = importlib.import_module(import_path)
                
                # Get version if available
                installed_version = None
                if hasattr(module, '__version__'):
                    installed_version = module.__version__
                elif hasattr(module, 'version'):
                    installed_version = module.version
                elif hasattr(module, 'VERSION'):
                    installed_version = module.VERSION
                
                # Check version compatibility
                status = DependencyStatus.SATISFIED
                if required_version and installed_version:
                    # Simple version comparison (could be enhanced)
                    if installed_version < required_version:
                        status = DependencyStatus.VERSION_MISMATCH
                
                dependencies[dep_name] = DependencyInfo(
                    name=dep_name,
                    required_version=required_version,
                    installed_version=installed_version,
                    status=status,
                    import_path=import_path,
                    critical=critical
                )
                
            except ImportError as e:
                dependencies[dep_name] = DependencyInfo(
                    name=dep_name,
                    required_version=required_version,
                    installed_version=None,
                    status=DependencyStatus.MISSING,
                    import_path=import_path,
                    critical=critical,
                    error_message=str(e)
                )
            except Exception as e:
                dependencies[dep_name] = DependencyInfo(
                    name=dep_name,
                    required_version=required_version,
                    installed_version=None,
                    status=DependencyStatus.IMPORT_ERROR,
                    import_path=import_path,
                    critical=critical,
                    error_message=str(e)
                )
        
        return dependencies
    
    async def scan_all_capabilities(self) -> Dict[str, ServiceCapabilities]:
        """Scan capabilities for all Brain services"""
        try:
            logger.info("🔍 Scanning all service capabilities...")
            
            capabilities = {}
            
            # Scan each Brain service
            capabilities["brain1_embedding"] = await self.scan_brain1_capabilities()
            capabilities["brain2_reranker"] = await self.scan_brain2_capabilities()
            capabilities["brain4_docling"] = await self.scan_brain4_capabilities()
            
            # Update cache
            self.service_capabilities = capabilities
            self.last_scan_time = time.time()
            
            logger.info(f"✅ Capability scan complete: {len(capabilities)} services scanned")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"❌ Capability scan failed: {str(e)}")
            return {}
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status for a specific service"""
        try:
            # Check if we need to refresh capabilities
            if time.time() - self.last_scan_time > self.scan_interval:
                await self.scan_all_capabilities()
            
            if service_name in self.service_capabilities:
                capabilities = self.service_capabilities[service_name]
                return {
                    "service_name": service_name,
                    "status": capabilities.overall_status.value,
                    "capabilities": {
                        name: {
                            "available": cap.available,
                            "reason": cap.reason,
                            "performance_impact": cap.performance_impact,
                            "fallback_available": cap.fallback_available
                        }
                        for name, cap in capabilities.capabilities.items()
                    },
                    "limitations": capabilities.limitations,
                    "recommendations": capabilities.recommendations,
                    "performance_metrics": capabilities.performance_metrics,
                    "last_updated": capabilities.last_updated
                }
            else:
                return {
                    "service_name": service_name,
                    "status": ServiceStatus.UNKNOWN.value,
                    "error": "Service not found or not scanned",
                    "available_services": list(self.service_capabilities.keys())
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get service status: {str(e)}")
            return {
                "service_name": service_name,
                "status": ServiceStatus.ERROR.value,
                "error": str(e)
            }
    
    async def get_dependency_report(self) -> Dict[str, Any]:
        """Get comprehensive dependency report for all services"""
        try:
            # Ensure capabilities are up to date
            if time.time() - self.last_scan_time > self.scan_interval:
                await self.scan_all_capabilities()
            
            all_dependencies = {}
            dependency_summary = {
                "total_dependencies": 0,
                "satisfied": 0,
                "missing": 0,
                "version_mismatch": 0,
                "import_errors": 0,
                "critical_missing": 0
            }
            
            # Collect dependencies from all services
            for service_name, capabilities in self.service_capabilities.items():
                for dep_name, dep_info in capabilities.dependencies.items():
                    # Use service-specific key to avoid conflicts
                    key = f"{service_name}_{dep_name}"
                    all_dependencies[key] = {
                        "service": service_name,
                        "dependency": dep_name,
                        "required_version": dep_info.required_version,
                        "installed_version": dep_info.installed_version,
                        "status": dep_info.status.value,
                        "critical": dep_info.critical,
                        "import_path": dep_info.import_path,
                        "error_message": dep_info.error_message
                    }
                    
                    # Update summary
                    dependency_summary["total_dependencies"] += 1
                    if dep_info.status == DependencyStatus.SATISFIED:
                        dependency_summary["satisfied"] += 1
                    elif dep_info.status == DependencyStatus.MISSING:
                        dependency_summary["missing"] += 1
                        if dep_info.critical:
                            dependency_summary["critical_missing"] += 1
                    elif dep_info.status == DependencyStatus.VERSION_MISMATCH:
                        dependency_summary["version_mismatch"] += 1
                    elif dep_info.status == DependencyStatus.IMPORT_ERROR:
                        dependency_summary["import_errors"] += 1
            
            return {
                "scan_timestamp": self.last_scan_time,
                "summary": dependency_summary,
                "dependencies": all_dependencies,
                "services_scanned": list(self.service_capabilities.keys()),
                "recommendations": self._generate_dependency_recommendations(all_dependencies)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate dependency report: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _generate_dependency_recommendations(self, dependencies: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dependency analysis"""
        recommendations = []
        
        # Check for missing critical dependencies
        critical_missing = [
            dep for dep in dependencies.values() 
            if dep["critical"] and dep["status"] == "missing"
        ]
        
        if critical_missing:
            recommendations.append(f"Install {len(critical_missing)} critical missing dependencies")
            for dep in critical_missing[:3]:  # Show first 3
                recommendations.append(f"  - Install {dep['dependency']} for {dep['service']}")
        
        # Check for version mismatches
        version_issues = [
            dep for dep in dependencies.values() 
            if dep["status"] == "version_mismatch"
        ]
        
        if version_issues:
            recommendations.append(f"Update {len(version_issues)} dependencies with version mismatches")
        
        # Check for TensorRT availability
        tensorrt_deps = [
            dep for dep in dependencies.values() 
            if "tensorrt" in dep["dependency"].lower()
        ]
        
        if any(dep["status"] != "satisfied" for dep in tensorrt_deps):
            recommendations.append("Install TensorRT for GPU acceleration across all services")
        
        # Check for cuDF availability
        cudf_deps = [
            dep for dep in dependencies.values() 
            if "cudf" in dep["dependency"].lower()
        ]
        
        if any(dep["status"] != "satisfied" for dep in cudf_deps):
            recommendations.append("Install cuDF for GPU-accelerated data processing")
        
        return recommendations

# Global transparency endpoints instance
_transparency_endpoints = None

def get_transparency_endpoints() -> TransparencyEndpoints:
    """Get global transparency endpoints instance"""
    global _transparency_endpoints
    if _transparency_endpoints is None:
        _transparency_endpoints = TransparencyEndpoints()
    return _transparency_endpoints
