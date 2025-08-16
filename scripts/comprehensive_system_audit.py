#!/usr/bin/env python3
"""
Comprehensive System Audit for Augment Agent Evolution
Validates entire project structure, connections, and integrity
"""

import os
import sys
import json
import subprocess
import redis
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

class AugmentSystemAuditor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.audit_results = {}
        self.redis_client = None
        self.issues = []
        self.warnings = []
        
    def connect_redis(self):
        """Connect to Redis for storing audit results"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            print("✅ Connected to Redis")
        except Exception as e:
            self.issues.append(f"❌ Redis connection failed: {e}")
            
    def audit_project_structure(self):
        """Audit the overall project structure based on README.md requirements"""
        print("\n🔍 AUDITING PROJECT STRUCTURE...")

        # Based on README.md - Four-Brain Architecture requirements
        expected_structure = {
            'containers/four-brain': {
                'required': True,
                'description': 'Main Four-Brain system container',
                'subdirs': ['docker', 'src', 'config', 'monitoring', 'scripts']
            },
            'docs': {
                'required': True,
                'description': 'Complete documentation system',
                'subdirs': ['architecture', 'api', 'deployment', 'monitoring', 'operations']
            },
            'scripts': {
                'required': True,
                'description': 'Main project scripts',
                'files': ['comprehensive_system_audit.py']
            },
            'config': {
                'required': True,
                'description': 'System configuration'
            },
            'models': {
                'required': True,
                'description': 'AI model storage'
            },
            'data': {
                'required': True,
                'description': 'Data storage'
            }
        }

        print("\n📋 FOUR-BRAIN ARCHITECTURE COMPLIANCE:")
        for path, requirements in expected_structure.items():
            full_path = self.project_root / path
            if not full_path.exists():
                if requirements['required']:
                    self.issues.append(f"❌ Missing required: {path} - {requirements['description']}")
                else:
                    self.warnings.append(f"⚠️ Missing optional: {path}")
            else:
                print(f"✅ {path}: {requirements['description']}")

                # Check subdirectories if specified
                if 'subdirs' in requirements:
                    for subdir in requirements['subdirs']:
                        subdir_path = full_path / subdir
                        if subdir_path.exists():
                            print(f"  ✅ {subdir}/")
                        else:
                            self.warnings.append(f"⚠️ Missing subdir: {path}/{subdir}")

                # Check required files if specified
                if 'files' in requirements:
                    for file_name in requirements['files']:
                        file_path = full_path / file_name
                        if file_path.exists():
                            print(f"  ✅ {file_name}")
                        else:
                            self.issues.append(f"❌ Missing required file: {path}/{file_name}")

        # Analyze Four-Brain service structure
        self.audit_four_brain_services()

    def audit_four_brain_services(self):
        """Audit Four-Brain service implementation - Unified Architecture"""
        print("\n🧠 AUDITING FOUR-BRAIN SERVICES:")

        src_path = self.project_root / 'containers/four-brain/src'
        if not src_path.exists():
            self.issues.append("❌ Four-Brain src directory missing")
            return

        # Check for unified main.py entry point
        main_py = src_path / 'main.py'
        if main_py.exists():
            print("✅ main.py: Unified Four-Brain System Entry Point")
            self.audit_service_structure(main_py, "Unified Four-Brain System")
        else:
            self.issues.append("❌ Missing main.py - Unified system entry point")

        # Check for brain modules in brains/ directory
        brain_modules = {
            'brains/embedding_service': 'Brain-1: Vector Embeddings (Qwen3-8B NVFP4)',
            'brains/reranker_service': 'Brain-2: Result Ranking (Qwen3-Reranker-8B NVFP4)',
            'brains/intelligence_service': 'Brain-3: AI Reasoning (HRM Manager)',
            'brains/document_processor': 'Brain-4: Document Processing (Docling)',
            'orchestrator_hub': 'Orchestrator Hub: Four-Brain Coordination'
        }

        for module_path, description in brain_modules.items():
            module_dir = src_path / module_path
            if module_dir.exists():
                print(f"✅ {module_path}/: {description}")
                # Check for key files in module
                self.audit_brain_module(module_dir, description)
            else:
                self.issues.append(f"❌ Missing brain module: {module_path} - {description}")

    def audit_brain_module(self, module_path, description):
        """Audit individual brain module structure"""
        key_files = ['__init__.py']
        python_files = list(module_path.glob('*.py'))

        if len(python_files) == 0:
            self.warnings.append(f"⚠️ {module_path.name} has no Python files")
        else:
            print(f"  📁 {len(python_files)} Python files found")

        # Check for core functionality files
        core_files = ['core', 'modules', 'config']
        for core_dir in core_files:
            core_path = module_path / core_dir
            if core_path.exists():
                print(f"  ✅ {core_dir}/ directory")
            else:
                self.warnings.append(f"⚠️ {module_path.name} missing {core_dir}/ directory")

    def audit_service_structure(self, service_path, description):
        """Audit individual service structure"""
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for essential components
            required_patterns = {
                'FastAPI': 'from fastapi import',
                'Health endpoint': '/health',
                'Error handling': 'try:',
                'Logging': 'import logging'
            }

            missing_components = []
            for component, pattern in required_patterns.items():
                if pattern not in content:
                    missing_components.append(component)

            if missing_components:
                self.warnings.append(f"⚠️ {service_path.name} missing: {missing_components}")

        except Exception as e:
            self.warnings.append(f"⚠️ Could not analyze {service_path.name}: {e}")
                
    def audit_docker_integration(self):
        """Audit Docker and Linux integration"""
        print("\n🐳 AUDITING DOCKER INTEGRATION...")
        
        dockerfile_path = self.project_root / 'containers/four-brain/docker/Dockerfile'
        if not dockerfile_path.exists():
            self.issues.append("❌ Main Dockerfile not found")
            return
            
        # Check what's being copied to Linux containers
        with open(dockerfile_path, 'r', encoding='utf-8') as f:
            dockerfile_content = f.read()
            
        copy_commands = []
        for line in dockerfile_content.split('\n'):
            if line.strip().startswith('COPY'):
                copy_commands.append(line.strip())
                
        print("📋 COPY COMMANDS IN DOCKERFILE:")
        for cmd in copy_commands:
            print(f"  {cmd}")
            
        # Check if main scripts are copied
        main_scripts_copied = any('scripts/' in cmd for cmd in copy_commands)
        if not main_scripts_copied:
            self.issues.append("❌ Main /scripts folder not copied to Linux containers")
            
        # Check for duplicate stage names
        if 'AS base' in dockerfile_content:
            base_count = dockerfile_content.count('AS base')
            if base_count > 1:
                self.issues.append(f"❌ Duplicate 'AS base' stages found: {base_count}")
                
    def audit_monitoring_system(self):
        """Audit monitoring and observability setup"""
        print("\n📊 AUDITING MONITORING SYSTEM...")

        try:
            result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'],
                                  capture_output=True, text=True)

            running_containers = result.stdout
            print("🔍 RUNNING CONTAINERS:")
            print(running_containers)

            # Check for monitoring containers based on README requirements
            required_monitoring = {
                'grafana': 'Visualization Dashboard (Port 3000)',
                'prometheus': 'Time-Series DB (Port 9090)',
                'loki': 'Log Aggregation (Port 3100)',
                'alloy': 'Unified Observability (Port 12345)',
                'redis': 'Inter-brain Communication (Port 6380)'
            }

            missing_monitoring = []
            for service, description in required_monitoring.items():
                if service not in running_containers.lower():
                    missing_monitoring.append(f"{service} ({description})")
                else:
                    print(f"✅ {service}: {description}")

            if missing_monitoring:
                self.warnings.append(f"⚠️ Missing monitoring services: {missing_monitoring}")

        except Exception as e:
            self.issues.append(f"❌ Docker status check failed: {e}")

    def audit_docker_images(self):
        """Audit built Docker images for Four-Brain services"""
        print("\n🐳 AUDITING DOCKER IMAGES...")

        try:
            result = subprocess.run(['docker', 'images', '--format', 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}'],
                                  capture_output=True, text=True)

            images_output = result.stdout

            # Expected Four-Brain images based on README
            expected_images = {
                'docker-embedding-service': 'Brain-1: Vector Embeddings',
                'docker-reranker-service': 'Brain-2: Result Ranking',
                'docker-intelligence-service': 'Brain-3: AI Reasoning',
                'docker-document-processor': 'Brain-4: Document Processing',
                'docker-orchestrator-hub': 'Orchestrator: System Coordination'
            }

            print("🔍 FOUR-BRAIN DOCKER IMAGES:")
            missing_images = []

            for image_name, description in expected_images.items():
                if image_name in images_output:
                    # Extract size and creation info
                    for line in images_output.split('\n'):
                        if image_name in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                size = parts[2]
                                print(f"✅ {image_name}: {description} ({size})")
                            break
                else:
                    missing_images.append(f"{image_name} ({description})")

            if missing_images:
                self.issues.append(f"❌ Missing Docker images: {missing_images}")

            # Check image sizes for optimization
            self.audit_image_optimization(images_output)

        except Exception as e:
            self.issues.append(f"❌ Docker images check failed: {e}")

    def audit_image_optimization(self, images_output):
        """Check Docker image sizes for optimization opportunities"""
        print("\n⚡ DOCKER IMAGE OPTIMIZATION ANALYSIS:")

        large_images = []
        for line in images_output.split('\n'):
            if 'docker-' in line and ('GB' in line):
                parts = line.split()
                if len(parts) >= 3:
                    repo = parts[0]
                    size_str = parts[2]
                    if 'GB' in size_str:
                        try:
                            size_gb = float(size_str.replace('GB', ''))
                            if size_gb > 30:  # Flag images over 30GB
                                large_images.append(f"{repo}: {size_str}")
                        except ValueError:
                            pass

        if large_images:
            self.warnings.append(f"⚠️ Large Docker images (>30GB): {large_images}")
            print("💡 Consider multi-stage builds and layer optimization")
        else:
            print("✅ Docker images are reasonably sized")
            
    def audit_file_redundancy(self):
        """Check for duplicate or redundant files"""
        print("\n🔄 AUDITING FILE REDUNDANCY...")
        
        # Look for potential duplicates
        script_files = {}
        
        for script_dir in ['scripts', 'containers/four-brain/scripts']:
            script_path = self.project_root / script_dir
            if script_path.exists():
                for file_path in script_path.rglob('*.py'):
                    filename = file_path.name
                    if filename in script_files:
                        script_files[filename].append(str(file_path))
                    else:
                        script_files[filename] = [str(file_path)]
                        
        duplicates = {k: v for k, v in script_files.items() if len(v) > 1}
        if duplicates:
            print("⚠️ POTENTIAL DUPLICATE FILES:")
            for filename, paths in duplicates.items():
                print(f"  {filename}: {paths}")
                self.warnings.append(f"Duplicate file: {filename} in {len(paths)} locations")
                
    def audit_configuration_consistency(self):
        """Check configuration file consistency"""
        print("\n⚙️ AUDITING CONFIGURATION CONSISTENCY...")
        
        config_files = [
            'containers/four-brain/docker/docker-compose.yml',
            'containers/four-brain/config/production.env',
            'containers/four-brain/monitoring/prometheus.yml'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"✅ Found: {config_file}")
            else:
                self.issues.append(f"❌ Missing config: {config_file}")
                
    def store_results_in_redis(self):
        """Store audit results in Redis"""
        if not self.redis_client:
            return
            
        timestamp = datetime.now().isoformat()
        
        # Store summary
        self.redis_client.hset('system_audit_summary', mapping={
            'timestamp': timestamp,
            'total_issues': len(self.issues),
            'total_warnings': len(self.warnings),
            'status': 'FAILED' if self.issues else 'PASSED'
        })
        
        # Store detailed results
        for i, issue in enumerate(self.issues):
            self.redis_client.hset(f'system_audit_issues', f'issue_{i}', issue)
            
        for i, warning in enumerate(self.warnings):
            self.redis_client.hset(f'system_audit_warnings', f'warning_{i}', warning)
            
        print(f"\n💾 Audit results stored in Redis")
        
    def generate_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE SYSTEM AUDIT REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Issues Found: {len(self.issues)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Overall Status: {'❌ FAILED' if self.issues else '✅ PASSED'}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  {issue}")
                
        if self.warnings:
            print(f"\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
                
        print(f"\n🔧 RECOMMENDATIONS:")
        if self.issues:
            print("  1. Fix critical issues before deployment")
            print("  2. Review Docker integration and Linux file copying")
            print("  3. Start missing monitoring services")
        else:
            print("  System appears healthy - ready for operation")
            
    def run_full_audit(self):
        """Run complete system audit"""
        print("🚀 STARTING COMPREHENSIVE FOUR-BRAIN SYSTEM AUDIT...")
        print("📋 Based on README.md Production-Ready Architecture")

        self.connect_redis()
        self.audit_project_structure()
        self.audit_docker_integration()
        self.audit_docker_images()
        self.audit_monitoring_system()
        self.audit_file_redundancy()
        self.audit_configuration_consistency()
        self.store_results_in_redis()
        self.generate_report()

        return len(self.issues) == 0

if __name__ == "__main__":
    auditor = AugmentSystemAuditor()
    success = auditor.run_full_audit()
    sys.exit(0 if success else 1)
