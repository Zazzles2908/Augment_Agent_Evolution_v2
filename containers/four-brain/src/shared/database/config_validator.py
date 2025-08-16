"""
Database Configuration Validator - Environment Validation
Validates database configuration and environment variables

This module provides comprehensive validation of database configuration,
environment variables, and system requirements for the Four-Brain system.

Created: 2025-07-29 AEST
Purpose: Validate database configuration and environment
Module Size: 150 lines (modular design)
"""

import os
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
import socket
import asyncio

logger = logging.getLogger(__name__)


class DatabaseConfigValidator:
    """
    Database Configuration Validator
    
    Provides comprehensive validation of database configuration,
    environment variables, and connectivity requirements.
    """
    
    def __init__(self, brain_id: str):
        """Initialize configuration validator"""
        self.brain_id = brain_id
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
        # Required environment variables
        self.required_env_vars = {
            'supabase': [
                'SUPABASE_URL',
                'SUPABASE_SERVICE_ROLE_KEY'
            ],
            'postgresql': [
                'POSTGRES_HOST',
                'POSTGRES_PORT', 
                'POSTGRES_DB',
                'POSTGRES_USER',
                'POSTGRES_PASSWORD'
            ],
            'general': [
                'AUGMENT_SCHEMA'
            ]
        }
        
        # Optional but recommended environment variables
        self.optional_env_vars = [
            'SUPABASE_ANON_KEY',
            'DB_ENCRYPTION_KEY',
            'DATABASE_URL'
        ]
        
        logger.info(f"🔍 Database Config Validator initialized for {brain_id}")
    
    def validate_all(self) -> Dict[str, Any]:
        """Perform comprehensive configuration validation"""
        logger.info("🔍 Starting comprehensive database configuration validation...")
        
        # Reset validation state
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
        # Validate environment variables
        env_validation = self._validate_environment_variables()
        self.validation_results['environment'] = env_validation
        
        # Validate Supabase configuration
        supabase_validation = self._validate_supabase_config()
        self.validation_results['supabase'] = supabase_validation
        
        # Validate PostgreSQL configuration
        postgres_validation = self._validate_postgresql_config()
        self.validation_results['postgresql'] = postgres_validation
        
        # Validate network connectivity
        network_validation = self._validate_network_connectivity()
        self.validation_results['network'] = network_validation
        
        # Validate schema configuration
        schema_validation = self._validate_schema_config()
        self.validation_results['schema'] = schema_validation
        
        # Generate overall assessment
        overall_status = self._generate_overall_assessment()
        
        result = {
            'brain_id': self.brain_id,
            'overall_status': overall_status,
            'validation_results': self.validation_results,
            'warnings': self.warnings,
            'errors': self.errors,
            'recommendations': self._generate_recommendations()
        }
        
        logger.info(f"✅ Configuration validation completed - Status: {overall_status}")
        return result
    
    def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validate required and optional environment variables"""
        env_status = {
            'required_missing': [],
            'optional_missing': [],
            'present': [],
            'invalid_format': []
        }
        
        # Check required variables
        for category, vars_list in self.required_env_vars.items():
            for var in vars_list:
                value = os.getenv(var)
                if not value:
                    env_status['required_missing'].append(var)
                    self.errors.append(f"Required environment variable missing: {var}")
                else:
                    env_status['present'].append(var)
                    
                    # Validate format for specific variables
                    if not self._validate_env_var_format(var, value):
                        env_status['invalid_format'].append(var)
                        self.errors.append(f"Invalid format for {var}")
        
        # Check optional variables
        for var in self.optional_env_vars:
            value = os.getenv(var)
            if not value:
                env_status['optional_missing'].append(var)
                self.warnings.append(f"Optional environment variable missing: {var}")
            else:
                env_status['present'].append(var)
        
        return env_status
    
    def _validate_env_var_format(self, var_name: str, value: str) -> bool:
        """Validate format of specific environment variables"""
        
        if var_name == 'SUPABASE_URL':
            return bool(re.match(r'^https://[a-z0-9]{20}\.supabase\.co$', value))
        
        elif var_name == 'SUPABASE_SERVICE_ROLE_KEY':
            # Should be a JWT token
            return len(value.split('.')) == 3
        
        elif var_name == 'POSTGRES_PORT':
            try:
                port = int(value)
                return 1 <= port <= 65535
            except ValueError:
                return False
        
        elif var_name == 'POSTGRES_HOST':
            # Basic hostname validation
            return bool(re.match(r'^[a-zA-Z0-9.-]+$', value))
        
        elif var_name == 'POSTGRES_DB':
            # Database name validation
            return bool(re.match(r'^[a-zA-Z0-9_]+$', value))
        
        elif var_name == 'POSTGRES_USER':
            # Username validation
            return bool(re.match(r'^[a-zA-Z0-9_]+$', value))
        
        elif var_name == 'AUGMENT_SCHEMA':
            # Schema name validation
            return bool(re.match(r'^[a-zA-Z0-9_]+$', value))
        
        return True  # Default to valid for unknown variables
    
    def _validate_supabase_config(self) -> Dict[str, Any]:
        """Validate Supabase configuration"""
        supabase_status = {
            'url_valid': False,
            'service_key_valid': False,
            'project_ref_match': False,
            'region_detected': None
        }
        
        supabase_url = os.getenv('SUPABASE_URL')
        service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if supabase_url:
            # Validate URL format
            if re.match(r'^https://[a-z0-9]{20}\.supabase\.co$', supabase_url):
                supabase_status['url_valid'] = True
                
                # Extract project reference
                project_ref = supabase_url.split('//')[1].split('.')[0]
                
                # Detect region (this is a best guess based on common patterns)
                # In practice, you'd need to query Supabase API for actual region
                supabase_status['region_detected'] = 'ap-southeast-2'  # Default from project info
                
                if service_key:
                    # Validate service key format
                    if len(service_key.split('.')) == 3:
                        supabase_status['service_key_valid'] = True
                        
                        # Check if project ref in JWT matches URL
                        try:
                            import jwt
                            payload = jwt.decode(service_key, options={"verify_signature": False})
                            token_ref = payload.get('ref')
                            if token_ref == project_ref:
                                supabase_status['project_ref_match'] = True
                            else:
                                self.errors.append("Project reference mismatch between URL and service key")
                        except Exception as e:
                            self.warnings.append(f"Could not validate JWT payload: {e}")
            else:
                self.errors.append("Invalid Supabase URL format")
        
        return supabase_status
    
    def _validate_postgresql_config(self) -> Dict[str, Any]:
        """Validate PostgreSQL configuration"""
        postgres_status = {
            'host_valid': False,
            'port_valid': False,
            'credentials_present': False,
            'connection_string_valid': False
        }
        
        host = os.getenv('POSTGRES_HOST')
        port = os.getenv('POSTGRES_PORT')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        database = os.getenv('POSTGRES_DB')
        
        # Validate host
        if host and re.match(r'^[a-zA-Z0-9.-]+$', host):
            postgres_status['host_valid'] = True
        
        # Validate port
        if port:
            try:
                port_num = int(port)
                if 1 <= port_num <= 65535:
                    postgres_status['port_valid'] = True
            except ValueError:
                self.errors.append("Invalid PostgreSQL port number")
        
        # Check credentials
        if user and password and database:
            postgres_status['credentials_present'] = True
        
        # Validate connection string if present
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            try:
                parsed = urlparse(database_url)
                if parsed.scheme == 'postgresql' and parsed.hostname and parsed.port:
                    postgres_status['connection_string_valid'] = True
            except Exception:
                self.warnings.append("Invalid DATABASE_URL format")
        
        return postgres_status
    
    def _validate_network_connectivity(self) -> Dict[str, Any]:
        """Validate network connectivity (basic checks)"""
        network_status = {
            'supabase_reachable': False,
            'postgres_reachable': False,
            'dns_resolution': False
        }
        
        # Test Supabase connectivity
        supabase_url = os.getenv('SUPABASE_URL')
        if supabase_url:
            try:
                hostname = supabase_url.split('//')[1].split('/')[0]
                socket.gethostbyname(hostname)
                network_status['supabase_reachable'] = True
                network_status['dns_resolution'] = True
            except Exception:
                self.warnings.append("Cannot resolve Supabase hostname")
        
        # Test PostgreSQL connectivity
        postgres_host = os.getenv('POSTGRES_HOST')
        postgres_port = os.getenv('POSTGRES_PORT')
        if postgres_host and postgres_port:
            try:
                port_num = int(postgres_port)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((postgres_host, port_num))
                sock.close()
                
                if result == 0:
                    network_status['postgres_reachable'] = True
                else:
                    self.warnings.append(f"Cannot connect to PostgreSQL at {postgres_host}:{postgres_port}")
            except Exception as e:
                self.warnings.append(f"PostgreSQL connectivity test failed: {e}")
        
        return network_status
    
    def _validate_schema_config(self) -> Dict[str, Any]:
        """Validate schema configuration"""
        schema_status = {
            'schema_name_valid': False,
            'table_config_present': False
        }
        
        schema_name = os.getenv('AUGMENT_SCHEMA')
        if schema_name and re.match(r'^[a-zA-Z0-9_]+$', schema_name):
            schema_status['schema_name_valid'] = True
        
        # Check for table configuration
        table_vars = [
            'AUGMENT_SESSIONS_TABLE',
            'AUGMENT_KNOWLEDGE_TABLE',
            'AUGMENT_PATTERNS_TABLE'
        ]
        
        if any(os.getenv(var) for var in table_vars):
            schema_status['table_config_present'] = True
        
        return schema_status
    
    def _generate_overall_assessment(self) -> str:
        """Generate overall configuration assessment"""
        if self.errors:
            return "critical_issues"
        elif len(self.warnings) > 3:
            return "needs_attention"
        elif self.warnings:
            return "minor_issues"
        else:
            return "healthy"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate configuration recommendations"""
        recommendations = []
        
        # Check for missing encryption key
        if not os.getenv('DB_ENCRYPTION_KEY'):
            recommendations.append("Set DB_ENCRYPTION_KEY for credential encryption")
        
        # Check for missing optional Supabase key
        if not os.getenv('SUPABASE_ANON_KEY'):
            recommendations.append("Set SUPABASE_ANON_KEY for client-side operations")
        
        # Check for connection string
        if not os.getenv('DATABASE_URL'):
            recommendations.append("Set DATABASE_URL for simplified connection management")
        
        # Performance recommendations
        recommendations.append("Consider connection pooling for high-load scenarios")
        recommendations.append("Enable SSL/TLS for production database connections")
        
        return recommendations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results"""
        return {
            'brain_id': self.brain_id,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'critical_issues': [error for error in self.errors],
            'attention_needed': [warning for warning in self.warnings]
        }


# Factory function for easy creation
def create_config_validator(brain_id: str) -> DatabaseConfigValidator:
    """Factory function to create database configuration validator"""
    return DatabaseConfigValidator(brain_id)
