"""
Database Authentication Handler - Secure Credential Management
Handles authentication, credential validation, and security for database connections

This module provides secure authentication management for database connections,
including credential validation, token management, and security enforcement.

Created: 2025-07-29 AEST
Purpose: Secure database authentication management
Module Size: 150 lines (modular design)
"""

import os
import logging
import time
import hashlib
import base64
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
import re

logger = logging.getLogger(__name__)


class DatabaseAuthenticationHandler:
    """
    Database Authentication Handler
    
    Provides secure credential management, validation, and authentication
    for database connections across the Four-Brain system.
    """
    
    def __init__(self, brain_id: str):
        """Initialize authentication handler"""
        self.brain_id = brain_id
        self.credentials_cache = {}
        self.validation_cache = {}
        self.failed_attempts = {}
        
        # Security settings
        self.max_failed_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        self.credential_ttl = 3600   # 1 hour
        
        # Initialize encryption key for sensitive data
        self._init_encryption()
        
        logger.info(f"🔐 Database Authentication Handler initialized for {brain_id}")
    
    def _init_encryption(self):
        """Initialize encryption for sensitive credential storage"""
        try:
            # Try to get encryption key from environment
            key = os.getenv('DB_ENCRYPTION_KEY')
            if not key:
                # Generate a key for this session (not persistent)
                key = Fernet.generate_key()
                logger.warning("⚠️ Using session-only encryption key - credentials won't persist")
            else:
                key = key.encode()
            
            self.cipher = Fernet(key)
            logger.debug("🔒 Encryption initialized for credential protection")
            
        except Exception as e:
            logger.warning(f"⚠️ Encryption initialization failed: {e}")
            self.cipher = None
    
    def validate_credentials(self, credential_type: str, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate database credentials based on type"""
        
        # Check for lockout
        if self._is_locked_out(credential_type):
            return False, f"Authentication locked due to failed attempts. Try again later."
        
        try:
            if credential_type == "supabase":
                return self._validate_supabase_credentials(credentials)
            elif credential_type == "postgresql":
                return self._validate_postgresql_credentials(credentials)
            elif credential_type == "local_postgres":
                return self._validate_local_postgres_credentials(credentials)
            else:
                return False, f"Unknown credential type: {credential_type}"
                
        except Exception as e:
            self._record_failed_attempt(credential_type)
            logger.error(f"❌ Credential validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_supabase_credentials(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Supabase credentials"""
        required_fields = ['url', 'service_role_key']
        
        # Check required fields
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                return False, f"Missing required field: {field}"
        
        # Validate URL format
        url = credentials['url']
        if not url.startswith('https://') or not '.supabase.co' in url:
            return False, "Invalid Supabase URL format"
        
        # Validate service role key format (JWT)
        service_key = credentials['service_role_key']
        if not self._validate_jwt_format(service_key):
            return False, "Invalid service role key format"
        
        # Extract and validate project reference
        try:
            project_ref = url.split('//')[1].split('.')[0]
            if not re.match(r'^[a-z0-9]{20}$', project_ref):
                return False, "Invalid project reference format"
            
            # Validate JWT payload contains correct project reference
            if not self._validate_jwt_project_ref(service_key, project_ref):
                return False, "Service key doesn't match project reference"
                
        except Exception as e:
            return False, f"Project reference validation failed: {e}"
        
        logger.info("✅ Supabase credentials validated successfully")
        return True, "Supabase credentials valid"
    
    def _validate_postgresql_credentials(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate PostgreSQL credentials"""
        required_fields = ['host', 'port', 'database', 'username', 'password']
        
        # Check required fields
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                return False, f"Missing required field: {field}"
        
        # Validate port
        try:
            port = int(credentials['port'])
            if port < 1 or port > 65535:
                return False, "Invalid port number"
        except ValueError:
            return False, "Port must be a number"
        
        # Validate host format
        host = credentials['host']
        if not re.match(r'^[a-zA-Z0-9.-]+$', host):
            return False, "Invalid host format"
        
        # Validate database name
        database = credentials['database']
        if not re.match(r'^[a-zA-Z0-9_]+$', database):
            return False, "Invalid database name format"
        
        # Validate username
        username = credentials['username']
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Invalid username format"
        
        # Password strength check
        password = credentials['password']
        if len(password) < 8:
            return False, "Password too short (minimum 8 characters)"
        
        logger.info("✅ PostgreSQL credentials validated successfully")
        return True, "PostgreSQL credentials valid"
    
    def _validate_local_postgres_credentials(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate local PostgreSQL credentials (more lenient)"""
        required_fields = ['username', 'password', 'database']
        
        # Check required fields
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                return False, f"Missing required field: {field}"
        
        logger.info("✅ Local PostgreSQL credentials validated successfully")
        return True, "Local PostgreSQL credentials valid"
    
    def _validate_jwt_format(self, token: str) -> bool:
        """Validate JWT token format"""
        try:
            # JWT should have 3 parts separated by dots
            parts = token.split('.')
            if len(parts) != 3:
                return False
            
            # Each part should be valid base64
            for part in parts:
                # Add padding if needed
                padded = part + '=' * (4 - len(part) % 4)
                base64.urlsafe_b64decode(padded)
            
            return True
            
        except Exception:
            return False
    
    def _validate_jwt_project_ref(self, token: str, expected_ref: str) -> bool:
        """Validate JWT contains correct project reference"""
        try:
            # Decode JWT without verification (just to check payload)
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Check if 'ref' field matches expected project reference
            token_ref = payload.get('ref')
            return token_ref == expected_ref
            
        except Exception as e:
            logger.warning(f"⚠️ JWT payload validation failed: {e}")
            return False
    
    def store_credentials(self, credential_type: str, credentials: Dict[str, Any]) -> bool:
        """Securely store validated credentials"""
        try:
            # Validate credentials first
            is_valid, message = self.validate_credentials(credential_type, credentials)
            if not is_valid:
                logger.error(f"❌ Cannot store invalid credentials: {message}")
                return False
            
            # Encrypt sensitive data if encryption available
            if self.cipher:
                encrypted_creds = self._encrypt_credentials(credentials)
            else:
                encrypted_creds = credentials.copy()
                logger.warning("⚠️ Storing credentials without encryption")
            
            # Store with timestamp
            self.credentials_cache[credential_type] = {
                'credentials': encrypted_creds,
                'stored_at': time.time(),
                'brain_id': self.brain_id
            }
            
            logger.info(f"🔐 Credentials stored securely for {credential_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store credentials: {e}")
            return False
    
    def get_credentials(self, credential_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt stored credentials"""
        try:
            if credential_type not in self.credentials_cache:
                return None
            
            cached_data = self.credentials_cache[credential_type]
            
            # Check if credentials have expired
            if time.time() - cached_data['stored_at'] > self.credential_ttl:
                del self.credentials_cache[credential_type]
                logger.info(f"🕐 Expired credentials removed for {credential_type}")
                return None
            
            # Decrypt credentials if encryption was used
            if self.cipher:
                return self._decrypt_credentials(cached_data['credentials'])
            else:
                return cached_data['credentials'].copy()
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve credentials: {e}")
            return None
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive credential fields"""
        encrypted = {}
        sensitive_fields = ['password', 'service_role_key', 'anon_key']
        
        for key, value in credentials.items():
            if key in sensitive_fields and isinstance(value, str):
                encrypted[key] = self.cipher.encrypt(value.encode()).decode()
            else:
                encrypted[key] = value
        
        return encrypted
    
    def _decrypt_credentials(self, encrypted_credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive credential fields"""
        decrypted = {}
        sensitive_fields = ['password', 'service_role_key', 'anon_key']
        
        for key, value in encrypted_credentials.items():
            if key in sensitive_fields and isinstance(value, str):
                decrypted[key] = self.cipher.decrypt(value.encode()).decode()
            else:
                decrypted[key] = value
        
        return decrypted
    
    def _record_failed_attempt(self, credential_type: str):
        """Record failed authentication attempt"""
        current_time = time.time()
        
        if credential_type not in self.failed_attempts:
            self.failed_attempts[credential_type] = []
        
        self.failed_attempts[credential_type].append(current_time)
        
        # Clean old attempts
        cutoff_time = current_time - self.lockout_duration
        self.failed_attempts[credential_type] = [
            attempt for attempt in self.failed_attempts[credential_type]
            if attempt > cutoff_time
        ]
        
        logger.warning(f"⚠️ Failed authentication attempt recorded for {credential_type}")
    
    def _is_locked_out(self, credential_type: str) -> bool:
        """Check if credential type is locked out due to failed attempts"""
        if credential_type not in self.failed_attempts:
            return False
        
        current_time = time.time()
        cutoff_time = current_time - self.lockout_duration
        
        # Count recent failed attempts
        recent_failures = [
            attempt for attempt in self.failed_attempts[credential_type]
            if attempt > cutoff_time
        ]
        
        return len(recent_failures) >= self.max_failed_attempts
    
    def clear_failed_attempts(self, credential_type: str):
        """Clear failed attempts for successful authentication"""
        if credential_type in self.failed_attempts:
            del self.failed_attempts[credential_type]
            logger.info(f"🧹 Failed attempts cleared for {credential_type}")
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        return {
            "brain_id": self.brain_id,
            "stored_credentials": list(self.credentials_cache.keys()),
            "failed_attempts": {
                cred_type: len(attempts) 
                for cred_type, attempts in self.failed_attempts.items()
            },
            "encryption_enabled": self.cipher is not None,
            "credential_ttl": self.credential_ttl,
            "max_failed_attempts": self.max_failed_attempts
        }


# Factory function for easy creation
def create_auth_handler(brain_id: str) -> DatabaseAuthenticationHandler:
    """Factory function to create database authentication handler"""
    return DatabaseAuthenticationHandler(brain_id)
