#!/usr/bin/env python3.11
"""
Redis Authentication Manager for Four-Brain System
Production-ready Redis security with authentication and encryption

Author: Zazzles's Agent
Date: 2025-08-02
Purpose: Secure Redis connections with authentication and monitoring
"""

import os
import sys
import logging
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import redis
import ssl

# Configure logging
logger = logging.getLogger(__name__)

class RedisSecurityLevel(Enum):
    """Redis security levels"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class RedisAuthConfig:
    """Redis authentication configuration"""
    host: str
    port: int
    password: Optional[str] = None
    username: Optional[str] = None
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    connection_timeout: int = 30
    socket_timeout: int = 30
    max_connections: int = 100
    security_level: RedisSecurityLevel = RedisSecurityLevel.DEVELOPMENT

class RedisAuthManager:
    """Manages Redis authentication and secure connections"""
    
    def __init__(self, config: RedisAuthConfig):
        self.config = config
        self.connection_pool = None
        self.authenticated = False
        
        # Security monitoring
        self.failed_auth_attempts = 0
        self.last_auth_attempt = 0.0
        self.max_auth_attempts = 5
        self.auth_lockout_duration = 300  # 5 minutes
        
        logger.info("🔐 Redis Authentication Manager initialized")
        logger.info(f"  Security Level: {config.security_level.value}")
        logger.info(f"  SSL Enabled: {config.ssl_enabled}")
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate cryptographically secure password for Redis"""
        # Use secrets module for cryptographically secure random generation
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        logger.info(f"✅ Generated secure Redis password ({length} characters)")
        return password
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for Redis connections"""
        if not self.config.ssl_enabled:
            return None
        
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Configure SSL settings for production
            if self.config.security_level == RedisSecurityLevel.PRODUCTION:
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                # Development/staging - less strict
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Load certificates if provided
            if self.config.ssl_cert_path and self.config.ssl_key_path:
                ssl_context.load_cert_chain(
                    certfile=self.config.ssl_cert_path,
                    keyfile=self.config.ssl_key_path
                )
                logger.info(f"✅ SSL client certificate loaded")
            
            if self.config.ssl_ca_path:
                ssl_context.load_verify_locations(cafile=self.config.ssl_ca_path)
                logger.info(f"✅ SSL CA certificate loaded")
            
            logger.info(f"✅ SSL context created for {self.config.security_level.value}")
            return ssl_context
            
        except Exception as e:
            logger.error(f"❌ Failed to create SSL context: {str(e)}")
            return None
    
    def create_connection_pool(self) -> redis.ConnectionPool:
        """Create authenticated Redis connection pool"""
        try:
            # Check for auth lockout
            if self._is_auth_locked_out():
                raise Exception("Authentication locked out due to too many failed attempts")
            
            # SSL context
            ssl_context = self.create_ssl_context()
            
            # Connection parameters
            connection_params = {
                'host': self.config.host,
                'port': self.config.port,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.connection_timeout,
                'max_connections': self.config.max_connections,
                'retry_on_timeout': True,
                'health_check_interval': 30
            }
            
            # Add authentication
            if self.config.password:
                connection_params['password'] = self.config.password
                
            if self.config.username:
                connection_params['username'] = self.config.username
            
            # Add SSL if enabled
            if ssl_context:
                connection_params['ssl'] = True
                connection_params['ssl_context'] = ssl_context
            
            # Create connection pool
            pool = redis.ConnectionPool(**connection_params)
            
            # Test connection
            test_client = redis.Redis(connection_pool=pool)
            test_client.ping()
            
            self.connection_pool = pool
            self.authenticated = True
            self.failed_auth_attempts = 0
            
            logger.info("✅ Redis authenticated connection pool created")
            return pool
            
        except redis.AuthenticationError as e:
            self._handle_auth_failure()
            logger.error(f"❌ Redis authentication failed: {str(e)}")
            raise
        except Exception as e:
            self._handle_auth_failure()
            logger.error(f"❌ Redis connection failed: {str(e)}")
            raise
    
    def get_authenticated_client(self) -> redis.Redis:
        """Get authenticated Redis client"""
        if not self.connection_pool:
            self.create_connection_pool()
        
        return redis.Redis(connection_pool=self.connection_pool)
    
    def _is_auth_locked_out(self) -> bool:
        """Check if authentication is locked out"""
        if self.failed_auth_attempts >= self.max_auth_attempts:
            time_since_last_attempt = time.time() - self.last_auth_attempt
            if time_since_last_attempt < self.auth_lockout_duration:
                remaining_time = self.auth_lockout_duration - time_since_last_attempt
                logger.warning(f"⚠️ Auth lockout active: {remaining_time:.0f}s remaining")
                return True
            else:
                # Reset after lockout period
                self.failed_auth_attempts = 0
                logger.info("🔓 Auth lockout expired, resetting attempts")
        
        return False
    
    def _handle_auth_failure(self):
        """Handle authentication failure"""
        self.failed_auth_attempts += 1
        self.last_auth_attempt = time.time()
        
        logger.warning(f"⚠️ Auth failure #{self.failed_auth_attempts}/{self.max_auth_attempts}")
        
        if self.failed_auth_attempts >= self.max_auth_attempts:
            logger.error(f"🔒 Auth lockout activated for {self.auth_lockout_duration}s")
    
    def validate_redis_security(self) -> Dict[str, Any]:
        """Validate Redis security configuration"""
        security_report = {
            "timestamp": time.time(),
            "security_level": self.config.security_level.value,
            "checks": {},
            "recommendations": [],
            "score": 0,
            "max_score": 100
        }
        
        # Check password strength
        if self.config.password:
            password_strength = self._assess_password_strength(self.config.password)
            security_report["checks"]["password_strength"] = password_strength
            security_report["score"] += password_strength["score"]
        else:
            security_report["checks"]["password_strength"] = {
                "status": "FAIL",
                "score": 0,
                "message": "No password configured"
            }
            security_report["recommendations"].append("Configure strong Redis password")
        
        # Check SSL configuration
        if self.config.ssl_enabled:
            security_report["checks"]["ssl_enabled"] = {
                "status": "PASS",
                "score": 25,
                "message": "SSL encryption enabled"
            }
            security_report["score"] += 25
        else:
            security_report["checks"]["ssl_enabled"] = {
                "status": "FAIL",
                "score": 0,
                "message": "SSL encryption disabled"
            }
            security_report["recommendations"].append("Enable SSL encryption for production")
        
        # Check authentication method
        if self.config.username and self.config.password:
            security_report["checks"]["authentication"] = {
                "status": "PASS",
                "score": 25,
                "message": "Username/password authentication configured"
            }
            security_report["score"] += 25
        elif self.config.password:
            security_report["checks"]["authentication"] = {
                "status": "PARTIAL",
                "score": 15,
                "message": "Password-only authentication configured"
            }
            security_report["score"] += 15
            security_report["recommendations"].append("Consider username-based authentication")
        else:
            security_report["checks"]["authentication"] = {
                "status": "FAIL",
                "score": 0,
                "message": "No authentication configured"
            }
            security_report["recommendations"].append("Configure Redis authentication")
        
        # Check connection security
        connection_security_score = 0
        if self.config.connection_timeout <= 30:
            connection_security_score += 5
        if self.config.max_connections <= 100:
            connection_security_score += 5
        
        security_report["checks"]["connection_security"] = {
            "status": "PASS" if connection_security_score >= 8 else "PARTIAL",
            "score": connection_security_score,
            "message": f"Connection security score: {connection_security_score}/10"
        }
        security_report["score"] += connection_security_score
        
        # Security level assessment
        if self.config.security_level == RedisSecurityLevel.PRODUCTION:
            if security_report["score"] < 80:
                security_report["recommendations"].append("Production security score too low - address critical issues")
        
        logger.info(f"🔍 Redis security assessment: {security_report['score']}/{security_report['max_score']}")
        
        return security_report
    
    def _assess_password_strength(self, password: str) -> Dict[str, Any]:
        """Assess Redis password strength"""
        score = 0
        issues = []
        
        # Length check
        if len(password) >= 16:
            score += 10
        elif len(password) >= 12:
            score += 7
        elif len(password) >= 8:
            score += 5
        else:
            issues.append("Password too short (minimum 12 characters recommended)")
        
        # Character variety
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        variety_score = sum([has_lower, has_upper, has_digit, has_special]) * 5
        score += variety_score
        
        if not has_lower:
            issues.append("Missing lowercase letters")
        if not has_upper:
            issues.append("Missing uppercase letters")
        if not has_digit:
            issues.append("Missing digits")
        if not has_special:
            issues.append("Missing special characters")
        
        # Common patterns check
        common_patterns = ["123", "abc", "password", "admin", "redis"]
        for pattern in common_patterns:
            if pattern.lower() in password.lower():
                score -= 5
                issues.append(f"Contains common pattern: {pattern}")
        
        # Final assessment
        if score >= 25:
            status = "STRONG"
        elif score >= 15:
            status = "MODERATE"
        else:
            status = "WEAK"
        
        return {
            "status": status,
            "score": min(score, 30),  # Cap at 30 for password component
            "issues": issues,
            "message": f"Password strength: {status} ({score}/30)"
        }
    
    def rotate_password(self, new_password: Optional[str] = None) -> str:
        """Rotate Redis password"""
        try:
            if not new_password:
                new_password = self.generate_secure_password()
            
            # Update configuration
            old_password = self.config.password
            self.config.password = new_password
            
            # Test new connection
            test_config = RedisAuthConfig(
                host=self.config.host,
                port=self.config.port,
                password=new_password,
                username=self.config.username,
                ssl_enabled=self.config.ssl_enabled
            )
            
            test_manager = RedisAuthManager(test_config)
            test_pool = test_manager.create_connection_pool()
            test_client = redis.Redis(connection_pool=test_pool)
            test_client.ping()
            
            # Update connection pool
            self.connection_pool = test_pool
            
            logger.info("✅ Redis password rotated successfully")
            return new_password
            
        except Exception as e:
            # Rollback on failure
            self.config.password = old_password
            logger.error(f"❌ Password rotation failed: {str(e)}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection information"""
        return {
            "host": self.config.host,
            "port": self.config.port,
            "ssl_enabled": self.config.ssl_enabled,
            "authenticated": self.authenticated,
            "security_level": self.config.security_level.value,
            "max_connections": self.config.max_connections,
            "failed_auth_attempts": self.failed_auth_attempts,
            "auth_locked_out": self._is_auth_locked_out()
        }

def create_production_redis_config() -> RedisAuthConfig:
    """Create production Redis configuration from environment"""
    return RedisAuthConfig(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        username=os.getenv("REDIS_USERNAME", "default"),
        ssl_enabled=os.getenv("REDIS_SSL_ENABLED", "false").lower() == "true",
        ssl_cert_path=os.getenv("REDIS_SSL_CERT_PATH"),
        ssl_key_path=os.getenv("REDIS_SSL_KEY_PATH"),
        ssl_ca_path=os.getenv("REDIS_SSL_CA_PATH"),
        security_level=RedisSecurityLevel(os.getenv("REDIS_SECURITY_LEVEL", "development"))
    )

# Global Redis auth manager
_redis_auth_manager = None

def get_redis_auth_manager() -> RedisAuthManager:
    """Get global Redis authentication manager"""
    global _redis_auth_manager
    if _redis_auth_manager is None:
        config = create_production_redis_config()
        _redis_auth_manager = RedisAuthManager(config)
    return _redis_auth_manager
