#!/usr/bin/env python3
"""
JWT Token Validator for K2-Vector-Hub
Implements comprehensive JWT token validation and management for the Four-Brain System

This module provides robust JWT token validation capabilities including token
verification, expiration checking, signature validation, claim verification,
and token refresh mechanisms.

Key Features:
- JWT token validation and verification
- Token signature verification
- Expiration and timing validation
- Claim verification and extraction
- Token refresh and renewal
- Blacklist and revocation support
- Multi-issuer token support
- Security audit logging

Zero Fabrication Policy: ENFORCED
All token validation uses real JWT standards and verified cryptographic implementations.
"""

import asyncio
import logging
import time
import jwt
import hashlib
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import secrets

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """JWT token types"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    API_TOKEN = "api_token"
    SERVICE_TOKEN = "service_token"


class TokenStatus(Enum):
    """Token validation status"""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_FORMAT = "invalid_format"
    REVOKED = "revoked"
    NOT_YET_VALID = "not_yet_valid"
    MISSING_CLAIMS = "missing_claims"


class ValidationResult(Enum):
    """Token validation results"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


@dataclass
class TokenClaims:
    """JWT token claims"""
    user_id: str
    username: str
    role: str
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    not_before: Optional[datetime] = None
    issuer: str = "four-brain-system"
    audience: str = "four-brain-api"
    token_id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    custom_claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Token validation context"""
    token: str
    token_type: TokenType
    expected_issuer: Optional[str] = None
    expected_audience: Optional[str] = None
    required_claims: List[str] = field(default_factory=list)
    max_age_seconds: Optional[int] = None
    allow_refresh: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationLog:
    """Token validation audit log"""
    log_id: str
    token_id: str
    user_id: str
    validation_result: ValidationResult
    token_status: TokenStatus
    reason: str
    timestamp: datetime
    ip_address: str = "unknown"
    user_agent: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)


class JWTTokenValidator:
    """
    Comprehensive JWT token validator for the Four-Brain System
    """
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256"):
        """Initialize JWT token validator"""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        
        # Token management
        self.revoked_tokens: Set[str] = set()
        self.token_blacklist: Set[str] = set()
        self.validation_logs: List[ValidationLog] = []
        
        # Validation settings
        self.default_expiration_hours = 24
        self.refresh_token_expiration_days = 30
        self.max_token_age_seconds = 86400  # 24 hours
        self.clock_skew_seconds = 300  # 5 minutes
        
        # Issuer and audience settings
        self.default_issuer = "four-brain-system"
        self.default_audience = "four-brain-api"
        self.trusted_issuers = {self.default_issuer}
        
        # Token usage tracking
        self.token_usage_count: Dict[str, int] = {}
        self.failed_validation_count: Dict[str, int] = {}
        
        logger.info("🔐 JWTTokenValidator initialized")
    
    async def validate_token(self, validation_context: ValidationContext) -> Dict[str, Any]:
        """
        Validate JWT token with comprehensive checks
        
        Args:
            validation_context: Token validation context
            
        Returns:
            Dict containing validation result and token claims
        """
        token = validation_context.token
        token_type = validation_context.token_type
        
        logger.debug(f"🔐 Validating {token_type.value} token")
        
        try:
            # Basic format validation
            if not token or not isinstance(token, str):
                return self._create_validation_result(
                    ValidationResult.FAILURE,
                    TokenStatus.INVALID_FORMAT,
                    "Token is empty or invalid format",
                    validation_context
                )
            
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in self.token_blacklist:
                return self._create_validation_result(
                    ValidationResult.FAILURE,
                    TokenStatus.REVOKED,
                    "Token is blacklisted",
                    validation_context
                )
            
            # Decode and validate JWT
            decode_result = await self._decode_jwt_token(token, validation_context)
            if not decode_result["success"]:
                return decode_result
            
            payload = decode_result["payload"]
            
            # Extract token ID for tracking
            token_id = payload.get("jti", "unknown")
            
            # Check if token is revoked
            if token_id in self.revoked_tokens:
                return self._create_validation_result(
                    ValidationResult.FAILURE,
                    TokenStatus.REVOKED,
                    "Token has been revoked",
                    validation_context,
                    token_id=token_id
                )
            
            # Validate claims
            claims_result = await self._validate_claims(payload, validation_context)
            if not claims_result["success"]:
                return claims_result
            
            # Validate timing
            timing_result = await self._validate_timing(payload, validation_context)
            if not timing_result["success"]:
                return timing_result
            
            # Validate issuer and audience
            issuer_result = await self._validate_issuer_audience(payload, validation_context)
            if not issuer_result["success"]:
                return issuer_result
            
            # Update token usage tracking
            self.token_usage_count[token_id] = self.token_usage_count.get(token_id, 0) + 1
            
            # Create token claims object
            token_claims = self._extract_token_claims(payload)
            
            # Log successful validation
            await self._log_validation(
                ValidationResult.SUCCESS,
                TokenStatus.VALID,
                "Token validation successful",
                validation_context,
                token_id,
                payload.get("sub", "unknown")
            )
            
            return {
                "success": True,
                "result": ValidationResult.SUCCESS,
                "status": TokenStatus.VALID,
                "claims": token_claims,
                "token_id": token_id,
                "user_id": payload.get("sub"),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0)).isoformat(),
                "message": "Token is valid"
            }
            
        except Exception as e:
            logger.error(f"❌ Token validation error: {e}")
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.INVALID_FORMAT,
                f"Token validation error: {str(e)}",
                validation_context
            )
    
    async def _decode_jwt_token(self, token: str, context: ValidationContext) -> Dict[str, Any]:
        """Decode JWT token with signature verification"""
        try:
            # Decode token with verification
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True
                }
            )
            
            return {"success": True, "payload": payload}
            
        except jwt.ExpiredSignatureError:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.EXPIRED,
                "Token has expired",
                context
            )
        except jwt.InvalidSignatureError:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.INVALID_SIGNATURE,
                "Token signature is invalid",
                context
            )
        except jwt.InvalidTokenError as e:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.INVALID_FORMAT,
                f"Token format is invalid: {str(e)}",
                context
            )
    
    async def _validate_claims(self, payload: Dict[str, Any], 
                             context: ValidationContext) -> Dict[str, Any]:
        """Validate required JWT claims"""
        required_claims = context.required_claims or ["sub", "iat", "exp"]
        
        missing_claims = []
        for claim in required_claims:
            if claim not in payload:
                missing_claims.append(claim)
        
        if missing_claims:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.MISSING_CLAIMS,
                f"Missing required claims: {', '.join(missing_claims)}",
                context
            )
        
        # Validate specific claim values
        if "sub" in payload and not payload["sub"]:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.MISSING_CLAIMS,
                "Subject (sub) claim is empty",
                context
            )
        
        return {"success": True}
    
    async def _validate_timing(self, payload: Dict[str, Any], 
                             context: ValidationContext) -> Dict[str, Any]:
        """Validate token timing claims"""
        current_time = datetime.utcnow().timestamp()
        
        # Check expiration
        exp = payload.get("exp")
        if exp and current_time > exp + self.clock_skew_seconds:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.EXPIRED,
                "Token has expired",
                context
            )
        
        # Check not before
        nbf = payload.get("nbf")
        if nbf and current_time < nbf - self.clock_skew_seconds:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.NOT_YET_VALID,
                "Token is not yet valid",
                context
            )
        
        # Check issued at
        iat = payload.get("iat")
        if iat and current_time < iat - self.clock_skew_seconds:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.NOT_YET_VALID,
                "Token issued in the future",
                context
            )
        
        # Check maximum age
        if context.max_age_seconds and iat:
            token_age = current_time - iat
            if token_age > context.max_age_seconds:
                return self._create_validation_result(
                    ValidationResult.FAILURE,
                    TokenStatus.EXPIRED,
                    f"Token exceeds maximum age: {token_age}s > {context.max_age_seconds}s",
                    context
                )
        
        return {"success": True}
    
    async def _validate_issuer_audience(self, payload: Dict[str, Any], 
                                      context: ValidationContext) -> Dict[str, Any]:
        """Validate issuer and audience claims"""
        # Validate issuer
        iss = payload.get("iss")
        expected_issuer = context.expected_issuer or self.default_issuer
        
        if expected_issuer and iss != expected_issuer:
            if iss not in self.trusted_issuers:
                return self._create_validation_result(
                    ValidationResult.FAILURE,
                    TokenStatus.INVALID_FORMAT,
                    f"Untrusted issuer: {iss}",
                    context
                )
        
        # Validate audience
        aud = payload.get("aud")
        expected_audience = context.expected_audience or self.default_audience
        
        if expected_audience and aud != expected_audience:
            return self._create_validation_result(
                ValidationResult.FAILURE,
                TokenStatus.INVALID_FORMAT,
                f"Invalid audience: {aud}, expected: {expected_audience}",
                context
            )
        
        return {"success": True}
    
    def _extract_token_claims(self, payload: Dict[str, Any]) -> TokenClaims:
        """Extract token claims from JWT payload"""
        return TokenClaims(
            user_id=payload.get("sub", ""),
            username=payload.get("username", ""),
            role=payload.get("role", "user"),
            session_id=payload.get("session_id"),
            permissions=payload.get("permissions", []),
            issued_at=datetime.fromtimestamp(payload.get("iat", 0)),
            expires_at=datetime.fromtimestamp(payload.get("exp", 0)),
            not_before=datetime.fromtimestamp(payload["nbf"]) if payload.get("nbf") else None,
            issuer=payload.get("iss", self.default_issuer),
            audience=payload.get("aud", self.default_audience),
            token_id=payload.get("jti", "unknown"),
            custom_claims={k: v for k, v in payload.items() 
                          if k not in ["sub", "username", "role", "session_id", "permissions", 
                                     "iat", "exp", "nbf", "iss", "aud", "jti"]}
        )
    
    def _create_validation_result(self, result: ValidationResult, status: TokenStatus,
                                reason: str, context: ValidationContext,
                                token_id: str = "unknown") -> Dict[str, Any]:
        """Create validation result dictionary"""
        return {
            "success": result == ValidationResult.SUCCESS,
            "result": result,
            "status": status,
            "reason": reason,
            "token_type": context.token_type.value,
            "token_id": token_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _log_validation(self, result: ValidationResult, status: TokenStatus,
                            reason: str, context: ValidationContext,
                            token_id: str = "unknown", user_id: str = "unknown"):
        """Log token validation attempt"""
        log_entry = ValidationLog(
            log_id=f"validation_{int(time.time() * 1000)}",
            token_id=token_id,
            user_id=user_id,
            validation_result=result,
            token_status=status,
            reason=reason,
            timestamp=datetime.utcnow(),
            ip_address=context.context.get("ip_address", "unknown"),
            user_agent=context.context.get("user_agent", "unknown"),
            context=context.context
        )
        
        self.validation_logs.append(log_entry)
        
        # Keep log size manageable
        if len(self.validation_logs) > 10000:
            self.validation_logs = self.validation_logs[-5000:]
        
        # Log security events
        if result == ValidationResult.FAILURE:
            logger.warning(f"🔐 Token validation failed: {token_id} - {reason}")
            self.failed_validation_count[token_id] = self.failed_validation_count.get(token_id, 0) + 1
        else:
            logger.debug(f"🔐 Token validation successful: {token_id}")
    
    async def create_token(self, claims: TokenClaims, token_type: TokenType = TokenType.ACCESS_TOKEN) -> str:
        """Create JWT token with specified claims"""
        payload = {
            "sub": claims.user_id,
            "username": claims.username,
            "role": claims.role,
            "permissions": claims.permissions,
            "iat": int(claims.issued_at.timestamp()),
            "exp": int(claims.expires_at.timestamp()),
            "iss": claims.issuer,
            "aud": claims.audience,
            "jti": claims.token_id,
            "token_type": token_type.value
        }
        
        # Add session ID if provided
        if claims.session_id:
            payload["session_id"] = claims.session_id
        
        # Add not before if provided
        if claims.not_before:
            payload["nbf"] = int(claims.not_before.timestamp())
        
        # Add custom claims
        payload.update(claims.custom_claims)
        
        # Create JWT token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"🔐 Created {token_type.value} token for user {claims.user_id}")
        
        return token
    
    async def revoke_token(self, token_id: str) -> Dict[str, Any]:
        """Revoke a specific token"""
        self.revoked_tokens.add(token_id)
        
        logger.info(f"🔐 Token revoked: {token_id}")
        
        return {
            "success": True,
            "message": f"Token {token_id} has been revoked"
        }
    
    async def blacklist_token(self, token: str) -> Dict[str, Any]:
        """Add token to blacklist"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self.token_blacklist.add(token_hash)
        
        logger.info(f"🔐 Token blacklisted: {token_hash[:16]}...")
        
        return {
            "success": True,
            "message": "Token has been blacklisted"
        }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        # Validate refresh token
        validation_context = ValidationContext(
            token=refresh_token,
            token_type=TokenType.REFRESH_TOKEN,
            required_claims=["sub", "username", "role"]
        )
        
        validation_result = await self.validate_token(validation_context)
        if not validation_result["success"]:
            return {
                "success": False,
                "error": "Invalid refresh token",
                "details": validation_result["reason"]
            }
        
        # Extract claims from refresh token
        old_claims = validation_result["claims"]
        
        # Create new access token
        new_claims = TokenClaims(
            user_id=old_claims.user_id,
            username=old_claims.username,
            role=old_claims.role,
            permissions=old_claims.permissions,
            expires_at=datetime.utcnow() + timedelta(hours=self.default_expiration_hours)
        )
        
        new_access_token = await self.create_token(new_claims, TokenType.ACCESS_TOKEN)
        
        logger.info(f"🔐 Token refreshed for user {old_claims.user_id}")
        
        return {
            "success": True,
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": self.default_expiration_hours * 3600,
            "expires_at": new_claims.expires_at.isoformat()
        }
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens from tracking"""
        current_time = datetime.utcnow().timestamp()
        
        # Clean up validation logs older than 30 days
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.validation_logs = [
            log for log in self.validation_logs
            if log.timestamp > cutoff_time
        ]
        
        logger.info(f"🔐 Cleaned up expired token data")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get token validation statistics"""
        total_validations = len(self.validation_logs)
        if total_validations == 0:
            return {"total_validations": 0}
        
        # Calculate success rate
        successful_validations = len([
            log for log in self.validation_logs
            if log.validation_result == ValidationResult.SUCCESS
        ])
        
        # Token status distribution
        status_distribution = {}
        for log in self.validation_logs:
            status = log.token_status.value
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        # Most active tokens
        token_usage_sorted = sorted(
            self.token_usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations,
            "status_distribution": status_distribution,
            "revoked_tokens": len(self.revoked_tokens),
            "blacklisted_tokens": len(self.token_blacklist),
            "most_active_tokens": token_usage_sorted,
            "failed_validation_attempts": len(self.failed_validation_count)
        }
