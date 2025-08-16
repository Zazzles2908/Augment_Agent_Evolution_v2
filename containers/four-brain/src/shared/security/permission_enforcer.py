"""
Permission Enforcement System for Four-Brain System v2
Fine-grained permission enforcement with policy-based access control

Created: 2025-07-30 AEST
Purpose: Enforce permissions and access policies across all system components
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    """Permission levels for hierarchical access control"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    SUPER_ADMIN = 5

class ResourceType(Enum):
    """System resource types"""
    BRAIN = "brain"
    API = "api"
    DATABASE = "database"
    FILE = "file"
    SYSTEM = "system"
    MONITORING = "monitoring"
    SECURITY = "security"

class ActionType(Enum):
    """Action types for permission checking"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"

@dataclass
class Permission:
    """Individual permission definition"""
    resource_type: ResourceType
    resource_id: str
    action: ActionType
    level: PermissionLevel
    conditions: Dict[str, Any]
    expires_at: Optional[datetime] = None
    
    def matches(self, resource_type: ResourceType, resource_id: str, action: ActionType) -> bool:
        """Check if this permission matches the requested access"""
        return (
            self.resource_type == resource_type and
            (self.resource_id == "*" or self.resource_id == resource_id) and
            self.action == action and
            (not self.expires_at or datetime.now() < self.expires_at)
        )

@dataclass
class AccessPolicy:
    """Access policy definition"""
    policy_id: str
    name: str
    description: str
    permissions: List[Permission]
    conditions: Dict[str, Any]
    priority: int
    active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class AccessRequest:
    """Access request for permission checking"""
    user_id: str
    username: str
    role: str
    resource_type: ResourceType
    resource_id: str
    action: ActionType
    context: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class AccessResult:
    """Result of permission check"""
    granted: bool
    reason: str
    matched_permissions: List[Permission]
    policy_violations: List[str]
    security_warnings: List[str]
    timestamp: datetime

class PermissionEnforcer:
    """
    Comprehensive permission enforcement system
    
    Features:
    - Fine-grained permission checking
    - Policy-based access control
    - Hierarchical permission levels
    - Conditional access rules
    - Time-based permissions
    - Audit logging
    - Performance optimization
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/6"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Permission cache for performance
        self.permission_cache: Dict[str, List[Permission]] = {}
        self.policy_cache: Dict[str, AccessPolicy] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Default role permissions
        self.default_role_permissions = {
            'super_admin': [
                Permission(ResourceType.SYSTEM, "*", ActionType.ADMIN, PermissionLevel.SUPER_ADMIN, {}),
                Permission(ResourceType.BRAIN, "*", ActionType.ADMIN, PermissionLevel.SUPER_ADMIN, {}),
                Permission(ResourceType.DATABASE, "*", ActionType.ADMIN, PermissionLevel.SUPER_ADMIN, {}),
                Permission(ResourceType.SECURITY, "*", ActionType.ADMIN, PermissionLevel.SUPER_ADMIN, {}),
            ],
            'admin': [
                Permission(ResourceType.BRAIN, "*", ActionType.ADMIN, PermissionLevel.ADMIN, {}),
                Permission(ResourceType.API, "*", ActionType.ADMIN, PermissionLevel.ADMIN, {}),
                Permission(ResourceType.MONITORING, "*", ActionType.READ, PermissionLevel.READ, {}),
            ],
            'brain_service': [
                Permission(ResourceType.BRAIN, "*", ActionType.EXECUTE, PermissionLevel.EXECUTE, {}),
                Permission(ResourceType.DATABASE, "brain_data", ActionType.WRITE, PermissionLevel.WRITE, {}),
                Permission(ResourceType.API, "internal", ActionType.EXECUTE, PermissionLevel.EXECUTE, {}),
            ],
            'user': [
                Permission(ResourceType.API, "public", ActionType.READ, PermissionLevel.READ, {}),
                Permission(ResourceType.BRAIN, "chat", ActionType.EXECUTE, PermissionLevel.EXECUTE, {}),
            ],
            'guest': [
                Permission(ResourceType.API, "public", ActionType.READ, PermissionLevel.READ, {}),
            ]
        }
        
        # Security policies
        self.security_policies = {
            'rate_limiting': {
                'enabled': True,
                'max_requests_per_minute': 100,
                'max_requests_per_hour': 1000
            },
            'ip_restrictions': {
                'enabled': False,
                'allowed_ips': [],
                'blocked_ips': []
            },
            'time_restrictions': {
                'enabled': False,
                'allowed_hours': list(range(24))
            }
        }
        
        # Access statistics
        self.stats = {
            'total_requests': 0,
            'granted_requests': 0,
            'denied_requests': 0,
            'policy_violations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("🛡️ Permission Enforcer initialized")
    
    async def initialize(self):
        """Initialize Redis connection and load policies"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load policies from Redis
            await self._load_policies_from_redis()
            
            logger.info("✅ Permission Enforcer Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Permission Enforcer: {e}")
            raise
    
    async def check_permission(self, access_request: AccessRequest) -> AccessResult:
        """Check if access should be granted based on permissions and policies"""
        try:
            self.stats['total_requests'] += 1
            
            # Get user permissions
            user_permissions = await self._get_user_permissions(access_request.user_id, access_request.role)
            
            # Check basic permission match
            matched_permissions = []
            for permission in user_permissions:
                if permission.matches(access_request.resource_type, access_request.resource_id, access_request.action):
                    matched_permissions.append(permission)
            
            # Check security policies
            policy_violations = await self._check_security_policies(access_request)
            security_warnings = await self._check_security_warnings(access_request)
            
            # Determine access result
            granted = len(matched_permissions) > 0 and len(policy_violations) == 0
            
            if granted:
                # Additional condition checks
                granted = await self._check_permission_conditions(matched_permissions, access_request)
            
            # Create result
            reason = self._generate_access_reason(granted, matched_permissions, policy_violations)
            
            result = AccessResult(
                granted=granted,
                reason=reason,
                matched_permissions=matched_permissions,
                policy_violations=policy_violations,
                security_warnings=security_warnings,
                timestamp=datetime.now()
            )
            
            # Update statistics
            if granted:
                self.stats['granted_requests'] += 1
            else:
                self.stats['denied_requests'] += 1
                if policy_violations:
                    self.stats['policy_violations'] += 1
            
            # Log access attempt
            await self._log_access_attempt(access_request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Permission check failed: {e}")
            return AccessResult(
                granted=False,
                reason=f"Permission check error: {str(e)}",
                matched_permissions=[],
                policy_violations=["system_error"],
                security_warnings=[],
                timestamp=datetime.now()
            )
    
    async def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant a specific permission to a user"""
        try:
            # Get current permissions
            current_permissions = await self._get_user_permissions(user_id)
            
            # Add new permission
            current_permissions.append(permission)
            
            # Store updated permissions
            await self._store_user_permissions(user_id, current_permissions)
            
            # Clear cache
            self.permission_cache.pop(user_id, None)
            
            logger.info(f"✅ Permission granted to user {user_id}: {permission.resource_type.value}:{permission.resource_id}:{permission.action.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to grant permission: {e}")
            return False
    
    async def revoke_permission(self, user_id: str, resource_type: ResourceType, 
                              resource_id: str, action: ActionType) -> bool:
        """Revoke a specific permission from a user"""
        try:
            # Get current permissions
            current_permissions = await self._get_user_permissions(user_id)
            
            # Remove matching permissions
            updated_permissions = [
                p for p in current_permissions 
                if not (p.resource_type == resource_type and 
                       p.resource_id == resource_id and 
                       p.action == action)
            ]
            
            # Store updated permissions
            await self._store_user_permissions(user_id, updated_permissions)
            
            # Clear cache
            self.permission_cache.pop(user_id, None)
            
            logger.info(f"✅ Permission revoked from user {user_id}: {resource_type.value}:{resource_id}:{action.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to revoke permission: {e}")
            return False
    
    async def _get_user_permissions(self, user_id: str, role: str = None) -> List[Permission]:
        """Get all permissions for a user"""
        # Check cache first
        cache_key = f"{user_id}:{role or 'unknown'}"
        if cache_key in self.permission_cache:
            self.stats['cache_hits'] += 1
            return self.permission_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Get role-based permissions
        role_permissions = self.default_role_permissions.get(role, []).copy()
        
        # Get user-specific permissions from Redis
        user_permissions = await self._load_user_permissions_from_redis(user_id)
        
        # Combine permissions
        all_permissions = role_permissions + user_permissions
        
        # Cache permissions
        self.permission_cache[cache_key] = all_permissions
        
        return all_permissions
    
    async def _check_security_policies(self, access_request: AccessRequest) -> List[str]:
        """Check security policies for violations"""
        violations = []
        
        # Rate limiting check
        if self.security_policies['rate_limiting']['enabled']:
            if await self._check_rate_limit(access_request.user_id):
                violations.append("rate_limit_exceeded")
        
        # IP restrictions check
        if self.security_policies['ip_restrictions']['enabled'] and access_request.ip_address:
            if access_request.ip_address in self.security_policies['ip_restrictions']['blocked_ips']:
                violations.append("ip_blocked")
            elif (self.security_policies['ip_restrictions']['allowed_ips'] and 
                  access_request.ip_address not in self.security_policies['ip_restrictions']['allowed_ips']):
                violations.append("ip_not_allowed")
        
        # Time restrictions check
        if self.security_policies['time_restrictions']['enabled']:
            current_hour = datetime.now().hour
            if current_hour not in self.security_policies['time_restrictions']['allowed_hours']:
                violations.append("time_restricted")
        
        return violations
    
    async def _check_security_warnings(self, access_request: AccessRequest) -> List[str]:
        """Check for security warnings (non-blocking)"""
        warnings = []
        
        # Check for unusual access patterns
        if access_request.action == ActionType.ADMIN:
            warnings.append("admin_access_requested")
        
        # Check for sensitive resource access
        if access_request.resource_type == ResourceType.SECURITY:
            warnings.append("security_resource_access")
        
        return warnings
    
    async def _check_permission_conditions(self, permissions: List[Permission], 
                                         access_request: AccessRequest) -> bool:
        """Check additional conditions on matched permissions"""
        for permission in permissions:
            if permission.conditions:
                # Time-based conditions
                if 'time_range' in permission.conditions:
                    time_range = permission.conditions['time_range']
                    current_hour = datetime.now().hour
                    if not (time_range['start'] <= current_hour <= time_range['end']):
                        return False
                
                # IP-based conditions
                if 'allowed_ips' in permission.conditions and access_request.ip_address:
                    if access_request.ip_address not in permission.conditions['allowed_ips']:
                        return False
                
                # Context-based conditions
                if 'required_context' in permission.conditions:
                    required_keys = permission.conditions['required_context']
                    if not all(key in access_request.context for key in required_keys):
                        return False
        
        return True
    
    def _generate_access_reason(self, granted: bool, matched_permissions: List[Permission], 
                               policy_violations: List[str]) -> str:
        """Generate human-readable reason for access decision"""
        if granted:
            return f"Access granted based on {len(matched_permissions)} matching permission(s)"
        else:
            if policy_violations:
                return f"Access denied due to policy violations: {', '.join(policy_violations)}"
            elif not matched_permissions:
                return "Access denied: no matching permissions found"
            else:
                return "Access denied: permission conditions not met"
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        if not self.redis_client:
            return False
        
        try:
            current_time = int(time.time())
            minute_key = f"rate_limit:minute:{user_id}:{current_time // 60}"
            hour_key = f"rate_limit:hour:{user_id}:{current_time // 3600}"
            
            # Check minute limit
            minute_count = await self.redis_client.incr(minute_key)
            await self.redis_client.expire(minute_key, 60)
            
            if minute_count > self.security_policies['rate_limiting']['max_requests_per_minute']:
                return True
            
            # Check hour limit
            hour_count = await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)
            
            if hour_count > self.security_policies['rate_limiting']['max_requests_per_hour']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    async def _log_access_attempt(self, access_request: AccessRequest, result: AccessResult):
        """Log access attempt for audit purposes"""
        if self.redis_client:
            try:
                log_entry = {
                    'user_id': access_request.user_id,
                    'username': access_request.username,
                    'role': access_request.role,
                    'resource_type': access_request.resource_type.value,
                    'resource_id': access_request.resource_id,
                    'action': access_request.action.value,
                    'granted': result.granted,
                    'reason': result.reason,
                    'ip_address': access_request.ip_address,
                    'session_id': access_request.session_id,
                    'timestamp': access_request.timestamp.isoformat(),
                    'policy_violations': result.policy_violations,
                    'security_warnings': result.security_warnings
                }
                
                # Store in Redis with TTL
                key = f"access_log:{access_request.user_id}:{int(time.time())}"
                await self.redis_client.setex(key, 86400 * 7, json.dumps(log_entry))  # 7 days retention
                
            except Exception as e:
                logger.error(f"Failed to log access attempt: {e}")
    
    async def _load_user_permissions_from_redis(self, user_id: str) -> List[Permission]:
        """Load user-specific permissions from Redis"""
        if not self.redis_client:
            return []
        
        try:
            key = f"user_permissions:{user_id}"
            data = await self.redis_client.get(key)
            if data:
                permissions_data = json.loads(data)
                return [Permission(**p) for p in permissions_data]
        except Exception as e:
            logger.error(f"Failed to load user permissions: {e}")
        
        return []
    
    async def _store_user_permissions(self, user_id: str, permissions: List[Permission]):
        """Store user permissions in Redis"""
        if self.redis_client:
            try:
                key = f"user_permissions:{user_id}"
                permissions_data = [asdict(p) for p in permissions]
                await self.redis_client.set(key, json.dumps(permissions_data, default=str))
            except Exception as e:
                logger.error(f"Failed to store user permissions: {e}")
    
    async def _load_policies_from_redis(self):
        """Load access policies from Redis"""
        # Implementation for loading custom policies
        # For now, using default policies
        pass
    
    async def get_enforcement_statistics(self) -> Dict[str, Any]:
        """Get permission enforcement statistics"""
        return {
            'statistics': self.stats.copy(),
            'cache_size': len(self.permission_cache),
            'policies': len(self.security_policies),
            'timestamp': datetime.now().isoformat()
        }

# Decorator for enforcing permissions
def require_permission(resource_type: ResourceType, resource_id: str, action: ActionType):
    """Decorator to enforce permissions on functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user context (implementation depends on framework)
            user_id = kwargs.get('user_id') or 'unknown'
            username = kwargs.get('username') or 'unknown'
            role = kwargs.get('role') or 'guest'
            
            # Create access request
            access_request = AccessRequest(
                user_id=user_id,
                username=username,
                role=role,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                context=kwargs,
                timestamp=datetime.now()
            )
            
            # Check permission
            enforcer = PermissionEnforcer()
            await enforcer.initialize()
            result = await enforcer.check_permission(access_request)
            
            if not result.granted:
                raise PermissionError(f"Access denied: {result.reason}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global permission enforcer instance
permission_enforcer = PermissionEnforcer()

async def initialize_permission_enforcer():
    """Initialize the global permission enforcer"""
    await permission_enforcer.initialize()

if __name__ == "__main__":
    # Test the permission enforcer
    async def test_permission_enforcer():
        await initialize_permission_enforcer()
        
        # Create test access request
        access_request = AccessRequest(
            user_id="test_user",
            username="testuser",
            role="user",
            resource_type=ResourceType.API,
            resource_id="public",
            action=ActionType.READ,
            context={},
            timestamp=datetime.now(),
            ip_address="127.0.0.1"
        )
        
        # Check permission
        result = await permission_enforcer.check_permission(access_request)
        print(f"Access granted: {result.granted}")
        print(f"Reason: {result.reason}")
        
        # Get statistics
        stats = await permission_enforcer.get_enforcement_statistics()
        print(f"Statistics: {stats}")
    
    asyncio.run(test_permission_enforcer())
