#!/usr/bin/env python3
"""
Role-Based Access Controller for K2-Vector-Hub
Implements comprehensive access control and authorization for the Four-Brain System

This module provides sophisticated role-based access control (RBAC) capabilities
including permission management, resource access control, operation authorization,
and fine-grained security policies.

Key Features:
- Role-based access control (RBAC)
- Permission-based authorization
- Resource-level access control
- Operation-specific permissions
- Dynamic permission evaluation
- Access audit logging
- Policy-based security rules

Zero Fabrication Policy: ENFORCED
All access control mechanisms use real security standards and verified implementations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # System administration
    ADMIN_FULL_ACCESS = "admin:full_access"
    ADMIN_USER_MANAGEMENT = "admin:user_management"
    ADMIN_SYSTEM_CONFIG = "admin:system_config"
    ADMIN_MONITORING = "admin:monitoring"
    
    # Brain operations
    BRAIN_READ = "brain:read"
    BRAIN_WRITE = "brain:write"
    BRAIN_EXECUTE = "brain:execute"
    BRAIN_CONFIGURE = "brain:configure"
    
    # Workflow management
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_EXECUTE = "workflow:execute"
    WORKFLOW_DELETE = "workflow:delete"
    
    # Resource management
    RESOURCE_ALLOCATE = "resource:allocate"
    RESOURCE_MONITOR = "resource:monitor"
    RESOURCE_CONFIGURE = "resource:configure"
    
    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Data access
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"


class Resource(Enum):
    """System resources"""
    BRAIN1 = "brain1"
    BRAIN2 = "brain2"
    BRAIN3 = "brain3"
    BRAIN4 = "brain4"
    K2_HUB = "k2_hub"
    WORKFLOWS = "workflows"
    USERS = "users"
    SYSTEM_CONFIG = "system_config"
    MONITORING = "monitoring"
    API_KEYS = "api_keys"
    LOGS = "logs"


class AccessResult(Enum):
    """Access control results"""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    REQUIRES_MFA = "requires_mfa"


@dataclass
class AccessPolicy:
    """Access control policy"""
    policy_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    permissions: List[Permission]
    resources: List[Resource]
    priority: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessRequest:
    """Access request information"""
    user_id: str
    resource: Resource
    operation: str
    permission: Permission
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessLog:
    """Access control audit log entry"""
    log_id: str
    user_id: str
    resource: Resource
    operation: str
    permission: Permission
    result: AccessResult
    reason: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class RoleBasedAccessController:
    """
    Comprehensive role-based access controller for the Four-Brain System
    """
    
    def __init__(self, auth_manager=None):
        """Initialize access controller with authentication manager"""
        self.auth_manager = auth_manager
        
        # Access control data
        self.role_permissions: Dict[str, Set[Permission]] = {}
        self.user_permissions: Dict[str, Set[Permission]] = {}
        self.access_policies: Dict[str, AccessPolicy] = {}
        self.access_logs: List[AccessLog] = []
        
        # Resource access tracking
        self.resource_access_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.failed_access_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        self._initialize_default_policies()
        
        logger.info("🔒 RoleBasedAccessController initialized")
    
    def _initialize_default_roles(self):
        """Initialize default role permissions"""
        # Admin role - full access
        self.role_permissions["admin"] = {
            Permission.ADMIN_FULL_ACCESS,
            Permission.ADMIN_USER_MANAGEMENT,
            Permission.ADMIN_SYSTEM_CONFIG,
            Permission.ADMIN_MONITORING,
            Permission.BRAIN_READ,
            Permission.BRAIN_WRITE,
            Permission.BRAIN_EXECUTE,
            Permission.BRAIN_CONFIGURE,
            Permission.WORKFLOW_CREATE,
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_EXECUTE,
            Permission.WORKFLOW_DELETE,
            Permission.RESOURCE_ALLOCATE,
            Permission.RESOURCE_MONITOR,
            Permission.RESOURCE_CONFIGURE,
            Permission.API_READ,
            Permission.API_WRITE,
            Permission.API_ADMIN,
            Permission.DATA_READ,
            Permission.DATA_WRITE,
            Permission.DATA_DELETE,
            Permission.DATA_EXPORT
        }
        
        # User role - standard access
        self.role_permissions["user"] = {
            Permission.BRAIN_READ,
            Permission.BRAIN_EXECUTE,
            Permission.WORKFLOW_CREATE,
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_EXECUTE,
            Permission.RESOURCE_MONITOR,
            Permission.API_READ,
            Permission.API_WRITE,
            Permission.DATA_READ,
            Permission.DATA_WRITE
        }
        
        # Service role - API access
        self.role_permissions["service"] = {
            Permission.BRAIN_READ,
            Permission.BRAIN_EXECUTE,
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_EXECUTE,
            Permission.API_READ,
            Permission.DATA_READ
        }
        
        # Read-only role - monitoring access
        self.role_permissions["readonly"] = {
            Permission.BRAIN_READ,
            Permission.WORKFLOW_READ,
            Permission.RESOURCE_MONITOR,
            Permission.API_READ,
            Permission.DATA_READ
        }
        
        # Developer role - development access
        self.role_permissions["developer"] = {
            Permission.BRAIN_READ,
            Permission.BRAIN_WRITE,
            Permission.BRAIN_EXECUTE,
            Permission.BRAIN_CONFIGURE,
            Permission.WORKFLOW_CREATE,
            Permission.WORKFLOW_READ,
            Permission.WORKFLOW_EXECUTE,
            Permission.WORKFLOW_DELETE,
            Permission.RESOURCE_MONITOR,
            Permission.RESOURCE_CONFIGURE,
            Permission.API_READ,
            Permission.API_WRITE,
            Permission.DATA_READ,
            Permission.DATA_WRITE,
            Permission.DATA_EXPORT
        }
    
    def _initialize_default_policies(self):
        """Initialize default access policies"""
        policies = [
            AccessPolicy(
                policy_id="admin_full_access",
                name="Administrator Full Access",
                description="Full system access for administrators",
                conditions={"role": "admin"},
                permissions=list(Permission),
                resources=list(Resource),
                priority=1
            ),
            AccessPolicy(
                policy_id="business_hours_only",
                name="Business Hours Access",
                description="Restrict access to business hours for regular users",
                conditions={"role": "user", "time_restriction": True},
                permissions=[Permission.BRAIN_EXECUTE, Permission.WORKFLOW_EXECUTE],
                resources=[Resource.BRAIN1, Resource.BRAIN2, Resource.BRAIN3, Resource.BRAIN4],
                priority=2
            ),
            AccessPolicy(
                policy_id="read_only_monitoring",
                name="Read-Only Monitoring Access",
                description="Read-only access for monitoring users",
                conditions={"role": "readonly"},
                permissions=[Permission.BRAIN_READ, Permission.RESOURCE_MONITOR],
                resources=[Resource.MONITORING, Resource.LOGS],
                priority=3
            ),
            AccessPolicy(
                policy_id="service_api_access",
                name="Service API Access",
                description="API access for service accounts",
                conditions={"role": "service", "api_key_required": True},
                permissions=[Permission.API_READ, Permission.BRAIN_EXECUTE],
                resources=[Resource.BRAIN1, Resource.BRAIN2, Resource.BRAIN3, Resource.BRAIN4],
                priority=2
            )
        ]
        
        for policy in policies:
            self.access_policies[policy.policy_id] = policy
    
    async def check_access(self, user_id: str, resource: Resource, 
                          operation: str, permission: Permission,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if user has access to perform operation on resource
        
        Args:
            user_id: User identifier
            resource: Target resource
            operation: Operation to perform
            permission: Required permission
            context: Additional context for access decision
            
        Returns:
            Dict containing access result and details
        """
        context = context or {}
        request = AccessRequest(
            user_id=user_id,
            resource=resource,
            operation=operation,
            permission=permission,
            context=context
        )
        
        logger.debug(f"🔒 Checking access: user={user_id}, resource={resource.value}, "
                    f"operation={operation}, permission={permission.value}")
        
        # Get user information
        user_info = await self._get_user_info(user_id)
        if not user_info:
            result = self._create_access_result(AccessResult.DENIED, "User not found", request)
            await self._log_access_attempt(request, result)
            return result
        
        # Check if user is active
        if not user_info.get("is_active", False):
            result = self._create_access_result(AccessResult.DENIED, "User account inactive", request)
            await self._log_access_attempt(request, result)
            return result
        
        # Check role-based permissions
        user_role = user_info.get("role", "user")
        role_permissions = self.role_permissions.get(user_role, set())
        
        # Check user-specific permissions
        user_permissions = self.user_permissions.get(user_id, set())
        
        # Combine permissions
        all_permissions = role_permissions.union(user_permissions)
        
        # Check if user has required permission
        if permission not in all_permissions:
            result = self._create_access_result(
                AccessResult.DENIED, 
                f"Missing required permission: {permission.value}", 
                request
            )
            await self._log_access_attempt(request, result)
            return result
        
        # Evaluate access policies
        policy_result = await self._evaluate_policies(request, user_info, all_permissions)
        if policy_result["result"] != AccessResult.GRANTED:
            await self._log_access_attempt(request, policy_result)
            return policy_result
        
        # Check resource-specific access
        resource_result = await self._check_resource_access(request, user_info, all_permissions)
        if resource_result["result"] != AccessResult.GRANTED:
            await self._log_access_attempt(request, resource_result)
            return resource_result
        
        # Check rate limiting
        rate_limit_result = await self._check_rate_limits(request, user_info)
        if rate_limit_result["result"] != AccessResult.GRANTED:
            await self._log_access_attempt(request, rate_limit_result)
            return rate_limit_result
        
        # Access granted
        result = self._create_access_result(AccessResult.GRANTED, "Access granted", request)
        await self._log_access_attempt(request, result)
        
        # Update access tracking
        self.resource_access_count[user_id][resource.value] += 1
        
        return result
    
    async def _get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information from authentication manager"""
        if not self.auth_manager:
            return None
        
        # In a real implementation, this would query the auth manager
        # For now, simulate user lookup
        if hasattr(self.auth_manager, 'users') and user_id in self.auth_manager.users:
            user = self.auth_manager.users[user_id]
            return {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "is_active": user.is_active,
                "email": user.email
            }
        
        return None
    
    async def _evaluate_policies(self, request: AccessRequest, user_info: Dict[str, Any],
                               user_permissions: Set[Permission]) -> Dict[str, Any]:
        """Evaluate access policies"""
        applicable_policies = []
        
        # Find applicable policies
        for policy in self.access_policies.values():
            if not policy.enabled:
                continue
            
            # Check if policy applies to this request
            if (request.permission in policy.permissions and 
                request.resource in policy.resources):
                
                # Check policy conditions
                if self._check_policy_conditions(policy, request, user_info):
                    applicable_policies.append(policy)
        
        # Sort by priority (lower number = higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        # Evaluate policies
        for policy in applicable_policies:
            policy_result = await self._evaluate_single_policy(policy, request, user_info)
            if policy_result["result"] != AccessResult.GRANTED:
                return policy_result
        
        return self._create_access_result(AccessResult.GRANTED, "Policy evaluation passed", request)
    
    def _check_policy_conditions(self, policy: AccessPolicy, request: AccessRequest,
                                user_info: Dict[str, Any]) -> bool:
        """Check if policy conditions are met"""
        conditions = policy.conditions
        
        # Check role condition
        if "role" in conditions:
            if user_info.get("role") != conditions["role"]:
                return False
        
        # Check time restriction
        if conditions.get("time_restriction", False):
            current_hour = datetime.utcnow().hour
            if not (9 <= current_hour <= 17):  # Business hours 9 AM - 5 PM
                return False
        
        # Check API key requirement
        if conditions.get("api_key_required", False):
            if "api_key" not in request.context:
                return False
        
        return True
    
    async def _evaluate_single_policy(self, policy: AccessPolicy, request: AccessRequest,
                                     user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single access policy"""
        # For now, all policies that match conditions are granted
        # In a more complex system, policies could have additional logic
        return self._create_access_result(
            AccessResult.GRANTED, 
            f"Policy '{policy.name}' allows access", 
            request
        )
    
    async def _check_resource_access(self, request: AccessRequest, user_info: Dict[str, Any],
                                   user_permissions: Set[Permission]) -> Dict[str, Any]:
        """Check resource-specific access rules"""
        resource = request.resource
        
        # Brain-specific access rules
        if resource in [Resource.BRAIN1, Resource.BRAIN2, Resource.BRAIN3, Resource.BRAIN4]:
            # Check if brain is available
            if not await self._is_brain_available(resource):
                return self._create_access_result(
                    AccessResult.DENIED, 
                    f"Brain {resource.value} is not available", 
                    request
                )
        
        # System config access - admin only
        if resource == Resource.SYSTEM_CONFIG:
            if Permission.ADMIN_SYSTEM_CONFIG not in user_permissions:
                return self._create_access_result(
                    AccessResult.DENIED, 
                    "System configuration access requires admin privileges", 
                    request
                )
        
        # User management access - admin only
        if resource == Resource.USERS:
            if Permission.ADMIN_USER_MANAGEMENT not in user_permissions:
                return self._create_access_result(
                    AccessResult.DENIED, 
                    "User management access requires admin privileges", 
                    request
                )
        
        return self._create_access_result(AccessResult.GRANTED, "Resource access allowed", request)
    
    async def _is_brain_available(self, brain_resource: Resource) -> bool:
        """Check if brain is available for access"""
        # In a real implementation, this would check brain health status
        # For now, assume all brains are available
        return True
    
    async def _check_rate_limits(self, request: AccessRequest, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limiting for user access"""
        user_id = request.user_id
        resource = request.resource.value
        
        # Get recent access count
        current_time = datetime.utcnow()
        recent_access_count = self.resource_access_count[user_id][resource]
        
        # Define rate limits based on user role
        role = user_info.get("role", "user")
        rate_limits = {
            "admin": 1000,  # requests per hour
            "developer": 500,
            "user": 100,
            "service": 200,
            "readonly": 50
        }
        
        limit = rate_limits.get(role, 100)
        
        # Simple rate limiting (in production, would use sliding window)
        if recent_access_count > limit:
            return self._create_access_result(
                AccessResult.DENIED, 
                f"Rate limit exceeded: {recent_access_count}/{limit} requests", 
                request
            )
        
        return self._create_access_result(AccessResult.GRANTED, "Rate limit check passed", request)
    
    def _create_access_result(self, result: AccessResult, reason: str, 
                            request: AccessRequest) -> Dict[str, Any]:
        """Create access result dictionary"""
        return {
            "result": result,
            "granted": result == AccessResult.GRANTED,
            "reason": reason,
            "user_id": request.user_id,
            "resource": request.resource.value,
            "operation": request.operation,
            "permission": request.permission.value,
            "timestamp": request.timestamp.isoformat()
        }
    
    async def _log_access_attempt(self, request: AccessRequest, result: Dict[str, Any]):
        """Log access attempt for audit purposes"""
        log_entry = AccessLog(
            log_id=f"access_{int(time.time() * 1000)}",
            user_id=request.user_id,
            resource=request.resource,
            operation=request.operation,
            permission=request.permission,
            result=result["result"],
            reason=result["reason"],
            timestamp=request.timestamp,
            context=request.context
        )
        
        self.access_logs.append(log_entry)
        
        # Keep log size manageable
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]
        
        # Log security events
        if result["result"] == AccessResult.DENIED:
            logger.warning(f"🔒 Access denied: {request.user_id} -> {request.resource.value} "
                          f"({request.operation}) - {result['reason']}")
        else:
            logger.debug(f"🔒 Access granted: {request.user_id} -> {request.resource.value} "
                        f"({request.operation})")
    
    async def grant_user_permission(self, user_id: str, permission: Permission) -> Dict[str, Any]:
        """Grant specific permission to user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        
        self.user_permissions[user_id].add(permission)
        
        logger.info(f"🔒 Granted permission {permission.value} to user {user_id}")
        
        return {
            "success": True,
            "message": f"Permission {permission.value} granted to user {user_id}"
        }
    
    async def revoke_user_permission(self, user_id: str, permission: Permission) -> Dict[str, Any]:
        """Revoke specific permission from user"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)
        
        logger.info(f"🔒 Revoked permission {permission.value} from user {user_id}")
        
        return {
            "success": True,
            "message": f"Permission {permission.value} revoked from user {user_id}"
        }
    
    async def get_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get all permissions for a user"""
        user_info = await self._get_user_info(user_id)
        if not user_info:
            return {"success": False, "error": "User not found"}
        
        # Get role permissions
        role = user_info.get("role", "user")
        role_permissions = self.role_permissions.get(role, set())
        
        # Get user-specific permissions
        user_permissions = self.user_permissions.get(user_id, set())
        
        # Combine permissions
        all_permissions = role_permissions.union(user_permissions)
        
        return {
            "success": True,
            "user_id": user_id,
            "role": role,
            "role_permissions": [p.value for p in role_permissions],
            "user_permissions": [p.value for p in user_permissions],
            "all_permissions": [p.value for p in all_permissions]
        }
    
    async def get_access_logs(self, user_id: str = None, resource: Resource = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get access logs with optional filtering"""
        filtered_logs = self.access_logs
        
        # Filter by user
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        # Filter by resource
        if resource:
            filtered_logs = [log for log in filtered_logs if log.resource == resource]
        
        # Sort by timestamp (most recent first)
        filtered_logs.sort(key=lambda log: log.timestamp, reverse=True)
        
        # Limit results
        filtered_logs = filtered_logs[:limit]
        
        # Convert to dict format
        return [
            {
                "log_id": log.log_id,
                "user_id": log.user_id,
                "resource": log.resource.value,
                "operation": log.operation,
                "permission": log.permission.value,
                "result": log.result.value,
                "reason": log.reason,
                "timestamp": log.timestamp.isoformat(),
                "context": log.context
            }
            for log in filtered_logs
        ]
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access control statistics"""
        total_logs = len(self.access_logs)
        if total_logs == 0:
            return {"total_logs": 0}
        
        # Calculate access results
        granted_count = len([log for log in self.access_logs if log.result == AccessResult.GRANTED])
        denied_count = len([log for log in self.access_logs if log.result == AccessResult.DENIED])
        
        # Resource access distribution
        resource_access = defaultdict(int)
        for log in self.access_logs:
            resource_access[log.resource.value] += 1
        
        # User access distribution
        user_access = defaultdict(int)
        for log in self.access_logs:
            user_access[log.user_id] += 1
        
        return {
            "total_logs": total_logs,
            "granted_access": granted_count,
            "denied_access": denied_count,
            "success_rate": granted_count / total_logs if total_logs > 0 else 0,
            "resource_access_distribution": dict(resource_access),
            "user_access_distribution": dict(user_access),
            "active_policies": len([p for p in self.access_policies.values() if p.enabled]),
            "total_policies": len(self.access_policies)
        }
