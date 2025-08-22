#!/usr/bin/env python3
"""
Four-Brain System Authentication and Authorization Manager
Production-grade JWT-based authentication with RBAC
Version: Production v1.0
"""

import os
import sys
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps

# Add security to path
sys.path.append('/app/security')
from secrets_manager import ProductionSecretsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"
    BRAIN_SERVICE = "brain_service"

class Permission(Enum):
    """System permissions"""
    READ_METRICS = "read_metrics"
    WRITE_METRICS = "write_metrics"
    READ_LOGS = "read_logs"
    MANAGE_USERS = "manage_users"
    MANAGE_SECRETS = "manage_secrets"
    EXECUTE_BRAIN_OPERATIONS = "execute_brain_operations"
    MANAGE_SYSTEM = "manage_system"
    VIEW_DASHBOARD = "view_dashboard"
    API_ACCESS = "api_access"

@dataclass
class User:
    """User data model"""
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    password_hash: Optional[str] = None

@dataclass
class AuthToken:
    """Authentication token data"""
    user_id: str
    username: str
    role: str
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.secrets_manager = ProductionSecretsManager()
        self.jwt_config = self.secrets_manager.get_jwt_config()
        
        # Role-based permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [
                Permission.READ_METRICS,
                Permission.WRITE_METRICS,
                Permission.READ_LOGS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SECRETS,
                Permission.EXECUTE_BRAIN_OPERATIONS,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_DASHBOARD,
                Permission.API_ACCESS
            ],
            UserRole.OPERATOR: [
                Permission.READ_METRICS,
                Permission.READ_LOGS,
                Permission.EXECUTE_BRAIN_OPERATIONS,
                Permission.VIEW_DASHBOARD,
                Permission.API_ACCESS
            ],
            UserRole.VIEWER: [
                Permission.READ_METRICS,
                Permission.READ_LOGS,
                Permission.VIEW_DASHBOARD
            ],
            UserRole.API_USER: [
                Permission.API_ACCESS,
                Permission.EXECUTE_BRAIN_OPERATIONS
            ],
            UserRole.BRAIN_SERVICE: [
                Permission.READ_METRICS,
                Permission.WRITE_METRICS,
                Permission.EXECUTE_BRAIN_OPERATIONS,
                Permission.API_ACCESS
            ]
        }
        
        # Initialize default users
        self._initialize_default_users()
        
        logger.info("Authentication manager initialized")
    
    def _initialize_default_users(self):
        """Initialize default system users"""
        # Create admin user if not exists
        admin_password = self.secrets_manager.secrets_manager.get_secret('admin_password')
        if not admin_password:
            admin_password = self.secrets_manager.secrets_manager.generate_secret(
                'admin_password', 
                length=32,
                metadata={'user': 'admin', 'role': 'admin'}
            )
            logger.info(f"Generated admin password: {admin_password}")
        
        # Create brain service tokens
        for brain_id in ['brain1', 'brain2', 'brain3', 'brain4']:
            token_key = f'{brain_id}_service_token'
            if not self.secrets_manager.secrets_manager.get_secret(token_key):
                service_token = self.secrets_manager.secrets_manager.generate_secret(
                    token_key,
                    length=64,
                    metadata={'service': brain_id, 'role': 'brain_service'}
                )
                logger.info(f"Generated service token for {brain_id}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create a new user"""
        password_hash = self.hash_password(password)
        permissions = self.role_permissions.get(role, [])
        
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=datetime.now(),
            password_hash=password_hash
        )
        
        # Store user data (in production, use database)
        user_key = f'user_{username}'
        user_data = asdict(user)
        user_data['role'] = user.role.value
        user_data['permissions'] = [p.value for p in user.permissions]
        user_data['created_at'] = user.created_at.isoformat()
        
        self.secrets_manager.secrets_manager.set_secret(
            user_key,
            str(user_data),
            metadata={'type': 'user', 'username': username}
        )
        
        logger.info(f"User created: {username} with role {role.value}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user_key = f'user_{username}'
        user_data_str = self.secrets_manager.secrets_manager.get_secret(user_key)
        
        if not user_data_str:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        try:
            # Parse user data (in production, use proper serialization)
            user_data = eval(user_data_str)  # Note: Use proper JSON in production
            
            # Verify password
            if not self.verify_password(password, user_data['password_hash']):
                logger.warning(f"Authentication failed: invalid password for {username}")
                return None
            
            # Create user object
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                role=UserRole(user_data['role']),
                permissions=[Permission(p) for p in user_data['permissions']],
                created_at=datetime.fromisoformat(user_data['created_at']),
                last_login=datetime.now(),
                is_active=user_data.get('is_active', True),
                password_hash=user_data['password_hash']
            )
            
            logger.info(f"User authenticated: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user {username}: {e}")
            return None
    
    def generate_token(self, user: User, token_type: str = "access") -> str:
        """Generate JWT token for user"""
        now = datetime.now()
        expires_at = now + timedelta(hours=self.jwt_config['expiration_hours'])
        
        payload = {
            'user_id': user.username,
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'iat': now.timestamp(),
            'exp': expires_at.timestamp(),
            'token_type': token_type
        }
        
        token = jwt.encode(
            payload,
            self.jwt_config['secret_key'],
            algorithm=self.jwt_config['algorithm']
        )
        
        logger.info(f"Token generated for user: {user.username}")
        return token
    
    def verify_token(self, token: str) -> Optional[AuthToken]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_config['secret_key'],
                algorithms=[self.jwt_config['algorithm']]
            )
            
            auth_token = AuthToken(
                user_id=payload['user_id'],
                username=payload['username'],
                role=payload['role'],
                permissions=payload['permissions'],
                issued_at=datetime.fromtimestamp(payload['iat']),
                expires_at=datetime.fromtimestamp(payload['exp']),
                token_type=payload.get('token_type', 'access')
            )
            
            return auth_token
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """Check if token has required permission"""
        auth_token = self.verify_token(token)
        
        if not auth_token:
            return False
        
        return required_permission.value in auth_token.permissions
    
    def generate_service_token(self, service_name: str) -> str:
        """Generate service token for brain services"""
        # Create service user
        service_user = User(
            username=f"service_{service_name}",
            email=f"{service_name}@fourbrain.local",
            role=UserRole.BRAIN_SERVICE,
            permissions=self.role_permissions[UserRole.BRAIN_SERVICE],
            created_at=datetime.now()
        )
        
        return self.generate_token(service_user)

def require_auth(required_permission: Permission = None):
    """Decorator for requiring authentication and authorization"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request headers (implementation depends on web framework)
            token = kwargs.get('auth_token') or os.getenv('AUTH_TOKEN')
            
            if not token:
                raise PermissionError("Authentication token required")
            
            auth_manager = AuthManager()
            
            # Verify token
            auth_token = auth_manager.verify_token(token)
            if not auth_token:
                raise PermissionError("Invalid or expired token")
            
            # Check permission if required
            if required_permission:
                if not auth_manager.check_permission(token, required_permission):
                    raise PermissionError(f"Permission denied: {required_permission.value}")
            
            # Add auth context to kwargs
            kwargs['auth_context'] = auth_token
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def main():
    """Main function for testing"""
    try:
        auth_manager = AuthManager()
        
        # Create test admin user
        admin_user = auth_manager.create_user(
            username="admin",
            email="admin@fourbrain.local",
            password="secure_admin_password_2024",
            role=UserRole.ADMIN
        )
        
        # Generate token
        token = auth_manager.generate_token(admin_user)
        logger.info(f"Generated token: {token[:50]}...")
        
        # Verify token
        auth_token = auth_manager.verify_token(token)
        if auth_token:
            logger.info(f"Token verified for user: {auth_token.username}")
        
        # Test permission check
        has_permission = auth_manager.check_permission(token, Permission.MANAGE_SYSTEM)
        logger.info(f"Has MANAGE_SYSTEM permission: {has_permission}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
