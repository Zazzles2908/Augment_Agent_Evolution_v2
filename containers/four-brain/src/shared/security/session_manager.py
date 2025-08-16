"""
Session Management System for Four-Brain System v2
Comprehensive session lifecycle management with security features

Created: 2025-07-30 AEST
Purpose: Secure session management with Redis backend
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

@dataclass
class SessionInfo:
    """Session information data structure"""
    session_id: str
    user_id: str
    username: str
    role: str
    permissions: List[str]
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['status'] = SessionStatus(data['status'])
        return cls(**data)

class SessionManager:
    """
    Comprehensive session management system
    
    Features:
    - Secure session creation and validation
    - Automatic session expiration
    - Session activity tracking
    - Multi-device session management
    - Session security monitoring
    - Redis-backed persistence
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/5"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Session configuration
        self.session_config = {
            'default_ttl_hours': 24,
            'max_sessions_per_user': 10,
            'session_id_length': 32,
            'cleanup_interval_minutes': 30,
            'activity_timeout_minutes': 120,
            'max_inactive_sessions': 5
        }
        
        # Security settings
        self.security_config = {
            'require_ip_validation': True,
            'require_user_agent_validation': False,
            'max_concurrent_sessions': 5,
            'suspicious_activity_threshold': 10,
            'session_hijack_detection': True
        }
        
        # Active sessions cache
        self.active_sessions: Dict[str, SessionInfo] = {}
        
        # Session statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_expired': 0,
            'sessions_terminated': 0,
            'suspicious_activities': 0,
            'cleanup_runs': 0
        }
        
        logger.info("🔐 Session Manager initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start background tasks"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load active sessions from Redis
            await self._load_sessions_from_redis()
            
            # Start background cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
            
            logger.info("✅ Session Manager Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Session Manager: {e}")
            raise
    
    async def create_session(self, user_id: str, username: str, role: str, 
                           permissions: List[str], ip_address: str, 
                           user_agent: str, ttl_hours: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new session with security validation"""
        try:
            # Check concurrent session limits
            user_sessions = await self._get_user_sessions(user_id)
            if len(user_sessions) >= self.security_config['max_concurrent_sessions']:
                # Terminate oldest session
                oldest_session = min(user_sessions, key=lambda s: s.last_accessed)
                await self.terminate_session(oldest_session.session_id)
                logger.warning(f"Terminated oldest session for user {user_id} due to limit")
            
            # Generate secure session ID
            session_id = self._generate_session_id()
            
            # Calculate expiration
            ttl = ttl_hours or self.session_config['default_ttl_hours']
            expires_at = datetime.now() + timedelta(hours=ttl)
            
            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                username=username,
                role=role,
                permissions=permissions,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                status=SessionStatus.ACTIVE,
                metadata=metadata or {}
            )
            
            # Store in Redis and cache
            await self._store_session(session_info)
            self.active_sessions[session_id] = session_info
            
            # Update statistics
            self.stats['sessions_created'] += 1
            
            logger.info(f"✅ Session created for user {username}: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise
    
    async def validate_session(self, session_id: str, ip_address: Optional[str] = None,
                             user_agent: Optional[str] = None) -> Optional[SessionInfo]:
        """Validate session with security checks"""
        try:
            # Get session from cache or Redis
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                session_info = await self._load_session_from_redis(session_id)
                if session_info:
                    self.active_sessions[session_id] = session_info
            
            if not session_info:
                return None
            
            # Check expiration
            if datetime.now() > session_info.expires_at:
                await self.terminate_session(session_id)
                return None
            
            # Check status
            if session_info.status != SessionStatus.ACTIVE:
                return None
            
            # Security validations
            if self.security_config['require_ip_validation'] and ip_address:
                if session_info.ip_address != ip_address:
                    logger.warning(f"IP mismatch for session {session_id}: {session_info.ip_address} vs {ip_address}")
                    await self._flag_suspicious_activity(session_id, "ip_mismatch")
                    return None
            
            if self.security_config['require_user_agent_validation'] and user_agent:
                if session_info.user_agent != user_agent:
                    logger.warning(f"User agent mismatch for session {session_id}")
                    await self._flag_suspicious_activity(session_id, "user_agent_mismatch")
                    return None
            
            # Update last accessed time
            session_info.last_accessed = datetime.now()
            await self._store_session(session_info)
            
            return session_info
            
        except Exception as e:
            logger.error(f"❌ Failed to validate session: {e}")
            return None
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate session and cleanup"""
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                session_info = await self._load_session_from_redis(session_id)
            
            if session_info:
                session_info.status = SessionStatus.TERMINATED
                await self._store_session(session_info)
                
                # Remove from active cache
                self.active_sessions.pop(session_id, None)
                
                # Update statistics
                self.stats['sessions_terminated'] += 1
                
                logger.info(f"✅ Session terminated: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to terminate session: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user"""
        return await self._get_user_sessions(user_id)
    
    async def terminate_all_user_sessions(self, user_id: str) -> int:
        """Terminate all sessions for a user"""
        user_sessions = await self._get_user_sessions(user_id)
        terminated_count = 0
        
        for session in user_sessions:
            if await self.terminate_session(session.session_id):
                terminated_count += 1
        
        logger.info(f"✅ Terminated {terminated_count} sessions for user {user_id}")
        return terminated_count
    
    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID"""
        return secrets.token_urlsafe(self.session_config['session_id_length'])
    
    async def _store_session(self, session_info: SessionInfo):
        """Store session in Redis"""
        if self.redis_client:
            key = f"session:{session_info.session_id}"
            data = json.dumps(session_info.to_dict())
            ttl_seconds = int((session_info.expires_at - datetime.now()).total_seconds())
            await self.redis_client.setex(key, ttl_seconds, data)
    
    async def _load_session_from_redis(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from Redis"""
        if self.redis_client:
            try:
                key = f"session:{session_id}"
                data = await self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    return SessionInfo.from_dict(session_data)
            except Exception as e:
                logger.error(f"Failed to load session from Redis: {e}")
        return None
    
    async def _load_sessions_from_redis(self):
        """Load all active sessions from Redis on startup"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("session:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        session_data = json.loads(data)
                        session_info = SessionInfo.from_dict(session_data)
                        if session_info.status == SessionStatus.ACTIVE:
                            self.active_sessions[session_info.session_id] = session_info
                
                logger.info(f"✅ Loaded {len(self.active_sessions)} active sessions from Redis")
                
            except Exception as e:
                logger.error(f"Failed to load sessions from Redis: {e}")
    
    async def _get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all sessions for a specific user"""
        user_sessions = []
        
        # Check active cache
        for session in self.active_sessions.values():
            if session.user_id == user_id and session.status == SessionStatus.ACTIVE:
                user_sessions.append(session)
        
        # Also check Redis for any missed sessions
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("session:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        session_data = json.loads(data)
                        session_info = SessionInfo.from_dict(session_data)
                        if (session_info.user_id == user_id and 
                            session_info.status == SessionStatus.ACTIVE and
                            session_info.session_id not in [s.session_id for s in user_sessions]):
                            user_sessions.append(session_info)
            except Exception as e:
                logger.error(f"Failed to get user sessions from Redis: {e}")
        
        return user_sessions
    
    async def _flag_suspicious_activity(self, session_id: str, activity_type: str):
        """Flag suspicious session activity"""
        self.stats['suspicious_activities'] += 1
        
        # Log security event
        logger.warning(f"🚨 Suspicious activity detected: {activity_type} for session {session_id}")
        
        # Store in Redis for security monitoring
        if self.redis_client:
            key = f"security:suspicious:{session_id}:{int(time.time())}"
            data = {
                'session_id': session_id,
                'activity_type': activity_type,
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            await self.redis_client.setex(key, 86400, json.dumps(data))  # 24 hour retention
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.session_config['cleanup_interval_minutes'] * 60)
                
                expired_sessions = []
                current_time = datetime.now()
                
                # Find expired sessions
                for session_id, session_info in list(self.active_sessions.items()):
                    if current_time > session_info.expires_at:
                        expired_sessions.append(session_id)
                
                # Cleanup expired sessions
                for session_id in expired_sessions:
                    await self.terminate_session(session_id)
                    self.stats['sessions_expired'] += 1
                
                self.stats['cleanup_runs'] += 1
                
                if expired_sessions:
                    logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get session management statistics"""
        return {
            'active_sessions': len(self.active_sessions),
            'statistics': self.stats.copy(),
            'configuration': {
                'session_config': self.session_config,
                'security_config': self.security_config
            },
            'timestamp': datetime.now().isoformat()
        }

# Global session manager instance
session_manager = SessionManager()

async def initialize_session_manager():
    """Initialize the global session manager"""
    await session_manager.initialize()

if __name__ == "__main__":
    # Test the session manager
    async def test_session_manager():
        await initialize_session_manager()
        
        # Create test session
        session_id = await session_manager.create_session(
            user_id="test_user_1",
            username="testuser",
            role="user",
            permissions=["read", "write"],
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        print(f"Created session: {session_id}")
        
        # Validate session
        session_info = await session_manager.validate_session(session_id, "127.0.0.1")
        print(f"Session valid: {session_info is not None}")
        
        # Get statistics
        stats = await session_manager.get_session_statistics()
        print(f"Statistics: {stats}")
    
    asyncio.run(test_session_manager())
