"""
Incident Response System for Four-Brain System v2
Automated incident detection, response, and management

Created: 2025-07-30 AEST
Purpose: Automated incident response with escalation and remediation
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status tracking"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    RESPONDING = "responding"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentType(Enum):
    """Types of security incidents"""
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_COMPROMISE = "system_compromise"
    DENIAL_OF_SERVICE = "denial_of_service"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"

class ResponseAction(Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    TERMINATE_SESSION = "terminate_session"
    ISOLATE_SYSTEM = "isolate_system"
    ALERT_ADMIN = "alert_admin"
    LOG_INCIDENT = "log_incident"
    COLLECT_EVIDENCE = "collect_evidence"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    affected_systems: List[str]
    affected_users: List[str]
    source_ip: str
    detection_time: datetime
    response_time: Optional[datetime]
    resolution_time: Optional[datetime]
    evidence: List[str]
    response_actions: List[ResponseAction]
    assigned_to: str
    escalated: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['incident_type'] = self.incident_type.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['detection_time'] = self.detection_time.isoformat()
        data['response_time'] = self.response_time.isoformat() if self.response_time else None
        data['resolution_time'] = self.resolution_time.isoformat() if self.resolution_time else None
        data['response_actions'] = [action.value for action in self.response_actions]
        return data

@dataclass
class ResponsePlaybook:
    """Incident response playbook"""
    playbook_id: str
    incident_type: IncidentType
    severity_threshold: IncidentSeverity
    automated_actions: List[ResponseAction]
    manual_steps: List[str]
    escalation_criteria: Dict[str, Any]
    containment_procedures: List[str]
    recovery_procedures: List[str]
    evidence_collection: List[str]
    notification_list: List[str]

@dataclass
class ResponseMetrics:
    """Incident response metrics"""
    total_incidents: int
    incidents_by_severity: Dict[str, int]
    incidents_by_type: Dict[str, int]
    average_response_time: float
    average_resolution_time: float
    automated_responses: int
    manual_responses: int
    escalated_incidents: int

class IncidentResponder:
    """
    Comprehensive incident response system
    
    Features:
    - Automated incident detection and classification
    - Playbook-based response automation
    - Evidence collection and preservation
    - Escalation management
    - Response metrics and reporting
    - Integration with security monitoring
    - Stakeholder notification
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/9"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Active incidents
        self.active_incidents: Dict[str, SecurityIncident] = {}
        
        # Response playbooks
        self.playbooks = self._initialize_playbooks()
        
        # Response callbacks
        self.response_callbacks: Dict[ResponseAction, List[Callable]] = {
            action: [] for action in ResponseAction
        }
        
        # Configuration
        self.config = {
            'auto_response_enabled': True,
            'escalation_timeout_minutes': 30,
            'evidence_retention_days': 365,
            'max_concurrent_incidents': 100,
            'response_timeout_minutes': 15,
            'notification_enabled': True
        }
        
        # Response metrics
        self.metrics = ResponseMetrics(
            total_incidents=0,
            incidents_by_severity={s.value: 0 for s in IncidentSeverity},
            incidents_by_type={t.value: 0 for t in IncidentType},
            average_response_time=0.0,
            average_resolution_time=0.0,
            automated_responses=0,
            manual_responses=0,
            escalated_incidents=0
        )
        
        logger.info("🚨 Incident Responder initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start response monitoring"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load active incidents
            await self._load_active_incidents()
            
            # Start background tasks
            asyncio.create_task(self._monitor_incident_timeouts())
            asyncio.create_task(self._cleanup_resolved_incidents())
            
            logger.info("✅ Incident Responder Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Incident Responder: {e}")
            raise
    
    async def create_incident(self, incident_type: IncidentType, severity: IncidentSeverity,
                            title: str, description: str, affected_systems: List[str],
                            affected_users: List[str], source_ip: str,
                            evidence: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Create new security incident and initiate response"""
        try:
            # Generate incident ID
            incident_id = hashlib.sha256(
                f"{incident_type.value}:{source_ip}:{time.time()}".encode()
            ).hexdigest()[:16]
            
            # Create incident
            incident = SecurityIncident(
                incident_id=incident_id,
                incident_type=incident_type,
                severity=severity,
                status=IncidentStatus.DETECTED,
                title=title,
                description=description,
                affected_systems=affected_systems,
                affected_users=affected_users,
                source_ip=source_ip,
                detection_time=datetime.now(),
                response_time=None,
                resolution_time=None,
                evidence=evidence or [],
                response_actions=[],
                assigned_to="automated_system",
                escalated=False,
                metadata=metadata or {}
            )
            
            # Store incident
            self.active_incidents[incident_id] = incident
            await self._store_incident(incident)
            
            # Update metrics
            self.metrics.total_incidents += 1
            self.metrics.incidents_by_severity[severity.value] += 1
            self.metrics.incidents_by_type[incident_type.value] += 1
            
            # Initiate automated response
            if self.config['auto_response_enabled']:
                await self._initiate_automated_response(incident)
            
            logger.warning(f"🚨 Security incident created: {incident_id} - {title}")
            return incident_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create incident: {e}")
            raise
    
    async def _initiate_automated_response(self, incident: SecurityIncident):
        """Initiate automated response based on playbooks"""
        try:
            # Find matching playbook
            playbook = self._find_matching_playbook(incident)
            if not playbook:
                logger.warning(f"No matching playbook found for incident {incident.incident_id}")
                return
            
            # Update incident status
            incident.status = IncidentStatus.RESPONDING
            incident.response_time = datetime.now()
            
            # Execute automated actions
            for action in playbook.automated_actions:
                try:
                    await self._execute_response_action(action, incident)
                    incident.response_actions.append(action)
                    self.metrics.automated_responses += 1
                except Exception as e:
                    logger.error(f"Failed to execute response action {action.value}: {e}")
            
            # Check escalation criteria
            if self._should_escalate(incident, playbook):
                await self._escalate_incident(incident)
            
            # Store updated incident
            await self._store_incident(incident)
            
            logger.info(f"✅ Automated response initiated for incident {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"❌ Automated response failed: {e}")
    
    async def _execute_response_action(self, action: ResponseAction, incident: SecurityIncident):
        """Execute specific response action"""
        try:
            if action == ResponseAction.BLOCK_IP:
                await self._block_ip_address(incident.source_ip)
            
            elif action == ResponseAction.DISABLE_USER:
                for user in incident.affected_users:
                    await self._disable_user_account(user)
            
            elif action == ResponseAction.TERMINATE_SESSION:
                for user in incident.affected_users:
                    await self._terminate_user_sessions(user)
            
            elif action == ResponseAction.ISOLATE_SYSTEM:
                for system in incident.affected_systems:
                    await self._isolate_system(system)
            
            elif action == ResponseAction.ALERT_ADMIN:
                await self._alert_administrators(incident)
            
            elif action == ResponseAction.LOG_INCIDENT:
                await self._log_incident_details(incident)
            
            elif action == ResponseAction.COLLECT_EVIDENCE:
                await self._collect_additional_evidence(incident)
            
            elif action == ResponseAction.NOTIFY_STAKEHOLDERS:
                await self._notify_stakeholders(incident)
            
            # Execute registered callbacks
            for callback in self.response_callbacks.get(action, []):
                try:
                    await callback(incident)
                except Exception as e:
                    logger.error(f"Response callback error: {e}")
            
            logger.info(f"✅ Response action executed: {action.value} for incident {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute response action {action.value}: {e}")
            raise
    
    async def _block_ip_address(self, ip_address: str):
        """Block IP address at network level"""
        # This would integrate with firewall/network security
        logger.info(f"🚫 IP address blocked: {ip_address}")
        
        # Store blocked IP in Redis
        if self.redis_client:
            await self.redis_client.sadd("blocked_ips", ip_address)
            await self.redis_client.expire("blocked_ips", 86400)  # 24 hour block
    
    async def _disable_user_account(self, user_id: str):
        """Disable user account"""
        # This would integrate with user management system
        logger.info(f"🔒 User account disabled: {user_id}")
        
        # Store disabled user in Redis
        if self.redis_client:
            await self.redis_client.sadd("disabled_users", user_id)
    
    async def _terminate_user_sessions(self, user_id: str):
        """Terminate all user sessions"""
        # This would integrate with session management
        logger.info(f"🔌 User sessions terminated: {user_id}")
        
        try:
            from shared.security.session_manager import session_manager
            await session_manager.terminate_all_user_sessions(user_id)
        except ImportError:
            logger.warning("Session manager not available for session termination")
    
    async def _isolate_system(self, system_id: str):
        """Isolate affected system"""
        # This would integrate with system management
        logger.info(f"🏝️ System isolated: {system_id}")
    
    async def _alert_administrators(self, incident: SecurityIncident):
        """Alert system administrators"""
        alert_message = f"SECURITY INCIDENT: {incident.title} (ID: {incident.incident_id})"
        logger.critical(f"🚨 ADMIN ALERT: {alert_message}")
        
        # Store alert in Redis for admin dashboard
        if self.redis_client:
            alert_data = {
                'incident_id': incident.incident_id,
                'severity': incident.severity.value,
                'title': incident.title,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.lpush("admin_alerts", json.dumps(alert_data))
            await self.redis_client.ltrim("admin_alerts", 0, 99)  # Keep last 100 alerts
    
    async def _log_incident_details(self, incident: SecurityIncident):
        """Log detailed incident information"""
        incident_log = {
            'incident_id': incident.incident_id,
            'type': incident.incident_type.value,
            'severity': incident.severity.value,
            'description': incident.description,
            'affected_systems': incident.affected_systems,
            'affected_users': incident.affected_users,
            'source_ip': incident.source_ip,
            'detection_time': incident.detection_time.isoformat(),
            'evidence': incident.evidence
        }
        
        logger.info(f"📝 Incident logged: {json.dumps(incident_log)}")
    
    async def _collect_additional_evidence(self, incident: SecurityIncident):
        """Collect additional evidence for the incident"""
        # This would integrate with logging and monitoring systems
        additional_evidence = [
            f"System logs collected at {datetime.now().isoformat()}",
            f"Network traffic analysis for IP {incident.source_ip}",
            f"User activity logs for affected users"
        ]
        
        incident.evidence.extend(additional_evidence)
        logger.info(f"🔍 Additional evidence collected for incident {incident.incident_id}")
    
    async def _notify_stakeholders(self, incident: SecurityIncident):
        """Notify relevant stakeholders"""
        if incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            logger.info(f"📧 Stakeholders notified for critical incident {incident.incident_id}")
    
    def _find_matching_playbook(self, incident: SecurityIncident) -> Optional[ResponsePlaybook]:
        """Find matching response playbook for incident"""
        for playbook in self.playbooks.values():
            if (playbook.incident_type == incident.incident_type and
                self._severity_meets_threshold(incident.severity, playbook.severity_threshold)):
                return playbook
        return None
    
    def _severity_meets_threshold(self, incident_severity: IncidentSeverity, 
                                threshold: IncidentSeverity) -> bool:
        """Check if incident severity meets playbook threshold"""
        severity_levels = {
            IncidentSeverity.LOW: 1,
            IncidentSeverity.MEDIUM: 2,
            IncidentSeverity.HIGH: 3,
            IncidentSeverity.CRITICAL: 4
        }
        return severity_levels[incident_severity] >= severity_levels[threshold]
    
    def _should_escalate(self, incident: SecurityIncident, playbook: ResponsePlaybook) -> bool:
        """Determine if incident should be escalated"""
        # Check severity-based escalation
        if incident.severity == IncidentSeverity.CRITICAL:
            return True
        
        # Check time-based escalation
        if incident.response_time:
            time_since_response = datetime.now() - incident.response_time
            if time_since_response.total_seconds() > self.config['escalation_timeout_minutes'] * 60:
                return True
        
        # Check custom escalation criteria
        criteria = playbook.escalation_criteria
        if criteria.get('auto_escalate', False):
            return True
        
        return False
    
    async def _escalate_incident(self, incident: SecurityIncident):
        """Escalate incident to higher level"""
        incident.escalated = True
        incident.assigned_to = "security_team"
        self.metrics.escalated_incidents += 1
        
        logger.warning(f"⬆️ Incident escalated: {incident.incident_id}")
        
        # Additional escalation actions
        await self._alert_administrators(incident)
        await self._notify_stakeholders(incident)
    
    def _initialize_playbooks(self) -> Dict[str, ResponsePlaybook]:
        """Initialize incident response playbooks"""
        playbooks = {}
        
        # Brute force attack playbook
        playbooks['brute_force'] = ResponsePlaybook(
            playbook_id='brute_force',
            incident_type=IncidentType.BRUTE_FORCE_ATTACK,
            severity_threshold=IncidentSeverity.MEDIUM,
            automated_actions=[
                ResponseAction.BLOCK_IP,
                ResponseAction.LOG_INCIDENT,
                ResponseAction.COLLECT_EVIDENCE
            ],
            manual_steps=[
                "Review attack patterns",
                "Check for compromised accounts",
                "Update security policies"
            ],
            escalation_criteria={'failed_attempts': 10},
            containment_procedures=["Block source IP", "Monitor for additional attempts"],
            recovery_procedures=["Review blocked IPs", "Update detection rules"],
            evidence_collection=["Authentication logs", "Network traffic", "Failed login attempts"],
            notification_list=["security_team@company.com"]
        )
        
        # Privilege escalation playbook
        playbooks['privilege_escalation'] = ResponsePlaybook(
            playbook_id='privilege_escalation',
            incident_type=IncidentType.PRIVILEGE_ESCALATION,
            severity_threshold=IncidentSeverity.HIGH,
            automated_actions=[
                ResponseAction.DISABLE_USER,
                ResponseAction.TERMINATE_SESSION,
                ResponseAction.ALERT_ADMIN,
                ResponseAction.COLLECT_EVIDENCE
            ],
            manual_steps=[
                "Review user permissions",
                "Check for unauthorized changes",
                "Investigate attack vector"
            ],
            escalation_criteria={'auto_escalate': True},
            containment_procedures=["Disable affected accounts", "Review access logs"],
            recovery_procedures=["Restore proper permissions", "Update access controls"],
            evidence_collection=["Access logs", "Permission changes", "User activity"],
            notification_list=["security_team@company.com", "management@company.com"]
        )
        
        return playbooks
    
    async def _monitor_incident_timeouts(self):
        """Monitor incidents for timeout conditions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                timeout_threshold = timedelta(minutes=self.config['response_timeout_minutes'])
                
                for incident in list(self.active_incidents.values()):
                    if (incident.status in [IncidentStatus.DETECTED, IncidentStatus.INVESTIGATING] and
                        current_time - incident.detection_time > timeout_threshold):
                        
                        logger.warning(f"⏰ Incident timeout: {incident.incident_id}")
                        await self._escalate_incident(incident)
                
            except Exception as e:
                logger.error(f"❌ Incident timeout monitoring error: {e}")
    
    async def _cleanup_resolved_incidents(self):
        """Cleanup resolved incidents"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                resolved_incidents = []
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for incident_id, incident in list(self.active_incidents.items()):
                    if (incident.status == IncidentStatus.CLOSED and
                        incident.resolution_time and
                        incident.resolution_time < cutoff_time):
                        resolved_incidents.append(incident_id)
                
                # Move to historical storage
                for incident_id in resolved_incidents:
                    incident = self.active_incidents.pop(incident_id)
                    await self._archive_incident(incident)
                
                logger.info(f"🧹 Archived {len(resolved_incidents)} resolved incidents")
                
            except Exception as e:
                logger.error(f"❌ Incident cleanup error: {e}")
    
    async def _store_incident(self, incident: SecurityIncident):
        """Store incident in Redis"""
        if self.redis_client:
            try:
                key = f"incident:{incident.incident_id}"
                data = json.dumps(incident.to_dict())
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store incident: {e}")
    
    async def _archive_incident(self, incident: SecurityIncident):
        """Archive resolved incident"""
        if self.redis_client:
            try:
                key = f"incident_archive:{incident.incident_id}"
                data = json.dumps(incident.to_dict())
                ttl = self.config['evidence_retention_days'] * 86400
                await self.redis_client.setex(key, ttl, data)
            except Exception as e:
                logger.error(f"Failed to archive incident: {e}")
    
    async def _load_active_incidents(self):
        """Load active incidents from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("incident:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        incident_data = json.loads(data)
                        # Convert back to SecurityIncident object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load active incidents: {e}")
    
    def register_response_callback(self, action: ResponseAction, callback: Callable):
        """Register callback for response action"""
        self.response_callbacks[action].append(callback)
    
    async def get_incident_metrics(self) -> Dict[str, Any]:
        """Get incident response metrics"""
        return {
            'metrics': asdict(self.metrics),
            'active_incidents': len(self.active_incidents),
            'playbooks': len(self.playbooks),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global incident responder instance
incident_responder = IncidentResponder()

async def initialize_incident_responder():
    """Initialize the global incident responder"""
    await incident_responder.initialize()

if __name__ == "__main__":
    # Test the incident responder
    async def test_incident_responder():
        await initialize_incident_responder()
        
        # Create test incident
        incident_id = await incident_responder.create_incident(
            IncidentType.BRUTE_FORCE_ATTACK,
            IncidentSeverity.HIGH,
            "Brute Force Attack Detected",
            "Multiple failed login attempts from suspicious IP",
            ["web_server"],
            ["test_user"],
            "192.168.1.100",
            ["Failed login logs", "Network traffic analysis"]
        )
        
        print(f"Incident created: {incident_id}")
        
        # Wait for automated response
        await asyncio.sleep(2)
        
        # Get metrics
        metrics = await incident_responder.get_incident_metrics()
        print(f"Incident metrics: {metrics}")
    
    asyncio.run(test_incident_responder())
