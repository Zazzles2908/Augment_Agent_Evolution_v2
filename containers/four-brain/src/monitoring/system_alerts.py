#!/usr/bin/env python3
"""
Four-Brain System Alerting
Intelligent alerting system for Four-Brain architecture

Created: 2025-07-27 AEST
Author: AugmentAI - Alerting Implementation
"""

import asyncio
import aiohttp
import redis
import smtplib
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    BRAIN_DOWN = "brain_down"
    HIGH_RESPONSE_TIME = "high_response_time"
    REDIS_MEMORY_HIGH = "redis_memory_high"
    AI_COMMUNICATION_STOPPED = "ai_communication_stopped"
    SYSTEM_DEGRADED = "system_degraded"
    CONTAINER_UNHEALTHY = "container_unhealthy"

@dataclass
class Alert:
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    component: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_timestamp: Optional[float] = None

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, alert_type: AlertType, severity: AlertSeverity, threshold: float):
        self.alert_type = alert_type
        self.severity = severity
        self.threshold = threshold
        self.last_triggered = 0
        self.cooldown_seconds = 300  # 5 minutes
    
    def should_trigger(self, current_time: float) -> bool:
        """Check if enough time has passed since last trigger"""
        return current_time - self.last_triggered > self.cooldown_seconds
    
    def trigger(self, current_time: float):
        """Mark rule as triggered"""
        self.last_triggered = current_time
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule against metrics - to be implemented by subclasses"""
        raise NotImplementedError

class BrainDownRule(AlertRule):
    """Alert when a brain service is down"""
    
    def __init__(self):
        super().__init__(AlertType.BRAIN_DOWN, AlertSeverity.CRITICAL, 0)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        current_time = time.time()
        
        if not self.should_trigger(current_time):
            return None
        
        brains = metrics.get('brains', {})
        for brain_name, brain_data in brains.items():
            if brain_data.get('status') == 'unhealthy':
                self.trigger(current_time)
                return Alert(
                    id=f"brain_down_{brain_name}_{int(current_time)}",
                    type=self.alert_type,
                    severity=self.severity,
                    title=f"Brain Service Down: {brain_name}",
                    description=f"Brain service {brain_name} is unhealthy and not responding",
                    timestamp=current_time,
                    component=brain_name
                )
        return None

class HighResponseTimeRule(AlertRule):
    """Alert when response times are too high"""
    
    def __init__(self, threshold_ms: float = 5000):
        super().__init__(AlertType.HIGH_RESPONSE_TIME, AlertSeverity.WARNING, threshold_ms)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        current_time = time.time()
        
        if not self.should_trigger(current_time):
            return None
        
        brains = metrics.get('brains', {})
        for brain_name, brain_data in brains.items():
            response_time = brain_data.get('response_time_ms', 0)
            if response_time > self.threshold:
                self.trigger(current_time)
                return Alert(
                    id=f"high_response_time_{brain_name}_{int(current_time)}",
                    type=self.alert_type,
                    severity=self.severity,
                    title=f"High Response Time: {brain_name}",
                    description=f"Brain service {brain_name} response time is {response_time:.1f}ms (threshold: {self.threshold}ms)",
                    timestamp=current_time,
                    component=brain_name,
                    metric_value=response_time,
                    threshold=self.threshold
                )
        return None

class RedisMemoryHighRule(AlertRule):
    """Alert when Redis memory usage is high"""
    
    def __init__(self, threshold_mb: float = 1600):  # 80% of 2GB
        super().__init__(AlertType.REDIS_MEMORY_HIGH, AlertSeverity.WARNING, threshold_mb)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        current_time = time.time()
        
        if not self.should_trigger(current_time):
            return None
        
        redis_memory = metrics.get('redis_memory_mb', 0)
        if redis_memory > self.threshold:
            self.trigger(current_time)
            return Alert(
                id=f"redis_memory_high_{int(current_time)}",
                type=self.alert_type,
                severity=self.severity,
                title="Redis Memory Usage High",
                description=f"Redis memory usage is {redis_memory:.1f}MB (threshold: {self.threshold}MB)",
                timestamp=current_time,
                component="redis",
                metric_value=redis_memory,
                threshold=self.threshold
            )
        return None

class AICommunicationStoppedRule(AlertRule):
    """Alert when AI communication stops"""
    
    def __init__(self):
        super().__init__(AlertType.AI_COMMUNICATION_STOPPED, AlertSeverity.CRITICAL, 0)
        self.last_message_count = 0
        self.no_activity_threshold = 600  # 10 minutes
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        current_time = time.time()
        
        if not self.should_trigger(current_time):
            return None
        
        current_messages = metrics.get('redis_messages', 0)
        
        # If message count hasn't changed and we haven't seen activity
        if current_messages == self.last_message_count:
            if not hasattr(self, 'last_activity_time'):
                self.last_activity_time = current_time
            elif current_time - self.last_activity_time > self.no_activity_threshold:
                self.trigger(current_time)
                return Alert(
                    id=f"ai_communication_stopped_{int(current_time)}",
                    type=self.alert_type,
                    severity=self.severity,
                    title="AI Communication Stopped",
                    description=f"No new AI messages detected for {self.no_activity_threshold/60:.1f} minutes",
                    timestamp=current_time,
                    component="redis_streams"
                )
        else:
            self.last_activity_time = current_time
        
        self.last_message_count = current_messages
        return None

class SystemDegradedRule(AlertRule):
    """Alert when overall system health is degraded"""
    
    def __init__(self):
        super().__init__(AlertType.SYSTEM_DEGRADED, AlertSeverity.WARNING, 0)
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        current_time = time.time()
        
        if not self.should_trigger(current_time):
            return None
        
        overall_health = metrics.get('overall_health', 'unknown')
        if overall_health in ['degraded', 'unhealthy']:
            healthy_containers = metrics.get('healthy_containers', 0)
            total_containers = metrics.get('total_containers', 0)
            
            self.trigger(current_time)
            return Alert(
                id=f"system_degraded_{int(current_time)}",
                type=self.alert_type,
                severity=AlertSeverity.CRITICAL if overall_health == 'unhealthy' else AlertSeverity.WARNING,
                title=f"System Health {overall_health.title()}",
                description=f"System health is {overall_health}: {healthy_containers}/{total_containers} containers healthy",
                timestamp=current_time,
                component="system"
            )
        return None

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.rules = [
            BrainDownRule(),
            HighResponseTimeRule(),
            RedisMemoryHighRule(),
            AICommunicationStoppedRule(),
            SystemDegradedRule()
        ]
        
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
        # Add console notification by default
        self.add_notification_handler(self.console_notification)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def console_notification(self, alert: Alert):
        """Console notification handler"""
        severity_icons = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(alert.severity, "📢")
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
        
        if alert.resolved:
            logger.info(f"✅ {timestamp} RESOLVED: {alert.title}")
        else:
            logger.warning(f"{icon} {timestamp} {alert.severity.value.upper()}: {alert.title} - {alert.description}")
    
    def email_notification(self, alert: Alert, smtp_config: Dict[str, str]):
        """Email notification handler"""
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email']
            msg['Subject'] = f"Four-Brain Alert: {alert.title}"
            
            body = f"""
            Alert Details:
            - Type: {alert.type.value}
            - Severity: {alert.severity.value.upper()}
            - Component: {alert.component}
            - Time: {datetime.fromtimestamp(alert.timestamp)}
            - Description: {alert.description}
            
            Alert ID: {alert.id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate all rules against current metrics"""
        new_alerts = []
        
        for rule in self.rules:
            try:
                alert = rule.evaluate(metrics)
                if alert:
                    new_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.__class__.__name__}: {e}")
        
        # Process new alerts
        for alert in new_alerts:
            if alert.id not in self.active_alerts:
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                
                # Send notifications
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
        
        # Auto-resolve alerts that are no longer active
        self.auto_resolve_alerts(metrics)
    
    def auto_resolve_alerts(self, metrics: Dict[str, Any]):
        """Automatically resolve alerts when conditions improve"""
        current_time = time.time()
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            should_resolve = False
            
            if alert.type == AlertType.BRAIN_DOWN:
                # Check if brain is now healthy
                brains = metrics.get('brains', {})
                brain_data = brains.get(alert.component, {})
                if brain_data.get('status') == 'healthy':
                    should_resolve = True
            
            elif alert.type == AlertType.HIGH_RESPONSE_TIME:
                # Check if response time is now acceptable
                brains = metrics.get('brains', {})
                brain_data = brains.get(alert.component, {})
                response_time = brain_data.get('response_time_ms', 0)
                if response_time < alert.threshold * 0.8:  # 20% buffer
                    should_resolve = True
            
            elif alert.type == AlertType.REDIS_MEMORY_HIGH:
                # Check if memory usage is now acceptable
                redis_memory = metrics.get('redis_memory_mb', 0)
                if redis_memory < alert.threshold * 0.8:  # 20% buffer
                    should_resolve = True
            
            elif alert.type == AlertType.SYSTEM_DEGRADED:
                # Check if system health improved
                overall_health = metrics.get('overall_health', 'unknown')
                if overall_health == 'healthy':
                    should_resolve = True
            
            if should_resolve:
                alert.resolved = True
                alert.resolved_timestamp = current_time
                resolved_alerts.append(alert_id)
                
                # Send resolution notification
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in resolution notification: {e}")
        
        # Remove resolved alerts from active list
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        active_alerts = self.get_active_alerts()
        recent_history = self.get_alert_history(24)
        
        return {
            "active_alerts_count": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "warning_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "alerts_last_24h": len(recent_history),
            "most_recent_alert": max([a.timestamp for a in recent_history]) if recent_history else None,
            "active_alerts": [
                {
                    "id": alert.id,
                    "type": alert.type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "component": alert.component,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts
            ]
        }

# Example usage
async def main():
    """Example usage of alert manager"""
    alert_manager = AlertManager()
    
    # Example metrics (would come from monitoring system)
    sample_metrics = {
        "overall_health": "healthy",
        "total_containers": 5,
        "healthy_containers": 5,
        "redis_memory_mb": 50.0,
        "redis_messages": 385,
        "brains": {
            "brain1_embedding": {"status": "healthy", "response_time_ms": 150.0},
            "brain2_reranker": {"status": "healthy", "response_time_ms": 200.0},
            "brain3_augment": {"status": "healthy", "response_time_ms": 180.0},
            "brain4_docling": {"status": "healthy", "response_time_ms": 250.0},
            "k2_vector_hub": {"status": "healthy", "response_time_ms": 120.0}
        }
    }
    
    # Evaluate metrics
    alert_manager.evaluate_metrics(sample_metrics)
    
    # Get summary
    summary = alert_manager.get_alert_summary()
    print(f"Alert Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
