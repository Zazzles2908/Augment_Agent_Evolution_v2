"""
Alert Manager - System Alerts and Notifications
Handles alerts and notifications for the Four-Brain monitoring system

This module provides comprehensive alerting capabilities including threshold-based
alerts, notification routing, alert aggregation, and escalation management.

Created: 2025-07-31 AEST
Purpose: System alerts and notification management
Module Size: 300 lines (modular design)
"""

import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    REDIS = "redis"
    FILE = "file"


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "greater_than", "less_than", "equals"
    threshold: float
    severity: AlertSeverity
    enabled: bool
    cooldown_seconds: int
    notification_channels: List[NotificationChannel]
    metadata: Dict[str, Any]


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    metric_value: float
    threshold: float
    condition: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    notification_count: int
    metadata: Dict[str, Any]


@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]  # Channel-specific configuration


class AlertManager:
    """
    Comprehensive Alert Management System
    
    Features:
    - Threshold-based alerting
    - Multiple notification channels
    - Alert aggregation and deduplication
    - Escalation management
    - Alert history and analytics
    - Configurable alert rules
    - Cooldown and suppression
    """
    
    def __init__(self, brain_id: str):
        self.brain_id = brain_id
        self.enabled = True
        
        # Alert state
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppressed_rules: Set[str] = set()
        
        # Notification configuration
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # Performance tracking
        self.metrics = {
            'total_alerts_triggered': 0,
            'alerts_acknowledged': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0,
            'notification_failures': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        
        # Initialize default notification channels
        self._initialize_default_notifications()
        
        logger.info(f"🚨 Alert Manager initialized for {brain_id}")
    
    def _initialize_default_notifications(self):
        """Initialize default notification channels"""
        # Console notifications (always enabled)
        self.notification_configs[NotificationChannel.CONSOLE] = NotificationConfig(
            channel=NotificationChannel.CONSOLE,
            enabled=True,
            config={}
        )
        
        # Redis notifications for inter-brain communication
        self.notification_configs[NotificationChannel.REDIS] = NotificationConfig(
            channel=NotificationChannel.REDIS,
            enabled=True,
            config={
                'redis_url': 'redis://localhost:6379/4',
                'channel': f'alerts:{self.brain_id}'
            }
        )
        
        # File logging
        self.notification_configs[NotificationChannel.FILE] = NotificationConfig(
            channel=NotificationChannel.FILE,
            enabled=True,
            config={
                'log_file': f'/tmp/alerts_{self.brain_id}.log',
                'max_size_mb': 100,
                'backup_count': 5
            }
        )
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add or update an alert rule"""
        try:
            with self._lock:
                self.alert_rules[rule.rule_id] = rule
                logger.info(f"📋 Added alert rule: {rule.name} ({rule.rule_id})")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add alert rule {rule.rule_id}: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            with self._lock:
                if rule_id in self.alert_rules:
                    del self.alert_rules[rule_id]
                    logger.info(f"🗑️ Removed alert rule: {rule_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to remove alert rule {rule_id}: {e}")
            return False
    
    def check_metric(self, metric_name: str, metric_value: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Check a metric against all applicable alert rules"""
        triggered_alerts = []
        
        if not self.enabled:
            return triggered_alerts
        
        try:
            with self._lock:
                for rule_id, rule in self.alert_rules.items():
                    if not rule.enabled or rule_id in self.suppressed_rules:
                        continue
                    
                    if rule.metric_name != metric_name:
                        continue
                    
                    # Check if alert should trigger
                    should_trigger = self._evaluate_condition(
                        metric_value, rule.condition, rule.threshold
                    )
                    
                    if should_trigger:
                        # Check cooldown
                        if self._is_in_cooldown(rule_id):
                            continue
                        
                        # Create alert
                        alert = self._create_alert(rule, metric_value, metadata or {})
                        
                        # Add to active alerts
                        self.active_alerts[alert.alert_id] = alert
                        self.alert_history.append(alert)
                        
                        # Send notifications
                        asyncio.create_task(self._send_notifications(alert))
                        
                        triggered_alerts.append(alert.alert_id)
                        self.metrics['total_alerts_triggered'] += 1
                        
                        logger.warning(f"🚨 Alert triggered: {alert.name} - {metric_name}={metric_value} {rule.condition} {rule.threshold}")
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"❌ Error checking metric {metric_name}: {e}")
            return []
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        elif condition == "greater_equal":
            return value >= threshold
        elif condition == "less_equal":
            return value <= threshold
        else:
            logger.warning(f"⚠️ Unknown condition: {condition}")
            return False
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        # Find most recent alert for this rule
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.rule_id == rule_id and alert.triggered_at > datetime.utcnow() - timedelta(seconds=rule.cooldown_seconds)
        ]
        
        return len(recent_alerts) > 0
    
    def _create_alert(self, rule: AlertRule, metric_value: float, metadata: Dict[str, Any]) -> Alert:
        """Create a new alert instance"""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        return Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            condition=rule.condition,
            triggered_at=datetime.utcnow(),
            acknowledged_at=None,
            resolved_at=None,
            acknowledged_by=None,
            resolved_by=None,
            notification_count=0,
            metadata={**rule.metadata, **metadata}
        )

    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return

        for channel in rule.notification_channels:
            try:
                config = self.notification_configs.get(channel)
                if not config or not config.enabled:
                    continue

                success = await self._send_notification(alert, channel, config)
                if success:
                    alert.notification_count += 1
                    self.metrics['notifications_sent'] += 1
                else:
                    self.metrics['notification_failures'] += 1

            except Exception as e:
                logger.error(f"❌ Failed to send notification via {channel.value}: {e}")
                self.metrics['notification_failures'] += 1

    async def _send_notification(self, alert: Alert, channel: NotificationChannel,
                               config: NotificationConfig) -> bool:
        """Send notification via specific channel"""
        try:
            if channel == NotificationChannel.CONSOLE:
                return self._send_console_notification(alert)
            elif channel == NotificationChannel.REDIS:
                return await self._send_redis_notification(alert, config)
            elif channel == NotificationChannel.FILE:
                return self._send_file_notification(alert, config)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(alert, config)
            elif channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(alert, config)
            else:
                logger.warning(f"⚠️ Unsupported notification channel: {channel.value}")
                return False

        except Exception as e:
            logger.error(f"❌ Notification failed for {channel.value}: {e}")
            return False

    def _send_console_notification(self, alert: Alert) -> bool:
        """Send console notification"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }

        emoji = severity_emoji.get(alert.severity, "🔔")
        message = f"{emoji} ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.metric_name}={alert.metric_value} {alert.condition} {alert.threshold}"

        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            logger.error(message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(message)
        else:
            logger.info(message)

        return True

    async def _send_redis_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send Redis notification"""
        try:
            import redis.asyncio as aioredis

            redis_url = config.config.get('redis_url', 'redis://localhost:6379/4')
            channel = config.config.get('channel', f'alerts:{self.brain_id}')

            redis_client = aioredis.from_url(redis_url)

            alert_data = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'name': alert.name,
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'condition': alert.condition,
                'triggered_at': alert.triggered_at.isoformat(),
                'brain_id': self.brain_id,
                'metadata': alert.metadata
            }

            await redis_client.publish(channel, json.dumps(alert_data))
            await redis_client.close()

            return True

        except Exception as e:
            logger.error(f"❌ Redis notification failed: {e}")
            return False

    def _send_file_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send file notification"""
        try:
            log_file = config.config.get('log_file', f'/tmp/alerts_{self.brain_id}.log')

            alert_line = f"[{alert.triggered_at.isoformat()}] {alert.severity.value.upper()} - {alert.name}: {alert.metric_name}={alert.metric_value} {alert.condition} {alert.threshold} (ID: {alert.alert_id})\n"

            with open(log_file, 'a') as f:
                f.write(alert_line)

            return True

        except Exception as e:
            logger.error(f"❌ File notification failed: {e}")
            return False

    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp

            webhook_url = config.config.get('url')
            if not webhook_url:
                return False

            payload = {
                'alert_id': alert.alert_id,
                'name': alert.name,
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'condition': alert.condition,
                'triggered_at': alert.triggered_at.isoformat(),
                'brain_id': self.brain_id
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status < 400

        except Exception as e:
            logger.error(f"❌ Webhook notification failed: {e}")
            return False

    async def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send email notification"""
        try:
            smtp_server = config.config.get('smtp_server')
            smtp_port = config.config.get('smtp_port', 587)
            username = config.config.get('username')
            password = config.config.get('password')
            to_emails = config.config.get('to_emails', [])

            if not all([smtp_server, username, password, to_emails]):
                return False

            subject = f"[{alert.severity.value.upper()}] {alert.name}"
            body = f"""
Alert Details:
- Name: {alert.name}
- Severity: {alert.severity.value.upper()}
- Metric: {alert.metric_name} = {alert.metric_value}
- Condition: {alert.condition} {alert.threshold}
- Triggered: {alert.triggered_at.isoformat()}
- Brain: {self.brain_id}
- Alert ID: {alert.alert_id}

Description: {alert.description}
"""

            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            logger.error(f"❌ Email notification failed: {e}")
            return False

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            with self._lock:
                alert = self.active_alerts.get(alert_id)
                if alert and alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.utcnow()
                    alert.acknowledged_by = acknowledged_by
                    self.metrics['alerts_acknowledged'] += 1

                    logger.info(f"✅ Alert acknowledged: {alert.name} by {acknowledged_by}")
                    return True
                return False

        except Exception as e:
            logger.error(f"❌ Failed to acknowledge alert {alert_id}: {e}")
            return False

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        try:
            with self._lock:
                alert = self.active_alerts.get(alert_id)
                if alert:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    alert.resolved_by = resolved_by

                    # Remove from active alerts
                    del self.active_alerts[alert_id]
                    self.metrics['alerts_resolved'] += 1

                    logger.info(f"✅ Alert resolved: {alert.name} by {resolved_by}")
                    return True
                return False

        except Exception as e:
            logger.error(f"❌ Failed to resolve alert {alert_id}: {e}")
            return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at > cutoff_time]

    def get_metrics(self) -> Dict[str, Any]:
        """Get alert manager metrics"""
        with self._lock:
            return {
                **self.metrics,
                'active_alerts_count': len(self.active_alerts),
                'alert_rules_count': len(self.alert_rules),
                'suppressed_rules_count': len(self.suppressed_rules)
            }

    def suppress_rule(self, rule_id: str, duration_seconds: int = 3600):
        """Suppress an alert rule for specified duration"""
        with self._lock:
            self.suppressed_rules.add(rule_id)

            # Schedule unsuppression
            def unsuppress():
                time.sleep(duration_seconds)
                with self._lock:
                    self.suppressed_rules.discard(rule_id)
                logger.info(f"🔓 Alert rule unsuppressed: {rule_id}")

            threading.Thread(target=unsuppress, daemon=True).start()
            logger.info(f"🔇 Alert rule suppressed for {duration_seconds}s: {rule_id}")


def create_alert_manager(brain_id: str) -> AlertManager:
    """Factory function to create an AlertManager instance"""
    return AlertManager(brain_id)
