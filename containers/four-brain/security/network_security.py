#!/usr/bin/env python3
"""
Four-Brain System Network Security Manager
Production-grade network security with firewall rules and monitoring
Version: Production v1.0
"""

import os
import sys
import json
import logging
import ipaddress
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkZone(Enum):
    """Network security zones"""
    PUBLIC = "public"
    DMZ = "dmz"
    INTERNAL = "internal"
    MANAGEMENT = "management"
    BRAIN_SERVICES = "brain_services"

class Protocol(Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"

class Action(Enum):
    """Firewall actions"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"

@dataclass
class FirewallRule:
    """Firewall rule definition"""
    name: str
    source_zone: NetworkZone
    destination_zone: NetworkZone
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Protocol = Protocol.TCP
    action: Action = Action.ALLOW
    priority: int = 100
    description: str = ""

@dataclass
class NetworkSegment:
    """Network segment configuration"""
    name: str
    zone: NetworkZone
    cidr: str
    vlan_id: Optional[int] = None
    gateway: Optional[str] = None
    dns_servers: List[str] = None
    allowed_services: List[str] = None

class NetworkSecurityManager:
    """Network security management system"""
    
    def __init__(self):
        self.config_file = '/app/security/network_config.json'
        self.rules_file = '/app/security/firewall_rules.json'
        
        # Default network segments
        self.network_segments = {
            'public': NetworkSegment(
                name='public',
                zone=NetworkZone.PUBLIC,
                cidr='0.0.0.0/0',
                allowed_services=['nginx', 'load_balancer']
            ),
            'dmz': NetworkSegment(
                name='dmz',
                zone=NetworkZone.DMZ,
                cidr='172.20.1.0/24',
                allowed_services=['nginx', 'grafana', 'prometheus']
            ),
            'internal': NetworkSegment(
                name='internal',
                zone=NetworkZone.INTERNAL,
                cidr='172.20.2.0/24',
                allowed_services=['postgres', 'redis']
            ),
            'brain_services': NetworkSegment(
                name='brain_services',
                zone=NetworkZone.BRAIN_SERVICES,
                cidr='172.20.3.0/24',
                allowed_services=['brain1', 'brain2', 'brain3', 'brain4']
            ),
            'management': NetworkSegment(
                name='management',
                zone=NetworkZone.MANAGEMENT,
                cidr='172.20.4.0/24',
                allowed_services=['monitoring', 'logging', 'backup']
            )
        }
        
        # Initialize firewall rules
        self.firewall_rules = self._create_default_firewall_rules()
        
        # Trusted IP ranges
        self.trusted_ips = set([
            '127.0.0.1/32',      # Localhost
            '172.20.0.0/16',     # Docker network
            '10.0.0.0/8',        # Private network
            '192.168.0.0/16'     # Private network
        ])
        
        # Rate limiting configuration
        self.rate_limits = {
            'api_requests': {'limit': 1000, 'window': 3600},  # 1000 requests per hour
            'auth_attempts': {'limit': 5, 'window': 300},     # 5 attempts per 5 minutes
            'brain_requests': {'limit': 10000, 'window': 3600} # 10000 requests per hour
        }
        
        logger.info("Network security manager initialized")
    
    def _create_default_firewall_rules(self) -> List[FirewallRule]:
        """Create default firewall rules"""
        rules = [
            # Public to DMZ rules
            FirewallRule(
                name="public_to_nginx",
                source_zone=NetworkZone.PUBLIC,
                destination_zone=NetworkZone.DMZ,
                destination_port=80,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=10,
                description="Allow HTTP traffic to Nginx"
            ),
            FirewallRule(
                name="public_to_nginx_ssl",
                source_zone=NetworkZone.PUBLIC,
                destination_zone=NetworkZone.DMZ,
                destination_port=443,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=10,
                description="Allow HTTPS traffic to Nginx"
            ),
            
            # DMZ to Brain Services rules
            FirewallRule(
                name="dmz_to_brain_services",
                source_zone=NetworkZone.DMZ,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                destination_port=8001,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=20,
                description="Allow DMZ to Brain 1"
            ),
            FirewallRule(
                name="dmz_to_brain2",
                source_zone=NetworkZone.DMZ,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                destination_port=8002,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=20,
                description="Allow DMZ to Brain 2"
            ),
            FirewallRule(
                name="dmz_to_brain3",
                source_zone=NetworkZone.DMZ,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                destination_port=8003,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=20,
                description="Allow DMZ to Brain 3"
            ),
            FirewallRule(
                name="dmz_to_brain4",
                source_zone=NetworkZone.DMZ,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                destination_port=8004,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=20,
                description="Allow DMZ to Brain 4"
            ),
            
            # Brain Services to Internal rules
            FirewallRule(
                name="brain_to_postgres",
                source_zone=NetworkZone.BRAIN_SERVICES,
                destination_zone=NetworkZone.INTERNAL,
                destination_port=5432,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=30,
                description="Allow Brain services to PostgreSQL"
            ),
            FirewallRule(
                name="brain_to_redis",
                source_zone=NetworkZone.BRAIN_SERVICES,
                destination_zone=NetworkZone.INTERNAL,
                destination_port=6379,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=30,
                description="Allow Brain services to Redis"
            ),
            
            # Management zone rules
            FirewallRule(
                name="management_to_all",
                source_zone=NetworkZone.MANAGEMENT,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=40,
                description="Allow management access to all services"
            ),
            
            # Inter-brain communication
            FirewallRule(
                name="brain_to_brain",
                source_zone=NetworkZone.BRAIN_SERVICES,
                destination_zone=NetworkZone.BRAIN_SERVICES,
                protocol=Protocol.TCP,
                action=Action.ALLOW,
                priority=50,
                description="Allow inter-brain communication"
            ),
            
            # Default deny rule
            FirewallRule(
                name="default_deny",
                source_zone=NetworkZone.PUBLIC,
                destination_zone=NetworkZone.INTERNAL,
                action=Action.DENY,
                priority=1000,
                description="Default deny all other traffic"
            )
        ]
        
        return rules
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def is_trusted_ip(self, ip: str) -> bool:
        """Check if IP is in trusted ranges"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for trusted_range in self.trusted_ips:
                if ip_addr in ipaddress.ip_network(trusted_range):
                    return True
            return False
        except ValueError:
            return False
    
    def add_trusted_ip(self, ip_range: str):
        """Add IP range to trusted list"""
        try:
            # Validate IP range
            ipaddress.ip_network(ip_range)
            self.trusted_ips.add(ip_range)
            logger.info(f"Added trusted IP range: {ip_range}")
        except ValueError as e:
            logger.error(f"Invalid IP range {ip_range}: {e}")
            raise
    
    def remove_trusted_ip(self, ip_range: str):
        """Remove IP range from trusted list"""
        if ip_range in self.trusted_ips:
            self.trusted_ips.remove(ip_range)
            logger.info(f"Removed trusted IP range: {ip_range}")
        else:
            logger.warning(f"IP range not found: {ip_range}")
    
    def add_firewall_rule(self, rule: FirewallRule):
        """Add firewall rule"""
        self.firewall_rules.append(rule)
        self.firewall_rules.sort(key=lambda r: r.priority)
        logger.info(f"Added firewall rule: {rule.name}")
    
    def remove_firewall_rule(self, rule_name: str):
        """Remove firewall rule by name"""
        self.firewall_rules = [r for r in self.firewall_rules if r.name != rule_name]
        logger.info(f"Removed firewall rule: {rule_name}")
    
    def check_access(self, source_ip: str, destination_ip: str, 
                    destination_port: int, protocol: Protocol = Protocol.TCP) -> bool:
        """Check if access is allowed based on firewall rules"""
        # Determine zones based on IP addresses
        source_zone = self._get_zone_for_ip(source_ip)
        destination_zone = self._get_zone_for_ip(destination_ip)
        
        # Check rules in priority order
        for rule in sorted(self.firewall_rules, key=lambda r: r.priority):
            if self._rule_matches(rule, source_zone, destination_zone, 
                                source_ip, destination_ip, destination_port, protocol):
                if rule.action == Action.ALLOW:
                    logger.debug(f"Access allowed by rule: {rule.name}")
                    return True
                elif rule.action == Action.DENY:
                    logger.warning(f"Access denied by rule: {rule.name}")
                    return False
        
        # Default deny
        logger.warning(f"Access denied: no matching rule for {source_ip}:{destination_port}")
        return False
    
    def _get_zone_for_ip(self, ip: str) -> NetworkZone:
        """Determine network zone for IP address"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            for segment in self.network_segments.values():
                if ip_addr in ipaddress.ip_network(segment.cidr):
                    return segment.zone
            
            # Default to public if not in any defined segment
            return NetworkZone.PUBLIC
            
        except ValueError:
            return NetworkZone.PUBLIC
    
    def _rule_matches(self, rule: FirewallRule, source_zone: NetworkZone, 
                     destination_zone: NetworkZone, source_ip: str, 
                     destination_ip: str, destination_port: int, 
                     protocol: Protocol) -> bool:
        """Check if firewall rule matches the connection"""
        # Check zones
        if rule.source_zone != source_zone or rule.destination_zone != destination_zone:
            return False
        
        # Check protocol
        if rule.protocol != protocol:
            return False
        
        # Check destination port
        if rule.destination_port and rule.destination_port != destination_port:
            return False
        
        # Check source IP if specified
        if rule.source_ip:
            try:
                if ipaddress.ip_address(source_ip) not in ipaddress.ip_network(rule.source_ip):
                    return False
            except ValueError:
                return False
        
        # Check destination IP if specified
        if rule.destination_ip:
            try:
                if ipaddress.ip_address(destination_ip) not in ipaddress.ip_network(rule.destination_ip):
                    return False
            except ValueError:
                return False
        
        return True
    
    def generate_iptables_rules(self) -> List[str]:
        """Generate iptables rules from firewall configuration"""
        iptables_rules = [
            "# Four-Brain System Firewall Rules",
            "# Generated automatically - do not edit manually",
            "",
            "# Flush existing rules",
            "iptables -F",
            "iptables -X",
            "iptables -t nat -F",
            "iptables -t nat -X",
            "",
            "# Set default policies",
            "iptables -P INPUT DROP",
            "iptables -P FORWARD DROP",
            "iptables -P OUTPUT ACCEPT",
            "",
            "# Allow loopback traffic",
            "iptables -A INPUT -i lo -j ACCEPT",
            "iptables -A OUTPUT -o lo -j ACCEPT",
            "",
            "# Allow established connections",
            "iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
            ""
        ]
        
        # Add rules for each firewall rule
        for rule in sorted(self.firewall_rules, key=lambda r: r.priority):
            if rule.action == Action.ALLOW:
                iptables_cmd = self._generate_iptables_command(rule)
                if iptables_cmd:
                    iptables_rules.append(f"# {rule.description}")
                    iptables_rules.append(iptables_cmd)
                    iptables_rules.append("")
        
        return iptables_rules
    
    def _generate_iptables_command(self, rule: FirewallRule) -> str:
        """Generate iptables command for a firewall rule"""
        cmd_parts = ["iptables", "-A", "INPUT"]
        
        # Add protocol
        if rule.protocol != Protocol.ICMP:
            cmd_parts.extend(["-p", rule.protocol.value])
        
        # Add source IP
        if rule.source_ip:
            cmd_parts.extend(["-s", rule.source_ip])
        
        # Add destination port
        if rule.destination_port:
            cmd_parts.extend(["--dport", str(rule.destination_port)])
        
        # Add action
        if rule.action == Action.ALLOW:
            cmd_parts.extend(["-j", "ACCEPT"])
        elif rule.action == Action.DENY:
            cmd_parts.extend(["-j", "DROP"])
        
        return " ".join(cmd_parts)
    
    def save_configuration(self):
        """Save network security configuration to file"""
        config = {
            'network_segments': {
                name: asdict(segment) for name, segment in self.network_segments.items()
            },
            'firewall_rules': [asdict(rule) for rule in self.firewall_rules],
            'trusted_ips': list(self.trusted_ips),
            'rate_limits': self.rate_limits
        }
        
        # Convert enums to strings for JSON serialization
        for segment_data in config['network_segments'].values():
            segment_data['zone'] = segment_data['zone'].value if hasattr(segment_data['zone'], 'value') else segment_data['zone']
        
        for rule_data in config['firewall_rules']:
            rule_data['source_zone'] = rule_data['source_zone'].value if hasattr(rule_data['source_zone'], 'value') else rule_data['source_zone']
            rule_data['destination_zone'] = rule_data['destination_zone'].value if hasattr(rule_data['destination_zone'], 'value') else rule_data['destination_zone']
            rule_data['protocol'] = rule_data['protocol'].value if hasattr(rule_data['protocol'], 'value') else rule_data['protocol']
            rule_data['action'] = rule_data['action'].value if hasattr(rule_data['action'], 'value') else rule_data['action']
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Network security configuration saved to {self.config_file}")

def main():
    """Main function for testing"""
    try:
        net_security = NetworkSecurityManager()
        
        # Test access check
        allowed = net_security.check_access(
            source_ip="172.20.1.10",
            destination_ip="172.20.3.10",
            destination_port=8001,
            protocol=Protocol.TCP
        )
        logger.info(f"Access check result: {allowed}")
        
        # Generate iptables rules
        iptables_rules = net_security.generate_iptables_rules()
        logger.info(f"Generated {len(iptables_rules)} iptables rules")
        
        # Save configuration
        net_security.save_configuration()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
