#!/usr/bin/env python3.11
"""
SSL Certificate Manager for Four-Brain System
Production-ready SSL certificate management with Let's Encrypt and monitoring

Author: AugmentAI
Date: 2025-08-02
Purpose: Manage SSL certificates for nginx reverse proxy and secure communications
"""

import os
import sys
import logging
import time
import subprocess
import ssl
import socket
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import cryptography
from cryptography import x509
from cryptography.x509.oid import NameOID
import OpenSSL

# Configure logging
logger = logging.getLogger(__name__)

class CertificateType(Enum):
    """SSL certificate types"""
    SELF_SIGNED = "self_signed"
    LETS_ENCRYPT = "lets_encrypt"
    CUSTOM_CA = "custom_ca"
    WILDCARD = "wildcard"

class CertificateStatus(Enum):
    """Certificate status"""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    INVALID = "invalid"
    NOT_FOUND = "not_found"

@dataclass
class SSLCertificateConfig:
    """SSL certificate configuration"""
    domain: str
    cert_type: CertificateType
    cert_path: str
    key_path: str
    ca_path: Optional[str] = None
    email: Optional[str] = None  # For Let's Encrypt
    staging: bool = False  # Let's Encrypt staging
    auto_renew: bool = True
    renewal_days_before: int = 30
    key_size: int = 2048

@dataclass
class CertificateInfo:
    """SSL certificate information"""
    domain: str
    issuer: str
    subject: str
    valid_from: datetime
    valid_until: datetime
    days_until_expiry: int
    status: CertificateStatus
    fingerprint: str
    key_size: int
    signature_algorithm: str

class SSLCertificateManager:
    """Manages SSL certificates for Four-Brain system"""
    
    def __init__(self, config: SSLCertificateConfig):
        self.config = config
        self.certbot_available = self._check_certbot_availability()
        self.openssl_available = self._check_openssl_availability()
        
        # Certificate monitoring
        self.last_check_time = 0.0
        self.check_interval = 3600  # 1 hour
        
        logger.info("🔐 SSL Certificate Manager initialized")
        logger.info(f"  Domain: {config.domain}")
        logger.info(f"  Type: {config.cert_type.value}")
        logger.info(f"  Auto-renew: {config.auto_renew}")
    
    def _check_certbot_availability(self) -> bool:
        """Check if certbot is available for Let's Encrypt"""
        try:
            result = subprocess.run(
                ["certbot", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ Certbot available: {result.stdout.strip()}")
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Certbot not available")
            return False
    
    def _check_openssl_availability(self) -> bool:
        """Check if OpenSSL is available"""
        try:
            result = subprocess.run(
                ["openssl", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ OpenSSL available: {result.stdout.strip()}")
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ OpenSSL not available")
            return False
    
    def generate_self_signed_certificate(self) -> bool:
        """Generate self-signed SSL certificate"""
        try:
            if not self.openssl_available:
                logger.error("❌ OpenSSL not available for self-signed certificate generation")
                return False
            
            logger.info(f"🔧 Generating self-signed certificate for {self.config.domain}")
            
            # Create directories
            os.makedirs(os.path.dirname(self.config.cert_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.key_path), exist_ok=True)
            
            # Generate private key
            key_cmd = [
                "openssl", "genrsa",
                "-out", self.config.key_path,
                str(self.config.key_size)
            ]
            
            result = subprocess.run(key_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"❌ Private key generation failed: {result.stderr}")
                return False
            
            # Generate certificate
            cert_cmd = [
                "openssl", "req", "-new", "-x509",
                "-key", self.config.key_path,
                "-out", self.config.cert_path,
                "-days", "365",
                "-subj", f"/CN={self.config.domain}/O=Four-Brain System/C=US"
            ]
            
            result = subprocess.run(cert_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"❌ Certificate generation failed: {result.stderr}")
                return False
            
            # Set proper permissions
            os.chmod(self.config.key_path, 0o600)
            os.chmod(self.config.cert_path, 0o644)
            
            logger.info(f"✅ Self-signed certificate generated for {self.config.domain}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Self-signed certificate generation failed: {str(e)}")
            return False
    
    def obtain_lets_encrypt_certificate(self) -> bool:
        """Obtain Let's Encrypt SSL certificate"""
        try:
            if not self.certbot_available:
                logger.error("❌ Certbot not available for Let's Encrypt certificate")
                return False
            
            if not self.config.email:
                logger.error("❌ Email required for Let's Encrypt certificate")
                return False
            
            logger.info(f"🔧 Obtaining Let's Encrypt certificate for {self.config.domain}")
            
            # Build certbot command
            certbot_cmd = [
                "certbot", "certonly",
                "--webroot",
                "--webroot-path", "/var/www/certbot",
                "--email", self.config.email,
                "--agree-tos",
                "--no-eff-email",
                "-d", self.config.domain
            ]
            
            if self.config.staging:
                certbot_cmd.append("--staging")
                logger.info("🧪 Using Let's Encrypt staging environment")
            
            # Execute certbot
            result = subprocess.run(
                certbot_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                # Copy certificates to configured paths
                le_cert_path = f"/etc/letsencrypt/live/{self.config.domain}/fullchain.pem"
                le_key_path = f"/etc/letsencrypt/live/{self.config.domain}/privkey.pem"
                
                if os.path.exists(le_cert_path) and os.path.exists(le_key_path):
                    # Create directories
                    os.makedirs(os.path.dirname(self.config.cert_path), exist_ok=True)
                    os.makedirs(os.path.dirname(self.config.key_path), exist_ok=True)
                    
                    # Copy certificates
                    subprocess.run(["cp", le_cert_path, self.config.cert_path])
                    subprocess.run(["cp", le_key_path, self.config.key_path])
                    
                    # Set permissions
                    os.chmod(self.config.key_path, 0o600)
                    os.chmod(self.config.cert_path, 0o644)
                    
                    logger.info(f"✅ Let's Encrypt certificate obtained for {self.config.domain}")
                    return True
                else:
                    logger.error("❌ Let's Encrypt certificate files not found")
                    return False
            else:
                logger.error(f"❌ Certbot failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Let's Encrypt certificate generation failed: {str(e)}")
            return False
    
    def renew_certificate(self) -> bool:
        """Renew SSL certificate"""
        try:
            if self.config.cert_type == CertificateType.LETS_ENCRYPT:
                return self._renew_lets_encrypt_certificate()
            elif self.config.cert_type == CertificateType.SELF_SIGNED:
                return self.generate_self_signed_certificate()
            else:
                logger.warning(f"⚠️ Certificate renewal not supported for type: {self.config.cert_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Certificate renewal failed: {str(e)}")
            return False
    
    def _renew_lets_encrypt_certificate(self) -> bool:
        """Renew Let's Encrypt certificate"""
        try:
            if not self.certbot_available:
                return False
            
            logger.info(f"🔄 Renewing Let's Encrypt certificate for {self.config.domain}")
            
            # Renew certificate
            renew_cmd = ["certbot", "renew", "--quiet"]
            
            result = subprocess.run(
                renew_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Let's Encrypt certificate renewed for {self.config.domain}")
                
                # Reload nginx if available
                self._reload_nginx()
                
                return True
            else:
                logger.error(f"❌ Certificate renewal failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Let's Encrypt renewal failed: {str(e)}")
            return False
    
    def _reload_nginx(self):
        """Reload nginx configuration"""
        try:
            result = subprocess.run(
                ["nginx", "-s", "reload"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("✅ Nginx reloaded successfully")
            else:
                logger.warning(f"⚠️ Nginx reload failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"⚠️ Nginx reload error: {str(e)}")
    
    def get_certificate_info(self) -> Optional[CertificateInfo]:
        """Get SSL certificate information"""
        try:
            if not os.path.exists(self.config.cert_path):
                logger.warning(f"⚠️ Certificate file not found: {self.config.cert_path}")
                return None
            
            # Load certificate
            with open(self.config.cert_path, 'rb') as cert_file:
                cert_data = cert_file.read()
            
            # Parse certificate
            cert = x509.load_pem_x509_certificate(cert_data)
            
            # Extract information
            subject = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            issuer = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            
            valid_from = cert.not_valid_before
            valid_until = cert.not_valid_after
            
            # Calculate days until expiry
            now = datetime.now()
            days_until_expiry = (valid_until - now).days
            
            # Determine status
            if days_until_expiry < 0:
                status = CertificateStatus.EXPIRED
            elif days_until_expiry <= self.config.renewal_days_before:
                status = CertificateStatus.EXPIRING_SOON
            else:
                status = CertificateStatus.VALID
            
            # Get fingerprint
            fingerprint = cert.fingerprint(cryptography.hazmat.primitives.hashes.SHA256()).hex()
            
            # Get key size
            public_key = cert.public_key()
            key_size = public_key.key_size if hasattr(public_key, 'key_size') else 0
            
            # Get signature algorithm
            signature_algorithm = cert.signature_algorithm_oid._name
            
            cert_info = CertificateInfo(
                domain=self.config.domain,
                issuer=issuer,
                subject=subject,
                valid_from=valid_from,
                valid_until=valid_until,
                days_until_expiry=days_until_expiry,
                status=status,
                fingerprint=fingerprint,
                key_size=key_size,
                signature_algorithm=signature_algorithm
            )
            
            logger.debug(f"📋 Certificate info retrieved for {self.config.domain}")
            return cert_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get certificate info: {str(e)}")
            return None
    
    def validate_certificate_chain(self) -> Dict[str, Any]:
        """Validate SSL certificate chain"""
        try:
            validation_result = {
                "valid": False,
                "issues": [],
                "chain_length": 0,
                "root_ca_trusted": False
            }
            
            if not os.path.exists(self.config.cert_path):
                validation_result["issues"].append("Certificate file not found")
                return validation_result
            
            # Test SSL connection to domain
            try:
                context = ssl.create_default_context()
                with socket.create_connection((self.config.domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=self.config.domain) as ssock:
                        cert = ssock.getpeercert()
                        validation_result["valid"] = True
                        validation_result["chain_length"] = len(ssock.getpeercert_chain() or [])
                        validation_result["root_ca_trusted"] = True
                        
                        logger.info(f"✅ Certificate chain validation successful for {self.config.domain}")
                        
            except ssl.SSLError as e:
                validation_result["issues"].append(f"SSL validation failed: {str(e)}")
            except Exception as e:
                validation_result["issues"].append(f"Connection failed: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Certificate chain validation failed: {str(e)}")
            return {"valid": False, "issues": [str(e)]}
    
    def check_certificate_expiry(self) -> bool:
        """Check if certificate needs renewal"""
        cert_info = self.get_certificate_info()
        
        if not cert_info:
            logger.warning("⚠️ Cannot check expiry - certificate info unavailable")
            return True  # Assume renewal needed
        
        if cert_info.status in [CertificateStatus.EXPIRED, CertificateStatus.EXPIRING_SOON]:
            logger.warning(f"⚠️ Certificate expiring in {cert_info.days_until_expiry} days")
            return True
        
        logger.info(f"✅ Certificate valid for {cert_info.days_until_expiry} days")
        return False
    
    def auto_renew_if_needed(self) -> bool:
        """Automatically renew certificate if needed"""
        if not self.config.auto_renew:
            logger.info("🔒 Auto-renewal disabled")
            return False
        
        # Check if renewal is needed
        if not self.check_certificate_expiry():
            return False
        
        logger.info(f"🔄 Auto-renewing certificate for {self.config.domain}")
        return self.renew_certificate()
    
    def get_ssl_security_report(self) -> Dict[str, Any]:
        """Generate SSL security report"""
        cert_info = self.get_certificate_info()
        chain_validation = self.validate_certificate_chain()
        
        report = {
            "timestamp": time.time(),
            "domain": self.config.domain,
            "certificate_type": self.config.cert_type.value,
            "certificate_info": cert_info.__dict__ if cert_info else None,
            "chain_validation": chain_validation,
            "security_score": 0,
            "recommendations": []
        }
        
        # Calculate security score
        score = 0
        
        if cert_info:
            # Certificate validity
            if cert_info.status == CertificateStatus.VALID:
                score += 30
            elif cert_info.status == CertificateStatus.EXPIRING_SOON:
                score += 20
                report["recommendations"].append("Certificate expiring soon - schedule renewal")
            
            # Key size
            if cert_info.key_size >= 2048:
                score += 20
            else:
                report["recommendations"].append("Increase key size to at least 2048 bits")
            
            # Signature algorithm
            if "sha256" in cert_info.signature_algorithm.lower():
                score += 15
            else:
                report["recommendations"].append("Use SHA-256 or stronger signature algorithm")
        
        # Chain validation
        if chain_validation["valid"]:
            score += 25
        else:
            report["recommendations"].append("Fix certificate chain validation issues")
        
        # Auto-renewal
        if self.config.auto_renew:
            score += 10
        else:
            report["recommendations"].append("Enable auto-renewal for production")
        
        report["security_score"] = score
        
        logger.info(f"🔍 SSL security score: {score}/100 for {self.config.domain}")
        
        return report

def create_production_ssl_config(domain: str) -> SSLCertificateConfig:
    """Create production SSL configuration"""
    return SSLCertificateConfig(
        domain=domain,
        cert_type=CertificateType(os.getenv("SSL_CERT_TYPE", "lets_encrypt")),
        cert_path=f"/etc/ssl/certs/{domain}.crt",
        key_path=f"/etc/ssl/private/{domain}.key",
        ca_path=os.getenv("SSL_CA_PATH"),
        email=os.getenv("SSL_EMAIL"),
        staging=os.getenv("SSL_STAGING", "false").lower() == "true",
        auto_renew=os.getenv("SSL_AUTO_RENEW", "true").lower() == "true",
        renewal_days_before=int(os.getenv("SSL_RENEWAL_DAYS", "30"))
    )

# Global SSL certificate managers
_ssl_managers: Dict[str, SSLCertificateManager] = {}

def get_ssl_manager(domain: str) -> SSLCertificateManager:
    """Get SSL certificate manager for domain"""
    if domain not in _ssl_managers:
        config = create_production_ssl_config(domain)
        _ssl_managers[domain] = SSLCertificateManager(config)
    return _ssl_managers[domain]
