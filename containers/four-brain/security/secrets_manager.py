#!/usr/bin/env python3
"""
Four-Brain System Secrets Manager
Production-grade secrets management with encryption and rotation
Version: Production v1.0
"""

import os
import sys
import json
import base64
import hashlib
import secrets
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecretsManager:
    """Production secrets management system"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.secrets_file = '/app/security/secrets.enc'
        self.metadata_file = '/app/security/secrets_metadata.json'
        
        # Initialize encryption
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = os.getenv('MASTER_KEY', '').encode()
            if not self.master_key:
                raise ValueError("Master key required for secrets encryption")
        
        self.fernet = self._create_fernet()
        self._ensure_secrets_directory()
        
        logger.info("Secrets manager initialized")
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance"""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'four_brain_salt_2024',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def _ensure_secrets_directory(self):
        """Ensure secrets directory exists with proper permissions"""
        secrets_dir = os.path.dirname(self.secrets_file)
        os.makedirs(secrets_dir, mode=0o700, exist_ok=True)
    
    def _load_secrets(self) -> Dict[str, Any]:
        """Load and decrypt secrets from file"""
        if not os.path.exists(self.secrets_file):
            return {}
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return {}
    
    def _save_secrets(self, secrets: Dict[str, Any]):
        """Encrypt and save secrets to file"""
        try:
            # Encrypt secrets
            secrets_json = json.dumps(secrets, indent=2)
            encrypted_data = self.fernet.encrypt(secrets_json.encode())
            
            # Write to file with secure permissions
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set secure file permissions
            os.chmod(self.secrets_file, 0o600)
            
            # Update metadata
            self._update_metadata()
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise
    
    def _update_metadata(self):
        """Update secrets metadata"""
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'version': 1,
            'encryption': 'Fernet',
            'checksum': self._calculate_checksum()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        os.chmod(self.metadata_file, 0o600)
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of secrets file"""
        if not os.path.exists(self.secrets_file):
            return ""
        
        with open(self.secrets_file, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def set_secret(self, key: str, value: str, metadata: Optional[Dict] = None):
        """Set a secret with optional metadata"""
        secrets = self._load_secrets()
        
        secret_data = {
            'value': value,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        secrets[key] = secret_data
        self._save_secrets(secrets)
        
        logger.info(f"Secret '{key}' updated")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value"""
        secrets = self._load_secrets()
        secret_data = secrets.get(key)
        
        if secret_data:
            return secret_data.get('value')
        
        return None
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        secrets = self._load_secrets()
        
        if key in secrets:
            del secrets[key]
            self._save_secrets(secrets)
            logger.info(f"Secret '{key}' deleted")
            return True
        
        return False
    
    def list_secrets(self) -> Dict[str, Dict]:
        """List all secrets (without values)"""
        secrets = self._load_secrets()
        
        return {
            key: {
                'created_at': data.get('created_at'),
                'metadata': data.get('metadata', {})
            }
            for key, data in secrets.items()
        }
    
    def rotate_secret(self, key: str, new_value: str):
        """Rotate a secret (keep old value for rollback)"""
        secrets = self._load_secrets()
        
        if key in secrets:
            old_secret = secrets[key]
            
            # Create new secret with rotation metadata
            new_secret = {
                'value': new_value,
                'created_at': datetime.now().isoformat(),
                'metadata': old_secret.get('metadata', {}),
                'previous_value': old_secret.get('value'),
                'rotated_at': datetime.now().isoformat()
            }
            
            secrets[key] = new_secret
            self._save_secrets(secrets)
            
            logger.info(f"Secret '{key}' rotated")
        else:
            raise KeyError(f"Secret '{key}' not found")
    
    def generate_secret(self, key: str, length: int = 32, metadata: Optional[Dict] = None):
        """Generate a new random secret"""
        # Generate cryptographically secure random string
        secret_value = secrets.token_urlsafe(length)
        self.set_secret(key, secret_value, metadata)
        return secret_value
    
    def export_secrets(self, output_file: str, include_values: bool = False):
        """Export secrets to file (for backup)"""
        secrets = self._load_secrets()
        
        if not include_values:
            # Export metadata only
            export_data = {
                key: {
                    'created_at': data.get('created_at'),
                    'metadata': data.get('metadata', {})
                }
                for key, data in secrets.items()
            }
        else:
            export_data = secrets
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        os.chmod(output_file, 0o600)
        logger.info(f"Secrets exported to {output_file}")

class ProductionSecretsManager:
    """Production-specific secrets management"""
    
    def __init__(self):
        self.secrets_manager = SecretsManager()
        self._initialize_production_secrets()
    
    def _initialize_production_secrets(self):
        """Initialize production secrets if they don't exist"""
        required_secrets = [
            'jwt_secret_key',
            'postgres_password',
            'redis_password',
            'supabase_service_key',
            'grafana_admin_password',
            'api_encryption_key'
        ]
        
        for secret_key in required_secrets:
            if not self.secrets_manager.get_secret(secret_key):
                # Generate secure random secret
                secret_value = self.secrets_manager.generate_secret(
                    secret_key,
                    length=64,
                    metadata={'auto_generated': True, 'production': True}
                )
                logger.info(f"Generated production secret: {secret_key}")
    
    def get_database_credentials(self) -> Dict[str, str]:
        """Get database credentials"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'ai_system_prod'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': self.secrets_manager.get_secret('postgres_password')
        }
    
    def get_redis_credentials(self) -> Dict[str, str]:
        """Get Redis credentials"""
        return {
            'host': os.getenv('REDIS_HOST', 'redis'),
            'port': os.getenv('REDIS_PORT', '6379'),
            'password': self.secrets_manager.get_secret('redis_password')
        }
    
    def get_jwt_config(self) -> Dict[str, str]:
        """Get JWT configuration"""
        return {
            'secret_key': self.secrets_manager.get_secret('jwt_secret_key'),
            'algorithm': os.getenv('JWT_ALGORITHM', 'HS256'),
            'expiration_hours': int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        }
    
    def rotate_all_secrets(self):
        """Rotate all production secrets"""
        secrets_to_rotate = [
            'jwt_secret_key',
            'api_encryption_key'
        ]
        
        for secret_key in secrets_to_rotate:
            new_value = secrets.token_urlsafe(64)
            self.secrets_manager.rotate_secret(secret_key, new_value)
            logger.info(f"Rotated secret: {secret_key}")

def main():
    """Main function for testing"""
    try:
        # Initialize production secrets manager
        prod_secrets = ProductionSecretsManager()
        
        # Test secret operations
        logger.info("Production secrets initialized successfully")
        
        # List secrets (without values)
        secrets_list = prod_secrets.secrets_manager.list_secrets()
        logger.info(f"Available secrets: {list(secrets_list.keys())}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
