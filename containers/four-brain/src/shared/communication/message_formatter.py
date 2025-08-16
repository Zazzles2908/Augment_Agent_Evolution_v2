"""
Message Formatter Module - Standardized Message Formatting
Ensures consistent message formats across all brains

This module provides standardized message formatting and validation
to ensure all inter-brain communications follow the same structure.

Created: 2025-07-29 AEST
Purpose: Standardize message formats across brains
Module Size: 150 lines (modular design)
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class MessageMetadata:
    """Metadata for message tracking and debugging"""
    created_at: str
    message_version: str = "2.0.0"
    encoding: str = "utf-8"
    compression: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class MessageHeaders:
    """Standard headers for all inter-brain messages"""
    message_id: str
    correlation_id: Optional[str]
    reply_to: Optional[str]
    expires_at: Optional[float]
    priority: int
    retry_count: int
    max_retries: int


class MessageFormatter:
    """
    Standardized Message Formatter
    
    Provides consistent message formatting, validation, and serialization
    across all brains in the Four-Brain system.
    """
    
    def __init__(self, brain_id: str):
        """Initialize message formatter for specific brain"""
        self.brain_id = brain_id
        self.message_version = "2.0.0"
        self.default_ttl = 300  # 5 minutes
        self.max_payload_size = 10 * 1024 * 1024  # 10MB
        
        logger.info(f"📝 Message Formatter initialized for {brain_id}")
    
    def format_request_message(self, target_brain: str, message_type: str, 
                              payload: Dict[str, Any], correlation_id: Optional[str] = None,
                              priority: int = 0, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Format a request message with standardized structure"""
        
        # Generate IDs
        message_id = self._generate_message_id()
        correlation_id = correlation_id or self._generate_correlation_id()
        
        # Calculate expiration
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds
        
        # Create headers
        headers = MessageHeaders(
            message_id=message_id,
            correlation_id=correlation_id,
            reply_to=f"{self.brain_id}:responses",
            expires_at=expires_at,
            priority=priority,
            retry_count=0,
            max_retries=3
        )
        
        # Create metadata
        metadata = MessageMetadata(
            created_at=datetime.utcnow().isoformat(),
            message_version=self.message_version
        )
        
        # Validate payload
        self._validate_payload(payload)
        
        # Create standardized message
        formatted_message = {
            "headers": asdict(headers),
            "routing": {
                "source_brain": self.brain_id,
                "target_brain": target_brain,
                "message_type": message_type,
                "timestamp": time.time()
            },
            "payload": payload,
            "metadata": asdict(metadata)
        }
        
        logger.debug(f"📝 Formatted request: {self.brain_id} → {target_brain} ({message_type})")
        return formatted_message
    
    def format_response_message(self, original_message: Dict[str, Any], 
                               response_payload: Dict[str, Any], 
                               success: bool = True, error: Optional[str] = None) -> Dict[str, Any]:
        """Format a response message based on original request"""
        
        # Extract original headers
        original_headers = original_message.get("headers", {})
        original_routing = original_message.get("routing", {})
        
        # Generate response ID
        response_id = self._generate_message_id()
        
        # Create response headers
        headers = MessageHeaders(
            message_id=response_id,
            correlation_id=original_headers.get("correlation_id"),
            reply_to=None,  # No reply expected for responses
            expires_at=None,
            priority=original_headers.get("priority", 0),
            retry_count=0,
            max_retries=0
        )
        
        # Create metadata
        metadata = MessageMetadata(
            created_at=datetime.utcnow().isoformat(),
            message_version=self.message_version
        )
        
        # Create response routing (swap source/target)
        response_routing = {
            "source_brain": self.brain_id,
            "target_brain": original_routing.get("source_brain"),
            "message_type": "response",
            "original_message_type": original_routing.get("message_type"),
            "timestamp": time.time()
        }
        
        # Create response payload with status
        formatted_payload = {
            "success": success,
            "data": response_payload if success else None,
            "error": error if not success else None,
            "processing_time_ms": 0,  # To be filled by caller
            "original_message_id": original_headers.get("message_id")
        }
        
        # Validate payload
        self._validate_payload(formatted_payload)
        
        # Create standardized response
        formatted_response = {
            "headers": asdict(headers),
            "routing": response_routing,
            "payload": formatted_payload,
            "metadata": asdict(metadata)
        }
        
        logger.debug(f"📝 Formatted response: {self.brain_id} → {response_routing['target_brain']}")
        return formatted_response
    
    def format_broadcast_message(self, message_type: str, payload: Dict[str, Any],
                                priority: int = 0) -> Dict[str, Any]:
        """Format a broadcast message to all brains"""
        
        # Generate IDs
        message_id = self._generate_message_id()
        correlation_id = self._generate_correlation_id()
        
        # Create headers
        headers = MessageHeaders(
            message_id=message_id,
            correlation_id=correlation_id,
            reply_to=None,  # No replies expected for broadcasts
            expires_at=time.time() + self.default_ttl,
            priority=priority,
            retry_count=0,
            max_retries=1  # Limited retries for broadcasts
        )
        
        # Create metadata
        metadata = MessageMetadata(
            created_at=datetime.utcnow().isoformat(),
            message_version=self.message_version
        )
        
        # Create broadcast routing
        routing = {
            "source_brain": self.brain_id,
            "target_brain": "broadcast",
            "message_type": message_type,
            "timestamp": time.time()
        }
        
        # Validate payload
        self._validate_payload(payload)
        
        # Create standardized broadcast
        formatted_broadcast = {
            "headers": asdict(headers),
            "routing": routing,
            "payload": payload,
            "metadata": asdict(metadata)
        }
        
        logger.debug(f"📝 Formatted broadcast: {self.brain_id} → all ({message_type})")
        return formatted_broadcast
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message format and structure"""
        try:
            # Check required top-level keys
            required_keys = ["headers", "routing", "payload", "metadata"]
            for key in required_keys:
                if key not in message:
                    logger.error(f"❌ Missing required key: {key}")
                    return False
            
            # Validate headers
            headers = message["headers"]
            required_header_keys = ["message_id", "correlation_id"]
            for key in required_header_keys:
                if key not in headers:
                    logger.error(f"❌ Missing required header: {key}")
                    return False
            
            # Validate routing
            routing = message["routing"]
            required_routing_keys = ["source_brain", "target_brain", "message_type"]
            for key in required_routing_keys:
                if key not in routing:
                    logger.error(f"❌ Missing required routing key: {key}")
                    return False
            
            # Validate payload
            if not self._validate_payload(message["payload"]):
                return False
            
            # Check message expiration
            expires_at = headers.get("expires_at")
            if expires_at and time.time() > expires_at:
                logger.warning(f"⚠️ Message expired: {headers['message_id']}")
                return False
            
            logger.debug(f"✅ Message validation passed: {headers['message_id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Message validation failed: {e}")
            return False
    
    def serialize_message(self, message: Dict[str, Any]) -> str:
        """Serialize message to JSON string"""
        try:
            return json.dumps(message, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            logger.error(f"❌ Message serialization failed: {e}")
            raise
    
    def deserialize_message(self, message_json: str) -> Dict[str, Any]:
        """Deserialize message from JSON string"""
        try:
            message = json.loads(message_json)
            if not self.validate_message(message):
                raise ValueError("Invalid message format")
            return message
        except Exception as e:
            logger.error(f"❌ Message deserialization failed: {e}")
            raise
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"{self.brain_id}_{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for message tracking"""
        return f"corr_{int(time.time())}_{uuid.uuid4().hex[:12]}"
    
    def _validate_payload(self, payload: Any) -> bool:
        """Validate payload size and structure"""
        try:
            # Check payload size
            payload_json = json.dumps(payload)
            if len(payload_json.encode('utf-8')) > self.max_payload_size:
                logger.error(f"❌ Payload too large: {len(payload_json)} bytes")
                return False
            
            # Check for required payload structure (if it's a dict)
            if isinstance(payload, dict):
                # Payload is valid dict
                return True
            elif payload is None:
                # Null payload is acceptable
                return True
            else:
                # Other types need to be serializable
                json.dumps(payload)
                return True
                
        except Exception as e:
            logger.error(f"❌ Payload validation failed: {e}")
            return False
    
    def get_formatter_stats(self) -> Dict[str, Any]:
        """Get formatter statistics"""
        return {
            "brain_id": self.brain_id,
            "message_version": self.message_version,
            "default_ttl": self.default_ttl,
            "max_payload_size": self.max_payload_size
        }


# Factory function for easy creation
def create_message_formatter(brain_id: str) -> MessageFormatter:
    """Factory function to create message formatter"""
    return MessageFormatter(brain_id)
