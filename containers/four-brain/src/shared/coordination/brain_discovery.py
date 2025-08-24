"""
Brain Discovery System for Four-Brain System v2
Automatic discovery and registration of brain instances

Created: 2025-07-30 AEST
Purpose: Discover and manage Brain1, Brain2, Brain3, and Brain4 instances dynamically
"""

import asyncio
import json
import logging
import time
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainType(Enum):
    """Types of AI brains in the system"""
    EMBEDDING = "embedding"  # Brain1 - Qwen3-4B Embedding
    RERANKER = "reranker"    # Brain2 - Qwen3-Reranker-4B
    Zazzles's Agent = "Zazzles's Agent"      # Brain3 - Zazzles's Agent API
    DOCLING = "docling"      # Brain4 - Docling PDF processing

class DiscoveryMethod(Enum):
    """Brain discovery methods"""
    STATIC_CONFIG = "static_config"
    NETWORK_SCAN = "network_scan"
    SERVICE_REGISTRY = "service_registry"
    HEARTBEAT = "heartbeat"
    DNS_SD = "dns_sd"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"

class BrainStatus(Enum):
    """Brain discovery status"""
    DISCOVERED = "discovered"
    REGISTERED = "registered"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    LOST = "lost"
    DEREGISTERED = "deregistered"

@dataclass
class DiscoveredBrain:
    """Discovered brain instance information"""
    brain_id: str
    brain_type: BrainType
    host: str
    port: int
    endpoints: Dict[str, str]
    capabilities: List[str]
    version: str
    status: BrainStatus
    discovered_at: datetime
    last_seen: datetime
    discovery_method: DiscoveryMethod
    health_endpoint: str
    metadata: Dict[str, Any]

@dataclass
class DiscoveryConfig:
    """Discovery configuration"""
    method: DiscoveryMethod
    enabled: bool
    scan_interval_seconds: int
    timeout_seconds: int
    retry_attempts: int
    parameters: Dict[str, Any]

class BrainDiscovery:
    """
    Comprehensive brain discovery system
    
    Features:
    - Multiple discovery methods (static, network scan, service registry)
    - Automatic brain registration and deregistration
    - Health-based brain status tracking
    - Dynamic capability detection
    - Service mesh integration
    - Load balancer integration
    - Comprehensive discovery metrics
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/19"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Discovery state
        self.discovered_brains: Dict[str, DiscoveredBrain] = {}
        self.discovery_configs: Dict[DiscoveryMethod, DiscoveryConfig] = {}
        
        # Configuration
        self.config = {
            'discovery_interval_seconds': 30,
            'brain_timeout_seconds': 120,
            'health_check_timeout': 5,
            'max_discovery_attempts': 3,
            'auto_registration_enabled': True,
            'capability_detection_enabled': True
        }
        
        # Network scanning configuration
        self.network_config = {
            'scan_ranges': ['127.0.0.1', '192.168.0.0/24', '10.0.0.0/24'],
            'port_ranges': [(8000, 8010), (9000, 9010)],
            'common_brain_ports': [8001, 8002, 8003, 8004]
        }
        
        # Static brain configurations
        self.static_brains = {
            'embedding_service': {
                'type': BrainType.EMBEDDING,
                'host': 'localhost',
                'port': 8001,
                'endpoints': {
                    'health': '/health',
                    'embed': '/embed',
                    'status': '/status'
                }
            },
            'reranker_service': {
                'type': BrainType.RERANKER,
                'host': 'localhost',
                'port': 8002,
                'endpoints': {
                    'health': '/health',
                    'rerank': '/rerank',
                    'status': '/status'
                }
            },
            'intelligence_service': {
                'type': BrainType.Zazzles's Agent,
                'host': 'localhost',
                'port': 8003,
                'endpoints': {
                    'health': '/health',
                    'chat': '/chat',
                    'status': '/status'
                }
            },
            'document_processor': {
                'type': BrainType.DOCLING,
                'host': 'localhost',
                'port': 8004,
                'endpoints': {
                    'health': '/health',
                    'process': '/process',
                    'status': '/status'
                }
            }
        }
        
        # Discovery metrics
        self.metrics = {
            'total_discoveries': 0,
            'successful_registrations': 0,
            'failed_registrations': 0,
            'active_brains': 0,
            'lost_brains': 0,
            'discovery_errors': 0
        }
        
        # HTTP session for discovery
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("🔍 Brain Discovery initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start discovery services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['health_check_timeout'])
            )
            
            # Initialize discovery configurations
            await self._initialize_discovery_configs()
            
            # Load existing discoveries
            await self._load_discovered_brains()
            
            # Start discovery services
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._cleanup_lost_brains())
            
            logger.info("✅ Brain Discovery Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Brain Discovery: {e}")
            raise
    
    async def _initialize_discovery_configs(self):
        """Initialize discovery method configurations"""
        self.discovery_configs = {
            DiscoveryMethod.STATIC_CONFIG: DiscoveryConfig(
                method=DiscoveryMethod.STATIC_CONFIG,
                enabled=True,
                scan_interval_seconds=60,
                timeout_seconds=5,
                retry_attempts=3,
                parameters={}
            ),
            DiscoveryMethod.NETWORK_SCAN: DiscoveryConfig(
                method=DiscoveryMethod.NETWORK_SCAN,
                enabled=True,
                scan_interval_seconds=300,  # 5 minutes
                timeout_seconds=2,
                retry_attempts=1,
                parameters={'aggressive_scan': False}
            ),
            DiscoveryMethod.HEARTBEAT: DiscoveryConfig(
                method=DiscoveryMethod.HEARTBEAT,
                enabled=True,
                scan_interval_seconds=30,
                timeout_seconds=5,
                retry_attempts=2,
                parameters={}
            )
        }
    
    async def _discovery_loop(self):
        """Main discovery loop"""
        while True:
            try:
                # Run enabled discovery methods
                for method, config in self.discovery_configs.items():
                    if config.enabled:
                        await self._run_discovery_method(method, config)
                
                # Wait for next discovery cycle
                await asyncio.sleep(self.config['discovery_interval_seconds'])
                
            except Exception as e:
                logger.error(f"❌ Discovery loop error: {e}")
                self.metrics['discovery_errors'] += 1
                await asyncio.sleep(5)
    
    async def _run_discovery_method(self, method: DiscoveryMethod, config: DiscoveryConfig):
        """Run specific discovery method"""
        try:
            if method == DiscoveryMethod.STATIC_CONFIG:
                await self._discover_static_brains()
            elif method == DiscoveryMethod.NETWORK_SCAN:
                await self._discover_network_brains()
            elif method == DiscoveryMethod.HEARTBEAT:
                await self._discover_heartbeat_brains()
            
        except Exception as e:
            logger.error(f"❌ Discovery method {method.value} failed: {e}")
    
    async def _discover_static_brains(self):
        """Discover brains from static configuration"""
        try:
            for brain_id, brain_config in self.static_brains.items():
                if brain_id not in self.discovered_brains:
                    # Check if brain is reachable
                    if await self._check_brain_reachability(brain_config['host'], brain_config['port']):
                        # Discover brain details
                        brain = await self._create_discovered_brain(
                            brain_id,
                            brain_config['type'],
                            brain_config['host'],
                            brain_config['port'],
                            brain_config['endpoints'],
                            DiscoveryMethod.STATIC_CONFIG
                        )
                        
                        if brain:
                            await self._register_discovered_brain(brain)
                
        except Exception as e:
            logger.error(f"❌ Static brain discovery failed: {e}")
    
    async def _discover_network_brains(self):
        """Discover brains through network scanning"""
        try:
            discovered_hosts = []
            
            # Scan configured network ranges
            for scan_range in self.network_config['scan_ranges']:
                if '/' in scan_range:
                    # CIDR notation - would implement subnet scanning
                    # For now, skip complex network scanning
                    continue
                else:
                    # Single host
                    discovered_hosts.append(scan_range)
            
            # Check common brain ports on discovered hosts
            for host in discovered_hosts:
                for port in self.network_config['common_brain_ports']:
                    if await self._check_brain_reachability(host, port):
                        # Try to identify brain type
                        brain_type = await self._identify_brain_type(host, port)
                        if brain_type:
                            brain_id = f"discovered_{host}_{port}"
                            
                            if brain_id not in self.discovered_brains:
                                brain = await self._create_discovered_brain(
                                    brain_id,
                                    brain_type,
                                    host,
                                    port,
                                    {},  # Will be detected
                                    DiscoveryMethod.NETWORK_SCAN
                                )
                                
                                if brain:
                                    await self._register_discovered_brain(brain)
            
        except Exception as e:
            logger.error(f"❌ Network brain discovery failed: {e}")
    
    async def _discover_heartbeat_brains(self):
        """Discover brains through heartbeat messages"""
        try:
            # Check Redis for heartbeat messages
            if self.redis_client:
                keys = await self.redis_client.keys("brain_heartbeat:*")
                
                for key in keys:
                    heartbeat_data = await self.redis_client.get(key)
                    if heartbeat_data:
                        heartbeat = json.loads(heartbeat_data)
                        brain_id = heartbeat.get('brain_id')
                        
                        if brain_id and brain_id not in self.discovered_brains:
                            # Create brain from heartbeat data
                            brain = DiscoveredBrain(
                                brain_id=brain_id,
                                brain_type=BrainType(heartbeat.get('brain_type', 'embedding')),
                                host=heartbeat.get('host', 'localhost'),
                                port=heartbeat.get('port', 8000),
                                endpoints=heartbeat.get('endpoints', {}),
                                capabilities=heartbeat.get('capabilities', []),
                                version=heartbeat.get('version', '1.0.0'),
                                status=BrainStatus.DISCOVERED,
                                discovered_at=datetime.now(),
                                last_seen=datetime.now(),
                                discovery_method=DiscoveryMethod.HEARTBEAT,
                                health_endpoint=heartbeat.get('health_endpoint', '/health'),
                                metadata=heartbeat.get('metadata', {})
                            )
                            
                            await self._register_discovered_brain(brain)
            
        except Exception as e:
            logger.error(f"❌ Heartbeat brain discovery failed: {e}")
    
    async def _check_brain_reachability(self, host: str, port: int) -> bool:
        """Check if brain is reachable"""
        try:
            # Try TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config['health_check_timeout'])
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
            
        except Exception as e:
            logger.debug(f"Brain reachability check failed for {host}:{port}: {e}")
            return False
    
    async def _identify_brain_type(self, host: str, port: int) -> Optional[BrainType]:
        """Identify brain type through API inspection"""
        try:
            if not self.http_session:
                return None
            
            base_url = f"http://{host}:{port}"
            
            # Try common endpoints to identify brain type
            endpoints_to_check = [
                ('/embed', BrainType.EMBEDDING),
                ('/rerank', BrainType.RERANKER),
                ('/chat', BrainType.Zazzles's Agent),
                ('/process', BrainType.DOCLING)
            ]
            
            for endpoint, brain_type in endpoints_to_check:
                try:
                    async with self.http_session.get(f"{base_url}{endpoint}") as response:
                        if response.status in [200, 405]:  # 405 = Method Not Allowed (endpoint exists)
                            return brain_type
                except:
                    continue
            
            # Try to get brain info from status endpoint
            try:
                async with self.http_session.get(f"{base_url}/status") as response:
                    if response.status == 200:
                        status_data = await response.json()
                        brain_type_str = status_data.get('brain_type', status_data.get('type'))
                        if brain_type_str:
                            return BrainType(brain_type_str)
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Brain type identification failed for {host}:{port}: {e}")
            return None
    
    async def _create_discovered_brain(self, brain_id: str, brain_type: BrainType,
                                     host: str, port: int, endpoints: Dict[str, str],
                                     discovery_method: DiscoveryMethod) -> Optional[DiscoveredBrain]:
        """Create discovered brain with full details"""
        try:
            # Detect capabilities if enabled
            capabilities = []
            if self.config['capability_detection_enabled']:
                capabilities = await self._detect_brain_capabilities(host, port, brain_type)
            
            # Get version information
            version = await self._get_brain_version(host, port)
            
            # Detect additional endpoints
            if not endpoints:
                endpoints = await self._detect_brain_endpoints(host, port, brain_type)
            
            brain = DiscoveredBrain(
                brain_id=brain_id,
                brain_type=brain_type,
                host=host,
                port=port,
                endpoints=endpoints,
                capabilities=capabilities,
                version=version,
                status=BrainStatus.DISCOVERED,
                discovered_at=datetime.now(),
                last_seen=datetime.now(),
                discovery_method=discovery_method,
                health_endpoint=endpoints.get('health', '/health'),
                metadata={}
            )
            
            return brain
            
        except Exception as e:
            logger.error(f"❌ Failed to create discovered brain: {e}")
            return None
    
    async def _detect_brain_capabilities(self, host: str, port: int, brain_type: BrainType) -> List[str]:
        """Detect brain capabilities"""
        try:
            capabilities = []
            
            # Default capabilities based on brain type
            if brain_type == BrainType.EMBEDDING:
                capabilities = ['embedding', 'encode', 'vector_search']
            elif brain_type == BrainType.RERANKER:
                capabilities = ['rerank', 'score', 'ranking']
            elif brain_type == BrainType.Zazzles's Agent:
                capabilities = ['chat', 'completion', 'reasoning']
            elif brain_type == BrainType.DOCLING:
                capabilities = ['pdf_processing', 'document_parsing', 'markdown_conversion']
            
            # Try to get capabilities from API
            if self.http_session:
                try:
                    async with self.http_session.get(f"http://{host}:{port}/capabilities") as response:
                        if response.status == 200:
                            api_capabilities = await response.json()
                            if isinstance(api_capabilities, list):
                                capabilities.extend(api_capabilities)
                except:
                    pass
            
            return list(set(capabilities))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Capability detection failed: {e}")
            return []
    
    async def _get_brain_version(self, host: str, port: int) -> str:
        """Get brain version information"""
        try:
            if self.http_session:
                async with self.http_session.get(f"http://{host}:{port}/version") as response:
                    if response.status == 200:
                        version_data = await response.json()
                        return version_data.get('version', '1.0.0')
        except:
            pass
        
        return '1.0.0'  # Default version
    
    async def _detect_brain_endpoints(self, host: str, port: int, brain_type: BrainType) -> Dict[str, str]:
        """Detect available brain endpoints"""
        try:
            endpoints = {'health': '/health', 'status': '/status'}
            
            # Add type-specific endpoints
            if brain_type == BrainType.EMBEDDING:
                endpoints.update({'embed': '/embed', 'encode': '/encode'})
            elif brain_type == BrainType.RERANKER:
                endpoints.update({'rerank': '/rerank', 'score': '/score'})
            elif brain_type == BrainType.Zazzles's Agent:
                endpoints.update({'chat': '/chat', 'completion': '/completion'})
            elif brain_type == BrainType.DOCLING:
                endpoints.update({'process': '/process', 'convert': '/convert'})
            
            # Verify endpoints exist
            verified_endpoints = {}
            if self.http_session:
                for name, path in endpoints.items():
                    try:
                        async with self.http_session.get(f"http://{host}:{port}{path}") as response:
                            if response.status in [200, 405, 422]:  # Endpoint exists
                                verified_endpoints[name] = path
                    except:
                        continue
            
            return verified_endpoints if verified_endpoints else endpoints
            
        except Exception as e:
            logger.error(f"❌ Endpoint detection failed: {e}")
            return {'health': '/health'}
    
    async def _register_discovered_brain(self, brain: DiscoveredBrain):
        """Register discovered brain"""
        try:
            # Add to discovered brains
            self.discovered_brains[brain.brain_id] = brain
            
            # Update status to registered
            brain.status = BrainStatus.REGISTERED
            
            # Store in Redis
            await self._store_discovered_brain(brain)
            
            # Update metrics
            self.metrics['total_discoveries'] += 1
            self.metrics['successful_registrations'] += 1
            self.metrics['active_brains'] = len([b for b in self.discovered_brains.values() 
                                               if b.status in [BrainStatus.REGISTERED, BrainStatus.HEALTHY]])
            
            logger.info(f"✅ Brain registered: {brain.brain_id} ({brain.brain_type.value}) at {brain.host}:{brain.port}")
            
        except Exception as e:
            logger.error(f"❌ Brain registration failed: {e}")
            self.metrics['failed_registrations'] += 1
    
    async def _health_monitor(self):
        """Monitor health of discovered brains"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for brain in list(self.discovered_brains.values()):
                    if brain.status in [BrainStatus.REGISTERED, BrainStatus.HEALTHY, BrainStatus.UNHEALTHY]:
                        await self._check_brain_health(brain)
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
    
    async def _check_brain_health(self, brain: DiscoveredBrain):
        """Check health of specific brain"""
        try:
            if self.http_session:
                health_url = f"http://{brain.host}:{brain.port}{brain.health_endpoint}"
                
                async with self.http_session.get(health_url) as response:
                    if response.status == 200:
                        # Brain is healthy
                        brain.status = BrainStatus.HEALTHY
                        brain.last_seen = datetime.now()
                    else:
                        # Brain is unhealthy
                        brain.status = BrainStatus.UNHEALTHY
            else:
                # Fallback to TCP check
                if await self._check_brain_reachability(brain.host, brain.port):
                    brain.status = BrainStatus.HEALTHY
                    brain.last_seen = datetime.now()
                else:
                    brain.status = BrainStatus.UNHEALTHY
            
            # Store updated status
            await self._store_discovered_brain(brain)
            
        except Exception as e:
            logger.debug(f"Health check failed for {brain.brain_id}: {e}")
            brain.status = BrainStatus.UNHEALTHY
    
    async def _cleanup_lost_brains(self):
        """Cleanup brains that have been lost"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                timeout_threshold = timedelta(seconds=self.config['brain_timeout_seconds'])
                
                lost_brains = []
                for brain in list(self.discovered_brains.values()):
                    if current_time - brain.last_seen > timeout_threshold:
                        lost_brains.append(brain)
                
                # Mark brains as lost
                for brain in lost_brains:
                    brain.status = BrainStatus.LOST
                    await self._store_discovered_brain(brain)
                    logger.warning(f"⚠️ Brain marked as lost: {brain.brain_id}")
                
                # Update metrics
                self.metrics['lost_brains'] = len([b for b in self.discovered_brains.values() 
                                                 if b.status == BrainStatus.LOST])
                self.metrics['active_brains'] = len([b for b in self.discovered_brains.values() 
                                                   if b.status in [BrainStatus.REGISTERED, BrainStatus.HEALTHY]])
                
            except Exception as e:
                logger.error(f"❌ Brain cleanup error: {e}")
    
    async def _store_discovered_brain(self, brain: DiscoveredBrain):
        """Store discovered brain in Redis"""
        if self.redis_client:
            try:
                key = f"discovered_brain:{brain.brain_id}"
                data = json.dumps(asdict(brain), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store discovered brain: {e}")
    
    async def _load_discovered_brains(self):
        """Load discovered brains from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("discovered_brain:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        brain_data = json.loads(data)
                        # Convert back to DiscoveredBrain object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load discovered brains: {e}")
    
    async def get_discovered_brains(self, status_filter: Optional[BrainStatus] = None) -> List[DiscoveredBrain]:
        """Get list of discovered brains"""
        brains = list(self.discovered_brains.values())
        
        if status_filter:
            brains = [brain for brain in brains if brain.status == status_filter]
        
        return brains
    
    async def get_brain_by_id(self, brain_id: str) -> Optional[DiscoveredBrain]:
        """Get specific brain by ID"""
        return self.discovered_brains.get(brain_id)
    
    async def deregister_brain(self, brain_id: str) -> bool:
        """Deregister a brain"""
        try:
            brain = self.discovered_brains.get(brain_id)
            if brain:
                brain.status = BrainStatus.DEREGISTERED
                await self._store_discovered_brain(brain)
                logger.info(f"✅ Brain deregistered: {brain_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Brain deregistration failed: {e}")
            return False
    
    async def get_discovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive discovery metrics"""
        return {
            'metrics': self.metrics.copy(),
            'discovered_brains': len(self.discovered_brains),
            'brains_by_status': {
                status.value: len([b for b in self.discovered_brains.values() if b.status == status])
                for status in BrainStatus
            },
            'brains_by_type': {
                brain_type.value: len([b for b in self.discovered_brains.values() if b.brain_type == brain_type])
                for brain_type in BrainType
            },
            'discovery_methods': {
                method.value: len([b for b in self.discovered_brains.values() if b.discovery_method == method])
                for method in DiscoveryMethod
            },
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global brain discovery instance
brain_discovery = BrainDiscovery()

async def initialize_brain_discovery():
    """Initialize the global brain discovery"""
    await brain_discovery.initialize()

if __name__ == "__main__":
    # Test the brain discovery
    async def test_brain_discovery():
        await initialize_brain_discovery()
        
        # Wait for discovery
        await asyncio.sleep(10)
        
        # Get discovered brains
        brains = await brain_discovery.get_discovered_brains()
        print(f"Discovered brains: {len(brains)}")
        
        for brain in brains:
            print(f"- {brain.brain_id}: {brain.brain_type.value} at {brain.host}:{brain.port} ({brain.status.value})")
        
        # Get metrics
        metrics = await brain_discovery.get_discovery_metrics()
        print(f"Discovery metrics: {metrics}")
    
    asyncio.run(test_brain_discovery())
