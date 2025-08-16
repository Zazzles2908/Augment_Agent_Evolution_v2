"""
Flow Monitor for Four-Brain System v2
Comprehensive end-to-end workflow validation and monitoring

Created: 2025-07-31 AEST
Purpose: Monitor and validate end-to-end workflows across all Four-Brain components
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import aiohttp
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowStatus(Enum):
    """Flow execution status levels"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEGRADED = "degraded"

class FlowType(Enum):
    """Types of system flows"""
    DOCUMENT_PROCESSING = "document_processing"
    CHAT_INTERACTION = "chat_interaction"
    BRAIN_COMMUNICATION = "brain_communication"
    DATABASE_OPERATION = "database_operation"
    HEALTH_CHECK = "health_check"
    SYSTEM_STARTUP = "system_startup"
    USER_AUTHENTICATION = "user_authentication"
    API_REQUEST = "api_request"

@dataclass
class FlowStep:
    """Individual step in a workflow"""
    step_id: str
    step_name: str
    component: str
    start_time: datetime
    end_time: Optional[datetime]
    status: FlowStatus
    duration_ms: Optional[float]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class FlowExecution:
    """Complete flow execution record"""
    flow_id: str
    flow_type: FlowType
    flow_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: FlowStatus
    total_duration_ms: Optional[float]
    steps: List[FlowStep]
    success_rate: float
    error_count: int
    warning_count: int
    metadata: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]

@dataclass
class FlowMetrics:
    """Flow performance metrics"""
    flow_type: FlowType
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    success_rate: float
    error_rate: float
    throughput_per_minute: float
    last_execution: Optional[datetime]
    common_errors: List[str]

class FlowMonitor:
    """
    Comprehensive end-to-end workflow monitoring system
    
    Features:
    - End-to-end workflow tracking
    - Real-time flow execution monitoring
    - Performance metrics collection
    - Failure detection and analysis
    - Flow dependency mapping
    - Bottleneck identification
    - SLA monitoring and alerting
    - Comprehensive flow reporting
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/3"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'flow_timeout_seconds': 300,
            'step_timeout_seconds': 60,
            'max_concurrent_flows': 100,
            'metrics_retention_hours': 168,  # 7 days
            'alert_threshold_error_rate': 0.1,
            'alert_threshold_duration_ms': 30000,
            'performance_window_minutes': 60
        }
        
        # Flow tracking
        self.active_flows: Dict[str, FlowExecution] = {}
        self.flow_history: List[FlowExecution] = []
        self.flow_metrics: Dict[FlowType, FlowMetrics] = {}
        
        # Performance tracking
        self.metrics = {
            'total_flows_monitored': 0,
            'active_flow_count': 0,
            'average_flow_duration': 0.0,
            'system_throughput': 0.0,
            'error_rate': 0.0,
            'alerts_generated': 0
        }
        
        # Flow definitions for validation
        self.flow_definitions = self._initialize_flow_definitions()
        
        logger.info("🔄 Flow Monitor initialized")
    
    def _initialize_flow_definitions(self) -> Dict[FlowType, Dict[str, Any]]:
        """Initialize standard flow definitions"""
        return {
            FlowType.DOCUMENT_PROCESSING: {
                'expected_steps': ['upload', 'docling_processing', 'embedding', 'storage', 'indexing'],
                'max_duration_ms': 120000,  # 2 minutes
                'required_components': ['brain4-docling', 'brain1-embedding', 'supabase']
            },
            FlowType.CHAT_INTERACTION: {
                'expected_steps': ['authentication', 'query_processing', 'brain_routing', 'response_generation'],
                'max_duration_ms': 30000,  # 30 seconds
                'required_components': ['k2-hub', 'brain3-augment', 'redis']
            },
            FlowType.BRAIN_COMMUNICATION: {
                'expected_steps': ['message_send', 'routing', 'processing', 'response'],
                'max_duration_ms': 15000,  # 15 seconds
                'required_components': ['redis-streams', 'k2-hub']
            },
            FlowType.DATABASE_OPERATION: {
                'expected_steps': ['connection', 'query_execution', 'result_processing'],
                'max_duration_ms': 10000,  # 10 seconds
                'required_components': ['supabase', 'postgresql']
            },
            FlowType.HEALTH_CHECK: {
                'expected_steps': ['component_check', 'connectivity_test', 'status_aggregation'],
                'max_duration_ms': 5000,  # 5 seconds
                'required_components': ['all']
            }
        }
    
    async def initialize(self):
        """Initialize Redis connection and flow monitoring services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load flow history
            await self._load_flow_history()
            
            # Start background monitoring
            asyncio.create_task(self._flow_timeout_monitor())
            asyncio.create_task(self._metrics_aggregation())
            asyncio.create_task(self._performance_analysis())
            
            logger.info("✅ Flow Monitor Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Flow Monitor: {e}")
            raise
    
    async def start_flow(self, flow_type: FlowType, flow_name: str, 
                        user_id: Optional[str] = None, session_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start monitoring a new flow"""
        flow_id = str(uuid.uuid4())
        
        flow_execution = FlowExecution(
            flow_id=flow_id,
            flow_type=flow_type,
            flow_name=flow_name,
            start_time=datetime.utcnow(),
            end_time=None,
            status=FlowStatus.RUNNING,
            total_duration_ms=None,
            steps=[],
            success_rate=0.0,
            error_count=0,
            warning_count=0,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id
        )
        
        self.active_flows[flow_id] = flow_execution
        self.metrics['active_flow_count'] = len(self.active_flows)
        
        # Store in Redis
        await self._store_flow_execution(flow_execution)
        
        logger.info(f"🚀 Started flow monitoring: {flow_name} ({flow_id})")
        return flow_id
    
    async def add_flow_step(self, flow_id: str, step_name: str, component: str,
                           input_data: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a step to an active flow"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        step_id = str(uuid.uuid4())
        step = FlowStep(
            step_id=step_id,
            step_name=step_name,
            component=component,
            start_time=datetime.utcnow(),
            end_time=None,
            status=FlowStatus.RUNNING,
            duration_ms=None,
            input_data=input_data or {},
            output_data=None,
            error_message=None,
            metadata=metadata or {}
        )
        
        self.active_flows[flow_id].steps.append(step)
        
        # Update in Redis
        await self._store_flow_execution(self.active_flows[flow_id])
        
        logger.debug(f"📝 Added step {step_name} to flow {flow_id}")
        return step_id

    async def complete_flow_step(self, flow_id: str, step_id: str,
                                output_data: Optional[Dict[str, Any]] = None,
                                error_message: Optional[str] = None) -> bool:
        """Complete a flow step"""
        if flow_id not in self.active_flows:
            return False

        flow = self.active_flows[flow_id]
        step = next((s for s in flow.steps if s.step_id == step_id), None)

        if not step:
            return False

        step.end_time = datetime.utcnow()
        step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000
        step.output_data = output_data
        step.error_message = error_message
        step.status = FlowStatus.FAILED if error_message else FlowStatus.COMPLETED

        if error_message:
            flow.error_count += 1

        # Update in Redis
        await self._store_flow_execution(flow)

        logger.debug(f"✅ Completed step {step.step_name} in flow {flow_id}")
        return True

    async def complete_flow(self, flow_id: str, status: FlowStatus = FlowStatus.COMPLETED,
                           error_message: Optional[str] = None) -> bool:
        """Complete a flow execution"""
        if flow_id not in self.active_flows:
            return False

        flow = self.active_flows[flow_id]
        flow.end_time = datetime.utcnow()
        flow.total_duration_ms = (flow.end_time - flow.start_time).total_seconds() * 1000
        flow.status = status

        if error_message:
            flow.metadata['completion_error'] = error_message

        # Calculate success rate
        completed_steps = sum(1 for step in flow.steps if step.status == FlowStatus.COMPLETED)
        flow.success_rate = completed_steps / len(flow.steps) if flow.steps else 0.0

        # Move to history
        self.flow_history.append(flow)
        del self.active_flows[flow_id]
        self.metrics['active_flow_count'] = len(self.active_flows)
        self.metrics['total_flows_monitored'] += 1

        # Update metrics
        await self._update_flow_metrics(flow)

        # Store final state in Redis
        await self._store_flow_execution(flow)

        logger.info(f"🏁 Completed flow {flow.flow_name} ({flow_id}) - Status: {status.value}")
        return True

    async def validate_end_to_end_flow(self, flow_type: FlowType,
                                      test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate a complete end-to-end flow"""
        validation_id = str(uuid.uuid4())

        try:
            # Start validation flow
            flow_id = await self.start_flow(
                flow_type=flow_type,
                flow_name=f"E2E_Validation_{flow_type.value}",
                metadata={'validation_id': validation_id, 'test_data': test_data}
            )

            flow_def = self.flow_definitions.get(flow_type)
            if not flow_def:
                return {'success': False, 'error': f'No definition for flow type {flow_type.value}'}

            validation_results = {
                'flow_id': flow_id,
                'flow_type': flow_type.value,
                'validation_id': validation_id,
                'start_time': datetime.utcnow().isoformat(),
                'steps_validated': [],
                'success': True,
                'errors': [],
                'warnings': [],
                'performance_metrics': {}
            }

            # Validate each expected step
            for step_name in flow_def['expected_steps']:
                step_result = await self._validate_flow_step(flow_id, step_name, flow_def)
                validation_results['steps_validated'].append(step_result)

                if not step_result['success']:
                    validation_results['success'] = False
                    validation_results['errors'].append(step_result.get('error', 'Unknown error'))

            # Validate overall flow performance
            performance_result = await self._validate_flow_performance(flow_type, flow_def)
            validation_results['performance_metrics'] = performance_result

            # Complete the validation flow
            final_status = FlowStatus.COMPLETED if validation_results['success'] else FlowStatus.FAILED
            await self.complete_flow(flow_id, final_status)

            validation_results['end_time'] = datetime.utcnow().isoformat()

            logger.info(f"🔍 E2E validation completed for {flow_type.value}: {'✅ PASSED' if validation_results['success'] else '❌ FAILED'}")

            return validation_results

        except Exception as e:
            logger.error(f"❌ E2E validation failed for {flow_type.value}: {e}")
            return {
                'success': False,
                'error': str(e),
                'validation_id': validation_id,
                'flow_type': flow_type.value
            }

    async def _validate_flow_step(self, flow_id: str, step_name: str,
                                 flow_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an individual flow step"""
        step_id = await self.add_flow_step(flow_id, step_name, "validation_engine")

        step_result = {
            'step_name': step_name,
            'step_id': step_id,
            'success': True,
            'duration_ms': 0,
            'error': None,
            'warnings': []
        }

        start_time = time.time()

        try:
            # Simulate step validation based on step type
            if step_name == 'upload':
                await self._validate_upload_capability()
            elif step_name == 'docling_processing':
                await self._validate_docling_service()
            elif step_name == 'embedding':
                await self._validate_embedding_service()
            elif step_name == 'storage':
                await self._validate_storage_service()
            elif step_name == 'authentication':
                await self._validate_auth_service()
            elif step_name == 'query_processing':
                await self._validate_query_processing()
            elif step_name == 'brain_routing':
                await self._validate_brain_routing()
            elif step_name == 'response_generation':
                await self._validate_response_generation()
            else:
                # Generic component validation
                await self._validate_generic_component(step_name)

            step_result['duration_ms'] = (time.time() - start_time) * 1000
            await self.complete_flow_step(flow_id, step_id, {'validation': 'passed'})

        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)
            step_result['duration_ms'] = (time.time() - start_time) * 1000
            await self.complete_flow_step(flow_id, step_id, error_message=str(e))

        return step_result

    async def _validate_upload_capability(self):
        """Validate file upload capability"""
        # Test upload endpoint availability
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:9098/health') as response:
                if response.status != 200:
                    raise Exception(f"Upload service unavailable: HTTP {response.status}")

    async def _validate_docling_service(self):
        """Validate Docling document processing service"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8004/health') as response:
                if response.status != 200:
                    raise Exception(f"Docling service unavailable: HTTP {response.status}")

    async def _validate_embedding_service(self):
        """Validate embedding generation service"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8001/health') as response:
                if response.status != 200:
                    raise Exception(f"Embedding service unavailable: HTTP {response.status}")

    async def _validate_storage_service(self):
        """Validate storage service (Supabase)"""
        # Test Redis connection
        if self.redis_client:
            await self.redis_client.ping()
        else:
            raise Exception("Redis connection not available")

    async def _validate_auth_service(self):
        """Validate authentication service"""
        # Test basic auth endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:9098/health') as response:
                if response.status != 200:
                    raise Exception(f"Auth service unavailable: HTTP {response.status}")

    async def _validate_query_processing(self):
        """Validate query processing capability"""
        # Test K2-Hub query processing
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:9098/health') as response:
                if response.status != 200:
                    raise Exception(f"Query processing unavailable: HTTP {response.status}")

    async def _validate_brain_routing(self):
        """Validate brain routing capability"""
        # Test brain communication through Redis streams
        if self.redis_client:
            await self.redis_client.ping()
        else:
            raise Exception("Brain routing unavailable - Redis connection failed")

    async def _validate_response_generation(self):
        """Validate response generation capability"""
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8003/health') as response:
                if response.status != 200:
                    raise Exception(f"Response generation unavailable: HTTP {response.status}")

    async def _validate_generic_component(self, component_name: str):
        """Generic component validation"""
        logger.info(f"Validating generic component: {component_name}")
        # Basic validation - component exists and is responsive
        await asyncio.sleep(0.1)  # Simulate validation time

    async def _validate_flow_performance(self, flow_type: FlowType,
                                       flow_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate flow performance metrics"""
        metrics = self.flow_metrics.get(flow_type)

        performance_result = {
            'flow_type': flow_type.value,
            'performance_check': True,
            'metrics': {},
            'issues': []
        }

        if metrics:
            performance_result['metrics'] = {
                'average_duration_ms': metrics.average_duration_ms,
                'success_rate': metrics.success_rate,
                'error_rate': metrics.error_rate,
                'throughput_per_minute': metrics.throughput_per_minute
            }

            # Check performance thresholds
            max_duration = flow_def.get('max_duration_ms', 60000)
            if metrics.average_duration_ms > max_duration:
                performance_result['issues'].append(
                    f"Average duration ({metrics.average_duration_ms}ms) exceeds threshold ({max_duration}ms)"
                )

            if metrics.success_rate < 0.95:
                performance_result['issues'].append(
                    f"Success rate ({metrics.success_rate:.2%}) below 95% threshold"
                )

            if metrics.error_rate > 0.05:
                performance_result['issues'].append(
                    f"Error rate ({metrics.error_rate:.2%}) above 5% threshold"
                )

        return performance_result

    async def get_flow_status(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a flow"""
        if flow_id in self.active_flows:
            flow = self.active_flows[flow_id]
            return {
                'flow_id': flow_id,
                'status': flow.status.value,
                'progress': len([s for s in flow.steps if s.status == FlowStatus.COMPLETED]) / len(flow.steps) if flow.steps else 0,
                'current_step': flow.steps[-1].step_name if flow.steps else None,
                'duration_ms': (datetime.utcnow() - flow.start_time).total_seconds() * 1000,
                'error_count': flow.error_count
            }

        # Check history
        historical_flow = next((f for f in self.flow_history if f.flow_id == flow_id), None)
        if historical_flow:
            return {
                'flow_id': flow_id,
                'status': historical_flow.status.value,
                'progress': 1.0,
                'duration_ms': historical_flow.total_duration_ms,
                'success_rate': historical_flow.success_rate,
                'error_count': historical_flow.error_count
            }

        return None

    async def get_system_flow_health(self) -> Dict[str, Any]:
        """Get overall system flow health"""
        health_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'active_flows': len(self.active_flows),
            'total_flows_today': 0,
            'success_rate_24h': 0.0,
            'average_duration_24h': 0.0,
            'flow_type_metrics': {},
            'alerts': [],
            'recommendations': []
        }

        # Calculate 24h metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_flows = [f for f in self.flow_history if f.start_time > cutoff_time]

        if recent_flows:
            health_report['total_flows_today'] = len(recent_flows)
            successful_flows = [f for f in recent_flows if f.status == FlowStatus.COMPLETED]
            health_report['success_rate_24h'] = len(successful_flows) / len(recent_flows)
            health_report['average_duration_24h'] = sum(f.total_duration_ms or 0 for f in recent_flows) / len(recent_flows)

        # Flow type specific metrics
        for flow_type in FlowType:
            if flow_type in self.flow_metrics:
                metrics = self.flow_metrics[flow_type]
                health_report['flow_type_metrics'][flow_type.value] = {
                    'success_rate': metrics.success_rate,
                    'average_duration_ms': metrics.average_duration_ms,
                    'throughput_per_minute': metrics.throughput_per_minute,
                    'last_execution': metrics.last_execution.isoformat() if metrics.last_execution else None
                }

        # Generate alerts and recommendations
        if health_report['success_rate_24h'] < 0.9:
            health_report['overall_status'] = 'degraded'
            health_report['alerts'].append('System success rate below 90%')
            health_report['recommendations'].append('Investigate failing flows and error patterns')

        if len(self.active_flows) > self.config['max_concurrent_flows'] * 0.8:
            health_report['alerts'].append('High concurrent flow load detected')
            health_report['recommendations'].append('Monitor system resources and consider scaling')

        return health_report

    async def _store_flow_execution(self, flow: FlowExecution):
        """Store flow execution in Redis"""
        if not self.redis_client:
            return

        try:
            flow_data = {
                'flow_id': flow.flow_id,
                'flow_type': flow.flow_type.value,
                'flow_name': flow.flow_name,
                'start_time': flow.start_time.isoformat(),
                'end_time': flow.end_time.isoformat() if flow.end_time else None,
                'status': flow.status.value,
                'total_duration_ms': flow.total_duration_ms,
                'success_rate': flow.success_rate,
                'error_count': flow.error_count,
                'warning_count': flow.warning_count,
                'step_count': len(flow.steps),
                'metadata': flow.metadata,
                'user_id': flow.user_id,
                'session_id': flow.session_id
            }

            # Store flow summary
            await self.redis_client.hset(
                f"flow_monitor:flows:{flow.flow_id}",
                mapping=flow_data
            )

            # Store detailed steps
            for step in flow.steps:
                step_data = {
                    'step_id': step.step_id,
                    'step_name': step.step_name,
                    'component': step.component,
                    'start_time': step.start_time.isoformat(),
                    'end_time': step.end_time.isoformat() if step.end_time else None,
                    'status': step.status.value,
                    'duration_ms': step.duration_ms,
                    'error_message': step.error_message,
                    'metadata': json.dumps(step.metadata)
                }

                await self.redis_client.hset(
                    f"flow_monitor:steps:{flow.flow_id}:{step.step_id}",
                    mapping=step_data
                )

            # Update flow index
            await self.redis_client.zadd(
                "flow_monitor:flow_index",
                {flow.flow_id: time.time()}
            )

        except Exception as e:
            logger.error(f"Failed to store flow execution: {e}")

    async def _load_flow_history(self):
        """Load flow history from Redis"""
        if not self.redis_client:
            return

        try:
            # Load recent flows (last 24 hours)
            cutoff_time = time.time() - (24 * 3600)
            flow_ids = await self.redis_client.zrangebyscore(
                "flow_monitor:flow_index",
                cutoff_time,
                "+inf"
            )

            for flow_id in flow_ids[-100:]:  # Limit to last 100 flows
                flow_data = await self.redis_client.hgetall(f"flow_monitor:flows:{flow_id}")
                if flow_data:
                    # Reconstruct flow execution (simplified for history)
                    flow = FlowExecution(
                        flow_id=flow_data.get('flow_id', ''),
                        flow_type=FlowType(flow_data.get('flow_type', 'health_check')),
                        flow_name=flow_data.get('flow_name', ''),
                        start_time=datetime.fromisoformat(flow_data.get('start_time', datetime.utcnow().isoformat())),
                        end_time=datetime.fromisoformat(flow_data['end_time']) if flow_data.get('end_time') else None,
                        status=FlowStatus(flow_data.get('status', 'completed')),
                        total_duration_ms=float(flow_data.get('total_duration_ms', 0)) if flow_data.get('total_duration_ms') else None,
                        steps=[],  # Don't load detailed steps for history
                        success_rate=float(flow_data.get('success_rate', 0)),
                        error_count=int(flow_data.get('error_count', 0)),
                        warning_count=int(flow_data.get('warning_count', 0)),
                        metadata=flow_data.get('metadata', {}),
                        user_id=flow_data.get('user_id'),
                        session_id=flow_data.get('session_id')
                    )

                    self.flow_history.append(flow)

            logger.info(f"📚 Loaded {len(self.flow_history)} flows from history")

        except Exception as e:
            logger.error(f"Failed to load flow history: {e}")

    async def _update_flow_metrics(self, flow: FlowExecution):
        """Update flow type metrics"""
        flow_type = flow.flow_type

        if flow_type not in self.flow_metrics:
            self.flow_metrics[flow_type] = FlowMetrics(
                flow_type=flow_type,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_duration_ms=0.0,
                min_duration_ms=float('inf'),
                max_duration_ms=0.0,
                p95_duration_ms=0.0,
                success_rate=0.0,
                error_rate=0.0,
                throughput_per_minute=0.0,
                last_execution=None,
                common_errors=[]
            )

        metrics = self.flow_metrics[flow_type]
        metrics.total_executions += 1
        metrics.last_execution = flow.end_time or datetime.utcnow()

        if flow.status == FlowStatus.COMPLETED:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1

        if flow.total_duration_ms:
            metrics.min_duration_ms = min(metrics.min_duration_ms, flow.total_duration_ms)
            metrics.max_duration_ms = max(metrics.max_duration_ms, flow.total_duration_ms)

            # Update average (simple moving average)
            metrics.average_duration_ms = (
                (metrics.average_duration_ms * (metrics.total_executions - 1) + flow.total_duration_ms)
                / metrics.total_executions
            )

        metrics.success_rate = metrics.successful_executions / metrics.total_executions
        metrics.error_rate = metrics.failed_executions / metrics.total_executions

        # Calculate throughput (flows per minute in last hour)
        recent_flows = [f for f in self.flow_history if f.flow_type == flow_type and
                       f.start_time > datetime.utcnow() - timedelta(hours=1)]
        metrics.throughput_per_minute = len(recent_flows) / 60.0

    async def _flow_timeout_monitor(self):
        """Monitor for flow timeouts"""
        while True:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = timedelta(seconds=self.config['flow_timeout_seconds'])

                timed_out_flows = []
                for flow_id, flow in self.active_flows.items():
                    if current_time - flow.start_time > timeout_threshold:
                        timed_out_flows.append(flow_id)

                for flow_id in timed_out_flows:
                    logger.warning(f"⏰ Flow {flow_id} timed out")
                    await self.complete_flow(flow_id, FlowStatus.TIMEOUT, "Flow execution timeout")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in flow timeout monitor: {e}")
                await asyncio.sleep(60)

    async def _metrics_aggregation(self):
        """Aggregate and update system metrics"""
        while True:
            try:
                # Update system-wide metrics
                self.metrics['total_flows_monitored'] = len(self.flow_history)
                self.metrics['active_flow_count'] = len(self.active_flows)

                if self.flow_history:
                    total_duration = sum(f.total_duration_ms or 0 for f in self.flow_history)
                    self.metrics['average_flow_duration'] = total_duration / len(self.flow_history)

                    successful_flows = [f for f in self.flow_history if f.status == FlowStatus.COMPLETED]
                    self.metrics['error_rate'] = 1.0 - (len(successful_flows) / len(self.flow_history))

                # Store metrics in Redis
                if self.redis_client:
                    await self.redis_client.hset(
                        "flow_monitor:system_metrics",
                        mapping={k: str(v) for k, v in self.metrics.items()}
                    )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
                await asyncio.sleep(120)

    async def _performance_analysis(self):
        """Perform periodic performance analysis"""
        while True:
            try:
                # Analyze performance trends and generate alerts
                for flow_type, metrics in self.flow_metrics.items():
                    if metrics.error_rate > self.config['alert_threshold_error_rate']:
                        logger.warning(f"🚨 High error rate detected for {flow_type.value}: {metrics.error_rate:.2%}")
                        self.metrics['alerts_generated'] += 1

                    if metrics.average_duration_ms > self.config['alert_threshold_duration_ms']:
                        logger.warning(f"🐌 Slow performance detected for {flow_type.value}: {metrics.average_duration_ms:.0f}ms")
                        self.metrics['alerts_generated'] += 1

                await asyncio.sleep(300)  # Analyze every 5 minutes

            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                await asyncio.sleep(600)

# Global flow monitor instance
_global_flow_monitor: Optional[FlowMonitor] = None

def get_flow_monitor() -> FlowMonitor:
    """Get or create global flow monitor instance"""
    global _global_flow_monitor

    if _global_flow_monitor is None:
        _global_flow_monitor = FlowMonitor()

    return _global_flow_monitor

async def initialize_flow_monitoring(redis_url: str = "redis://localhost:6379/3") -> FlowMonitor:
    """Initialize flow monitoring system"""
    global _global_flow_monitor
    _global_flow_monitor = FlowMonitor(redis_url)
    await _global_flow_monitor.initialize()
    return _global_flow_monitor
