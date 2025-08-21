"""
Brain Coordination System for Four-Brain System v2
Central coordination hub for managing multiple AI brain instances

Created: 2025-07-30 AEST
Purpose: Coordinate tasks and communication between Brain1, Brain2, Brain3, and Brain4
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainType(Enum):
    """Types of AI brains in the system"""
    EMBEDDING = "embedding"  # Brain1 - Qwen3-4B Embedding
    RERANKER = "reranker"    # Brain2 - Qwen3-Reranker-4B
    AUGMENT = "augment"      # Brain3 - Augment Agent API
    DOCLING = "docling"      # Brain4 - Docling PDF processing

class BrainStatus(Enum):
    """Brain operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BrainInstance:
    """Brain instance information"""
    brain_id: str
    brain_type: BrainType
    status: BrainStatus
    endpoint: str
    port: int
    capabilities: List[str]
    current_load: float
    max_capacity: int
    memory_usage: float
    gpu_usage: float
    last_heartbeat: datetime
    version: str
    metadata: Dict[str, Any]

@dataclass
class CoordinationTask:
    """Task for brain coordination"""
    task_id: str
    task_type: str
    priority: TaskPriority
    assigned_brain: Optional[str]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class CoordinationMetrics:
    """Brain coordination metrics"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    average_completion_time: float
    brain_utilization: Dict[str, float]
    task_distribution: Dict[str, int]
    error_rate: float

class BrainCoordinator:
    """
    Central brain coordination system
    
    Features:
    - Multi-brain task distribution and load balancing
    - Real-time brain health monitoring
    - Intelligent task routing based on capabilities
    - Fault tolerance and automatic failover
    - Performance optimization and resource allocation
    - Task dependency management
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/11"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Brain registry
        self.brain_instances: Dict[str, BrainInstance] = {}
        
        # Task management
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Coordination configuration
        self.config = {
            'heartbeat_interval_seconds': 30,
            'task_timeout_seconds': 300,
            'max_retries': 3,
            'load_balance_threshold': 0.8,
            'health_check_interval': 60,
            'failover_enabled': True,
            'auto_scaling_enabled': False
        }
        
        # Coordination metrics
        self.metrics = CoordinationMetrics(
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            active_tasks=0,
            average_completion_time=0.0,
            brain_utilization={},
            task_distribution={},
            error_rate=0.0
        )
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'brain_online': [],
            'brain_offline': [],
            'task_completed': [],
            'task_failed': [],
            'coordination_error': []
        }
        
        logger.info("🧠 Brain Coordinator initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start coordination services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Load existing brain instances
            await self._load_brain_instances()
            
            # Start background services
            asyncio.create_task(self._brain_health_monitor())
            asyncio.create_task(self._task_processor())
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._cleanup_completed_tasks())
            
            logger.info("✅ Brain Coordinator Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Brain Coordinator: {e}")
            raise
    
    async def register_brain(self, brain_id: str, brain_type: BrainType, endpoint: str, 
                           port: int, capabilities: List[str], version: str = "1.0.0",
                           metadata: Dict[str, Any] = None) -> bool:
        """Register a new brain instance"""
        try:
            brain_instance = BrainInstance(
                brain_id=brain_id,
                brain_type=brain_type,
                status=BrainStatus.ONLINE,
                endpoint=endpoint,
                port=port,
                capabilities=capabilities,
                current_load=0.0,
                max_capacity=100,
                memory_usage=0.0,
                gpu_usage=0.0,
                last_heartbeat=datetime.now(),
                version=version,
                metadata=metadata or {}
            )
            
            # Store brain instance
            self.brain_instances[brain_id] = brain_instance
            await self._store_brain_instance(brain_instance)
            
            # Initialize metrics for this brain
            self.metrics.brain_utilization[brain_id] = 0.0
            self.metrics.task_distribution[brain_id] = 0
            
            # Trigger event callbacks
            await self._trigger_event_callbacks('brain_online', brain_instance)
            
            logger.info(f"✅ Brain registered: {brain_id} ({brain_type.value}) at {endpoint}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register brain {brain_id}: {e}")
            return False
    
    async def submit_task(self, task_type: str, input_data: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         dependencies: List[str] = None, 
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a task for brain coordination"""
        try:
            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{len(self.active_tasks)}"
            
            # Create coordination task
            task = CoordinationTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                assigned_brain=None,
                input_data=input_data,
                output_data=None,
                status="queued",
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                error_message=None,
                retry_count=0,
                max_retries=self.config['max_retries'],
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            # Queue for processing
            await self.task_queue.put(task)
            
            # Update metrics
            self.metrics.total_tasks += 1
            self.metrics.active_tasks += 1
            
            logger.info(f"✅ Task submitted: {task_id} ({task_type}) with priority {priority.name}")
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit task: {e}")
            raise
    
    async def _task_processor(self):
        """Background task processor"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Check dependencies
                if not await self._check_task_dependencies(task):
                    # Re-queue task if dependencies not met
                    await asyncio.sleep(5)
                    await self.task_queue.put(task)
                    continue
                
                # Find suitable brain for task
                suitable_brain = await self._find_suitable_brain(task)
                if not suitable_brain:
                    logger.warning(f"No suitable brain found for task {task.task_id}")
                    await asyncio.sleep(10)
                    await self.task_queue.put(task)
                    continue
                
                # Assign and execute task
                await self._execute_task(task, suitable_brain)
                
            except Exception as e:
                logger.error(f"❌ Task processor error: {e}")
                await asyncio.sleep(1)
    
    async def _find_suitable_brain(self, task: CoordinationTask) -> Optional[BrainInstance]:
        """Find the most suitable brain for a task"""
        try:
            suitable_brains = []
            
            # Filter brains by capability and status
            for brain in self.brain_instances.values():
                if (brain.status == BrainStatus.ONLINE and
                    self._brain_can_handle_task(brain, task) and
                    brain.current_load < self.config['load_balance_threshold']):
                    suitable_brains.append(brain)
            
            if not suitable_brains:
                return None
            
            # Sort by load and priority
            suitable_brains.sort(key=lambda b: (b.current_load, -task.priority.value))
            
            return suitable_brains[0]
            
        except Exception as e:
            logger.error(f"❌ Failed to find suitable brain: {e}")
            return None
    
    def _brain_can_handle_task(self, brain: BrainInstance, task: CoordinationTask) -> bool:
        """Check if brain can handle the task type"""
        task_type = task.task_type.lower()
        
        # Map task types to brain capabilities
        if task_type in ['embedding', 'encode', 'vector']:
            return brain.brain_type == BrainType.EMBEDDING
        elif task_type in ['rerank', 'ranking', 'score']:
            return brain.brain_type == BrainType.RERANKER
        elif task_type in ['chat', 'completion', 'augment']:
            return brain.brain_type == BrainType.AUGMENT
        elif task_type in ['pdf', 'document', 'docling']:
            return brain.brain_type == BrainType.DOCLING
        else:
            # Check if task type is in brain capabilities
            return task_type in [cap.lower() for cap in brain.capabilities]
    
    async def _execute_task(self, task: CoordinationTask, brain: BrainInstance):
        """Execute task on assigned brain"""
        try:
            # Update task status
            task.assigned_brain = brain.brain_id
            task.status = "running"
            task.started_at = datetime.now()
            
            # Update brain load
            brain.current_load = min(1.0, brain.current_load + 0.1)
            
            # Execute task based on brain type
            result = await self._call_brain_api(brain, task)
            
            if result['success']:
                # Task completed successfully
                task.output_data = result['data']
                task.status = "completed"
                task.completed_at = datetime.now()
                
                # Update metrics
                self.metrics.completed_tasks += 1
                self.metrics.active_tasks -= 1
                self.metrics.task_distribution[brain.brain_id] += 1
                
                # Calculate completion time
                completion_time = (task.completed_at - task.started_at).total_seconds()
                self._update_average_completion_time(completion_time)
                
                # Trigger callbacks
                await self._trigger_event_callbacks('task_completed', task)
                
                logger.info(f"✅ Task completed: {task.task_id} on {brain.brain_id}")
                
            else:
                # Task failed
                await self._handle_task_failure(task, result.get('error', 'Unknown error'))
            
            # Update brain load
            brain.current_load = max(0.0, brain.current_load - 0.1)
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _call_brain_api(self, brain: BrainInstance, task: CoordinationTask) -> Dict[str, Any]:
        """Call brain API to execute task"""
        try:
            # This would make actual HTTP calls to brain endpoints
            # For now, simulating the call
            
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate success/failure based on brain type and task
            if brain.brain_type == BrainType.EMBEDDING and task.task_type == 'embedding':
                return {
                    'success': True,
                    'data': {
                        'embeddings': [0.1, 0.2, 0.3],  # Simulated embedding
                        'processing_time': 0.1
                    }
                }
            elif brain.brain_type == BrainType.RERANKER and task.task_type == 'rerank':
                return {
                    'success': True,
                    'data': {
                        'scores': [0.9, 0.7, 0.5],  # Simulated scores
                        'processing_time': 0.05
                    }
                }
            elif brain.brain_type == BrainType.AUGMENT and task.task_type == 'chat':
                return {
                    'success': True,
                    'data': {
                        'response': 'Simulated chat response',
                        'processing_time': 0.2
                    }
                }
            elif brain.brain_type == BrainType.DOCLING and task.task_type == 'pdf':
                return {
                    'success': True,
                    'data': {
                        'markdown': '# Simulated PDF content',
                        'processing_time': 0.3
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported task type {task.task_type} for brain {brain.brain_type.value}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _handle_task_failure(self, task: CoordinationTask, error_message: str):
        """Handle task failure with retry logic"""
        try:
            task.error_message = error_message
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Retry task
                task.status = "queued"
                task.assigned_brain = None
                await self.task_queue.put(task)
                logger.warning(f"⚠️ Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                # Task permanently failed
                task.status = "failed"
                task.completed_at = datetime.now()
                
                # Update metrics
                self.metrics.failed_tasks += 1
                self.metrics.active_tasks -= 1
                self._update_error_rate()
                
                # Trigger callbacks
                await self._trigger_event_callbacks('task_failed', task)
                
                logger.error(f"❌ Task permanently failed: {task.task_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to handle task failure: {e}")
    
    async def _check_task_dependencies(self, task: CoordinationTask) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        for dep_task_id in task.dependencies:
            dep_task = self.active_tasks.get(dep_task_id)
            if not dep_task or dep_task.status != "completed":
                return False
        
        return True
    
    async def _brain_health_monitor(self):
        """Monitor brain health and update status"""
        while True:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                current_time = datetime.now()
                heartbeat_timeout = timedelta(seconds=self.config['heartbeat_interval_seconds'] * 2)
                
                for brain in list(self.brain_instances.values()):
                    # Check heartbeat timeout
                    if current_time - brain.last_heartbeat > heartbeat_timeout:
                        if brain.status == BrainStatus.ONLINE:
                            brain.status = BrainStatus.OFFLINE
                            await self._trigger_event_callbacks('brain_offline', brain)
                            logger.warning(f"⚠️ Brain {brain.brain_id} marked offline due to heartbeat timeout")
                    
                    # Update brain utilization metrics
                    self.metrics.brain_utilization[brain.brain_id] = brain.current_load
                
            except Exception as e:
                logger.error(f"❌ Brain health monitor error: {e}")
    
    async def _metrics_collector(self):
        """Collect and update coordination metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Update metrics every minute
                
                # Update active tasks count
                self.metrics.active_tasks = len([t for t in self.active_tasks.values() if t.status in ['queued', 'running']])
                
                # Store metrics in Redis
                await self._store_metrics()
                
            except Exception as e:
                logger.error(f"❌ Metrics collector error: {e}")
    
    async def _cleanup_completed_tasks(self):
        """Cleanup old completed tasks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now() - timedelta(hours=24)
                completed_tasks = []
                
                for task_id, task in list(self.active_tasks.items()):
                    if (task.status in ['completed', 'failed'] and
                        task.completed_at and
                        task.completed_at < cutoff_time):
                        completed_tasks.append(task_id)
                
                # Archive and remove old tasks
                for task_id in completed_tasks:
                    task = self.active_tasks.pop(task_id)
                    await self._archive_task(task)
                
                logger.info(f"🧹 Archived {len(completed_tasks)} completed tasks")
                
            except Exception as e:
                logger.error(f"❌ Task cleanup error: {e}")
    
    def _update_average_completion_time(self, completion_time: float):
        """Update average completion time metric"""
        if self.metrics.completed_tasks == 1:
            self.metrics.average_completion_time = completion_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_completion_time = (
                alpha * completion_time + 
                (1 - alpha) * self.metrics.average_completion_time
            )
    
    def _update_error_rate(self):
        """Update error rate metric"""
        total_completed = self.metrics.completed_tasks + self.metrics.failed_tasks
        if total_completed > 0:
            self.metrics.error_rate = self.metrics.failed_tasks / total_completed
    
    async def _trigger_event_callbacks(self, event_type: str, data: Any):
        """Trigger event callbacks"""
        callbacks = self.event_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    async def _store_brain_instance(self, brain: BrainInstance):
        """Store brain instance in Redis"""
        if self.redis_client:
            try:
                key = f"brain_instance:{brain.brain_id}"
                data = json.dumps(asdict(brain), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store brain instance: {e}")
    
    async def _load_brain_instances(self):
        """Load brain instances from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("brain_instance:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        brain_data = json.loads(data)
                        # Convert back to BrainInstance object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load brain instances: {e}")
    
    async def _store_metrics(self):
        """Store coordination metrics in Redis"""
        if self.redis_client:
            try:
                key = "coordination_metrics"
                data = json.dumps(asdict(self.metrics), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store metrics: {e}")
    
    async def _archive_task(self, task: CoordinationTask):
        """Archive completed task"""
        if self.redis_client:
            try:
                key = f"task_archive:{task.task_id}"
                data = json.dumps(asdict(task), default=str)
                await self.redis_client.setex(key, 86400 * 7, data)  # 7 days retention
            except Exception as e:
                logger.error(f"Failed to archive task: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for coordination events"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    async def get_brain_status(self, brain_id: str) -> Optional[BrainInstance]:
        """Get status of specific brain"""
        return self.brain_instances.get(brain_id)
    
    async def get_task_status(self, task_id: str) -> Optional[CoordinationTask]:
        """Get status of specific task"""
        return self.active_tasks.get(task_id)
    
    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        return {
            'metrics': asdict(self.metrics),
            'brain_count': len(self.brain_instances),
            'active_brains': len([b for b in self.brain_instances.values() if b.status == BrainStatus.ONLINE]),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global brain coordinator instance
brain_coordinator = BrainCoordinator()

async def initialize_brain_coordinator():
    """Initialize the global brain coordinator"""
    await brain_coordinator.initialize()

if __name__ == "__main__":
    # Test the brain coordinator
    async def test_brain_coordinator():
        await initialize_brain_coordinator()
        
        # Register test brains
        await brain_coordinator.register_brain(
            "brain1", BrainType.EMBEDDING, "localhost", 8001, ["embedding", "encode"]
        )
        await brain_coordinator.register_brain(
            "brain2", BrainType.RERANKER, "localhost", 8002, ["rerank", "score"]
        )
        
        # Submit test task
        task_id = await brain_coordinator.submit_task(
            "embedding", {"text": "test input"}, TaskPriority.HIGH
        )
        
        print(f"Task submitted: {task_id}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get metrics
        metrics = await brain_coordinator.get_coordination_metrics()
        print(f"Coordination metrics: {metrics}")
    
    asyncio.run(test_brain_coordinator())
