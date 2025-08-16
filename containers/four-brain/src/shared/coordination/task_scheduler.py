"""
Task Scheduling System for Four-Brain System v2
Advanced task scheduling with priority management and resource optimization

Created: 2025-07-30 AEST
Purpose: Intelligent task scheduling across Brain1, Brain2, Brain3, and Brain4
"""

import asyncio
import json
import logging
import time
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class SchedulingStrategy(Enum):
    """Task scheduling strategies"""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based
    SHORTEST_JOB_FIRST = "shortest_job_first"
    ROUND_ROBIN = "round_robin"
    FAIR_SHARE = "fair_share"
    DEADLINE_AWARE = "deadline_aware"
    RESOURCE_AWARE = "resource_aware"

@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    task_id: str
    task_type: str
    priority: TaskPriority
    estimated_duration: float
    deadline: Optional[datetime]
    dependencies: List[str]
    resource_requirements: Dict[str, float]
    retry_count: int
    max_retries: int
    created_at: datetime
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    assigned_brain: Optional[str]
    status: TaskStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SchedulingPolicy:
    """Scheduling policy configuration"""
    policy_id: str
    strategy: SchedulingStrategy
    priority_weights: Dict[str, float]
    resource_limits: Dict[str, float]
    deadline_buffer_minutes: int
    max_queue_size: int
    preemption_enabled: bool
    load_balancing_enabled: bool

@dataclass
class SchedulerMetrics:
    """Task scheduler metrics"""
    total_tasks_scheduled: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    average_wait_time: float
    average_execution_time: float
    queue_length: int
    throughput_per_minute: float
    deadline_miss_rate: float
    resource_utilization: Dict[str, float]

class TaskScheduler:
    """
    Advanced task scheduling system for Four-Brain coordination
    
    Features:
    - Multiple scheduling strategies (FIFO, Priority, SJF, etc.)
    - Deadline-aware scheduling
    - Resource-aware task placement
    - Dynamic priority adjustment
    - Dependency management
    - Load balancing integration
    - Performance optimization
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/16"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Task queues (priority heaps)
        self.task_queue: List[Tuple[float, ScheduledTask]] = []  # (priority_score, task)
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        
        # Scheduling state
        self.brain_availability: Dict[str, bool] = {}
        self.brain_capacity: Dict[str, Dict[str, float]] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        
        # Configuration
        self.config = {
            'default_strategy': SchedulingStrategy.PRIORITY,
            'max_queue_size': 1000,
            'scheduling_interval_seconds': 5,
            'deadline_check_interval_seconds': 30,
            'metrics_update_interval_seconds': 60,
            'task_timeout_minutes': 30,
            'dependency_timeout_minutes': 60,
            'preemption_enabled': False
        }
        
        # Scheduling policies
        self.scheduling_policies: Dict[str, SchedulingPolicy] = {}
        
        # Scheduler metrics
        self.metrics = SchedulerMetrics(
            total_tasks_scheduled=0,
            completed_tasks=0,
            failed_tasks=0,
            cancelled_tasks=0,
            average_wait_time=0.0,
            average_execution_time=0.0,
            queue_length=0,
            throughput_per_minute=0.0,
            deadline_miss_rate=0.0,
            resource_utilization={}
        )
        
        # Event callbacks
        self.task_callbacks: Dict[str, List[Callable]] = {
            'task_scheduled': [],
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'deadline_missed': []
        }
        
        logger.info("📅 Task Scheduler initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start scheduling services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize default policies
            await self._initialize_default_policies()
            
            # Load existing tasks
            await self._load_pending_tasks()
            
            # Start background services
            asyncio.create_task(self._scheduling_loop())
            asyncio.create_task(self._deadline_monitor())
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._dependency_resolver())
            
            logger.info("✅ Task Scheduler Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Task Scheduler: {e}")
            raise
    
    async def schedule_task(self, task_type: str, input_data: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL,
                          estimated_duration: float = 60.0,
                          deadline: Optional[datetime] = None,
                          dependencies: List[str] = None,
                          resource_requirements: Dict[str, float] = None,
                          max_retries: int = 3,
                          metadata: Dict[str, Any] = None) -> str:
        """Schedule a new task"""
        try:
            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{len(self.task_queue)}"
            
            # Create scheduled task
            task = ScheduledTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                estimated_duration=estimated_duration,
                deadline=deadline,
                dependencies=dependencies or [],
                resource_requirements=resource_requirements or {},
                retry_count=0,
                max_retries=max_retries,
                created_at=datetime.now(),
                scheduled_at=None,
                started_at=None,
                completed_at=None,
                assigned_brain=None,
                status=TaskStatus.PENDING,
                input_data=input_data,
                output_data=None,
                error_message=None,
                metadata=metadata or {}
            )
            
            # Check queue capacity
            if len(self.task_queue) >= self.config['max_queue_size']:
                logger.warning(f"Task queue at capacity, rejecting task {task_id}")
                return None
            
            # Add to queue with priority score
            priority_score = await self._calculate_priority_score(task)
            heapq.heappush(self.task_queue, (-priority_score, task))  # Negative for max-heap
            
            # Update dependencies
            if task.dependencies:
                self.task_dependencies[task_id] = set(task.dependencies)
            
            # Update metrics
            self.metrics.total_tasks_scheduled += 1
            self.metrics.queue_length = len(self.task_queue)
            
            # Store task
            await self._store_task(task)
            
            # Trigger callbacks
            await self._trigger_callbacks('task_scheduled', task)
            
            logger.info(f"✅ Task scheduled: {task_id} ({task_type}) with priority {priority.name}")
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Failed to schedule task: {e}")
            return None
    
    async def _calculate_priority_score(self, task: ScheduledTask) -> float:
        """Calculate priority score for task scheduling"""
        try:
            # Base priority from enum value
            base_score = task.priority.value * 100
            
            # Deadline urgency factor
            deadline_factor = 0
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds()
                if time_to_deadline > 0:
                    # More urgent as deadline approaches
                    deadline_factor = max(0, 100 - (time_to_deadline / 3600))  # Hours to deadline
                else:
                    deadline_factor = 200  # Past deadline - highest urgency
            
            # Duration factor (shorter jobs get slight boost)
            duration_factor = max(0, 50 - task.estimated_duration / 60)  # Minutes
            
            # Age factor (older tasks get boost to prevent starvation)
            age_minutes = (datetime.now() - task.created_at).total_seconds() / 60
            age_factor = min(50, age_minutes / 10)  # Up to 50 points for age
            
            # Retry penalty (failed tasks get lower priority)
            retry_penalty = task.retry_count * 10
            
            total_score = base_score + deadline_factor + duration_factor + age_factor - retry_penalty
            return max(0, total_score)
            
        except Exception as e:
            logger.error(f"❌ Priority score calculation failed: {e}")
            return task.priority.value * 100  # Fallback to base priority
    
    async def _scheduling_loop(self):
        """Main scheduling loop"""
        while True:
            try:
                await asyncio.sleep(self.config['scheduling_interval_seconds'])
                
                # Process ready tasks
                await self._process_ready_tasks()
                
                # Update queue metrics
                self.metrics.queue_length = len(self.task_queue)
                
            except Exception as e:
                logger.error(f"❌ Scheduling loop error: {e}")
    
    async def _process_ready_tasks(self):
        """Process tasks that are ready to run"""
        try:
            ready_tasks = []
            
            # Find tasks with satisfied dependencies
            while self.task_queue:
                priority_score, task = heapq.heappop(self.task_queue)
                
                if await self._are_dependencies_satisfied(task):
                    ready_tasks.append(task)
                else:
                    # Put back in queue
                    heapq.heappush(self.task_queue, (priority_score, task))
                    break  # Stop processing if dependencies not met
            
            # Schedule ready tasks
            for task in ready_tasks:
                await self._assign_task_to_brain(task)
            
        except Exception as e:
            logger.error(f"❌ Ready task processing failed: {e}")
    
    async def _are_dependencies_satisfied(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are satisfied"""
        try:
            if not task.dependencies:
                return True
            
            for dep_task_id in task.dependencies:
                # Check if dependency is completed
                if dep_task_id not in self.completed_tasks:
                    # Check if dependency failed
                    dep_task = await self._find_task_by_id(dep_task_id)
                    if dep_task and dep_task.status == TaskStatus.FAILED:
                        # Dependency failed, mark this task as failed too
                        task.status = TaskStatus.FAILED
                        task.error_message = f"Dependency {dep_task_id} failed"
                        await self._handle_task_completion(task)
                        return False
                    
                    # Dependency not yet completed
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False
    
    async def _assign_task_to_brain(self, task: ScheduledTask):
        """Assign task to an available brain"""
        try:
            # Find suitable brain
            suitable_brain = await self._find_suitable_brain(task)
            if not suitable_brain:
                # No suitable brain available, put back in queue
                priority_score = await self._calculate_priority_score(task)
                heapq.heappush(self.task_queue, (-priority_score, task))
                return
            
            # Assign task
            task.assigned_brain = suitable_brain
            task.status = TaskStatus.SCHEDULED
            task.scheduled_at = datetime.now()
            
            # Add to running tasks
            self.running_tasks[task.task_id] = task
            
            # Execute task
            asyncio.create_task(self._execute_task(task))
            
            # Trigger callbacks
            await self._trigger_callbacks('task_started', task)
            
            logger.info(f"✅ Task assigned: {task.task_id} -> {suitable_brain}")
            
        except Exception as e:
            logger.error(f"❌ Task assignment failed: {e}")
    
    async def _find_suitable_brain(self, task: ScheduledTask) -> Optional[str]:
        """Find suitable brain for task execution"""
        try:
            # This would integrate with brain coordinator and load balancer
            # For now, simulate brain selection
            available_brains = []
            
            # Check brain availability and capacity
            for brain_id in ['brain1', 'brain2', 'brain3', 'brain4']:
                if self.brain_availability.get(brain_id, True):
                    # Check resource requirements
                    if await self._check_resource_availability(brain_id, task.resource_requirements):
                        available_brains.append(brain_id)
            
            if not available_brains:
                return None
            
            # Select brain based on strategy
            strategy = self.config['default_strategy']
            
            if strategy == SchedulingStrategy.ROUND_ROBIN:
                # Simple round-robin selection
                return available_brains[len(self.running_tasks) % len(available_brains)]
            elif strategy == SchedulingStrategy.RESOURCE_AWARE:
                # Select brain with most available resources
                best_brain = available_brains[0]
                best_score = await self._calculate_resource_score(best_brain)
                
                for brain_id in available_brains[1:]:
                    score = await self._calculate_resource_score(brain_id)
                    if score > best_score:
                        best_score = score
                        best_brain = brain_id
                
                return best_brain
            else:
                # Default to first available
                return available_brains[0]
            
        except Exception as e:
            logger.error(f"❌ Brain selection failed: {e}")
            return None
    
    async def _check_resource_availability(self, brain_id: str, requirements: Dict[str, float]) -> bool:
        """Check if brain has required resources available"""
        try:
            if not requirements:
                return True
            
            brain_capacity = self.brain_capacity.get(brain_id, {})
            
            for resource, required_amount in requirements.items():
                available_amount = brain_capacity.get(resource, 1.0)
                if available_amount < required_amount:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Resource availability check failed: {e}")
            return True  # Default to available
    
    async def _calculate_resource_score(self, brain_id: str) -> float:
        """Calculate resource availability score for brain"""
        try:
            brain_capacity = self.brain_capacity.get(brain_id, {})
            
            # Calculate average available resources
            if brain_capacity:
                total_availability = sum(brain_capacity.values())
                return total_availability / len(brain_capacity)
            else:
                return 1.0  # Default full availability
            
        except Exception as e:
            logger.error(f"❌ Resource score calculation failed: {e}")
            return 0.5  # Default neutral score
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute task on assigned brain"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Simulate task execution
            await asyncio.sleep(min(task.estimated_duration, 5))  # Cap simulation time
            
            # Simulate success/failure
            import random
            success_rate = 0.9  # 90% success rate
            
            if random.random() < success_rate:
                # Task succeeded
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.output_data = {'result': f'Task {task.task_id} completed successfully'}
                
                # Update metrics
                self.metrics.completed_tasks += 1
                execution_time = (task.completed_at - task.started_at).total_seconds()
                self._update_average_execution_time(execution_time)
                
                # Trigger callbacks
                await self._trigger_callbacks('task_completed', task)
                
                logger.info(f"✅ Task completed: {task.task_id}")
            else:
                # Task failed
                await self._handle_task_failure(task, "Simulated task failure")
            
            # Handle completion
            await self._handle_task_completion(task)
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _handle_task_failure(self, task: ScheduledTask, error_message: str):
        """Handle task failure with retry logic"""
        try:
            task.error_message = error_message
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Retry task
                task.status = TaskStatus.RETRYING
                task.assigned_brain = None
                
                # Add back to queue with lower priority
                priority_score = await self._calculate_priority_score(task)
                heapq.heappush(self.task_queue, (-priority_score * 0.8, task))  # Reduce priority
                
                logger.warning(f"⚠️ Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                # Task permanently failed
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                # Update metrics
                self.metrics.failed_tasks += 1
                
                # Trigger callbacks
                await self._trigger_callbacks('task_failed', task)
                
                logger.error(f"❌ Task permanently failed: {task.task_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"❌ Task failure handling failed: {e}")
    
    async def _handle_task_completion(self, task: ScheduledTask):
        """Handle task completion cleanup"""
        try:
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)
            
            # Add to completed tasks
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self.completed_tasks[task.task_id] = task
                
                # Calculate wait time
                if task.scheduled_at:
                    wait_time = (task.scheduled_at - task.created_at).total_seconds()
                    self._update_average_wait_time(wait_time)
            
            # Clean up dependencies
            self.task_dependencies.pop(task.task_id, None)
            
            # Store updated task
            await self._store_task(task)
            
        except Exception as e:
            logger.error(f"❌ Task completion handling failed: {e}")
    
    async def _deadline_monitor(self):
        """Monitor task deadlines"""
        while True:
            try:
                await asyncio.sleep(self.config['deadline_check_interval_seconds'])
                
                current_time = datetime.now()
                missed_deadlines = []
                
                # Check running tasks
                for task in self.running_tasks.values():
                    if task.deadline and current_time > task.deadline:
                        missed_deadlines.append(task)
                
                # Check queued tasks
                for _, task in self.task_queue:
                    if task.deadline and current_time > task.deadline:
                        missed_deadlines.append(task)
                
                # Handle missed deadlines
                for task in missed_deadlines:
                    await self._handle_missed_deadline(task)
                
            except Exception as e:
                logger.error(f"❌ Deadline monitor error: {e}")
    
    async def _handle_missed_deadline(self, task: ScheduledTask):
        """Handle missed deadline"""
        try:
            logger.warning(f"⏰ Deadline missed for task {task.task_id}")
            
            # Update metrics
            self.metrics.deadline_miss_rate = (
                self.metrics.deadline_miss_rate * 0.9 + 0.1
            )  # Exponential moving average
            
            # Trigger callbacks
            await self._trigger_callbacks('deadline_missed', task)
            
            # Optionally cancel task or increase priority
            if task.status == TaskStatus.PENDING:
                # Increase priority for queued tasks
                task.priority = TaskPriority.URGENT
            
        except Exception as e:
            logger.error(f"❌ Deadline handling failed: {e}")
    
    async def _dependency_resolver(self):
        """Resolve task dependencies"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for orphaned dependencies
                current_time = datetime.now()
                timeout = timedelta(minutes=self.config['dependency_timeout_minutes'])
                
                orphaned_tasks = []
                for task_id, dependencies in self.task_dependencies.items():
                    task = await self._find_task_by_id(task_id)
                    if task and current_time - task.created_at > timeout:
                        # Check if any dependencies are still missing
                        missing_deps = [dep for dep in dependencies if dep not in self.completed_tasks]
                        if missing_deps:
                            orphaned_tasks.append(task)
                
                # Handle orphaned tasks
                for task in orphaned_tasks:
                    task.status = TaskStatus.FAILED
                    task.error_message = "Dependency timeout"
                    await self._handle_task_completion(task)
                
            except Exception as e:
                logger.error(f"❌ Dependency resolver error: {e}")
    
    async def _metrics_collector(self):
        """Collect scheduling metrics"""
        while True:
            try:
                await asyncio.sleep(self.config['metrics_update_interval_seconds'])
                
                # Calculate throughput
                completed_in_last_minute = sum(
                    1 for task in self.completed_tasks.values()
                    if task.completed_at and 
                    datetime.now() - task.completed_at < timedelta(minutes=1)
                )
                self.metrics.throughput_per_minute = completed_in_last_minute
                
                # Update resource utilization
                await self._update_resource_utilization()
                
                # Store metrics
                await self._store_metrics()
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
    
    async def _update_resource_utilization(self):
        """Update resource utilization metrics"""
        try:
            # Calculate average resource utilization across all brains
            total_utilization = {}
            brain_count = 0
            
            for brain_id, capacity in self.brain_capacity.items():
                brain_count += 1
                for resource, available in capacity.items():
                    utilization = 1.0 - available  # Utilization = 1 - available
                    if resource not in total_utilization:
                        total_utilization[resource] = 0
                    total_utilization[resource] += utilization
            
            # Calculate averages
            if brain_count > 0:
                for resource in total_utilization:
                    total_utilization[resource] /= brain_count
            
            self.metrics.resource_utilization = total_utilization
            
        except Exception as e:
            logger.error(f"❌ Resource utilization update failed: {e}")
    
    def _update_average_wait_time(self, wait_time: float):
        """Update average wait time metric"""
        if self.metrics.completed_tasks == 1:
            self.metrics.average_wait_time = wait_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_wait_time = (
                alpha * wait_time + 
                (1 - alpha) * self.metrics.average_wait_time
            )
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        if self.metrics.completed_tasks == 1:
            self.metrics.average_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_execution_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.average_execution_time
            )
    
    async def _find_task_by_id(self, task_id: str) -> Optional[ScheduledTask]:
        """Find task by ID across all collections"""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check queue
        for _, task in self.task_queue:
            if task.task_id == task_id:
                return task
        
        return None
    
    async def _trigger_callbacks(self, event_type: str, task: ScheduledTask):
        """Trigger event callbacks"""
        callbacks = self.task_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"Task callback error: {e}")
    
    async def _initialize_default_policies(self):
        """Initialize default scheduling policies"""
        self.scheduling_policies = {
            'default_priority': SchedulingPolicy(
                policy_id='default_priority',
                strategy=SchedulingStrategy.PRIORITY,
                priority_weights={'critical': 5.0, 'urgent': 4.0, 'high': 3.0, 'normal': 2.0, 'low': 1.0},
                resource_limits={'cpu': 0.8, 'memory': 0.8, 'gpu': 0.9},
                deadline_buffer_minutes=30,
                max_queue_size=1000,
                preemption_enabled=False,
                load_balancing_enabled=True
            )
        }
    
    async def _store_task(self, task: ScheduledTask):
        """Store task in Redis"""
        if self.redis_client:
            try:
                key = f"scheduled_task:{task.task_id}"
                data = json.dumps(asdict(task), default=str)
                await self.redis_client.setex(key, 86400, data)  # 24 hour TTL
            except Exception as e:
                logger.error(f"Failed to store task: {e}")
    
    async def _store_metrics(self):
        """Store scheduler metrics in Redis"""
        if self.redis_client:
            try:
                key = "task_scheduler_metrics"
                data = json.dumps(asdict(self.metrics), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store scheduler metrics: {e}")
    
    async def _load_pending_tasks(self):
        """Load pending tasks from Redis"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("scheduled_task:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        task_data = json.loads(data)
                        # Convert back to ScheduledTask object
                        # This would need proper deserialization logic
                        pass
            except Exception as e:
                logger.error(f"Failed to load pending tasks: {e}")
    
    def register_task_callback(self, event_type: str, callback: Callable):
        """Register callback for task events"""
        if event_type in self.task_callbacks:
            self.task_callbacks[event_type].append(callback)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task"""
        try:
            task = await self._find_task_by_id(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False  # Cannot cancel completed tasks
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Remove from appropriate collections
            self.running_tasks.pop(task_id, None)
            
            # Remove from queue if present
            self.task_queue = [(score, t) for score, t in self.task_queue if t.task_id != task_id]
            heapq.heapify(self.task_queue)
            
            # Update metrics
            self.metrics.cancelled_tasks += 1
            
            # Store updated task
            await self._store_task(task)
            
            logger.info(f"✅ Task cancelled: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Task cancellation failed: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        task = await self._find_task_by_id(task_id)
        if task:
            return {
                'task_id': task.task_id,
                'status': task.status.value,
                'priority': task.priority.name,
                'assigned_brain': task.assigned_brain,
                'created_at': task.created_at.isoformat(),
                'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None,
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'retry_count': task.retry_count,
                'error_message': task.error_message
            }
        return None
    
    async def get_scheduler_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler metrics"""
        return {
            'metrics': asdict(self.metrics),
            'queue_length': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'brain_availability': self.brain_availability.copy(),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }

# Global task scheduler instance
task_scheduler = TaskScheduler()

async def initialize_task_scheduler():
    """Initialize the global task scheduler"""
    await task_scheduler.initialize()

if __name__ == "__main__":
    # Test the task scheduler
    async def test_task_scheduler():
        await initialize_task_scheduler()
        
        # Schedule test tasks
        task_id1 = await task_scheduler.schedule_task(
            "embedding", {"text": "test input"}, TaskPriority.HIGH, 30.0
        )
        task_id2 = await task_scheduler.schedule_task(
            "rerank", {"documents": ["doc1", "doc2"]}, TaskPriority.NORMAL, 15.0,
            dependencies=[task_id1]
        )
        
        print(f"Scheduled tasks: {task_id1}, {task_id2}")
        
        # Wait for processing
        await asyncio.sleep(10)
        
        # Get metrics
        metrics = await task_scheduler.get_scheduler_metrics()
        print(f"Scheduler metrics: {metrics}")
        
        # Get task status
        status1 = await task_scheduler.get_task_status(task_id1)
        status2 = await task_scheduler.get_task_status(task_id2)
        print(f"Task statuses: {status1}, {status2}")
    
    asyncio.run(test_task_scheduler())
