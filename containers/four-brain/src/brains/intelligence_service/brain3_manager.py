"""
Brain 3 Manager - Intelligence Service Manager
Real GLM4.5 Integration for Four-Brain Architecture

This module implements the Brain 3 manager class that was missing from the system.
It provides the main interface for the intelligence service brain.

Key Features:
- GLM4.5 API integration via Z.AI platform
- Task management and workflow orchestration
- Integration with existing four-brain architecture
- Redis communication for inter-brain messaging
- Zero fabrication compliance

Created: 2025-08-03 AEST
Purpose: Fix missing brain3_manager module causing import errors
"""

import os
import sys
import time
import asyncio
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import Brain 3 specific components
from .config.settings import Brain3Settings, get_brain3_settings
from .modules.brain3_integration import Brain3Integration
from .clients.glm_client import GLMClient
from .communication.brain_communicator import BrainCommunicator
from .agentic_loop import AgenticLoop

# Import shared components
sys.path.append('/workspace/src')
from shared.redis_client import RedisStreamsClient
from shared.logging import get_logger
from shared.monitoring import MetricsCollector

class Brain3Manager:
    """
    Brain 3 Manager - Intelligence Service Manager
    
    Manages the intelligence service brain with GLM4.5 integration,
    task orchestration, and inter-brain communication.
    """
    
    def __init__(self):
        """Initialize Brain 3 Manager"""
        self.logger = get_logger(__name__)
        self.settings = get_brain3_settings()
        
        # Initialize components
        self.glm_client = None
        self.brain_communicator = None
        self.agentic_loop = None
        self.redis_client = None
        self.metrics_collector = None
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.task_queue = asyncio.Queue()
        
        self.logger.info("Brain3Manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize all Brain 3 components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Brain 3 Manager...")
            
            # Initialize Redis client
            self.redis_client = RedisStreamsClient(
                redis_url=self.settings.redis_url,
                brain_id="intelligence"
            )
            await self.redis_client.connect()
            
            # Initialize GLM client
            self.glm_client = GLMClient(
                api_key=self.settings.glm_api_key,
                api_url=self.settings.glm_api_url,
                model=self.settings.glm_model
            )
            
            # Initialize brain communicator
            self.brain_communicator = BrainCommunicator(
                redis_client=self.redis_client,
                brain_id="brain3",
                logger=self.logger
            )
            
            # Initialize agentic loop
            self.agentic_loop = AgenticLoop(
                glm_client=self.glm_client,
                redis_client=self.redis_client,
                settings=self.settings
            )
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(
                service_name="brain3_intelligence",
                redis_client=self.redis_client
            )
            
            # Test connections
            await self._test_connections()
            
            self.is_initialized = True
            self.logger.info("Brain 3 Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Brain 3 Manager: {e}")
            return False
    
    async def _test_connections(self):
        """Test all external connections"""
        # Test Redis connection
        await self.redis_client.ping()
        
        # Test GLM API connection
        test_response = await self.glm_client.test_connection()
        if not test_response:
            raise Exception("GLM API connection test failed")
        
        self.logger.info("All connections tested successfully")
    
    async def start(self):
        """Start the Brain 3 Manager"""
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                raise Exception("Failed to initialize Brain 3 Manager")
        
        self.is_running = True
        self.logger.info("Brain 3 Manager started")
        
        # Start background tasks
        asyncio.create_task(self._process_task_queue())
        asyncio.create_task(self._monitor_health())
    
    async def stop(self):
        """Stop the Brain 3 Manager"""
        self.is_running = False
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Brain 3 Manager stopped")
    
    async def process_intelligence_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an intelligence request using GLM4.5
        
        Args:
            request: Intelligence request containing query and context
            
        Returns:
            Dict containing the intelligence response
        """
        try:
            start_time = time.time()
            
            # Extract request data
            query = request.get('query', '')
            context = request.get('context', {})
            task_type = request.get('task_type', 'general')
            
            self.logger.info(f"Processing intelligence request: {task_type}")
            
            # Use agentic loop for complex reasoning
            if self.settings.agentic_loop_enabled:
                response = await self.agentic_loop.process_request(
                    query=query,
                    context=context,
                    task_type=task_type
                )
            else:
                # Direct GLM API call
                response = await self.glm_client.generate_response(
                    query=query,
                    context=context
                )
            
            # Record metrics
            processing_time = time.time() - start_time
            await self.metrics_collector.record_request(
                request_type="intelligence",
                processing_time=processing_time,
                success=True
            )
            
            return {
                'status': 'success',
                'response': response,
                'processing_time': processing_time,
                'brain_id': 'brain3',
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing intelligence request: {e}")
            
            # Record error metrics
            await self.metrics_collector.record_request(
                request_type="intelligence",
                processing_time=0,
                success=False,
                error=str(e)
            )
            
            return {
                'status': 'error',
                'error': str(e),
                'brain_id': 'brain3',
                'timestamp': time.time()
            }
    
    async def _process_task_queue(self):
        """Process tasks from the task queue"""
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                await self.process_intelligence_request(task)
                
            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing task from queue: {e}")
    
    async def _monitor_health(self):
        """Monitor Brain 3 health and report status"""
        while self.is_running:
            try:
                # Check component health
                health_status = {
                    'brain_id': 'brain3',
                    'status': 'healthy' if self.is_running else 'unhealthy',
                    'redis_connected': await self._check_redis_health(),
                    'glm_api_available': await self._check_glm_health(),
                    'timestamp': time.time()
                }
                
                # Report health to Redis
                await self.redis_client.set(
                    'brain3:health',
                    health_status,
                    expire=30
                )
                
                # Wait before next health check
                await asyncio.sleep(self.settings.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.redis_client.ping()
            return True
        except:
            return False
    
    async def _check_glm_health(self) -> bool:
        """Check GLM API health"""
        try:
            return await self.glm_client.test_connection()
        except:
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current Brain 3 status"""
        return {
            'brain_id': 'brain3',
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'task_queue_size': self.task_queue.qsize(),
            'redis_connected': await self._check_redis_health(),
            'glm_api_available': await self._check_glm_health(),
            'timestamp': time.time()
        }

# Global instance for service use
brain3_manager = Brain3Manager()

async def get_brain3_manager() -> Brain3Manager:
    """Get the global Brain 3 manager instance"""
    if not brain3_manager.is_initialized:
        await brain3_manager.initialize()
    return brain3_manager
