"""
Fallback Manager - Graceful Degradation Strategies
Provides fallback mechanisms and graceful degradation for system resilience

This module implements fallback strategies to maintain system functionality
when primary services fail, ensuring graceful degradation of capabilities.

Created: 2025-07-29 AEST
Purpose: Graceful degradation and fallback strategies
Module Size: 150 lines (modular design)
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategy types"""
    CACHE = "cache"
    SIMPLIFIED = "simplified"
    ALTERNATIVE_SERVICE = "alternative_service"
    DEFAULT_RESPONSE = "default_response"
    DEGRADED_MODE = "degraded_mode"
    CUSTOM = "custom"


@dataclass
class FallbackConfig:
    """Fallback configuration"""
    strategy: FallbackStrategy
    priority: int
    enabled: bool
    timeout: float
    fallback_data: Dict[str, Any]
    condition_func: Optional[Callable] = None


@dataclass
class FallbackExecution:
    """Fallback execution record"""
    execution_id: str
    original_function: str
    strategy_used: FallbackStrategy
    success: bool
    execution_time: float
    result: Any
    timestamp: float


class FallbackManager:
    """
    Fallback Manager
    
    Provides graceful degradation strategies and fallback mechanisms
    to maintain system functionality when primary services fail.
    """
    
    def __init__(self, brain_id: str):
        """Initialize fallback manager"""
        self.brain_id = brain_id
        self.enabled = True
        
        # Fallback configurations
        self.fallback_configs: Dict[str, List[FallbackConfig]] = {}
        self.fallback_handlers: Dict[FallbackStrategy, Callable] = {}
        
        # Execution tracking
        self.fallback_executions: Dict[str, FallbackExecution] = {}
        self.max_executions = 1000
        
        # Statistics
        self.stats = {
            "total_fallbacks": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "fallbacks_by_strategy": {strategy.value: 0 for strategy in FallbackStrategy}
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default fallback handlers
        self._initialize_default_handlers()
        
        logger.info(f"🛡️ Fallback Manager initialized for {brain_id}")
    
    def _initialize_default_handlers(self):
        """Initialize default fallback handlers"""
        
        async def cache_fallback(original_func, args, kwargs, config):
            """Cache-based fallback"""
            cache_key = f"{original_func.__name__}_{hash(str(args) + str(kwargs))}"
            cached_result = config.fallback_data.get(cache_key)
            
            if cached_result:
                logger.info(f"🗄️ Using cached fallback for {original_func.__name__}")
                return cached_result
            
            return None
        
        async def simplified_fallback(original_func, args, kwargs, config):
            """Simplified processing fallback"""
            simplified_func = config.fallback_data.get("simplified_function")
            
            if simplified_func:
                logger.info(f"⚡ Using simplified fallback for {original_func.__name__}")
                if asyncio.iscoroutinefunction(simplified_func):
                    return await simplified_func(*args, **kwargs)
                else:
                    return simplified_func(*args, **kwargs)
            
            return None
        
        async def default_response_fallback(original_func, args, kwargs, config):
            """Default response fallback"""
            default_response = config.fallback_data.get("default_response")
            
            if default_response:
                logger.info(f"📋 Using default response fallback for {original_func.__name__}")
                return default_response
            
            return None
        
        # Register default handlers
        self.fallback_handlers[FallbackStrategy.CACHE] = cache_fallback
        self.fallback_handlers[FallbackStrategy.SIMPLIFIED] = simplified_fallback
        self.fallback_handlers[FallbackStrategy.DEFAULT_RESPONSE] = default_response_fallback
    
    def register_fallback(self, function_name: str, config: FallbackConfig):
        """Register fallback configuration for a function"""
        with self._lock:
            if function_name not in self.fallback_configs:
                self.fallback_configs[function_name] = []
            
            self.fallback_configs[function_name].append(config)
            
            # Sort by priority (higher priority first)
            self.fallback_configs[function_name].sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"🛡️ Fallback registered for {function_name}: {config.strategy.value}")
    
    def register_fallback_handler(self, strategy: FallbackStrategy, handler: Callable):
        """Register custom fallback handler"""
        self.fallback_handlers[strategy] = handler
        logger.info(f"🔧 Fallback handler registered: {strategy.value}")
    
    async def execute_fallback(self, original_func: Callable, exception: Exception, 
                             *args, **kwargs) -> Any:
        """Execute fallback for failed function"""
        function_name = original_func.__name__
        
        # Get fallback configurations
        configs = self.fallback_configs.get(function_name, [])
        
        if not configs:
            logger.warning(f"No fallback configured for {function_name}")
            raise exception
        
        execution_id = f"fallback_{self.brain_id}_{int(time.time() * 1000000)}"
        
        for config in configs:
            if not config.enabled:
                continue
            
            # Check condition if specified
            if config.condition_func and not config.condition_func(exception, *args, **kwargs):
                continue
            
            try:
                start_time = time.time()
                
                # Execute fallback strategy
                result = await self._execute_strategy(
                    config.strategy, original_func, args, kwargs, config
                )
                
                if result is not None:
                    execution_time = time.time() - start_time
                    
                    # Record successful fallback
                    execution = FallbackExecution(
                        execution_id=execution_id,
                        original_function=function_name,
                        strategy_used=config.strategy,
                        success=True,
                        execution_time=execution_time,
                        result=result,
                        timestamp=time.time()
                    )
                    
                    self._store_execution(execution)
                    self._update_statistics(execution)
                    
                    logger.info(f"✅ Fallback successful: {function_name} using {config.strategy.value}")
                    return result
            
            except Exception as fallback_error:
                logger.warning(f"Fallback strategy {config.strategy.value} failed: {fallback_error}")
                continue
        
        # All fallbacks failed
        execution = FallbackExecution(
            execution_id=execution_id,
            original_function=function_name,
            strategy_used=FallbackStrategy.CUSTOM,  # No specific strategy
            success=False,
            execution_time=0.0,
            result=None,
            timestamp=time.time()
        )
        
        self._store_execution(execution)
        self._update_statistics(execution)
        
        logger.error(f"❌ All fallbacks failed for {function_name}")
        raise exception
    
    async def _execute_strategy(self, strategy: FallbackStrategy, original_func: Callable,
                              args: tuple, kwargs: dict, config: FallbackConfig) -> Any:
        """Execute specific fallback strategy"""
        
        if strategy in self.fallback_handlers:
            handler = self.fallback_handlers[strategy]
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    handler(original_func, args, kwargs, config),
                    timeout=config.timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Fallback strategy {strategy.value} timed out")
                return None
        
        else:
            logger.warning(f"No handler registered for strategy: {strategy.value}")
            return None
    
    def _store_execution(self, execution: FallbackExecution):
        """Store fallback execution"""
        with self._lock:
            self.fallback_executions[execution.execution_id] = execution
            
            # Limit memory usage
            if len(self.fallback_executions) > self.max_executions:
                oldest_id = min(self.fallback_executions.keys())
                del self.fallback_executions[oldest_id]
    
    def _update_statistics(self, execution: FallbackExecution):
        """Update fallback statistics"""
        with self._lock:
            self.stats["total_fallbacks"] += 1
            
            if execution.success:
                self.stats["successful_fallbacks"] += 1
                self.stats["fallbacks_by_strategy"][execution.strategy_used.value] += 1
            else:
                self.stats["failed_fallbacks"] += 1
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        with self._lock:
            success_rate = 0.0
            if self.stats["total_fallbacks"] > 0:
                success_rate = (self.stats["successful_fallbacks"] / self.stats["total_fallbacks"]) * 100
            
            return {
                "brain_id": self.brain_id,
                "enabled": self.enabled,
                "statistics": self.stats.copy(),
                "success_rate": success_rate,
                "configured_functions": len(self.fallback_configs),
                "registered_handlers": len(self.fallback_handlers),
                "stored_executions": len(self.fallback_executions)
            }


# Factory function for easy creation
def create_fallback_manager(brain_id: str) -> FallbackManager:
    """Factory function to create fallback manager"""
    return FallbackManager(brain_id)
