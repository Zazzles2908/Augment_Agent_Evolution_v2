"""
Timeout Helper Utilities
Provides timeout wrappers and retry logic for Brain 3 operations
"""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def with_timeout(coro, timeout_seconds: float = 30, operation_name: str = "operation") -> Any:
    """
    Execute an async operation with timeout
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging
        
    Returns:
        Result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If operation times out
    """
    try:
        logger.debug(f"⏱️ Starting {operation_name} with {timeout_seconds}s timeout")
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        logger.debug(f"✅ {operation_name} completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error(f"⏰ {operation_name} timed out after {timeout_seconds}s")
        raise
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}")
        raise


async def with_retry(
    coro_func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    timeout_per_attempt: float = 10.0,
    operation_name: str = "operation"
) -> Any:
    """
    Execute an async operation with retry logic and exponential backoff
    
    Args:
        coro_func: Function that returns a coroutine
        max_attempts: Maximum number of attempts
        base_delay: Base delay between attempts
        timeout_per_attempt: Timeout for each attempt
        operation_name: Name for logging
        
    Returns:
        Result of the successful attempt
        
    Raises:
        Exception: Last exception if all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"🔄 {operation_name} attempt {attempt + 1}/{max_attempts}")
            
            # Create fresh coroutine for each attempt
            coro = coro_func()
            result = await with_timeout(coro, timeout_per_attempt, f"{operation_name} (attempt {attempt + 1})")
            
            logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            last_exception = e
            logger.warning(f"⚠️ {operation_name} attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(f"⏳ Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
    
    logger.error(f"❌ {operation_name} failed after {max_attempts} attempts")
    raise last_exception


def timeout_wrapper(timeout_seconds: float = 30, operation_name: Optional[str] = None):
    """
    Decorator to add timeout to async functions
    
    Args:
        timeout_seconds: Timeout in seconds
        operation_name: Optional operation name for logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            coro = func(*args, **kwargs)
            return await with_timeout(coro, timeout_seconds, name)
        return wrapper
    return decorator


def retry_wrapper(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    timeout_per_attempt: float = 10.0,
    operation_name: Optional[str] = None
):
    """
    Decorator to add retry logic to async functions
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between attempts
        timeout_per_attempt: Timeout for each attempt
        operation_name: Optional operation name for logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            
            async def coro_func():
                return await func(*args, **kwargs)
            
            return await with_retry(
                coro_func, max_attempts, base_delay, timeout_per_attempt, name
            )
        return wrapper
    return decorator


class TimeoutConfig:
    """Configuration for various timeout operations"""
    
    # Network operations
    REDIS_CONNECT_TIMEOUT = 10.0
    REDIS_OPERATION_TIMEOUT = 5.0
    SUPABASE_CONNECT_TIMEOUT = 15.0
    SUPABASE_OPERATION_TIMEOUT = 10.0
    HTTP_REQUEST_TIMEOUT = 30.0
    
    # Service initialization
    SERVICE_INIT_TIMEOUT = 60.0
    BRAIN_INIT_TIMEOUT = 45.0
    
    # Retry configuration
    MAX_RETRY_ATTEMPTS = 3
    BASE_RETRY_DELAY = 1.0
    
    @classmethod
    def get_timeout(cls, operation: str) -> float:
        """Get timeout for specific operation"""
        return getattr(cls, f"{operation.upper()}_TIMEOUT", 30.0)


async def safe_shutdown(tasks: list, timeout: float = 10.0):
    """
    Safely shutdown async tasks with timeout
    
    Args:
        tasks: List of asyncio tasks to shutdown
        timeout: Timeout for shutdown
    """
    if not tasks:
        return
    
    logger.info(f"🔄 Shutting down {len(tasks)} tasks...")
    
    try:
        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
        
        logger.info("✅ All tasks shut down successfully")
        
    except asyncio.TimeoutError:
        logger.warning(f"⏰ Task shutdown timed out after {timeout}s")
    except Exception as e:
        logger.error(f"❌ Error during task shutdown: {e}")


# Convenience functions for common operations
async def safe_redis_connect(redis_client, timeout: float = None) -> bool:
    """Safely connect to Redis with timeout"""
    timeout = timeout or TimeoutConfig.REDIS_CONNECT_TIMEOUT
    try:
        await with_timeout(redis_client.ping(), timeout, "Redis connection")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False


async def safe_supabase_connect(supabase_client, timeout: float = None) -> bool:
    """Safely test Supabase connection with timeout"""
    timeout = timeout or TimeoutConfig.SUPABASE_CONNECT_TIMEOUT
    try:
        # Simple query to test connection
        await with_timeout(
            supabase_client.table('sessions').select('*').limit(1).execute(),
            timeout,
            "Supabase connection"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False
