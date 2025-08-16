"""
Centralized Database Connection Manager - Fix Authentication Issues
Provides unified database connection management for all Four-Brain services

This module fixes the PostgreSQL authentication failures by providing
a centralized, properly configured connection management system.

Created: 2025-07-29 AEST
Purpose: Fix database authentication failures across all services
Module Size: 150 lines (modular design)
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse, parse_qs
import asyncpg
from supabase import create_client, Client
import threading

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Centralized Database Connection Manager
    
    Fixes authentication issues by providing properly configured
    connections to both local PostgreSQL and Supabase databases.
    """
    
    def __init__(self, brain_id: str):
        """Initialize connection manager for specific brain"""
        self.brain_id = brain_id
        self.supabase_client: Optional[Client] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._lock = threading.Lock()
        
        # Load configuration from environment
        self._load_configuration()
        
        # Connection tracking
        self.connection_stats = {
            "supabase_connected": False,
            "postgres_connected": False,
            "last_connection_attempt": None,
            "connection_errors": [],
            "successful_connections": 0,
            "failed_connections": 0
        }
        
        logger.info(f"🔗 Database Connection Manager initialized for {brain_id}")
    
    def _load_configuration(self):
        """Load and validate database configuration from environment"""
        # Supabase configuration
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://ustcfwmonegxeoqeixgg.supabase.co')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.supabase_anon_key = os.getenv('SUPABASE_ANON_KEY')
        
        # PostgreSQL configuration
        self.postgres_host = os.getenv('POSTGRES_HOST', 'v6-ai-postgres')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', '5432'))
        self.postgres_db = os.getenv('POSTGRES_DB', 'augment_agent')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', 'ai_secure_2024')
        
        # Schema configuration
        self.schema_name = os.getenv('AUGMENT_SCHEMA', 'augment_agent')
        
        # Build connection strings
        self._build_connection_strings()
    
    def _build_connection_strings(self):
        """Build proper connection strings for different database types"""
        
        # Local PostgreSQL connection string
        self.local_pg_url = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        
        # Supabase direct PostgreSQL connection string (FIXED FORMAT)
        if self.supabase_service_key:
            # Extract project reference from Supabase URL
            project_ref = self.supabase_url.split('//')[1].split('.')[0]
            
            # Correct Supabase PostgreSQL connection format
            # Region: ap-southeast-2 (from Supabase project info)
            self.supabase_pg_url = (
                f"postgresql://postgres.{project_ref}:{self.supabase_service_key}"
                f"@aws-0-ap-southeast-2.pooler.supabase.com:5432/postgres"
            )
        else:
            self.supabase_pg_url = None
            logger.warning("⚠️ SUPABASE_SERVICE_ROLE_KEY not found - Supabase PostgreSQL unavailable")
        
        logger.info(f"🔧 Connection strings configured for {self.brain_id}")
    
    async def initialize(self, prefer_supabase: bool = True) -> bool:
        """Initialize database connections with preference order"""
        if self._initialized:
            return True
        
        with self._lock:
            if self._initialized:
                return True
            
            self.connection_stats["last_connection_attempt"] = time.time()
            
            supabase_success = False
            postgres_success = False
            
            if prefer_supabase:
                # Try Supabase first
                supabase_success = await self._initialize_supabase()
                if not supabase_success:
                    # Fallback to local PostgreSQL
                    postgres_success = await self._initialize_local_postgres()
            else:
                # Try local PostgreSQL first
                postgres_success = await self._initialize_local_postgres()
                if not postgres_success:
                    # Fallback to Supabase
                    supabase_success = await self._initialize_supabase()
            
            # Update connection status
            self.connection_stats["supabase_connected"] = supabase_success
            self.connection_stats["postgres_connected"] = postgres_success
            
            # Consider initialization successful if at least one connection works
            if supabase_success or postgres_success:
                self._initialized = True
                self.connection_stats["successful_connections"] += 1
                logger.info(f"✅ Database connections initialized for {self.brain_id}")
                return True
            else:
                self.connection_stats["failed_connections"] += 1
                logger.error(f"❌ All database connections failed for {self.brain_id}")
                return False
    
    async def _initialize_supabase(self) -> bool:
        """Initialize Supabase REST client and PostgreSQL connection"""
        try:
            if not self.supabase_service_key:
                logger.warning("⚠️ Supabase service key not available")
                return False
            
            # Initialize Supabase REST client
            logger.info("🔄 Initializing Supabase REST client...")
            self.supabase_client = create_client(
                self.supabase_url,
                self.supabase_service_key
            )
            
            # Test Supabase connection
            await self._test_supabase_connection()
            
            # Initialize Supabase PostgreSQL pool if URL available
            if self.supabase_pg_url:
                logger.info("🔄 Initializing Supabase PostgreSQL pool...")
                self.pg_pool = await asyncpg.create_pool(
                    self.supabase_pg_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60,
                    server_settings={
                        'search_path': f'{self.schema_name},public',
                        'application_name': f'four_brain_{self.brain_id}'
                    }
                )
                
                # Test PostgreSQL connection
                await self._test_postgres_connection()
            
            logger.info("✅ Supabase connections initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Supabase initialization failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.connection_stats["connection_errors"].append(error_msg)
            return False
    
    async def _initialize_local_postgres(self) -> bool:
        """Initialize local PostgreSQL connection"""
        try:
            logger.info("🔄 Initializing local PostgreSQL pool...")
            self.pg_pool = await asyncpg.create_pool(
                self.local_pg_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'search_path': f'{self.schema_name},public',
                    'application_name': f'four_brain_{self.brain_id}'
                }
            )
            
            # Test connection
            await self._test_postgres_connection()
            
            logger.info("✅ Local PostgreSQL connection initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Local PostgreSQL initialization failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.connection_stats["connection_errors"].append(error_msg)
            return False
    
    async def _test_supabase_connection(self):
        """Test Supabase REST API connection"""
        if not self.supabase_client:
            raise Exception("Supabase client not initialized")
        
        # Simple test query
        response = self.supabase_client.table('sessions').select('id').limit(1).execute()
        logger.debug("✅ Supabase REST API test successful")
    
    async def _test_postgres_connection(self):
        """Test PostgreSQL connection"""
        if not self.pg_pool:
            raise Exception("PostgreSQL pool not initialized")
        
        async with self.pg_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise Exception("PostgreSQL test query failed")
        
        logger.debug("✅ PostgreSQL connection test successful")
    
    async def get_supabase_client(self) -> Optional[Client]:
        """Get Supabase REST client"""
        if not self._initialized:
            await self.initialize()
        return self.supabase_client
    
    async def get_postgres_connection(self):
        """Get PostgreSQL connection from pool"""
        if not self._initialized:
            await self.initialize()
        
        if not self.pg_pool:
            raise Exception("PostgreSQL pool not available")
        
        return self.pg_pool.acquire()
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute query on PostgreSQL with automatic connection management"""
        if not self._initialized:
            await self.initialize()
        
        if not self.pg_pool:
            raise Exception("No PostgreSQL connection available")
        
        async with self.pg_pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_scalar(self, query: str, *args) -> Any:
        """Execute scalar query on PostgreSQL"""
        if not self._initialized:
            await self.initialize()
        
        if not self.pg_pool:
            raise Exception("No PostgreSQL connection available")
        
        async with self.pg_pool.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information and status"""
        return {
            "brain_id": self.brain_id,
            "initialized": self._initialized,
            "supabase_url": self.supabase_url,
            "postgres_host": self.postgres_host,
            "postgres_db": self.postgres_db,
            "schema_name": self.schema_name,
            "stats": self.connection_stats.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "brain_id": self.brain_id,
            "overall_status": "unknown",
            "supabase_status": "unknown",
            "postgres_status": "unknown",
            "errors": []
        }
        
        # Test Supabase if available
        if self.supabase_client:
            try:
                await self._test_supabase_connection()
                health_status["supabase_status"] = "healthy"
            except Exception as e:
                health_status["supabase_status"] = "unhealthy"
                health_status["errors"].append(f"Supabase: {str(e)}")
        
        # Test PostgreSQL if available
        if self.pg_pool:
            try:
                await self._test_postgres_connection()
                health_status["postgres_status"] = "healthy"
            except Exception as e:
                health_status["postgres_status"] = "unhealthy"
                health_status["errors"].append(f"PostgreSQL: {str(e)}")
        
        # Determine overall status
        if (health_status["supabase_status"] == "healthy" or 
            health_status["postgres_status"] == "healthy"):
            health_status["overall_status"] = "healthy"
        else:
            health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    async def cleanup(self):
        """Clean up database connections"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                logger.info(f"🧹 PostgreSQL pool closed for {self.brain_id}")
            
            # Supabase client doesn't need explicit cleanup
            self._initialized = False
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed for {self.brain_id}: {e}")


# Factory function for easy creation
def create_connection_manager(brain_id: str) -> DatabaseConnectionManager:
    """Factory function to create database connection manager"""
    return DatabaseConnectionManager(brain_id)
