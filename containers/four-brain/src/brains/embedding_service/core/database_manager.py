#!/usr/bin/env python3
"""
Hybrid Database Manager for Three-Brain Architecture
Implements Option C: Hybrid Approach
- Supabase REST API for simple operations (business logic)
- Direct PostgreSQL for vector operations (AI workloads)
"""

import os
import asyncio
import asyncpg
import threading
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class HybridDatabaseManager:
    """
    Hybrid database manager that provides:
    1. Supabase REST API for simple CRUD operations
    2. Direct PostgreSQL connections for vector similarity searches
    3. Connection pooling for performance
    4. Thread-safe operations for AI workloads
    """
    
    def __init__(self):
        """Initialize hybrid database manager."""
        self.supabase_client: Optional[Client] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._lock = threading.Lock()
        
        # Configuration from environment
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://ustcfwmonegxeoqeixgg.supabase.co')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.schema_name = 'augment_agent'
        
        # PostgreSQL connection string for direct access
        self.pg_connection_string = self._build_pg_connection_string()
        
        logger.info("🔗 Hybrid Database Manager initialized")
    
    def _build_pg_connection_string(self) -> str:
        """Build PostgreSQL connection string for Supabase direct connection."""
        # FIXED: Use service key for authentication with proper format
        project_ref = "ustcfwmonegxeoqeixgg"
        # Use service key as password for direct PostgreSQL connection
        password = self.supabase_service_key

        # Correct Supabase direct database connection format
        return f"postgresql://postgres.{project_ref}:{password}@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
    
    async def initialize(self):
        """Initialize both Supabase REST client and PostgreSQL connection pool with resilient error handling."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            supabase_success = False
            pg_success = False

            # RESILIENT INITIALIZATION: Try each component separately
            try:
                # Initialize Supabase REST client
                logger.info("🔄 Initializing Supabase REST client...")
                self.supabase_client = create_client(
                    self.supabase_url,
                    self.supabase_service_key
                )

                # Test Supabase connection
                await self._test_supabase_connection()
                supabase_success = True
                logger.info("✅ Supabase REST client initialized successfully")

            except Exception as e:
                logger.error(f"❌ Supabase initialization failed: {e}")
                logger.warning("⚠️ Continuing with PostgreSQL initialization...")

            try:
                # Initialize PostgreSQL connection pool for vector operations
                logger.info("🔄 Initializing PostgreSQL connection pool...")
                self.pg_pool = await asyncpg.create_pool(
                    self.pg_connection_string,
                    min_size=3,           # Minimum connections for AI workloads
                    max_size=15,          # Maximum connections for concurrent operations
                    command_timeout=120,  # Longer timeout for vector operations
                    max_queries=50000,    # High query limit for AI workloads
                    max_inactive_connection_lifetime=300,  # 5 minutes
                    server_settings={
                        'search_path': f'{self.schema_name},public',
                        'work_mem': '256MB',           # Increase work memory for vector ops
                        'maintenance_work_mem': '1GB', # For index operations
                        'effective_cache_size': '4GB'  # Optimize for vector similarity
                    }
                )

                # Test PostgreSQL connection
                await self._test_pg_connection()
                pg_success = True
                logger.info("✅ PostgreSQL connection pool initialized successfully")

            except Exception as e:
                logger.error(f"❌ PostgreSQL initialization failed: {e}")

            # Mark as initialized if at least one component works
            if supabase_success or pg_success:
                self._initialized = True
                logger.info(f"✅ Hybrid Database Manager initialized (Supabase: {supabase_success}, PostgreSQL: {pg_success})")
            else:
                logger.error("❌ Both Supabase and PostgreSQL initialization failed")
                raise Exception("Complete database initialization failure")
    
    async def _test_supabase_connection(self):
        """Test Supabase REST API connection with correct schema."""
        try:
            # FIXED: Use actual column name from schema
            response = self.supabase_client.schema(self.schema_name).table('knowledge').select('knowledge_id').limit(1).execute()
            logger.info("✅ Supabase REST API connection successful")
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            raise
    
    async def _test_pg_connection(self):
        """Test direct PostgreSQL connection with correct schema."""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema_name}.knowledge")
                logger.info(f"✅ PostgreSQL connection successful - {result} records in knowledge table")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection test failed: {e}")
            raise
    
    # SUPABASE REST API METHODS (for simple operations)
    
    def create_knowledge_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge entry using Supabase REST API with correct column mapping."""
        # Track database operation with flow monitoring
        try:
            from flow_monitoring import get_flow_monitor, DatabaseType
            flow_monitor = get_flow_monitor()

            with flow_monitor.track_database_operation(DatabaseType.POSTGRESQL, "insert"):
                # FIXED: Map application data to actual schema columns
                mapped_data = {
                    'knowledge_title': data.get('title', ''),
                    'knowledge_content': data.get('content', {}),  # Handle jsonb content
                    'domain': data.get('domain', 'unknown'),
                    'knowledge_type': data.get('knowledge_type', 'general'),
                    'confidence_level': data.get('confidence_level', 0.5),
                    'tags': data.get('tags', [])
                }

                response = self.supabase_client.schema(self.schema_name).table('knowledge').insert(mapped_data).execute()
                return response.data[0] if response.data else {}

        except ImportError:
            # Flow monitoring not available, proceed without tracking
            mapped_data = {
                'knowledge_title': data.get('title', ''),
                'knowledge_content': data.get('content', {}),  # Handle jsonb content
                'domain': data.get('domain', 'unknown'),
                'knowledge_type': data.get('knowledge_type', 'general'),
                'confidence_level': data.get('confidence_level', 0.5),
                'tags': data.get('tags', [])
            }

            response = self.supabase_client.schema(self.schema_name).table('knowledge').insert(mapped_data).execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"❌ Failed to create knowledge entry: {e}")
            raise
    
    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge entry by ID using Supabase REST API with correct column mapping."""
        try:
            # FIXED: Use actual primary key column name
            response = self.supabase_client.schema(self.schema_name).table('knowledge').select('*').eq('knowledge_id', knowledge_id).execute()

            if response.data:
                # Map back to expected format for backward compatibility
                entry = response.data[0]
                return {
                    'id': entry['knowledge_id'],
                    'title': entry['knowledge_title'],
                    'content': entry['knowledge_content'],
                    'domain': entry['domain'],
                    'knowledge_type': entry['knowledge_type'],
                    'confidence_level': entry['confidence_level'],
                    'tags': entry['tags'],
                    'created_at': entry['created_at'],
                    'updated_at': entry['updated_at']
                }
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get knowledge entry: {e}")
            raise
    
    def update_knowledge_entry(self, knowledge_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge entry using Supabase REST API with correct column mapping."""
        try:
            # FIXED: Map application data to actual schema columns
            mapped_data = {}
            if 'title' in data:
                mapped_data['knowledge_title'] = data['title']
            if 'content' in data:
                mapped_data['knowledge_content'] = data['content']
            if 'confidence_level' in data:
                mapped_data['confidence_level'] = data['confidence_level']
            if 'tags' in data:
                mapped_data['tags'] = data['tags']

            # FIXED: Use actual primary key column name
            response = self.supabase_client.schema(self.schema_name).table('knowledge').update(mapped_data).eq('knowledge_id', knowledge_id).execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"❌ Failed to update knowledge entry: {e}")
            raise
    
    # POSTGRESQL DIRECT METHODS (for vector operations)
    
    async def store_embedding(self, knowledge_id: str, embedding: List[float]) -> bool:
        """Store embedding vector using direct PostgreSQL connection with correct column names."""
        try:
            async with self.pg_pool.acquire() as conn:
                # FIXED: Use actual primary key column name
                await conn.execute(
                    f"UPDATE {self.schema_name}.knowledge SET embedding = $1 WHERE knowledge_id = $2",
                    embedding, knowledge_id
                )
                logger.info(f"✅ Embedding stored for knowledge ID: {knowledge_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to store embedding: {e}")
            return False
    
    async def vector_similarity_search(self, query_embedding: List[float],
                                     similarity_threshold: float = 0.7,
                                     max_results: int = 10,
                                     use_index: bool = True,
                                     include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Perform optimized vector similarity search using direct PostgreSQL connection.

        Args:
            query_embedding: Query vector (2000-dimensional)
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            max_results: Maximum number of results to return
            use_index: Whether to use vector index for faster search
            include_metadata: Whether to include full metadata in results
        """
        # Track database operation with flow monitoring
        try:
            from flow_monitoring import get_flow_monitor, DatabaseType
            flow_monitor = get_flow_monitor()

            async with flow_monitor.track_database_operation(DatabaseType.POSTGRESQL, "vector_search"):
                async with self.pg_pool.acquire() as conn:
                    # Optimized query with conditional metadata selection
                    if include_metadata:
                        select_fields = """
                            id, title, content, domain, knowledge_type,
                            confidence_level, tags, created_at, updated_at
                        """
                    else:
                        select_fields = "id, title, confidence_level"

                    # FIXED: Use actual column names with aliasing for backward compatibility
                    if use_index:
                        # Use IVFFlat index for faster approximate search
                        query = f"""
                        SELECT
                            knowledge_id as id,
                            knowledge_title as title,
                            knowledge_content as content,
                            domain, knowledge_type, confidence_level, tags, created_at, updated_at,
                            1 - (embedding <=> $1) as similarity_score
                        FROM {self.schema_name}.knowledge
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1
                        LIMIT $2
                        """
                        rows = await conn.fetch(query, query_embedding, max_results * 2)

                        # Filter by threshold after retrieval for better performance
                        filtered_rows = [row for row in rows if (1 - row['similarity_score']) >= similarity_threshold][:max_results]
                    else:
                        # Exact search with threshold filtering
                        query = f"""
                        SELECT
                            knowledge_id as id,
                            knowledge_title as title,
                            knowledge_content as content,
                            domain, knowledge_type, confidence_level, tags, created_at, updated_at,
                            1 - (embedding <=> $1) as similarity_score
                        FROM {self.schema_name}.knowledge
                        WHERE embedding IS NOT NULL
                        AND 1 - (embedding <=> $1) >= $2
                        ORDER BY embedding <=> $1
                        LIMIT $3
                        """
                        filtered_rows = await conn.fetch(query, query_embedding, similarity_threshold, max_results)

                results = []
                for row in filtered_rows:
                    result = {
                        'id': row['id'],
                        'title': row['title'],
                        'similarity_score': float(1 - row['similarity_score'])  # Convert distance to similarity
                    }

                    if include_metadata:
                        result.update({
                            'content': row['content'],
                            'domain': row['domain'],
                            'knowledge_type': row['knowledge_type'],
                            'confidence_level': row['confidence_level'],
                            'tags': row['tags'],
                            'created_at': row['created_at'],
                            'updated_at': row['updated_at']
                        })

                    results.append(result)

                logger.info(f"✅ Vector search returned {len(results)} results (threshold: {similarity_threshold})")
                return results

        except Exception as e:
            logger.error(f"❌ Vector similarity search failed: {e}")
            return []
    
    async def batch_store_embeddings(self, embeddings_data: List[Tuple[str, List[float]]]) -> int:
        """Batch store embeddings for performance with correct column names."""
        try:
            async with self.pg_pool.acquire() as conn:
                async with conn.transaction():
                    count = 0
                    for knowledge_id, embedding in embeddings_data:
                        # FIXED: Use actual primary key column name
                        await conn.execute(
                            f"UPDATE {self.schema_name}.knowledge SET embedding = $1 WHERE knowledge_id = $2",
                            embedding, knowledge_id
                        )
                        count += 1

                    logger.info(f"✅ Batch stored {count} embeddings")
                    return count

        except Exception as e:
            logger.error(f"❌ Batch embedding storage failed: {e}")
            return 0

    async def hybrid_search(self, query_text: str, query_embedding: List[float],
                          text_weight: float = 0.3, vector_weight: float = 0.7,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text search and vector similarity.

        Args:
            query_text: Text query for full-text search
            query_embedding: Vector for similarity search
            text_weight: Weight for text search score (0.0 to 1.0)
            vector_weight: Weight for vector search score (0.0 to 1.0)
            max_results: Maximum number of results
        """
        try:
            async with self.pg_pool.acquire() as conn:
                # Hybrid search query combining text and vector similarity
                query = f"""
                WITH text_search AS (
                    SELECT
                        id, title, content, domain, knowledge_type, confidence_level,
                        ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', $1)) as text_score
                    FROM {self.schema_name}.knowledge
                    WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', $1)
                ),
                vector_search AS (
                    SELECT
                        id, title, content, domain, knowledge_type, confidence_level,
                        1 - (embedding <=> $2) as vector_score
                    FROM {self.schema_name}.knowledge
                    WHERE embedding IS NOT NULL
                )
                SELECT
                    COALESCE(t.id, v.id) as id,
                    COALESCE(t.title, v.title) as title,
                    COALESCE(t.content, v.content) as content,
                    COALESCE(t.domain, v.domain) as domain,
                    COALESCE(t.knowledge_type, v.knowledge_type) as knowledge_type,
                    COALESCE(t.confidence_level, v.confidence_level) as confidence_level,
                    COALESCE(t.text_score, 0) * $3 + COALESCE(v.vector_score, 0) * $4 as combined_score,
                    COALESCE(t.text_score, 0) as text_score,
                    COALESCE(v.vector_score, 0) as vector_score
                FROM text_search t
                FULL OUTER JOIN vector_search v ON t.id = v.id
                WHERE COALESCE(t.text_score, 0) * $3 + COALESCE(v.vector_score, 0) * $4 > 0.1
                ORDER BY combined_score DESC
                LIMIT $5
                """

                rows = await conn.fetch(
                    query, query_text, query_embedding, text_weight, vector_weight, max_results
                )

                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'domain': row['domain'],
                        'knowledge_type': row['knowledge_type'],
                        'confidence_level': row['confidence_level'],
                        'combined_score': float(row['combined_score']),
                        'text_score': float(row['text_score']),
                        'vector_score': float(row['vector_score'])
                    })

                logger.info(f"✅ Hybrid search returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []

    async def batch_vector_search(self, query_embeddings: List[List[float]],
                                max_results_per_query: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Perform batch vector similarity search for multiple queries.
        Optimized for AI workloads that need to process multiple queries efficiently.
        """
        try:
            async with self.pg_pool.acquire() as conn:
                all_results = []

                # Process queries in batches to avoid overwhelming the connection
                batch_size = 10
                for i in range(0, len(query_embeddings), batch_size):
                    batch = query_embeddings[i:i + batch_size]
                    batch_results = []

                    for query_embedding in batch:
                        # Use the optimized vector search for each query
                        results = await self.vector_similarity_search(
                            query_embedding,
                            similarity_threshold=0.5,
                            max_results=max_results_per_query,
                            use_index=True,
                            include_metadata=False  # Faster for batch operations
                        )
                        batch_results.append(results)

                    all_results.extend(batch_results)

                logger.info(f"✅ Batch vector search completed for {len(query_embeddings)} queries")
                return all_results

        except Exception as e:
            logger.error(f"❌ Batch vector search failed: {e}")
            return []
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics using direct PostgreSQL."""
        try:
            async with self.pg_pool.acquire() as conn:
                total_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema_name}.knowledge")
                embedded_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema_name}.knowledge WHERE embedding IS NOT NULL")
                
                return {
                    'total_knowledge_entries': total_count,
                    'embedded_entries': embedded_count,
                    'embedding_coverage': embedded_count / total_count if total_count > 0 else 0.0
                }
        except Exception as e:
            logger.error(f"❌ Failed to get embedding stats: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown database connections."""
        logger.info("🔄 Shutting down database connections...")
        
        if self.pg_pool:
            await self.pg_pool.close()
            self.pg_pool = None
        
        # Supabase client doesn't need explicit shutdown
        self.supabase_client = None
        self._initialized = False
        
        logger.info("✅ Database connections closed")

    async def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics for monitoring."""
        if not self.pg_pool:
            return {}

        return {
            'pool_size': self.pg_pool.get_size(),
            'pool_min_size': self.pg_pool.get_min_size(),
            'pool_max_size': self.pg_pool.get_max_size(),
            'pool_idle_connections': self.pg_pool.get_idle_size(),
            'pool_active_connections': self.pg_pool.get_size() - self.pg_pool.get_idle_size()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all database connections."""
        health_status = {
            'supabase_healthy': False,
            'postgresql_healthy': False,
            'overall_healthy': False,
            'pool_stats': {}
        }

        try:
            # FIXED: Test Supabase connection with correct column name
            response = self.supabase_client.schema(self.schema_name).table('knowledge').select('knowledge_id').limit(1).execute()
            health_status['supabase_healthy'] = True
        except Exception as e:
            logger.warning(f"⚠️ Supabase health check failed: {e}")

        try:
            # Test PostgreSQL connection
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status['postgresql_healthy'] = True
            health_status['pool_stats'] = await self.get_connection_pool_stats()
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL health check failed: {e}")

        health_status['overall_healthy'] = (
            health_status['supabase_healthy'] and
            health_status['postgresql_healthy']
        )

        return health_status

    async def discover_schema(self) -> Dict[str, Any]:
        """Automatically discover schema structure to prevent hardcoding issues."""
        try:
            async with self.pg_pool.acquire() as conn:
                # Get column information
                columns = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = $1 AND table_name = 'knowledge'
                    ORDER BY ordinal_position
                """, self.schema_name)

                # Get indexes
                indexes = await conn.fetch("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = $1 AND tablename = 'knowledge'
                """, self.schema_name)

                schema_info = {
                    'columns': [dict(row) for row in columns],
                    'indexes': [dict(row) for row in indexes],
                    'primary_key': None,
                    'vector_column': None
                }

                # Identify key columns
                for col in schema_info['columns']:
                    if col['column_name'].endswith('_id') and 'uuid' in col['data_type']:
                        schema_info['primary_key'] = col['column_name']
                    if col['data_type'] == 'USER-DEFINED':  # Vector column
                        schema_info['vector_column'] = col['column_name']

                logger.info(f"📋 Schema discovered: PK={schema_info['primary_key']}, Vector={schema_info['vector_column']}")
                return schema_info

        except Exception as e:
            logger.error(f"❌ Schema discovery failed: {e}")
            return {}

# Global instance
_db_manager: Optional[HybridDatabaseManager] = None

async def get_database_manager() -> HybridDatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager

    if _db_manager is None:
        _db_manager = HybridDatabaseManager()
        await _db_manager.initialize()

    return _db_manager
