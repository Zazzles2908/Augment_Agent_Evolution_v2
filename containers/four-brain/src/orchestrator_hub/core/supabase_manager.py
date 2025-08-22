"""
Supabase Database Manager for Four-Brain System v2
Handles all database operations for the augment_agent schema
"""

import os
import json
import asyncio
import asyncpg
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    display_name: Optional[str] = None
    preferences: Dict[str, Any] = None
    personality_traits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                "communication_style": "balanced",
                "detail_level": "moderate", 
                "formality": "professional",
                "response_length": "medium"
            }
        if self.personality_traits is None:
            self.personality_traits = {
                "concise_verbose": 0.5,
                "formal_casual": 0.5,
                "technical_simple": 0.5,
                "direct_diplomatic": 0.5
            }

class SupabaseManager:
    """Manages all Supabase database operations for the Four-Brain System"""
    
    def __init__(self):
        self.connection_pool = None
        self.db_url = self._build_connection_string()
        self.schema = "augment_agent"
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables"""
        # Use Supabase database connection details
        # Format: postgresql://postgres.{project_ref}:{password}@{host}:{port}/postgres
        host = os.getenv('SUPABASE_DB_HOST', 'db.ustcfwmonegxeoqeixgg.supabase.co')
        port = os.getenv('SUPABASE_DB_PORT', '5432')
        database = os.getenv('SUPABASE_DB_NAME', 'postgres')
        user = os.getenv('SUPABASE_DB_USER', 'postgres.ustcfwmonegxeoqeixgg')
        password = os.getenv('SUPABASE_DB_PASSWORD', 'SLf9q35ERDQsT0!')

        # For external access, we need to use the service role key approach
        # Let's try a different approach using the service role for database access
        logger.info(f"Attempting Supabase connection to {host}:{port}/{database} as {user}")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            # Try to create connection pool with shorter timeout for external connections
            self.connection_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=5,
                command_timeout=10,
                server_settings={
                    'application_name': 'four_brain_orchestrator'
                }
            )
            logger.info("✅ Supabase connection pool initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Supabase connection failed (expected in isolated container): {e}")
            logger.info("🔄 Orchestrator Hub will continue with local-only features")
            return False
    
    async def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Supabase connection pool closed")

    def is_available(self) -> bool:
        """Check if database connection is available"""
        return self.connection_pool is not None
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        if not self.is_available():
            logger.debug("Database not available, returning None for user profile")
            return None

        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                SELECT user_id, display_name, preferences, personality_traits
                FROM {self.schema}.user_profiles 
                WHERE user_id = $1
                """
                row = await conn.fetchrow(query, user_id)
                
                if row:
                    return UserProfile(
                        user_id=str(row['user_id']),
                        display_name=row['display_name'],
                        preferences=row['preferences'],
                        personality_traits=row['personality_traits']
                    )
                return None
        except Exception as e:
            logger.error(f"Error fetching user profile for {user_id}: {e}")
            return None
    
    async def create_or_update_user_profile(self, profile: UserProfile) -> bool:
        """Create or update user profile"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                INSERT INTO {self.schema}.user_profiles 
                (user_id, display_name, preferences, personality_traits, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    display_name = EXCLUDED.display_name,
                    preferences = EXCLUDED.preferences,
                    personality_traits = EXCLUDED.personality_traits,
                    updated_at = NOW()
                """
                await conn.execute(
                    query,
                    profile.user_id,
                    profile.display_name,
                    json.dumps(profile.preferences),
                    json.dumps(profile.personality_traits)
                )
                logger.info(f"User profile updated for {profile.user_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating user profile for {profile.user_id}: {e}")
            return False
    
    async def log_task(self, task_id: str, user_id: str, task_type: str, 
                      input_data: Dict[str, Any], status: str = "pending") -> bool:
        """Log a new task"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                INSERT INTO {self.schema}.task_logs 
                (task_id, user_id, task_type, status, input_data)
                VALUES ($1, $2, $3, $4, $5)
                """
                await conn.execute(
                    query,
                    task_id,
                    user_id,
                    task_type,
                    status,
                    json.dumps(input_data)
                )
                logger.info(f"Task logged: {task_id} for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error logging task {task_id}: {e}")
            return False
    
    async def update_task_status(self, task_id: str, status: str, 
                                output_data: Optional[Dict[str, Any]] = None,
                                processing_time_ms: Optional[int] = None,
                                error_message: Optional[str] = None) -> bool:
        """Update task status and completion data"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                UPDATE {self.schema}.task_logs 
                SET status = $2, 
                    output_data = $3,
                    processing_time_ms = $4,
                    error_message = $5,
                    updated_at = NOW(),
                    completed_at = CASE WHEN $2 IN ('completed', 'failed') THEN NOW() ELSE completed_at END
                WHERE task_id = $1
                """
                await conn.execute(
                    query,
                    task_id,
                    status,
                    json.dumps(output_data) if output_data else None,
                    processing_time_ms,
                    error_message
                )
                logger.info(f"Task status updated: {task_id} -> {status}")
                return True
        except Exception as e:
            logger.error(f"Error updating task status for {task_id}: {e}")
            return False
    
    async def store_document(self, user_id: str, filename: str, document_type: str,
                           content_text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Store document metadata and content"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                INSERT INTO {self.schema}.documents 
                (user_id, filename, document_type, content_text, metadata, processing_status)
                VALUES ($1, $2, $3, $4, $5, 'processed')
                RETURNING id
                """
                doc_id = await conn.fetchval(
                    query,
                    user_id,
                    filename,
                    document_type,
                    content_text,
                    json.dumps(metadata)
                )
                logger.info(f"Document stored: {filename} for user {user_id}")
                return str(doc_id)
        except Exception as e:
            logger.error(f"Error storing document {filename}: {e}")
            return None
    
    async def store_embedding(self, user_id: str, text: str, embedding_vector: List[float],
                            model_version: str = "qwen3-4b") -> bool:
        """Store text embedding"""
        try:
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            async with self.connection_pool.acquire() as conn:
                query = f"""
                INSERT INTO {self.schema}.embeddings 
                (user_id, text_hash, original_text, embedding_vector, model_version)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (text_hash) DO NOTHING
                """
                await conn.execute(
                    query,
                    user_id,
                    text_hash,
                    text,
                    embedding_vector,
                    model_version
                )
                logger.info(f"Embedding stored for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    async def search_similar_embeddings(self, user_id: str, query_embedding: List[float],
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity"""
        if not self.is_available():
            logger.debug("Database not available, returning empty results for embedding search")
            return []

        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                SELECT 
                    original_text,
                    1 - (embedding_vector <=> $2) as similarity,
                    created_at
                FROM {self.schema}.embeddings 
                WHERE user_id = $1 OR user_id IS NULL
                ORDER BY embedding_vector <=> $2
                LIMIT $3
                """
                rows = await conn.fetch(query, user_id, query_embedding, limit)
                
                results = []
                for row in rows:
                    results.append({
                        "text": row['original_text'],
                        "similarity": float(row['similarity']),
                        "created_at": row['created_at'].isoformat()
                    })
                
                logger.info(f"Found {len(results)} similar embeddings for user {user_id}")
                return results
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    async def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = f"""
                SELECT id, filename, document_type, processing_status, 
                       metadata, created_at, updated_at
                FROM {self.schema}.documents 
                WHERE user_id = $1
                ORDER BY created_at DESC
                """
                rows = await conn.fetch(query, user_id)
                
                documents = []
                for row in rows:
                    documents.append({
                        "id": str(row['id']),
                        "filename": row['filename'],
                        "document_type": row['document_type'],
                        "processing_status": row['processing_status'],
                        "metadata": row['metadata'],
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                    })
                
                return documents
        except Exception as e:
            logger.error(f"Error fetching user documents: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check database connection health"""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global instance
supabase_manager = SupabaseManager()
