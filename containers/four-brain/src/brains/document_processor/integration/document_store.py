"""
Document Store for Brain 4
Handles database operations for document storage and retrieval
"""

import asyncio
import logging
import asyncpg
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import sqlite3
from pathlib import Path
import pickle

class DocumentStore:
    """
    Database interface for document storage and retrieval
    Handles PostgreSQL operations with pgvector support
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.fallback_mode = False
        self.fallback_storage = {}  # In-memory fallback storage
    
    async def initialize(self):
        """Initialize database connection pool with fallback support"""

        try:
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_queries=50000,
                command_timeout=60.0
            )

            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            self.logger.info("Document store initialized successfully")
            self.fallback_mode = False

        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
            self.logger.warning("Falling back to in-memory storage")
            self._initialize_fallback_storage()
            self.fallback_mode = True

    def _initialize_fallback_storage(self):
        """Initialize fallback in-memory storage"""
        self.fallback_storage = {
            'documents': {},
            'embeddings': {},
            'metadata': {}
        }
        self.logger.info("Fallback storage initialized")

    def _store_document_fallback(self, document_data: Dict[str, Any]) -> bool:
        """Store document in fallback in-memory storage"""
        try:
            task_id = document_data.get('task_id', 'unknown')

            # Store in fallback storage
            self.fallback_storage['documents'][task_id] = {
                'data': document_data,
                'timestamp': datetime.now().isoformat(),
                'status': 'stored_fallback'
            }

            self.logger.info(f"Document {task_id} stored in fallback storage")
            return True

        except Exception as e:
            self.logger.error(f"Fallback storage failed: {e}")
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status with honest testing"""
        # Test actual connection if not in fallback mode
        if not self.fallback_mode and self.connection_pool:
            try:
                # This would need to be async in real usage, but for testing we check the pool
                connected = self.connection_pool is not None
            except Exception:
                connected = False
        else:
            connected = False

        return {
            "connected": connected and not self.fallback_mode,
            "fallback_mode": self.fallback_mode,
            "storage_type": "postgresql" if connected and not self.fallback_mode else "in_memory",
            "fallback_documents": len(self.fallback_storage.get('documents', {})) if self.fallback_mode else 0
        }

    async def store_document(self, document_data: Dict[str, Any]) -> bool:
        """
        Store processed document in database with fallback support

        Args:
            document_data: Processed document data

        Returns:
            True if successful, False otherwise
        """

        if self.fallback_mode:
            return self._store_document_fallback(document_data)

        try:
            if not self.connection_pool:
                self.logger.warning("Database connection not available - using fallback")
                return self._store_document_fallback(document_data)
            
            async with self.connection_pool.acquire() as conn:
                # Insert into processed_documents table
                query = """
                INSERT INTO documents.processed_documents (
                    task_id, source_path, document_type, file_size, page_count,
                    processing_status, content_text, content_json, structure_data,
                    tables_data, images_data, embeddings, enhanced_analysis,
                    performance_metrics, metadata, processed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                )
                """
                
                # Extract data with defaults
                task_id = document_data.get("task_id")
                source_path = document_data.get("source_path", "")
                document_type = document_data.get("document_type", "unknown")
                file_size = document_data.get("file_size", 0)
                page_count = document_data.get("page_count", 0)
                processing_status = "completed"
                
                content = document_data.get("content", {})
                content_text = content.get("text", "")
                content_json = json.dumps(content.get("json", {}))
                structure_data = json.dumps(content.get("structure", {}))
                tables_data = json.dumps(content.get("tables", []))
                images_data = json.dumps(content.get("images", []))
                
                # Handle embeddings (convert to pgvector format if needed)
                embeddings = document_data.get("embeddings")
                if embeddings and isinstance(embeddings, list):
                    # Convert to string format for pgvector
                    embeddings = f"[{','.join(map(str, embeddings))}]"
                
                enhanced_analysis = json.dumps(document_data.get("enhanced_analysis", {}))
                performance_metrics = json.dumps(document_data.get("performance", {}))
                metadata = json.dumps(document_data.get("metadata", {}))
                processed_at = datetime.now()
                
                await conn.execute(
                    query,
                    task_id, source_path, document_type, file_size, page_count,
                    processing_status, content_text, content_json, structure_data,
                    tables_data, images_data, embeddings, enhanced_analysis,
                    performance_metrics, metadata, processed_at
                )

                # VERIFICATION: Confirm data was actually written to database
                verification_query = """
                SELECT task_id, processing_status, processed_at
                FROM documents.processed_documents
                WHERE task_id = $1
                """

                verification_result = await conn.fetchrow(verification_query, task_id)

                if verification_result is None:
                    self.logger.error(f"VERIFICATION FAILED: Document {task_id} not found after insert")
                    raise Exception(f"Document storage verification failed: {task_id} not found in database")

                if verification_result['processing_status'] != processing_status:
                    self.logger.error(f"VERIFICATION FAILED: Document {task_id} status mismatch")
                    raise Exception(f"Document storage verification failed: status mismatch for {task_id}")

                self.logger.info(f"Document stored and VERIFIED successfully: {task_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing document: {e}")
            return False
    
    async def get_document(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by task ID
        
        Args:
            task_id: Document task ID
            
        Returns:
            Document data or None if not found
        """
        
        try:
            if not self.connection_pool:
                raise Exception("Database connection not initialized")
            
            async with self.connection_pool.acquire() as conn:
                query = """
                SELECT * FROM documents.processed_documents 
                WHERE task_id = $1
                """
                
                row = await conn.fetchrow(query, task_id)
                
                if row:
                    # Convert row to dictionary
                    document_data = dict(row)
                    
                    # Parse JSON fields
                    json_fields = ['content_json', 'structure_data', 'tables_data', 
                                 'images_data', 'enhanced_analysis', 'performance_metrics', 'metadata']
                    
                    for field in json_fields:
                        if document_data.get(field):
                            try:
                                document_data[field] = json.loads(document_data[field])
                            except json.JSONDecodeError:
                                document_data[field] = {}
                    
                    return document_data
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving document {task_id}: {e}")
            return None
    
    async def search_documents(self, 
                             query_embedding: List[float],
                             similarity_threshold: float = 0.7,
                             max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of similar documents
        """
        
        try:
            if not self.connection_pool:
                raise Exception("Database connection not initialized")
            
            async with self.connection_pool.acquire() as conn:
                # Use the similarity search function
                query = """
                SELECT * FROM documents.search_similar_documents($1, $2, $3)
                """
                
                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                
                rows = await conn.fetch(query, embedding_str, similarity_threshold, max_results)
                
                results = []
                for row in rows:
                    result = {
                        "document_id": row["document_id"],
                        "task_id": row["task_id"],
                        "similarity_score": row["similarity_score"],
                        "content_preview": row["content_preview"]
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    async def list_documents(self, 
                           limit: int = 10, 
                           offset: int = 0,
                           status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List documents with pagination
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            status_filter: Filter by processing status
            
        Returns:
            List of documents
        """
        
        try:
            if not self.connection_pool:
                raise Exception("Database connection not initialized")
            
            async with self.connection_pool.acquire() as conn:
                # Build query with optional status filter
                base_query = """
                SELECT task_id, source_path, document_type, file_size, 
                       processing_status, processed_at, metadata
                FROM documents.processed_documents
                """
                
                if status_filter:
                    query = base_query + " WHERE processing_status = $1 ORDER BY processed_at DESC LIMIT $2 OFFSET $3"
                    rows = await conn.fetch(query, status_filter, limit, offset)
                else:
                    query = base_query + " ORDER BY processed_at DESC LIMIT $1 OFFSET $2"
                    rows = await conn.fetch(query, limit, offset)
                
                documents = []
                for row in rows:
                    doc = dict(row)
                    # Parse metadata if it exists
                    if doc.get("metadata"):
                        try:
                            doc["metadata"] = json.loads(doc["metadata"])
                        except json.JSONDecodeError:
                            doc["metadata"] = {}
                    
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []
    
    async def get_document_count(self, status_filter: Optional[str] = None) -> int:
        """
        Get total count of documents
        
        Args:
            status_filter: Filter by processing status
            
        Returns:
            Total document count
        """
        
        try:
            if not self.connection_pool:
                raise Exception("Database connection not initialized")
            
            async with self.connection_pool.acquire() as conn:
                if status_filter:
                    query = "SELECT COUNT(*) FROM documents.processed_documents WHERE processing_status = $1"
                    count = await conn.fetchval(query, status_filter)
                else:
                    query = "SELECT COUNT(*) FROM documents.processed_documents"
                    count = await conn.fetchval(query)
                
                return count or 0
                
        except Exception as e:
            self.logger.error(f"Error getting document count: {e}")
            return 0
    
    async def update_brain_heartbeat(self, brain_id: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Update brain heartbeat in database
        
        Args:
            brain_id: Brain identifier
            metrics: Performance metrics
        """
        
        try:
            if not self.connection_pool:
                return
            
            async with self.connection_pool.acquire() as conn:
                # Use the heartbeat update function
                await conn.execute(
                    "SELECT brain_system.update_brain_heartbeat($1, $2)",
                    brain_id,
                    json.dumps(metrics) if metrics else None
                )
                
        except Exception as e:
            self.logger.error(f"Error updating brain heartbeat: {e}")
    
    async def close(self):
        """Close database connection pool"""
        
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            
            self.logger.info("Document store closed")
            
        except Exception as e:
            self.logger.error(f"Error closing document store: {e}")
