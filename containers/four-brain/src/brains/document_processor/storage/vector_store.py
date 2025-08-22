"""
Vector Store Implementation for Phase 6 Four-Brain AI System
Handles embedding storage and similarity search using PostgreSQL with pgvector
"""

import asyncio
import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncpg
import numpy as np
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector storage and similarity search using PostgreSQL with pgvector extension.
    Optimized for Phase 6 Four-Brain AI System with RTX 5070 Ti GPU support.
    """
    
    def __init__(self, database_url: str, pool_size: int = 10):
        """
        Initialize Vector Store with connection pooling.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Maximum number of connections in pool
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize connection pool and validate database setup."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=60,
                server_settings={
                    'search_path': 'augment_agent,public',
                    'application_name': 'phase6_vector_store'
                }
            )
            
            # Validate database setup
            await self._validate_setup()
            self.logger.info("Vector Store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vector Store: {e}")
            raise
    
    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Vector Store connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            raise RuntimeError("Vector Store not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def _validate_setup(self) -> None:
        """Validate database schema and extensions."""
        async with self.get_connection() as conn:
            # Check pgvector extension
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if not result:
                raise RuntimeError("pgvector extension not installed")
            
            # Check required tables
            tables = ['documents', 'document_content', 'document_embeddings', 'knowledge']
            for table in tables:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'augment_agent' AND table_name = $1)",
                    table
                )
                if not exists:
                    raise RuntimeError(f"Required table 'augment_agent.{table}' not found")
            
            self.logger.info("Database schema validation successful")
    
    async def store_document(self, filename: str, file_path: str, 
                           file_size: int, mime_type: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store document metadata.
        
        Args:
            filename: Original filename
            file_path: Storage path
            file_size: File size in bytes
            mime_type: MIME type
            metadata: Additional metadata
            
        Returns:
            Document UUID
        """
        document_id = str(uuid.uuid4())
        
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO augment_agent.documents
                (id, filename, file_path, file_size, mime_type, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                document_id, filename, file_path, file_size, mime_type,
                json.dumps(metadata) if metadata else None
            )
        
        self.logger.info(f"Stored document metadata: {document_id}")
        return document_id
    
    async def store_document_content(self, document_id: str, content_type: str,
                                   content_text: str, page_number: Optional[int] = None,
                                   position_data: Optional[Dict[str, Any]] = None,
                                   extraction_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store document content.
        
        Args:
            document_id: Document UUID
            content_type: Type of content (text, table, image, etc.)
            content_text: Extracted text content
            page_number: Page number (if applicable)
            position_data: Position/coordinate data
            extraction_metadata: Extraction metadata
            
        Returns:
            Content UUID
        """
        content_id = str(uuid.uuid4())
        
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO augment_agent.document_content 
                (id, document_id, content_type, content_text, page_number, 
                 position_data, extraction_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                content_id, document_id, content_type, content_text,
                page_number, json.dumps(position_data) if position_data else None,
                json.dumps(extraction_metadata) if extraction_metadata else None
            )
        
        self.logger.debug(f"Stored document content: {content_id}")
        return content_id
    
    async def store_embeddings(self, document_id: str, content_id: Optional[str],
                             chunks: List[str], embeddings: List[List[float]],
                             model_name: str) -> List[str]:
        """
        Store text chunks and their embeddings.
        
        Args:
            document_id: Document UUID
            content_id: Content UUID (optional)
            chunks: List of text chunks
            embeddings: List of embedding vectors
            model_name: Name of embedding model used
            
        Returns:
            List of embedding UUIDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        embedding_ids = []
        
        async with self.get_connection() as conn:
            async with conn.transaction():
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding_id = str(uuid.uuid4())
                    
                    # Convert embedding list to vector string format
                    vector_str = '[' + ','.join(map(str, embedding)) + ']'

                    await conn.execute(
                        """
                        INSERT INTO augment_agent.document_embeddings
                        (id, document_id, content_id, chunk_text, chunk_index,
                         embedding, embedding_model)
                        VALUES ($1, $2, $3, $4, $5, $6::vector, $7)
                        """,
                        embedding_id, document_id, content_id, chunk, i,
                        vector_str, model_name
                    )
                    
                    embedding_ids.append(embedding_id)
        
        self.logger.info(f"Stored {len(embedding_ids)} embeddings for document {document_id}")
        return embedding_ids
    
    async def similarity_search(self, query_embedding: List[float], 
                              similarity_threshold: float = 0.7,
                              max_results: int = 10,
                              document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            document_id: Optional document ID filter
            
        Returns:
            List of similar chunks with metadata
        """
        # Convert query embedding to vector string format
        query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with self.get_connection() as conn:
            if document_id:
                query = """
                SELECT e.document_id, e.content_id, e.chunk_text, e.chunk_index,
                       1 - (e.embedding <=> $1::vector) as similarity,
                       d.filename, d.metadata as doc_metadata
                FROM augment_agent.document_embeddings e
                JOIN augment_agent.documents d ON e.document_id = d.id
                WHERE e.document_id = $4
                AND 1 - (e.embedding <=> $1::vector) > $2
                ORDER BY e.embedding <=> $1::vector
                LIMIT $3
                """
                results = await conn.fetch(query, query_vector_str, similarity_threshold,
                                         max_results, document_id)
            else:
                query = """
                SELECT e.document_id, e.content_id, e.chunk_text, e.chunk_index,
                       1 - (e.embedding <=> $1::vector) as similarity,
                       d.filename, d.metadata as doc_metadata
                FROM augment_agent.document_embeddings e
                JOIN augment_agent.documents d ON e.document_id = d.id
                WHERE 1 - (e.embedding <=> $1::vector) > $2
                ORDER BY e.embedding <=> $1::vector
                LIMIT $3
                """
                results = await conn.fetch(query, query_vector_str, similarity_threshold,
                                         max_results)
        
        # Convert to list of dictionaries
        search_results = []
        for row in results:
            search_results.append({
                'document_id': row['document_id'],
                'content_id': row['content_id'],
                'chunk_text': row['chunk_text'],
                'chunk_index': row['chunk_index'],
                'similarity': float(row['similarity']),
                'filename': row['filename'],
                'doc_metadata': row['doc_metadata']
            })
        
        self.logger.info(f"Found {len(search_results)} similar chunks")
        return search_results
    
    async def store_knowledge(self, document_id: Optional[str], title: str,
                            content: str, summary: Optional[str] = None,
                            keywords: Optional[List[str]] = None,
                            embedding: Optional[List[float]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store processed knowledge with optional embedding.
        
        Args:
            document_id: Source document UUID (optional)
            title: Knowledge title
            content: Knowledge content
            summary: Optional summary
            keywords: Optional keywords
            embedding: Optional embedding vector
            metadata: Additional metadata
            
        Returns:
            Knowledge UUID
        """
        knowledge_id = str(uuid.uuid4())
        
        # Convert embedding to vector string format if provided
        embedding_str = None
        if embedding:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        # Track database operation with flow monitoring
        try:
            from flow_monitoring import get_flow_monitor, DatabaseType
            flow_monitor = get_flow_monitor()

            async with flow_monitor.track_database_operation(DatabaseType.POSTGRESQL, "vector_insert"):
                async with self.get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO augment_agent.knowledge
                        (id, document_id, title, content, summary, keywords, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                        """,
                        knowledge_id, document_id, title, content, summary,
                        keywords, embedding_str, json.dumps(metadata) if metadata else None
                    )

        except ImportError:
            # Flow monitoring not available, proceed without tracking
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO augment_agent.knowledge
                    (id, document_id, title, content, summary, keywords, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                    """,
                    knowledge_id, document_id, title, content, summary,
                    keywords, embedding_str, json.dumps(metadata) if metadata else None
                )
        
        self.logger.info(f"Stored knowledge: {knowledge_id}")
        return knowledge_id
    
    async def search_knowledge(self, query_embedding: List[float],
                             similarity_threshold: float = 0.7,
                             max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search knowledge base using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similar knowledge entries
        """
        # Convert query embedding to vector string format
        query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Track database operation with flow monitoring
        try:
            from flow_monitoring import get_flow_monitor, DatabaseType
            flow_monitor = get_flow_monitor()

            async with flow_monitor.track_database_operation(DatabaseType.POSTGRESQL, "vector_search"):
                async with self.get_connection() as conn:
                    results = await conn.fetch(
                        """
                        SELECT id, document_id, title, content, summary,
                               1 - (embedding <=> $1::vector) as similarity
                        FROM augment_agent.knowledge
                        WHERE embedding IS NOT NULL
                        AND 1 - (embedding <=> $1::vector) > $2
                        ORDER BY embedding <=> $1::vector
                        LIMIT $3
                        """,
                        query_vector_str, similarity_threshold, max_results
                    )

        except ImportError:
            # Flow monitoring not available, proceed without tracking
            async with self.get_connection() as conn:
                results = await conn.fetch(
                    """
                    SELECT id, document_id, title, content, summary,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM augment_agent.knowledge
                    WHERE embedding IS NOT NULL
                    AND 1 - (embedding <=> $1::vector) > $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_vector_str, similarity_threshold, max_results
                )
        
        # Convert to list of dictionaries
        knowledge_results = []
        for row in results:
            knowledge_results.append({
                'id': row['id'],
                'document_id': row['document_id'],
                'title': row['title'],
                'content': row['content'],
                'summary': row['summary'],
                'similarity': float(row['similarity'])
            })
        
        self.logger.info(f"Found {len(knowledge_results)} similar knowledge entries")
        return knowledge_results
    
    async def get_document_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        async with self.get_connection() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    (SELECT COUNT(*) FROM augment_agent.documents) as documents,
                    (SELECT COUNT(*) FROM augment_agent.document_content) as content_items,
                    (SELECT COUNT(*) FROM augment_agent.document_embeddings) as embeddings,
                    (SELECT COUNT(*) FROM augment_agent.knowledge) as knowledge_items
                """
            )
        
        return dict(stats)
