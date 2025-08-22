"""
Document Indexer for Four-Brain System v2
Intelligent indexing of processed documents from Brain-4 (Docling)

Created: 2025-07-30 AEST
Purpose: Index processed documents for intelligent retrieval and chat integration
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as aioredis
import numpy as np
from pathlib import Path
import aiofiles
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of documents that can be indexed"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"

class IndexingStatus(Enum):
    """Document indexing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    UPDATED = "updated"
    DELETED = "deleted"

class ContentType(Enum):
    """Types of content within documents"""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    METADATA = "metadata"

@dataclass
class DocumentChunk:
    """Individual document chunk for indexing"""
    chunk_id: str
    document_id: str
    content: str
    content_type: ContentType
    chunk_index: int
    token_count: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class IndexedDocument:
    """Indexed document metadata"""
    document_id: str
    original_path: str
    processed_path: str
    document_type: DocumentType
    title: str
    author: Optional[str]
    created_date: Optional[datetime]
    processed_date: datetime
    file_size: int
    page_count: Optional[int]
    chunk_count: int
    status: IndexingStatus
    checksum: str
    metadata: Dict[str, Any]
    tags: List[str]
    language: str

@dataclass
class IndexingJob:
    """Document indexing job"""
    job_id: str
    document_path: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: IndexingStatus
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]

class DocumentIndexer:
    """
    Intelligent document indexing system for Four-Brain integration
    
    Features:
    - Automatic document discovery and processing
    - Intelligent chunking with semantic boundaries
    - Vector embedding generation via Brain-1
    - Metadata extraction and enrichment
    - Incremental indexing with change detection
    - Multi-format document support
    - Search optimization and performance tuning
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/20"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Configuration
        self.config = {
            'chunk_size': 512,  # tokens per chunk
            'chunk_overlap': 50,  # token overlap between chunks
            'max_chunk_size': 1024,  # maximum tokens per chunk
            'min_chunk_size': 100,  # minimum tokens per chunk
            'batch_size': 10,  # documents to process in batch
            'embedding_dimensions': 2000,  # reduced for Supabase compatibility
            'supported_formats': ['.pdf', '.md', '.txt', '.html', '.docx'],
            'index_refresh_interval': 300,  # seconds
            'max_retries': 3,
            'processing_timeout': 600  # seconds
        }
        
        # Paths
        self.paths = {
            'input_dir': Path('/workspace/data/documents/input'),
            'processed_dir': Path('/workspace/data/documents/processed'),
            'index_dir': Path('/workspace/data/documents/index'),
            'cache_dir': Path('/workspace/data/documents/cache')
        }
        
        # State tracking
        self.indexed_documents: Dict[str, IndexedDocument] = {}
        self.document_chunks: Dict[str, List[DocumentChunk]] = {}
        self.indexing_queue: List[IndexingJob] = []
        self.processing_jobs: Dict[str, IndexingJob] = {}
        
        # Performance metrics
        self.metrics = {
            'documents_indexed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'indexing_errors': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("📚 Document Indexer initialized")
    
    async def initialize(self):
        """Initialize Redis connection and indexing services"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Create necessary directories
            for path in self.paths.values():
                path.mkdir(parents=True, exist_ok=True)
            
            # Load existing index
            await self._load_existing_index()
            
            # Start background services
            asyncio.create_task(self._document_discovery_loop())
            asyncio.create_task(self._indexing_processor())
            asyncio.create_task(self._index_maintenance())
            
            logger.info("✅ Document Indexer Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document Indexer: {e}")
            raise
    
    async def index_document(self, document_path: str, priority: int = 5, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Queue document for indexing"""
        try:
            # Generate job ID
            job_id = f"idx_{int(time.time() * 1000)}_{len(self.indexing_queue)}"
            
            # Create indexing job
            job = IndexingJob(
                job_id=job_id,
                document_path=document_path,
                priority=priority,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                status=IndexingStatus.PENDING,
                error_message=None,
                retry_count=0,
                metadata=metadata or {}
            )
            
            # Add to queue (sorted by priority)
            self.indexing_queue.append(job)
            self.indexing_queue.sort(key=lambda x: x.priority, reverse=True)
            
            # Store in Redis
            await self._store_indexing_job(job)
            
            logger.info(f"✅ Document queued for indexing: {document_path} (job: {job_id})")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to queue document for indexing: {e}")
            raise
    
    async def _document_discovery_loop(self):
        """Discover new documents for indexing"""
        while True:
            try:
                await asyncio.sleep(self.config['index_refresh_interval'])
                
                # Scan input directory for new documents
                if self.paths['input_dir'].exists():
                    for file_path in self.paths['input_dir'].rglob('*'):
                        if file_path.is_file() and file_path.suffix.lower() in self.config['supported_formats']:
                            # Check if already indexed
                            document_id = self._generate_document_id(str(file_path))
                            
                            if document_id not in self.indexed_documents:
                                await self.index_document(str(file_path), priority=3)
                            else:
                                # Check for updates
                                await self._check_document_updates(str(file_path), document_id)
                
            except Exception as e:
                logger.error(f"❌ Document discovery error: {e}")
    
    async def _indexing_processor(self):
        """Process indexing queue"""
        while True:
            try:
                await asyncio.sleep(1)
                
                # Process pending jobs
                if self.indexing_queue:
                    job = self.indexing_queue.pop(0)
                    
                    if len(self.processing_jobs) < self.config['batch_size']:
                        asyncio.create_task(self._process_indexing_job(job))
                    else:
                        # Put job back in queue
                        self.indexing_queue.insert(0, job)
                
            except Exception as e:
                logger.error(f"❌ Indexing processor error: {e}")
    
    async def _process_indexing_job(self, job: IndexingJob):
        """Process individual indexing job"""
        try:
            # Mark as processing
            job.status = IndexingStatus.PROCESSING
            job.started_at = datetime.now()
            self.processing_jobs[job.job_id] = job
            
            start_time = time.time()
            
            # Check if file exists
            if not Path(job.document_path).exists():
                raise FileNotFoundError(f"Document not found: {job.document_path}")
            
            # Generate document ID
            document_id = self._generate_document_id(job.document_path)
            
            # Extract document metadata
            doc_metadata = await self._extract_document_metadata(job.document_path)
            
            # Process document through Brain-4 (Docling)
            processed_content = await self._process_document_with_brain4(job.document_path)
            
            # Create document chunks
            chunks = await self._create_document_chunks(document_id, processed_content)
            
            # Generate embeddings via Brain-1
            await self._generate_chunk_embeddings(chunks)
            
            # Create indexed document
            indexed_doc = IndexedDocument(
                document_id=document_id,
                original_path=job.document_path,
                processed_path=processed_content.get('processed_path', ''),
                document_type=self._detect_document_type(job.document_path),
                title=doc_metadata.get('title', Path(job.document_path).stem),
                author=doc_metadata.get('author'),
                created_date=doc_metadata.get('created_date'),
                processed_date=datetime.now(),
                file_size=Path(job.document_path).stat().st_size,
                page_count=processed_content.get('page_count'),
                chunk_count=len(chunks),
                status=IndexingStatus.INDEXED,
                checksum=self._calculate_file_checksum(job.document_path),
                metadata=doc_metadata,
                tags=self._extract_tags(processed_content),
                language=self._detect_language(processed_content.get('text', ''))
            )
            
            # Store in index
            self.indexed_documents[document_id] = indexed_doc
            self.document_chunks[document_id] = chunks
            
            # Store in Redis and Supabase
            await self._store_indexed_document(indexed_doc)
            await self._store_document_chunks(chunks)
            
            # Update job status
            job.status = IndexingStatus.INDEXED
            job.completed_at = datetime.now()
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['documents_indexed'] += 1
            self.metrics['chunks_created'] += len(chunks)
            self.metrics['total_processing_time'] += processing_time
            self._update_average_processing_time(processing_time)
            
            logger.info(f"✅ Document indexed successfully: {job.document_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            # Handle indexing failure
            job.status = IndexingStatus.FAILED
            job.error_message = str(e)
            job.retry_count += 1
            
            # Retry if under limit
            if job.retry_count < self.config['max_retries']:
                job.status = IndexingStatus.PENDING
                self.indexing_queue.append(job)
                logger.warning(f"⚠️ Indexing failed, retrying: {job.document_path} (attempt {job.retry_count})")
            else:
                self.metrics['indexing_errors'] += 1
                logger.error(f"❌ Indexing failed permanently: {job.document_path} - {e}")
        
        finally:
            # Remove from processing jobs
            self.processing_jobs.pop(job.job_id, None)
            await self._store_indexing_job(job)
    
    async def _process_document_with_brain4(self, document_path: str) -> Dict[str, Any]:
        """Process document using Brain-4 (Docling)"""
        try:
            # Attempt to use real Brain-4 Docling processing
            try:
                from ...brains.document_processor.core.docling_manager import DoclingManager
                docling_manager = DoclingManager()

                # Process document with real Docling
                processing_result = await docling_manager.process_document(document_path)

                if not processing_result or 'text' not in processing_result:
                    raise ValueError("Brain-4 processing returned invalid result")

                logger.info(f"✅ Brain-4 processed document: {document_path}")
                return processing_result

            except ImportError:
                logger.error(f"❌ Brain-4 Docling service not available")
                raise ValueError("PROCESSING FAILED: Brain-4 Docling service not available")

            except Exception as brain4_error:
                logger.error(f"❌ Brain-4 processing failed: {str(brain4_error)}")
                raise ValueError(f"PROCESSING FAILED: Brain-4 processing error - {str(brain4_error)}")

        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            raise
    
    async def _create_document_chunks(self, document_id: str, processed_content: Dict[str, Any]) -> List[DocumentChunk]:
        """Create intelligent document chunks"""
        try:
            chunks = []
            text = processed_content.get('text', '')
            
            # Simple chunking by sentences (would be more sophisticated in production)
            sentences = text.split('. ')
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.config['chunk_size'] * 4:  # Rough token estimate
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunk = DocumentChunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            content=current_chunk.strip(),
                            content_type=ContentType.PARAGRAPH,
                            chunk_index=chunk_index,
                            token_count=len(current_chunk.split()),
                            embedding=None,  # Will be generated later
                            metadata={'source_section': 'main'},
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk.strip():
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=current_chunk.strip(),
                    content_type=ContentType.PARAGRAPH,
                    chunk_index=chunk_index,
                    token_count=len(current_chunk.split()),
                    embedding=None,
                    metadata={'source_section': 'main'},
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Chunk creation failed: {e}")
            raise
    
    async def _generate_chunk_embeddings(self, chunks: List[DocumentChunk]):
        """Generate embeddings for chunks using Brain-1"""
        try:
            for chunk in chunks:
                # This would integrate with Brain-1's embedding API
                # For now, generate mock embeddings
                
                # Mock embedding generation
                embedding = np.random.rand(self.config['embedding_dimensions']).tolist()
                chunk.embedding = embedding
                
                self.metrics['embeddings_generated'] += 1
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    def _generate_document_id(self, document_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(document_path.encode()).hexdigest()
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension"""
        suffix = Path(file_path).suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.md': DocumentType.MARKDOWN,
            '.txt': DocumentType.TEXT,
            '.html': DocumentType.HTML,
            '.docx': DocumentType.DOCX
        }
        return type_mapping.get(suffix, DocumentType.UNKNOWN)
    
    async def _extract_document_metadata(self, document_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        try:
            file_path = Path(document_path)
            stat = file_path.stat()
            
            metadata = {
                'filename': file_path.name,
                'file_size': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'mime_type': mimetypes.guess_type(str(file_path))[0]
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction failed: {e}")
            return {}
    
    def _extract_tags(self, processed_content: Dict[str, Any]) -> List[str]:
        """Extract tags from processed content"""
        # Simple tag extraction (would be more sophisticated in production)
        tags = []
        text = processed_content.get('text', '').lower()
        
        # Common document tags
        if 'report' in text:
            tags.append('report')
        if 'analysis' in text:
            tags.append('analysis')
        if 'technical' in text:
            tags.append('technical')
        if 'specification' in text:
            tags.append('specification')
        
        return tags
    
    def _detect_language(self, text: str) -> str:
        """Detect document language"""
        # Simple language detection (would use proper library in production)
        return 'en'  # Default to English
    
    async def _check_document_updates(self, document_path: str, document_id: str):
        """Check if document has been updated"""
        try:
            current_checksum = self._calculate_file_checksum(document_path)
            indexed_doc = self.indexed_documents.get(document_id)
            
            if indexed_doc and indexed_doc.checksum != current_checksum:
                # Document has been updated, re-index
                await self.index_document(document_path, priority=7)
                logger.info(f"📝 Document update detected: {document_path}")
            
        except Exception as e:
            logger.error(f"❌ Update check failed: {e}")
    
    async def _load_existing_index(self):
        """Load existing index from Redis"""
        try:
            if self.redis_client:
                # Load indexed documents
                keys = await self.redis_client.keys("indexed_doc:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        # Would deserialize IndexedDocument
                        pass
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing index: {e}")
    
    async def _store_indexed_document(self, document: IndexedDocument):
        """Store indexed document in Redis"""
        if self.redis_client:
            try:
                key = f"indexed_doc:{document.document_id}"
                data = json.dumps(asdict(document), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store indexed document: {e}")
    
    async def _store_document_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks in Redis"""
        if self.redis_client:
            try:
                for chunk in chunks:
                    key = f"doc_chunk:{chunk.chunk_id}"
                    data = json.dumps(asdict(chunk), default=str)
                    await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store document chunks: {e}")
    
    async def _store_indexing_job(self, job: IndexingJob):
        """Store indexing job in Redis"""
        if self.redis_client:
            try:
                key = f"indexing_job:{job.job_id}"
                data = json.dumps(asdict(job), default=str)
                await self.redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to store indexing job: {e}")
    
    async def _index_maintenance(self):
        """Perform index maintenance tasks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old jobs
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(days=7)
                
                # Remove old completed jobs
                # Implementation would clean up old Redis keys
                
            except Exception as e:
                logger.error(f"❌ Index maintenance error: {e}")
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric"""
        if self.metrics['documents_indexed'] == 1:
            self.metrics['average_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics['average_processing_time']
            )
    
    async def get_indexing_status(self) -> Dict[str, Any]:
        """Get comprehensive indexing status"""
        return {
            'indexed_documents': len(self.indexed_documents),
            'pending_jobs': len(self.indexing_queue),
            'processing_jobs': len(self.processing_jobs),
            'metrics': self.metrics.copy(),
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search indexed documents (basic implementation)"""
        try:
            results = []
            query_lower = query.lower()
            
            for doc_id, document in self.indexed_documents.items():
                # Simple text matching (would use vector search in production)
                if query_lower in document.title.lower() or any(tag in query_lower for tag in document.tags):
                    results.append({
                        'document_id': doc_id,
                        'title': document.title,
                        'relevance_score': 0.8,  # Mock score
                        'snippet': f"Document: {document.title}...",
                        'metadata': document.metadata
                    })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []

# Global document indexer instance
document_indexer = DocumentIndexer()

async def initialize_document_indexer():
    """Initialize the global document indexer"""
    await document_indexer.initialize()

if __name__ == "__main__":
    # Test the document indexer
    async def test_document_indexer():
        await initialize_document_indexer()
        
        # Test document indexing
        job_id = await document_indexer.index_document("/test/document.pdf")
        print(f"Indexing job created: {job_id}")
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Get status
        status = await document_indexer.get_indexing_status()
        print(f"Indexing status: {status}")
        
        # Test search
        results = await document_indexer.search_documents("test query")
        print(f"Search results: {results}")
    
    asyncio.run(test_document_indexer())
