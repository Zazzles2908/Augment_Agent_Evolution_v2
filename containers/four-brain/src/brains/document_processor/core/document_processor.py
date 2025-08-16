"""
Document Processor for Brain 4
Advanced document processing with multi-format support and RTX 5070 Ti optimization
"""

import asyncio
import logging
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from docling.datamodel.document import DoclingDocument, ConversionResult
from ..integration.brain1_api_client import Brain1APIClient

from ..models.document_models import ProcessedDocument, DocumentChunk
from ..utils.performance_monitor import PerformanceMonitor
from ..utils.smart_model_manager import get_model
from ..storage.vector_store import VectorStore

class DocumentProcessor:
    """
    Advanced document processor with multi-format support
    Optimized for RTX 5070 Ti performance characteristics
    """
    
    def __init__(self, config_settings):
        self.converter = None  # Will be loaded on demand
        self.settings = config_settings
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()

        # Initialize vector store for embedding storage
        database_url = getattr(config_settings, 'DATABASE_URL',
                              'postgresql://postgres:phase6_ai_system_secure_2025@postgres:5432/phase6_ai_system')
        self.vector_store = VectorStore(database_url)
        
        # Processing statistics
        self.processing_stats = {
            "documents_processed": 0,
            "total_pages_processed": 0,
            "total_processing_time": 0.0,
            "format_stats": {},
            "error_count": 0
        }
        
        # Supported MIME types mapping
        self.mime_type_mapping = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "text/html": "html",
            "text/markdown": "md",
            "text/plain": "txt",
            "application/rtf": "rtf",
            "application/vnd.oasis.opendocument.text": "odt"
        }
        
        self.logger.info("Document processor initialized")
    
    async def process_document(self, 
                             file_path: Union[str, Path],
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document and extract all relevant information
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            Dictionary with all extracted information
        """
        
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Validate file
            await self._validate_file(file_path)
            
            # Extract basic metadata
            doc_metadata = await self._extract_file_metadata(file_path, metadata)
            
            # Convert document using Docling
            self.logger.info(f"Converting document: {file_path.name}")
            conversion_result = await self._convert_document(file_path)
            
            # Extract structured content
            content_data = await self._extract_content(conversion_result)
            
            # Create semantic chunks
            chunks = await self._create_semantic_chunks(content_data["text"])

            # Generate and store embeddings
            embeddings_data = await self._generate_and_store_embeddings(
                file_path, doc_metadata, content_data, chunks
            )

            # Create processed document data
            processed_data = {
                "source_path": str(file_path),
                "filename": file_path.name,
                "document_type": doc_metadata["file_type"],
                "file_size": doc_metadata["file_size"],
                "file_hash": doc_metadata["file_hash"],
                "metadata": doc_metadata,
                "content": content_data,
                "chunks": chunks,
                "embeddings": embeddings_data,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "processed_at": datetime.now().isoformat()
            }
            
            # Update statistics
            self._update_processing_stats(processed_data)
            
            self.logger.info(
                f"Document processed successfully: {file_path.name} "
                f"({processed_data['processing_time']:.2f}s)"
            )
            
            return processed_data
            
        except Exception as e:
            self.processing_stats["error_count"] += 1
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    async def _validate_file(self, file_path: Path):
        """Validate file before processing"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.settings.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {self.settings.max_file_size_mb}MB)"
            )
        
        # Check file format
        file_extension = file_path.suffix.lower().lstrip('.')
        if file_extension not in self.settings.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type not in self.mime_type_mapping:
            self.logger.warning(f"Unknown MIME type: {mime_type}")
    
    async def _extract_file_metadata(self, 
                                   file_path: Path, 
                                   additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract comprehensive file metadata"""
        
        stat = file_path.stat()
        
        # Calculate file hash for deduplication
        file_hash = await self._calculate_file_hash(file_path)
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_type = self.mime_type_mapping.get(mime_type, file_path.suffix.lower().lstrip('.'))
        
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_hash": file_hash,
            "file_type": file_type,
            "mime_type": mime_type,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "additional_metadata": additional_metadata or {}
        }
        
        return metadata
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication"""
        
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _convert_document(self, file_path: Path) -> ConversionResult:
        """Convert document using Docling with smart model loading"""

        # Track document processing tool usage with flow monitoring
        try:
            from flow_monitoring import get_flow_monitor, ToolType
            flow_monitor = get_flow_monitor()

            async with flow_monitor.track_tool_call(ToolType.DOCUMENT_TOOL, "docling_converter"):
                # Get converter on-demand (loads to VRAM only when needed)
                converter = await get_model("docling_converter")

                # Run conversion in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    converter.convert,
                    str(file_path)
                )

                self.logger.debug(f"Document conversion completed for: {file_path.name}")
                return result

        except ImportError:
            # Flow monitoring not available, proceed without tracking
            converter = await get_model("docling_converter")

            # Run conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                converter.convert,
                str(file_path)
            )

            self.logger.debug(f"Document conversion completed for: {file_path.name}")
            return result

        except Exception as e:
            self.logger.error(f"Docling conversion failed: {e}")
            raise
    
    async def _extract_content(self, conversion_result: ConversionResult) -> Dict[str, Any]:
        """Extract comprehensive content from conversion result"""
        
        document = conversion_result.document
        
        content = {
            "text": document.export_to_markdown(),
            "json": document.export_to_dict(),
            "structure": await self._extract_document_structure(document),
            "tables": await self._extract_tables(document),
            "images": await self._extract_images(document),
            "metadata_extracted": await self._extract_document_metadata(document)
        }
        
        return content
    
    async def _extract_document_structure(self, document: DoclingDocument) -> Dict[str, Any]:
        """Extract hierarchical document structure"""
        
        structure = {
            "page_count": len(document.pages) if hasattr(document, 'pages') else 0,
            "sections": [],
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "hierarchy": [],
            "reading_order": []
        }
        
        # Process text elements
        if hasattr(document, 'texts'):
            for i, text_item in enumerate(document.texts):
                item_data = {
                    "index": i,
                    "text": getattr(text_item, 'text', ''),
                    "label": getattr(text_item, 'label', 'unknown'),
                    "bbox": self._extract_bbox(text_item),
                    "confidence": getattr(text_item, 'confidence', 1.0)
                }
                
                # Categorize by type
                label = item_data["label"].lower()
                if "heading" in label:
                    level = self._extract_heading_level(label)
                    heading_data = {**item_data, "level": level}
                    structure["headings"].append(heading_data)
                elif label == "paragraph":
                    structure["paragraphs"].append(item_data)
                elif label == "list":
                    structure["lists"].append(item_data)
                
                structure["reading_order"].append(item_data)
        
        # Build heading hierarchy
        structure["hierarchy"] = self._build_heading_hierarchy(structure["headings"])
        
        return structure
    
    async def _extract_tables(self, document: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract and process table data"""
        
        tables = []
        
        if hasattr(document, 'tables'):
            for i, table in enumerate(document.tables):
                table_data = {
                    "table_id": f"table_{i}",
                    "bbox": self._extract_bbox(table),
                    "num_rows": getattr(table, 'num_rows', 0),
                    "num_cols": getattr(table, 'num_cols', 0),
                    "data": [],
                    "headers": [],
                    "caption": getattr(table, 'caption', ''),
                    "confidence": getattr(table, 'confidence', 1.0)
                }
                
                # Extract table data
                try:
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe()
                        table_data["data"] = df.to_dict('records')
                        table_data["headers"] = df.columns.tolist()
                        table_data["shape"] = df.shape
                    elif hasattr(table, 'data'):
                        # Handle raw table data
                        table_data["data"] = table.data
                except Exception as e:
                    self.logger.warning(f"Could not extract table {i} data: {e}")
                
                tables.append(table_data)
        
        return tables
    
    async def _extract_images(self, document: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract image information and metadata"""
        
        images = []
        
        if hasattr(document, 'pictures'):
            for i, picture in enumerate(document.pictures):
                image_data = {
                    "image_id": f"image_{i}",
                    "bbox": self._extract_bbox(picture),
                    "caption": getattr(picture, 'caption', ''),
                    "alt_text": getattr(picture, 'alt_text', ''),
                    "size": getattr(picture, 'size', None),
                    "format": getattr(picture, 'format', None),
                    "confidence": getattr(picture, 'confidence', 1.0)
                }
                
                # Extract image bytes if available
                if hasattr(picture, 'image') and picture.image:
                    try:
                        # Save image to temp directory for later processing
                        image_path = self.settings.temp_dir / f"image_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        picture.image.save(image_path)
                        image_data["saved_path"] = str(image_path)
                    except Exception as e:
                        self.logger.warning(f"Could not save image {i}: {e}")
                
                images.append(image_data)
        
        return images
    
    async def _extract_document_metadata(self, document: DoclingDocument) -> Dict[str, Any]:
        """Extract document-level metadata from Docling document"""
        
        metadata = {
            "title": getattr(document, 'title', ''),
            "author": getattr(document, 'author', ''),
            "subject": getattr(document, 'subject', ''),
            "creator": getattr(document, 'creator', ''),
            "producer": getattr(document, 'producer', ''),
            "creation_date": getattr(document, 'creation_date', None),
            "modification_date": getattr(document, 'modification_date', None),
            "language": getattr(document, 'language', ''),
            "page_count": len(document.pages) if hasattr(document, 'pages') else 0
        }
        
        # Clean up None values
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}

        return metadata

    async def _create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks from document text"""

        chunks = []

        if not text.strip():
            return chunks

        # Simple sentence-aware chunking (can be enhanced with semantic splitting)
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # Check if adding this sentence would exceed chunk size
            if (current_length + sentence_length > self.settings.chunk_size and
                current_chunk and
                current_length > self.settings.chunk_size // 2):

                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk = {
                    "chunk_id": f"chunk_{chunk_index}",
                    "text": chunk_text,
                    "start_char": text.find(current_chunk[0]),
                    "end_char": text.find(current_chunk[-1]) + len(current_chunk[-1]),
                    "word_count": len(chunk_text.split()),
                    "sentence_count": len(current_chunk),
                    "chunk_index": chunk_index
                }
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self._calculate_overlap_sentences(current_chunk):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = {
                "chunk_id": f"chunk_{chunk_index}",
                "text": chunk_text,
                "start_char": text.find(current_chunk[0]) if current_chunk else 0,
                "end_char": text.find(current_chunk[-1]) + len(current_chunk[-1]) if current_chunk else 0,
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_chunk),
                "chunk_index": chunk_index
            }
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""

        # Simple sentence splitting (can be enhanced with NLTK or spaCy)
        import re

        # Split on sentence endings, but preserve abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out empty sentences and clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate number of sentences to overlap between chunks"""

        # Use 20% overlap or minimum 1 sentence
        overlap = max(1, len(sentences) // 5)
        return min(overlap, len(sentences) - 1)

    def _extract_bbox(self, item) -> Optional[Dict[str, float]]:
        """Extract bounding box coordinates from document item"""

        try:
            if hasattr(item, 'bbox'):
                bbox = item.bbox
                return {
                    "x": getattr(bbox, 'x', 0.0),
                    "y": getattr(bbox, 'y', 0.0),
                    "width": getattr(bbox, 'width', 0.0),
                    "height": getattr(bbox, 'height', 0.0)
                }
        except Exception:
            pass

        return None

    def _extract_heading_level(self, label: str) -> int:
        """Extract heading level from label"""

        import re

        # Look for heading level in label (e.g., "heading-1", "h1", etc.)
        match = re.search(r'(\d+)', label.lower())
        if match:
            return int(match.group(1))

        # Default heading levels based on common patterns
        label_lower = label.lower()
        if 'title' in label_lower:
            return 1
        elif 'subtitle' in label_lower:
            return 2
        elif 'heading' in label_lower:
            return 3

        return 1

    def _build_heading_hierarchy(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical structure from headings"""

        hierarchy = []
        stack = []

        for heading in headings:
            level = heading.get("level", 1)

            # Pop stack until we find a parent level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            # Create hierarchy entry
            hierarchy_entry = {
                "heading": heading,
                "level": level,
                "parent_index": stack[-1]["index"] if stack else None,
                "children": []
            }

            # Add to parent's children if exists
            if stack:
                parent_index = stack[-1]["index"]
                for h in hierarchy:
                    if h["heading"]["index"] == parent_index:
                        h["children"].append(len(hierarchy))
                        break

            hierarchy.append(hierarchy_entry)
            stack.append({"level": level, "index": heading["index"]})

        return hierarchy

    def _update_processing_stats(self, processed_data: Dict[str, Any]):
        """Update processing statistics"""

        self.processing_stats["documents_processed"] += 1
        self.processing_stats["total_processing_time"] += processed_data["processing_time"]

        # Update format statistics
        doc_type = processed_data["document_type"]
        if doc_type not in self.processing_stats["format_stats"]:
            self.processing_stats["format_stats"][doc_type] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }

        format_stats = self.processing_stats["format_stats"][doc_type]
        format_stats["count"] += 1
        format_stats["total_time"] += processed_data["processing_time"]
        format_stats["average_time"] = format_stats["total_time"] / format_stats["count"]

        # Update page count if available
        page_count = processed_data.get("content", {}).get("structure", {}).get("page_count", 0)
        self.processing_stats["total_pages_processed"] += page_count

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""

        stats = self.processing_stats.copy()

        # Calculate averages
        if stats["documents_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["documents_processed"]
            )
            stats["average_pages_per_document"] = (
                stats["total_pages_processed"] / stats["documents_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["average_pages_per_document"] = 0.0

        return stats

    async def _generate_and_store_embeddings(self, file_path: Path, doc_metadata: Dict[str, Any],
                                           content_data: Dict[str, Any], chunks: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for document chunks and store them in vector database.

        Args:
            file_path: Path to the source document
            doc_metadata: Document metadata
            content_data: Extracted content data
            chunks: List of text chunks

        Returns:
            Dictionary with embedding generation results
        """
        try:
            # Initialize vector store if not already done
            if not hasattr(self.vector_store, 'pool') or self.vector_store.pool is None:
                await self.vector_store.initialize()

            # Store document metadata in vector database
            document_id = await self.vector_store.store_document(
                filename=file_path.name,
                file_path=str(file_path),
                file_size=doc_metadata["file_size"],
                mime_type=doc_metadata["mime_type"],
                metadata=doc_metadata
            )

            # Store document content
            content_id = await self.vector_store.store_document_content(
                document_id=document_id,
                content_type="text",
                content_text=content_data["text"],
                page_number=content_data.get("page_count", 1),
                position_data={"extraction_method": "docling"},
                extraction_metadata={
                    "tables_count": len(content_data.get("tables", [])),
                    "images_count": len(content_data.get("images", [])),
                    "processing_time": content_data.get("processing_time", 0)
                }
            )

            # Generate embeddings using Brain 1 via HTTP API
            self.logger.info(f"Requesting embeddings for {len(chunks)} chunks from Brain 1 (Qwen3-4B)...")

            # Brain‑1 service base URL from env; default to compose service name
            base_url = os.getenv("BRAIN1_EMBED_URL_BASE", "http://embedding-service:8001")
            timeout_s = int(os.getenv("BRAIN1_EMBED_TIMEOUT", "60"))

            client = Brain1APIClient(base_url=base_url, timeout=timeout_s)
            ready = await client.wait_for_ready(max_attempts=30, delay=2.0)
            if not ready:
                raise RuntimeError("Brain 1 service not ready for embedding generation")

            # Batch request in manageable chunks
            embeddings: List[List[float]] = []
            batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_embeddings = await client.generate_batch_embeddings(batch, dimensions=2000)
                # Convert Nones to empty lists to keep alignment
                for emb in batch_embeddings:
                    embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else (emb if emb is not None else []))

            await client.close()

            # Validate we have embeddings for each chunk
            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: got {len(embeddings)} for {len(chunks)} chunks")

            # Store embeddings in vector database
            embedding_ids = await self.vector_store.store_embeddings(
                document_id=document_id,
                content_id=content_id,
                chunks=chunks,
                embeddings=embeddings,
                model_name="Qwen3-Embedding-4B"
            )

            self.logger.info(f"✅ Generated and stored {len(embedding_ids)} embeddings for {file_path.name}")

            return {
                "document_id": document_id,
                "content_id": content_id,
                "embedding_ids": embedding_ids,
                "embedding_count": len(embedding_ids),
                "embedding_dimension": len(embeddings[0]) if embeddings and embeddings[0] else 0,
                "model_used": "Qwen3-Embedding-4B"
            }

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {file_path.name}: {e}")
            # Return empty result on failure
            return {
                "document_id": None,
                "content_id": None,
                "embedding_ids": [],
                "embedding_count": 0,
                "embedding_dimension": 0,
                "model_used": "Qwen3-Embedding-4B",
                "error": str(e)
            }
