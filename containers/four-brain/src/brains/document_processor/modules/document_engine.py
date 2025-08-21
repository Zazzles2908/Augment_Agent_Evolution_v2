"""
Document Engine Module for Brain-4
Handles document processing workflow, semantic chunking, and content extraction

Extracted from brain4_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentEngine:
    """
    Document Engine for Brain-4 - Handles document processing workflow
    Extracted from brain4_manager.py for modular architecture
    """
    
    def __init__(self, docling_manager=None, config_manager=None):
        """Initialize Document Engine with dependencies"""
        self.docling_manager = docling_manager
        self.config_manager = config_manager
        
        # Configuration
        self.supported_formats = ["pdf", "docx", "pptx", "html", "md", "txt"]
        self.max_file_size_mb = 100
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Processing tracking
        self.total_documents_processed = 0
        self.total_processing_time = 0.0
        self.total_chunks_created = 0
        
        # Task management
        self.processing_queue = asyncio.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        
        logger.info("🔧 Document Engine initialized")
        logger.info(f"📄 Supported formats: {', '.join(self.supported_formats)}")
    
    async def process_document(self, file_path: Union[str, Path], 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document through the complete workflow
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            Dictionary with all extracted information
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            logger.info(f"📄 Processing document: {file_path.name}")
            
            # Validate file
            await self._validate_file(file_path)
            
            # Extract basic metadata
            doc_metadata = await self._extract_file_metadata(file_path, metadata)
            
            # Convert document using Docling
            conversion_result = await self.docling_manager.convert_document(file_path)
            
            # Extract structured content
            content_data = await self._extract_content(conversion_result)

            # Compute conversion quality metrics (heuristic; Docling doesn't expose a top-level confidence)
            quality = await self._compute_conversion_quality(conversion_result, content_data)
            content_data["quality"] = quality

            # Create semantic chunks
            chunks = await self._create_semantic_chunks(content_data["text"])
            
            # Compile final result
            result = {
                "document_id": str(uuid.uuid4()),
                "file_path": str(file_path),
                "metadata": doc_metadata,
                "content": content_data,
                "chunks": chunks,
                "processing_time": time.time() - start_time,
                "processed_at": datetime.now().isoformat(),
                "success": True,
                "quality": quality
            }
            
            # Update statistics
            self.total_documents_processed += 1
            self.total_processing_time += result["processing_time"]
            self.total_chunks_created += len(chunks)
            
            logger.info(f"✅ Document processed successfully: {file_path.name}")
            logger.info(f"📊 Created {len(chunks)} chunks in {result['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Document processing failed: {e}")
            
            return {
                "document_id": None,
                "file_path": str(file_path),
                "metadata": metadata or {},
                "content": {},
                "chunks": [],
                "processing_time": processing_time,
                "processed_at": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def _validate_file(self, file_path: Path):
        """Validate file exists and is supported format"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_extension}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
        
        logger.debug(f"✅ File validated: {file_path.name} ({file_size_mb:.1f}MB)")
    
    async def _extract_file_metadata(self, file_path: Path, 
                                   additional_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract basic file metadata"""
        stat = file_path.stat()
        
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "file_size_mb": stat.st_size / (1024 * 1024),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "mime_type": self._get_mime_type(file_path)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    async def _extract_content(self, conversion_result) -> Dict[str, Any]:
        """Extract comprehensive content from Docling conversion result.
        Be robust to different Docling return types: object with .document, a document-like object,
        or an iterable/generator of such objects.
        """
        try:
            document = None

            # Case 1: result has a .document attribute
            doc_attr = getattr(conversion_result, 'document', None)
            if doc_attr is not None:
                document = doc_attr
            else:
                # Case 2: result itself is document-like
                if hasattr(conversion_result, 'export_to_markdown') and hasattr(conversion_result, 'export_to_dict'):
                    document = conversion_result
                else:
                    # Case 3: result is iterable/generator; scan a window for a suitable item
                    try:
                        from itertools import islice
                        iterator = iter(conversion_result)
                        items = list(islice(iterator, 0, 50))
                        for item in items:
                            if hasattr(item, 'document') and getattr(item, 'document') is not None:
                                document = item.document
                                break
                            if hasattr(item, 'export_to_markdown') and hasattr(item, 'export_to_dict'):
                                document = item
                                break
                    except TypeError:
                        # Not iterable
                        pass
                    except Exception:
                        pass

            if document is None:
                raise ValueError("Docling conversion produced no document")

            # Fallbacks: if document is a path-like or plain string
            from pathlib import Path as _Path
            text_out = ""
            json_out = {}
            if isinstance(document, (str, _Path)):
                p = _Path(document)
                try:
                    if p.exists() and p.is_file():
                        if p.suffix.lower() in {'.md', '.txt'}:
                            text_out = p.read_text(encoding='utf-8', errors='ignore')
                        elif p.suffix.lower() == '.json':
                            import json as _json
                            json_out = _json.loads(p.read_text(encoding='utf-8', errors='ignore'))
                            # try to recover text from json if present
                            text_out = json_out.get('text') or text_out
                    else:
                        # treat as plain text payload
                        text_out = str(document)
                except Exception as fe:
                    logger.warning(f"⚠️ Path-like document fallback failed: {fe}")
                    text_out = str(document)

            # Normal path: document has export methods
            if not text_out and hasattr(document, 'export_to_markdown'):
                try:
                    text_out = document.export_to_markdown()
                except Exception as exm:
                    logger.warning(f"⚠️ export_to_markdown failed: {exm}")
            if not json_out and hasattr(document, 'export_to_dict'):
                try:
                    json_out = document.export_to_dict()
                except Exception as exd:
                    logger.warning(f"⚠️ export_to_dict failed: {exd}")

            content = {
                "text": text_out or "",
                "json": json_out or {},
                "structure": await self._extract_document_structure(document) if hasattr(document, 'export_to_markdown') else {},
                "tables": await self._extract_tables(document) if hasattr(document, 'export_to_markdown') else [],
                "images": await self._extract_images(document) if hasattr(document, 'export_to_markdown') else [],
                "metadata_extracted": await self._extract_document_metadata(document) if hasattr(document, 'export_to_markdown') else {}
            }

            logger.debug(f"📊 Content extracted: {len(content['text'])} chars")
            return content

        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return {
                "text": "",
                "json": {},
                "structure": {},
                "tables": [],
                "images": [],
                "metadata_extracted": {}
            }
    
    async def _extract_document_structure(self, document) -> Dict[str, Any]:
        """Extract document structure information"""
        try:
            # Extract headings, sections, etc.
            structure = {
                "page_count": getattr(document, 'page_count', 0),
                "headings": [],
                "sections": [],
                "paragraphs": 0
            }
            
            # Count paragraphs in text
            text = document.export_to_markdown()
            structure["paragraphs"] = len([p for p in text.split('\n\n') if p.strip()])
            
            return structure
            
        except Exception as e:
            logger.warning(f"⚠️ Structure extraction failed: {e}")
            return {}
    
    async def _extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        try:
            tables = []
            # Extract table information from document
            # This would be implemented based on Docling's table extraction capabilities
            return tables
            
        except Exception as e:
            logger.warning(f"⚠️ Table extraction failed: {e}")
            return []
    
    async def _extract_images(self, document) -> List[Dict[str, Any]]:
        """Extract images from document"""
        try:
            images = []
            # Extract image information from document
            # This would be implemented based on Docling's image extraction capabilities
            return images
            
        except Exception as e:
            logger.warning(f"⚠️ Image extraction failed: {e}")
            return []
    
    async def _extract_document_metadata(self, document) -> Dict[str, Any]:
        """Extract metadata from document"""
        try:
            metadata = {}
            # Extract document metadata
            # This would be implemented based on Docling's metadata extraction capabilities
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Document metadata extraction failed: {e}")
            return {}
    
    async def _create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks from text"""
        try:
            if not text.strip():
                return []
            
            chunks = []
            
            # Simple chunking strategy (can be enhanced with semantic analysis)
            words = text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                
                if current_length + word_length > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "chunk_id": len(chunks),
                        "text": chunk_text,
                        "length": len(chunk_text),
                        "word_count": len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-self.chunk_overlap//10:] if len(current_chunk) > self.chunk_overlap//10 else []
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "length": len(chunk_text),
                    "word_count": len(current_chunk)
                })
            
            logger.debug(f"📄 Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Chunking failed: {e}")
            return []
    
    async def _compute_conversion_quality(self, conversion_result: Any, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a heuristic quality report since ConversionResult lacks explicit confidence.
        Returns a dict with fields: score [0..1], coverage_text, tables_found, images_found, warnings, errors.
        """
        try:
            doc = getattr(conversion_result, 'document', None)
            text_len = len(content_data.get('text') or '')
            tables = content_data.get('tables') or []
            images = content_data.get('images') or []
            warnings = getattr(conversion_result, 'warnings', None)
            errors = getattr(conversion_result, 'errors', None)

            # Basic coverage: normalized by pages if available, else by thresholded lengths
            page_count = getattr(doc, 'page_count', None) or 0
            if page_count and text_len:
                avg_chars_per_page = text_len / max(page_count, 1)
                # Simple sigmoid-like scaling for coverage (empirical thresholds)
                coverage = min(1.0, max(0.0, (avg_chars_per_page / 1500.0)))
            else:
                coverage = 1.0 if text_len > 2000 else 0.6 if text_len > 500 else 0.3 if text_len > 50 else 0.1

            table_bonus = min(0.2, 0.02 * len(tables))
            image_bonus = min(0.1, 0.01 * len(images))
            warn_penalty = 0.1 if warnings else 0.0
            err_penalty = 0.3 if errors else 0.0

            score = max(0.0, min(1.0, coverage + table_bonus + image_bonus - warn_penalty - err_penalty))
            return {
                "score": round(score, 3),
                "coverage_text": round(coverage, 3),
                "tables_found": len(tables),
                "images_found": len(images),
                "warnings": bool(warnings),
                "errors": bool(errors)
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute conversion quality: {e}")
            return {"score": 0.0, "coverage_text": 0.0, "tables_found": 0, "images_found": 0, "warnings": False, "errors": True}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_documents_processed
            if self.total_documents_processed > 0 else 0
        )
        
        avg_chunks_per_doc = (
            self.total_chunks_created / self.total_documents_processed
            if self.total_documents_processed > 0 else 0
        )
        
        return {
            "total_documents_processed": self.total_documents_processed,
            "total_processing_time": self.total_processing_time,
            "total_chunks_created": self.total_chunks_created,
            "average_processing_time": avg_processing_time,
            "average_chunks_per_document": avg_chunks_per_doc,
            "supported_formats": self.supported_formats,
            "max_file_size_mb": self.max_file_size_mb,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.total_documents_processed = 0
        self.total_processing_time = 0.0
        self.total_chunks_created = 0
        logger.info("📊 Document engine statistics reset")
