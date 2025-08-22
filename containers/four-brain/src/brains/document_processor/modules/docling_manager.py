"""
Docling Manager Module for Brain-4
Handles IBM Docling DocumentConverter lifecycle and document conversion

Extracted from brain4_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import asyncio
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DoclingManager:
    """
    Docling Manager for Brain-4 - Handles IBM Docling DocumentConverter
    Extracted from brain4_manager.py for modular architecture
    """
    
    def __init__(self, config_manager=None):
        """Initialize Docling Manager with configuration"""
        self.config_manager = config_manager
        
        # Docling state
        self.converter = None
        self.converter_loaded = False
        self.loading_time = 0.0
        self.initialization_time = time.time()
        
        # Configuration
        self.model_cache_dir = "/workspace/models/cache"
        self.enable_ocr = True
        self.enable_table_extraction = True
        self.enable_image_extraction = True

        # GPU Configuration for Docling
        self.use_gpu = True
        self.device = "cuda" if self._check_cuda_available() else "cpu"
        logger.info(f"🎮 Docling configured for device: {self.device}")
        
        # Performance tracking
        self.total_conversions = 0
        self.total_conversion_time = 0.0
        
        logger.info("🔧 Docling Manager initialized")

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_converter(self, force_reload: bool = False) -> bool:
        """
        Load IBM Docling DocumentConverter
        
        Args:
            force_reload: Force reload even if converter is already loaded
            
        Returns:
            True if converter loaded successfully, False otherwise
        """
        if self.converter_loaded and not force_reload:
            logger.info("✅ Docling converter already loaded")
            return True
        
        logger.info("🧠 Loading IBM Docling DocumentConverter...")
        
        start_time = time.time()
        
        try:
            # Prefer local models path and set env BEFORE importing Docling components
            import os
            local_models = os.getenv('DOCLING_MODELS_PATH', '/workspace/.cache/docling')
            os.makedirs(local_models, exist_ok=True)
            # Ensure Docling/deepsearch_glm honor writable cache dir (resources only)
            os.environ.setdefault('DOCLING_MODELS_PATH', local_models)
            os.environ.setdefault('DEEPSEARCH_GLM_RESOURCES_DIR', local_models)

            # Copy models.json from package resources into local cache if missing
            import shutil
            package_base = '/usr/local/lib/python3.12/dist-packages/deepsearch_glm/resources'
            pkg_models = os.path.join(package_base, 'models.json')
            cache_models = os.path.join(local_models, 'models.json')
            try:
                if os.path.exists(pkg_models) and not os.path.exists(cache_models):
                    os.makedirs(os.path.dirname(cache_models), exist_ok=True)
                    shutil.copyfile(pkg_models, cache_models)
            except Exception as ce:
                logger.warning(f"Docling models.json copy skipped: {ce}")

            # Import Docling components after env is set
            from docling.document_converter import DocumentConverter, PipelineOptions

            # Configure pipeline options - favor ACCURATE table extraction and GPU OCR when supported
            pipeline_options = PipelineOptions()
            try:
                # These options are version-dependent; wrap in try/except to avoid fabrication
                if hasattr(pipeline_options, 'table_recognition_mode'):
                    pipeline_options.table_recognition_mode = 'ACCURATE'
                if hasattr(pipeline_options, 'ocr_engine'):
                    pipeline_options.ocr_engine = 'gpu' if self.device == 'cuda' else 'cpu'
                if hasattr(pipeline_options, 'enable_image_extraction'):
                    pipeline_options.enable_image_extraction = self.enable_image_extraction
                if hasattr(pipeline_options, 'enable_table_extraction'):
                    pipeline_options.enable_table_extraction = self.enable_table_extraction
            except Exception as pe:
                logger.warning(f"Docling pipeline options partial application: {pe}")

            # Prefer local models path; fetch missing deepsearch_glm assets if needed

            # Best-effort: ensure deepsearch_glm resources are present to avoid network fetch during runtime
            try:
                import json
                import urllib.request
                package_base = '/usr/local/lib/python3.12/dist-packages/deepsearch_glm/resources'
                models_path = os.path.join(package_base, 'models.json')
                if os.path.exists(models_path):
                    with open(models_path, 'r', encoding='utf-8') as f:
                        models = json.load(f)
                    obj = models.get('object-store')
                    nlp = models.get('nlp', {})
                    prefix = nlp.get('prefix', '')
                    trained = nlp.get('trained-models', {})
                    for k, v in trained.items():
                        try:
                            src_name, dest_rel = v[0], v[1]
                            url = f"{obj}/{prefix}/{src_name}"
                            # Write resources into writable cache instead of site-packages
                            dest = os.path.join(local_models, dest_rel)
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                                logger.info(f"⬇️ Fetching Docling resource {k} from {url}")
                                urllib.request.urlretrieve(url, dest)
                        except Exception as fe:
                            logger.warning(f"Docling resource fetch skipped for {k}: {fe}")
            except Exception as de:
                logger.warning(f"Docling resources prefetch skipped: {de}")

            # Create converter with custom cache/model dir if supported
            try:
                self.converter = DocumentConverter(models_cache_dir=local_models)
            except TypeError:
                # Older versions may not have this arg; instantiate default
                self.converter = DocumentConverter()
                # Many Docling components read env vars for model base; set for subprocess/tools
                os.environ.setdefault('DOCLING_MODELS_PATH', local_models)

            # Define a simple document shim for fallback cases
            class _SimpleDoc:
                def __init__(self, text:str = "", obj:Dict[str,Any] = None):
                    self._text = text or ""
                    self._obj = obj or {}
                def export_to_markdown(self):
                    return self._text
                def export_to_dict(self):
                    return self._obj

            self.converter_loaded = True
            self.loading_time = time.time() - start_time
            
            logger.info(f"✅ Docling converter loaded successfully in {self.loading_time:.2f} seconds")
            logger.info("📊 Converter specs: IBM Docling DocumentConverter")
            logger.info(f"🔧 OCR enabled: {self.enable_ocr}")
            logger.info(f"🔧 Table extraction: {self.enable_table_extraction}")
            logger.info(f"🔧 Image extraction: {self.enable_image_extraction}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Docling converter loading failed: {e}")
            return False
    
    def unload_converter(self):
        """Unload converter and free memory"""
        try:
            if self.converter is not None:
                del self.converter
                self.converter = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.converter_loaded = False
            logger.info("✅ Docling converter unloaded and memory freed")
            
        except Exception as e:
            logger.error(f"❌ Error unloading converter: {e}")
    
    def is_converter_loaded(self) -> bool:
        """Check if converter is loaded and ready"""
        return self.converter_loaded and self.converter is not None
    
    async def convert_document(self, file_path: Path) -> Any:
        """
        Convert document using Docling DocumentConverter
        
        Args:
            file_path: Path to document file
            
        Returns:
            Docling ConversionResult
        """
        if not self.is_converter_loaded():
            if not self.load_converter():
                raise RuntimeError("Docling converter not loaded")
        
        start_time = time.time()
        self.total_conversions += 1
        
        try:
            logger.info(f"🔄 Converting document: {file_path.name}")
            
            # Track document processing tool usage with flow monitoring
            try:
                from flow_monitoring import get_flow_monitor, ToolType
                flow_monitor = get_flow_monitor()

                async with flow_monitor.track_tool_call(ToolType.DOCUMENT_TOOL, "docling_converter"):
                    # Run conversion in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        self.converter.convert,
                        str(file_path)
                    )

            except ImportError:
                # Flow monitoring not available, proceed without tracking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.converter.convert,
                    str(file_path)
                )

            conversion_time = time.time() - start_time
            self.total_conversion_time += conversion_time

            logger.info(f"✅ Document conversion completed in {conversion_time:.2f}s")
            # Normalize Docling result to a document-like object
            doc_obj = None
            try:
                # Case A: Known container shapes
                if hasattr(result, 'document') and getattr(result, 'document') is not None:
                    doc_obj = result.document
                elif hasattr(result, 'documents') and isinstance(getattr(result, 'documents'), (list, tuple)) and getattr(result, 'documents'):
                    doc_obj = result.documents[0]
                elif hasattr(result, 'docs') and isinstance(getattr(result, 'docs'), (list, tuple)) and getattr(result, 'docs'):
                    doc_obj = result.docs[0]
                # Case B: Direct document-like
                elif hasattr(result, 'export_to_markdown') and hasattr(result, 'export_to_dict'):
                    doc_obj = result
                else:
                    # Case C: Iterable/generator, scan a reasonable number of items
                    try:
                        from itertools import islice
                        # Materialize a small window so we don't exhaust the generator for callers
                        iterator = iter(result)
                        items = list(islice(iterator, 0, 20))
                        for idx, item in enumerate(items):
                            try:
                                # Diagnostic logging to capture shape of items
                                try:
                                    if isinstance(item, dict):
                                        logger.info(f"🔍 Docling item[{idx}] dict keys: {list(item.keys())[:10]}")
                                    else:
                                        logger.info(f"🔍 Docling item[{idx}] type: {type(item)} attrs: {dir(item)[:10]}")
                                except Exception:
                                    pass

                                if hasattr(item, 'document') and getattr(item, 'document') is not None:
                                    doc_obj = item.document
                                    break
                                if hasattr(item, 'export_to_markdown') and hasattr(item, 'export_to_dict'):
                                    doc_obj = item
                                    break
                                if isinstance(item, dict):
                                    # Common patterns: item may contain 'document' or 'docs'
                                    maybe_doc = item.get('document') or item.get('doc')
                                    if maybe_doc is not None:
                                        if hasattr(maybe_doc, 'export_to_markdown') and hasattr(maybe_doc, 'export_to_dict'):
                                            doc_obj = maybe_doc
                                            break
                                    maybe_docs = item.get('documents') or item.get('docs')
                                    if isinstance(maybe_docs, (list, tuple)) and maybe_docs:
                                        cand = maybe_docs[0]
                                        if hasattr(cand, 'export_to_markdown') and hasattr(cand, 'export_to_dict'):
                                            doc_obj = cand
                                            break
                                if isinstance(item, (tuple, list)) and item:
                                    cand = item[0]
                                    if hasattr(cand, 'export_to_markdown') and hasattr(cand, 'export_to_dict'):
                                        doc_obj = cand
                                        break
                                if isinstance(item, (str, Path)):
                                    pfirst = Path(item)
                                    if pfirst.exists():
                                        if pfirst.is_dir():
                                            md = pfirst / 'document.md'
                                            js = pfirst / 'document.json'
                                            text, obj = "", {}
                                            try:
                                                if md.exists():
                                                    text = md.read_text(encoding='utf-8', errors='ignore')
                                                if js.exists():
                                                    import json as _json
                                                    obj = _json.loads(js.read_text(encoding='utf-8', errors='ignore'))
                                                doc_obj = _SimpleDoc(text, obj)
                                                break
                                            except Exception:
                                                pass
                                        elif pfirst.is_file():
                                            text, obj = "", {}
                                            try:
                                                if pfirst.suffix.lower() in {'.md', '.txt'}:
                                                    text = pfirst.read_text(encoding='utf-8', errors='ignore')
                                                elif pfirst.suffix.lower() == '.json':
                                                    import json as _json
                                                    obj = _json.loads(pfirst.read_text(encoding='utf-8', errors='ignore'))
                                                doc_obj = _SimpleDoc(text, obj)
                                                break
                                            except Exception:
                                                pass
                            except Exception as item_e:
                                logger.warning(f"Docling item[{idx}] processing error: {item_e}")
                                continue
                        # If we didn't find a document-like here, keep doc_obj as None
                        # Do not return raw items/generator; fall through to other fallbacks
                    except Exception:
                        pass
                # Case D: Non-iterable path-like
                if doc_obj is None and isinstance(result, (str, Path)):
                    p = Path(result)
                    if p.exists():
                        if p.is_dir():
                            md = p / 'document.md'
                            js = p / 'document.json'
                            text, obj = "", {}
                            try:
                                if md.exists():
                                    text = md.read_text(encoding='utf-8', errors='ignore')
                                if js.exists():
                                    import json as _json
                                    obj = _json.loads(js.read_text(encoding='utf-8', errors='ignore'))
                                doc_obj = _SimpleDoc(text, obj)
                            except Exception:
                                pass
                        elif p.is_file():
                            text, obj = "", {}
                            try:
                                if p.suffix.lower() in {'.md', '.txt'}:
                                    text = p.read_text(encoding='utf-8', errors='ignore')
                                elif p.suffix.lower() == '.json':
                                    import json as _json
                                    obj = _json.loads(p.read_text(encoding='utf-8', errors='ignore'))
                                doc_obj = _SimpleDoc(text, obj)
                            except Exception:
                                pass
            except Exception as norm_e:
                logger.warning(f"Docling result normalization warning: {norm_e}")

            logger.info(f"📊 Conversion result type: {type(result)} | normalized: {type(doc_obj)}")

            return doc_obj or result
            
        except Exception as e:
            conversion_time = time.time() - start_time
            self.total_conversion_time += conversion_time
            
            logger.error(f"❌ Document conversion failed: {e}")
            raise
    
    def get_converter_info(self) -> Dict[str, Any]:
        """Get converter information and statistics"""
        avg_conversion_time = (
            self.total_conversion_time / self.total_conversions
            if self.total_conversions > 0 else 0
        )
        
        return {
            "converter_loaded": self.converter_loaded,
            "loading_time_seconds": self.loading_time,
            "uptime_seconds": time.time() - self.initialization_time,
            "total_conversions": self.total_conversions,
            "total_conversion_time": self.total_conversion_time,
            "average_conversion_time": avg_conversion_time,
            "enable_ocr": self.enable_ocr,
            "enable_table_extraction": self.enable_table_extraction,
            "enable_image_extraction": self.enable_image_extraction,
            "model_cache_dir": self.model_cache_dir
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform converter health check"""
        try:
            if not self.is_converter_loaded():
                return {
                    "healthy": False,
                    "status": "converter_not_loaded",
                    "error": "Docling converter not loaded"
                }
            
            # Basic converter test (if possible)
            # Note: This would depend on Docling's API for health checking
            
            return {
                "healthy": True,
                "status": "operational",
                "converter_loaded": True,
                "total_conversions": self.total_conversions
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error": str(e)
            }
    
    def reload_converter(self) -> bool:
        """Reload converter (unload and load again)"""
        logger.info("🔄 Reloading Docling converter...")
        self.unload_converter()
        return self.load_converter(force_reload=True)
    
    def get_supported_formats(self) -> list:
        """Get list of supported document formats"""
        return ["pdf", "docx", "pptx", "html", "md", "txt"]
    
    def configure_pipeline(self, **options):
        """Configure Docling pipeline options"""
        try:
            if "enable_ocr" in options:
                self.enable_ocr = options["enable_ocr"]
            
            if "enable_table_extraction" in options:
                self.enable_table_extraction = options["enable_table_extraction"]
            
            if "enable_image_extraction" in options:
                self.enable_image_extraction = options["enable_image_extraction"]
            
            logger.info("🔧 Docling pipeline configuration updated")
            logger.info(f"   OCR: {self.enable_ocr}")
            logger.info(f"   Tables: {self.enable_table_extraction}")
            logger.info(f"   Images: {self.enable_image_extraction}")
            
            # If converter is loaded, may need to reload with new configuration
            if self.converter_loaded:
                logger.info("🔄 Reloading converter with new configuration...")
                return self.reload_converter()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline configuration failed: {e}")
            return False
