#!/usr/bin/env python3.11
"""
TensorRT Engine Cache Manager for Four-Brain System
Pre-build once, reuse forever - Intelligent TensorRT engine caching and management

Author: Zazzles's Agent
Date: 2025-08-02
Purpose: Eliminate TensorRT compilation overhead through intelligent engine caching
"""

import os
import sys
import logging
import time
import hashlib
import json
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import shutil

# Configure logging
logger = logging.getLogger(__name__)

class PrecisionMode(Enum):
    """TensorRT precision modes"""
    FP32 = "fp32"
    FP16 = "fp16"
    FP4 = "fp4"
    INT8 = "int8"

class EngineStatus(Enum):
    """TensorRT engine status"""
    NOT_BUILT = "not_built"
    BUILDING = "building"
    BUILT = "built"
    CACHED = "cached"
    FAILED = "failed"
    LOADING = "loading"
    LOADED = "loaded"

@dataclass
class EngineConfig:
    """TensorRT engine configuration"""
    model_name: str
    model_path: str
    precision: PrecisionMode
    max_batch_size: int = 1
    max_workspace_gb: float = 4.0
    device_memory_fraction: float = 0.6
    optimization_level: int = 3
    use_cuda_graphs: bool = True
    use_fused_attention: bool = True
    dla_core: Optional[int] = None
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        config_str = f"{self.model_name}_{self.precision.value}_{self.max_batch_size}_{self.max_workspace_gb}_{self.optimization_level}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

@dataclass
class EngineInfo:
    """TensorRT engine information"""
    config: EngineConfig
    engine_path: str
    build_time: float
    file_size_mb: float
    status: EngineStatus
    build_log: str = ""
    error_message: str = ""
    last_used: float = 0.0
    use_count: int = 0
    performance_metrics: Dict[str, Any] = None

class TensorRTEngineCacheManager:
    """Manages TensorRT engine caching for optimal performance"""
    
    def __init__(self, cache_dir: str = "/workspace/models/tensorrt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Engine tracking
        self.engines: Dict[str, EngineInfo] = {}
        self.build_lock = threading.Lock()
        self.load_lock = threading.Lock()
        
        # Cache management
        self.max_cache_size_gb = 20.0  # Maximum cache size
        self.max_engines = 50  # Maximum number of cached engines
        
        # TensorRT settings
        self.trtexec_path = self._find_trtexec()
        self.tensorrt_available = self._check_tensorrt_availability()
        
        # Load existing cache
        self._load_cache_index()
        
        logger.info("🏗️ TensorRT Engine Cache Manager initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  TensorRT available: {self.tensorrt_available}")
        logger.info(f"  Cached engines: {len(self.engines)}")
    
    def _find_trtexec(self) -> Optional[str]:
        """Find trtexec executable"""
        possible_paths = [
            "/usr/src/tensorrt/bin/trtexec",
            "/opt/tensorrt/bin/trtexec",
            "/usr/local/bin/trtexec",
            shutil.which("trtexec")
        ]
        
        for path in possible_paths:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"✅ Found trtexec at: {path}")
                return path
        
        logger.warning("⚠️ trtexec not found - TensorRT engine building disabled")
        return None
    
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            logger.info(f"✅ TensorRT available: version {trt.__version__}")
            return True
        except ImportError:
            logger.warning("⚠️ TensorRT not available")
            return False
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    cache_data = json.load(f)
                
                for engine_data in cache_data.get("engines", []):
                    # Reconstruct EngineConfig
                    config_data = engine_data["config"]
                    config = EngineConfig(
                        model_name=config_data["model_name"],
                        model_path=config_data["model_path"],
                        precision=PrecisionMode(config_data["precision"]),
                        max_batch_size=config_data.get("max_batch_size", 1),
                        max_workspace_gb=config_data.get("max_workspace_gb", 4.0),
                        device_memory_fraction=config_data.get("device_memory_fraction", 0.6),
                        optimization_level=config_data.get("optimization_level", 3),
                        use_cuda_graphs=config_data.get("use_cuda_graphs", True),
                        use_fused_attention=config_data.get("use_fused_attention", True)
                    )
                    
                    # Reconstruct EngineInfo
                    engine_info = EngineInfo(
                        config=config,
                        engine_path=engine_data["engine_path"],
                        build_time=engine_data["build_time"],
                        file_size_mb=engine_data["file_size_mb"],
                        status=EngineStatus(engine_data["status"]),
                        build_log=engine_data.get("build_log", ""),
                        error_message=engine_data.get("error_message", ""),
                        last_used=engine_data.get("last_used", 0.0),
                        use_count=engine_data.get("use_count", 0),
                        performance_metrics=engine_data.get("performance_metrics")
                    )
                    
                    cache_key = config.get_cache_key()
                    self.engines[cache_key] = engine_info
                
                logger.info(f"✅ Loaded {len(self.engines)} engines from cache index")
                
            except Exception as e:
                logger.error(f"❌ Failed to load cache index: {str(e)}")
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        
        try:
            cache_data = {
                "timestamp": time.time(),
                "engines": []
            }
            
            for cache_key, engine_info in self.engines.items():
                engine_data = {
                    "cache_key": cache_key,
                    "config": asdict(engine_info.config),
                    "engine_path": engine_info.engine_path,
                    "build_time": engine_info.build_time,
                    "file_size_mb": engine_info.file_size_mb,
                    "status": engine_info.status.value,
                    "build_log": engine_info.build_log,
                    "error_message": engine_info.error_message,
                    "last_used": engine_info.last_used,
                    "use_count": engine_info.use_count,
                    "performance_metrics": engine_info.performance_metrics
                }
                # Convert enum to string for JSON serialization
                engine_data["config"]["precision"] = engine_info.config.precision.value
                cache_data["engines"].append(engine_data)
            
            with open(index_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"✅ Cache index saved with {len(self.engines)} engines")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache index: {str(e)}")
    
    def get_or_build_engine(self, config: EngineConfig) -> Optional[str]:
        """Get cached engine or build new one"""
        cache_key = config.get_cache_key()
        
        # Check if engine is already cached
        if cache_key in self.engines:
            engine_info = self.engines[cache_key]
            
            # Verify engine file exists
            if os.path.exists(engine_info.engine_path):
                # Update usage statistics
                engine_info.last_used = time.time()
                engine_info.use_count += 1
                engine_info.status = EngineStatus.CACHED
                
                logger.info(f"✅ Using cached TensorRT engine: {config.model_name} ({config.precision.value})")
                return engine_info.engine_path
            else:
                logger.warning(f"⚠️ Cached engine file missing: {engine_info.engine_path}")
                # Remove from cache and rebuild
                del self.engines[cache_key]
        
        # Build new engine
        return self._build_engine(config)
    
    def _build_engine(self, config: EngineConfig) -> Optional[str]:
        """Build TensorRT engine using trtexec"""
        if not self.tensorrt_available or not self.trtexec_path:
            logger.error("❌ TensorRT not available for engine building")
            return None
        
        cache_key = config.get_cache_key()
        
        with self.build_lock:
            logger.info(f"🏗️ Building TensorRT engine: {config.model_name} ({config.precision.value})")
            
            # Create engine info
            engine_filename = f"{config.model_name}_{config.precision.value}_{cache_key}.engine"
            engine_path = str(self.cache_dir / engine_filename)
            
            engine_info = EngineInfo(
                config=config,
                engine_path=engine_path,
                build_time=0.0,
                file_size_mb=0.0,
                status=EngineStatus.BUILDING
            )
            
            self.engines[cache_key] = engine_info
            
            try:
                start_time = time.time()
                
                # Build trtexec command
                cmd = self._build_trtexec_command(config, engine_path)
                
                logger.info(f"🔧 Executing: {' '.join(cmd[:5])}... (truncated)")
                
                # Execute trtexec
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                build_time = time.time() - start_time
                engine_info.build_time = build_time
                engine_info.build_log = result.stdout + result.stderr
                
                if result.returncode == 0 and os.path.exists(engine_path):
                    # Success
                    file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
                    engine_info.file_size_mb = file_size
                    engine_info.status = EngineStatus.BUILT
                    
                    logger.info(f"✅ TensorRT engine built successfully:")
                    logger.info(f"  Model: {config.model_name}")
                    logger.info(f"  Precision: {config.precision.value}")
                    logger.info(f"  Build time: {build_time:.1f}s")
                    logger.info(f"  File size: {file_size:.1f}MB")
                    logger.info(f"  Path: {engine_path}")
                    
                    # Save cache index
                    self._save_cache_index()
                    
                    return engine_path
                else:
                    # Failure
                    engine_info.status = EngineStatus.FAILED
                    engine_info.error_message = f"trtexec failed with code {result.returncode}"
                    
                    logger.error(f"❌ TensorRT engine build failed:")
                    logger.error(f"  Return code: {result.returncode}")
                    logger.error(f"  Error: {result.stderr}")
                    
                    return None
                    
            except subprocess.TimeoutExpired:
                engine_info.status = EngineStatus.FAILED
                engine_info.error_message = "Build timeout"
                logger.error(f"❌ TensorRT engine build timed out after 30 minutes")
                return None
                
            except Exception as e:
                engine_info.status = EngineStatus.FAILED
                engine_info.error_message = str(e)
                logger.error(f"❌ TensorRT engine build failed: {str(e)}")
                return None
    
    def _build_trtexec_command(self, config: EngineConfig, engine_path: str) -> List[str]:
        """Build trtexec command for engine compilation"""
        cmd = [
            self.trtexec_path,
            f"--onnx={config.model_path}",
            f"--saveEngine={engine_path}",
            f"--maxBatch={config.max_batch_size}",
            f"--workspace={int(config.max_workspace_gb * 1024)}",  # Convert to MB
            f"--deviceMemoryFraction={config.device_memory_fraction}",
            f"--builderOptimizationLevel={config.optimization_level}",
            "--verbose"
        ]
        
        # Precision settings
        if config.precision == PrecisionMode.FP16:
            cmd.append("--fp16")
        elif config.precision == PrecisionMode.FP4:
            cmd.append("--fp4")  # Experimental FP4 support
        elif config.precision == PrecisionMode.INT8:
            cmd.append("--int8")
        
        # CUDA Graphs support
        if config.use_cuda_graphs:
            cmd.append("--enableCudaGraph")
        
        # Fused attention support
        if config.use_fused_attention:
            cmd.append("--use_fused_attention=true")
        
        # DLA core (for Jetson platforms)
        if config.dla_core is not None:
            cmd.extend([f"--useDLACore={config.dla_core}", "--allowGPUFallback"])
        
        # Additional optimizations for Blackwell architecture
        cmd.extend([
            "--tacticSources=+CUDNN,+CUBLAS,+EDGE_MASK_CONVOLUTIONS",
            "--precisionConstraints=obey",
            "--layerPrecisions=",
            "--layerOutputTypes=",
            "--stronglyTyped"
        ])
        
        return cmd
    
    def cleanup_cache(self, max_size_gb: Optional[float] = None, max_engines: Optional[int] = None):
        """Clean up cache based on size and usage patterns"""
        max_size = max_size_gb or self.max_cache_size_gb
        max_count = max_engines or self.max_engines
        
        logger.info(f"🧹 Cleaning up TensorRT engine cache...")
        
        # Calculate current cache size
        total_size = sum(info.file_size_mb for info in self.engines.values()) / 1024  # GB
        
        if len(self.engines) <= max_count and total_size <= max_size:
            logger.info(f"✅ Cache within limits: {len(self.engines)} engines, {total_size:.1f}GB")
            return
        
        # Sort engines by usage (least recently used first)
        sorted_engines = sorted(
            self.engines.items(),
            key=lambda x: (x[1].last_used, x[1].use_count)
        )
        
        removed_count = 0
        freed_size = 0.0
        
        for cache_key, engine_info in sorted_engines:
            if len(self.engines) <= max_count and total_size <= max_size:
                break
            
            # Remove engine file
            try:
                if os.path.exists(engine_info.engine_path):
                    os.remove(engine_info.engine_path)
                    freed_size += engine_info.file_size_mb / 1024  # GB
                
                del self.engines[cache_key]
                removed_count += 1
                total_size -= engine_info.file_size_mb / 1024
                
                logger.debug(f"🗑️ Removed cached engine: {engine_info.config.model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to remove engine {engine_info.engine_path}: {str(e)}")
        
        logger.info(f"✅ Cache cleanup complete:")
        logger.info(f"  Removed: {removed_count} engines")
        logger.info(f"  Freed: {freed_size:.1f}GB")
        logger.info(f"  Remaining: {len(self.engines)} engines, {total_size:.1f}GB")
        
        # Save updated cache index
        self._save_cache_index()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size_mb = sum(info.file_size_mb for info in self.engines.values())
        total_build_time = sum(info.build_time for info in self.engines.values())
        total_uses = sum(info.use_count for info in self.engines.values())
        
        status_counts = {}
        for status in EngineStatus:
            status_counts[status.value] = sum(
                1 for info in self.engines.values() if info.status == status
            )
        
        return {
            "timestamp": time.time(),
            "total_engines": len(self.engines),
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "total_build_time_seconds": total_build_time,
            "total_uses": total_uses,
            "average_file_size_mb": total_size_mb / len(self.engines) if self.engines else 0,
            "status_distribution": status_counts,
            "cache_directory": str(self.cache_dir),
            "tensorrt_available": self.tensorrt_available,
            "trtexec_path": self.trtexec_path
        }
    
    def precompile_common_engines(self) -> Dict[str, bool]:
        """Pre-compile commonly used engine configurations"""
        logger.info("🏗️ Pre-compiling common TensorRT engines...")
        
        # Common configurations for Four-Brain system
        common_configs = [
            EngineConfig(
                model_name="qwen3_embedding",
                model_path="/workspace/models/onnx/qwen3_embedding.onnx",
                precision=PrecisionMode.FP16,
                max_batch_size=1,
                max_workspace_gb=4.0
            ),
            EngineConfig(
                model_name="qwen3_embedding",
                model_path="/workspace/models/onnx/qwen3_embedding.onnx",
                precision=PrecisionMode.FP4,
                max_batch_size=1,
                max_workspace_gb=4.0
            ),
            EngineConfig(
                model_name="qwen3_reranker",
                model_path="/workspace/models/onnx/qwen3_reranker.onnx",
                precision=PrecisionMode.FP16,
                max_batch_size=1,
                max_workspace_gb=4.0
            )
        ]
        
        results = {}
        
        for config in common_configs:
            try:
                engine_path = self.get_or_build_engine(config)
                results[f"{config.model_name}_{config.precision.value}"] = engine_path is not None
            except Exception as e:
                logger.error(f"❌ Failed to pre-compile {config.model_name}: {str(e)}")
                results[f"{config.model_name}_{config.precision.value}"] = False
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"✅ Pre-compilation complete: {successful}/{len(common_configs)} engines built")
        
        return results

# Global engine cache manager instance
_engine_cache_manager = None

def get_engine_cache_manager() -> TensorRTEngineCacheManager:
    """Get global TensorRT engine cache manager instance"""
    global _engine_cache_manager
    if _engine_cache_manager is None:
        _engine_cache_manager = TensorRTEngineCacheManager()
    return _engine_cache_manager

def get_or_build_engine(model_name: str, model_path: str, precision: str = "FP16") -> Optional[str]:
    """Convenience function to get or build TensorRT engine"""
    manager = get_engine_cache_manager()
    config = EngineConfig(
        model_name=model_name,
        model_path=model_path,
        precision=PrecisionMode(precision.lower())
    )
    return manager.get_or_build_engine(config)
