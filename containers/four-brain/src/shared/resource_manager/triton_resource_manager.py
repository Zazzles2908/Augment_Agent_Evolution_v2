#!/usr/bin/env python3
"""
Triton-centric ResourceManager (general)
- Tracks VRAM budgets and model footprints
- LRU eviction for on-demand models
- Integrates with TritonRepositoryClient for load/unload

Dtype discipline (client-side):
- ONNX Runtime models typically expect INT64 inputs (e.g., qwen3_embedding)
- TensorRT explicit-batch models typically expect INT32 inputs (e.g., *_trt variants)

Policies:
- Supporting models (Qwen3 embedding/reranker, Docling, GLM generation) may be loaded on-demand
- Soft pressure at 75%, hard at 85% of total VRAM
- Dynamic pool target ~8GB on 16GB GPU
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..triton_repository_client import TritonRepositoryClient

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    footprint_gb: float
    evictable: bool = True


@dataclass
class ResourceManagerConfig:
    total_vram_gb: float = 16.0
    reserved_gb: float = 1.5
    always_loaded: Dict[str, float] = field(default_factory=lambda: {
        # Keep empty by default; adjust per deployment needs (e.g., always-on generation model)
    })
    registry: Dict[str, float] = field(default_factory=lambda: {
        # Qwen3 models (~2.0GB each) and Docling GPU (~2.0GB)
        "qwen3_embedding_trt": 2.0,
        "qwen3_reranker_trt": 2.0,
        "docling_gpu": 2.0,
        # Optionally add generation model footprint if managed here, e.g.: "glm45_air": 12.0
    })
    soft_threshold: float = 0.75
    hard_threshold: float = 0.85


class TritonResourceManager:
    def __init__(self, client: TritonRepositoryClient, cfg: Optional[ResourceManagerConfig] = None) -> None:
        self.client = client
        self.cfg = cfg or ResourceManagerConfig()
        self.loaded: Dict[str, bool] = {k: True for k in self.cfg.always_loaded.keys()}
        self.lru: List[str] = []  # most recent at end
        self.last_touch: Dict[str, float] = {}
        self._preload_always()

    # ---------- VRAM accounting ----------
    def available_gb(self) -> float:
        used_dynamic = sum(self.cfg.registry.get(m, 0.0) for m in self.loaded.keys() if m in self.cfg.registry)
        always = sum(self.cfg.always_loaded.values())
        return self.cfg.total_vram_gb - self.cfg.reserved_gb - always - used_dynamic

    # ---------- LRU helpers ----------
    def touch(self, model: str) -> None:
        if model in self.lru:
            self.lru.remove(model)
        if model in self.cfg.registry:
            self.lru.append(model)
        self.last_touch[model] = time.time()

    def pick_victim(self, protected: List[str]) -> Optional[str]:
        for m in list(self.lru):
            if m not in protected and m not in self.cfg.always_loaded:
                self.lru.remove(m)
                return m
        return None

    # ---------- Load/unload ----------
    def _preload_always(self) -> None:
        for m in self.cfg.always_loaded.keys():
            ok = self.client.ensure_loaded(m)
            if ok:
                self.loaded[m] = True
                logger.info(f"Preloaded always-on model: {m}")
            else:
                logger.warning(f"Failed to preload always-on model: {m}")

    def ensure_loaded(self, needed: List[str]) -> None:
        needed = [m for m in needed if m in self.cfg.registry or m in self.cfg.always_loaded]
        protected = list(self.cfg.always_loaded.keys()) + needed
        need_gb = sum(self.cfg.registry.get(m, 0.0) for m in needed if m not in self.loaded)
        logger.info(f"Ensure models loaded: {needed} (need_gb={need_gb:.2f}, avail_gb={self.available_gb():.2f})")

        while self.available_gb() < need_gb:
            victim = self.pick_victim(protected)
            if victim is None:
                raise MemoryError("Cannot satisfy model loads; no evictable models left")
            if self.loaded.get(victim):
                if self.client.unload_model(victim):
                    self.loaded.pop(victim, None)
                    logger.info(f"Evicted model: {victim}")
                else:
                    logger.warning(f"Failed to unload victim: {victim}")
                    break

        for m in needed:
            if not self.loaded.get(m):
                if self.client.load_model(m):
                    self.loaded[m] = True
                    self.touch(m)
                    logger.info(f"Loaded model: {m}")
                else:
                    logger.error(f"Failed to load model: {m}")

    def unload(self, model: str) -> bool:
        if model in self.cfg.always_loaded:
            logger.info(f"Skip unload for always-on model: {model}")
            return False
        if self.loaded.get(model):
            ok = self.client.unload_model(model)
            if ok:
                self.loaded.pop(model, None)
                logger.info(f"Unloaded model: {model}")
            return ok
        return True

    # ---------- Telemetry ----------
    def status(self) -> Dict[str, any]:
        return {
            "available_gb": round(self.available_gb(), 2),
            "loaded": sorted(list(self.loaded.keys())),
            "lru": list(self.lru),
            "soft_threshold": self.cfg.soft_threshold,
            "hard_threshold": self.cfg.hard_threshold,
        }

