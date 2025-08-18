Title: Resource Manager — VRAM Tracking, LRU Unloading, Memory Pressure Handling

Goals
- Keep HRM H-Module always resident (~0.5GB)
- Manage an ~8GB dynamic pool for on-demand models on a 16GB VRAM GPU (RTX 5070 Ti)
- Load required models on demand; unload least-recently used when under pressure

Budget
- VRAM total: 16GB
- Reserved/system: 1.5GB
- Always-loaded: HRM H (~0.5GB)
- Dynamic pool target: ~8.0GB

Tracked Models (name → approx GB)
- hrm_l_trt → 0.3
- qwen3_embedding_trt → 2.0
- qwen3_reranker_trt → 2.0
- docling_gpu → 2.0

Design
- Central class ResourceManager with:
  - query_vram() → reads nvidia-smi or CUDA APIs for used/free
  - model_registry: memory_footprint_gb per model
  - residency: {model: loaded? non_evictable?}
  - lru_list: usage order for evictable models
  - ensure_loaded(models: list[str]) → plans evictions and calls Triton load API
  - unload(model) → calls Triton unload API
  - on_infer(model) → updates LRU and usage counters
  - pressure thresholds: soft at 75%, hard at 85% of VRAM

Pseudocode
```
class ResourceManager:
    def __init__(self, triton_client):
        self.client = triton_client
        self.total_gb = 16.0
        self.reserved_gb = 1.5
        self.always = {"hrm_h_trt": 0.5}
        self.registry = {
            "hrm_l_trt": 0.3,
            "qwen3_embedding_trt": 2.0,
            "qwen3_reranker_trt": 2.0,
            "docling_gpu": 2.0,
        }
        self.loaded = {"hrm_h_trt": True}
        self.lru = []  # most-recent at end

    def available_gb(self):
        used_dynamic = sum(self.registry[m] for m in self.loaded if m in self.registry)
        return self.total_gb - self.reserved_gb - sum(self.always.values()) - used_dynamic

    def ensure_loaded(self, needed: list[str]):
        # Filter unknowns
        needed = [m for m in needed if m in self.registry or m in self.always]
        # Load always-on first (noop if already loaded)
        if "hrm_h_trt" not in self.loaded:
            self.client.load("hrm_h_trt")
            self.loaded["hrm_h_trt"] = True
        # Plan capacity
        need_gb = sum(self.registry.get(m,0) for m in needed if m not in self.loaded)
        while self.available_gb() < need_gb:
            victim = self.pick_victim()
            if victim is None:
                raise MemoryError("Cannot satisfy model loads; no evictable models left")
            self.client.unload(victim)
            self.loaded.pop(victim, None)
        # Load missing
        for m in needed:
            if m not in self.loaded:
                self.client.load(m)
                self.loaded[m] = True
                if m in self.registry:
                    self.touch(m)

    def pick_victim(self):
        # skip always-on and currently-needed
        for m in list(self.lru):
            if m not in self.always:
                self.lru.remove(m)
                return m
        return None

    def touch(self, model):
        if model in self.lru:
            self.lru.remove(model)
        self.lru.append(model)
```

Memory Pressure Handling
- Soft threshold: reduce batch sizes (client), prefer smaller models, delay prefetch
- Hard threshold: immediately unload oldest evictable model before issuing next inference
- Expose /metrics for VRAM and evictions; log decisions with reasons

Smart Caching
- Persist embeddings to Supabase; cache hot vectors in Redis keyed by hash(text)
- Track usage to prefetch qwen3_reranker_trt when reranking spikes

Failure Modes
- Load failure → retry with backoff; if repeated, degrade by skipping optional models (e.g., reranker)
- OOM → unload, shrink batch, and reattempt; surface warning to HRM H-Module

See also: 08_triton_config_multi_models.md and 10_hrm_processing_flow.md.

