# ResourceManager API — Load/Unload and Telemetry

## Purpose
Central coordination for GPU residency, model load/unload, and pressure handling as per the master plan.

## Endpoints (example)
- POST /orchestrator/models/ensure_loaded
  - Body: { "models": ["qwen3_embedding_trt", "qwen3_reranker_trt"] }
  - 200: { "loaded": [...], "evicted": [...], "available_gb": 7.3 }
- POST /orchestrator/models/unload
  - Body: { "model": "qwen3_reranker_trt" }
  - 200: { "status": "unloaded" }
- GET /orchestrator/metrics
  - 200: { "vram_used_gb": 8.2, "evictions": 3, "soft_pressure": false, "hard_pressure": false }

## Contracts
- HRM H-Module is non-evictable; LRU for others
- Pressure thresholds: soft 75%/hard 85% of VRAM
- Dtype discipline enforced at client layer; ResourceManager handles residency only

## Logs and Metrics
- Structured logs (JSON): task_id, models_used, vram, evictions, durations
- Prometheus counters: model_load_total, model_unload_total, evictions_total
- Gauges: vram_used_gb, available_gb, queue_delay_ms

## Notes
- Names are generic to allow model swaps
- Evidence links required for performance claims

