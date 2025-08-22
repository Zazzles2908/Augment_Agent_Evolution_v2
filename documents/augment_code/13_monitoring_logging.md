Title: Monitoring, Logging, and Error Handling — HRM Multi-Brain

Goals
- Observe model health, VRAM pressure, queue latency, and cache efficacy
- Handle failures gracefully with retries and degraded modes

1) Health Endpoints
- Triton: /v2/health/ready, /v2/models/{name}, /metrics
- App: /healthz (overall), /metrics (Prometheus)

2) Key Metrics
- VRAM used/free; evictions count; model load/unload latency
- Triton queue time; dynamic batching effectiveness
- Supabase latency/error rate; Redis cache hit rate
- HRM loop iterations per task; acceptance ratio

3) Logging
- Structured logs (JSON) with fields: task_id, models_used, vram, cache_hits, durations
- Log retries and reasons for evictions/unloads

4) Alerts
- Triton unhealthy; load failure spikes; OOM warnings
- Supabase down; Redis down; cache miss rate spike

5) Error Handling Patterns
- Retry with exponential backoff for model load/unload and inference RPC
- Graceful degradation:
  - Skip reranker if low memory
  - Switch to L-Module-only path on severe pressure
  - Serve from cache-only if DB down (reads)
- Circuit breaker around repeated failing components

6) Verification Checks
- Startup: assert hrm_h_trt is READY; others UNAVAILABLE but loadable
- Scheduled smoke: load/unload each on-demand model and run a 1-sample inference

See also: 08_triton_config_multi_models.md and 09_resource_manager_design.md.

