# GLM-4.5 Implementation Recommendations – 07_memory_and_performance.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. VRAM/RAM Budgets (Per Model + Services)
- GPU 16GB target with dynamic buffer 15–20%
- Approx VRAM: Qwen3-4B embed ~8GB; Qwen3-0.6B rerank ~2GB; GLM-4.5 Air ~5GB; overhead ~0.5GB
- System RAM: Redis 4–8GB; Supabase/Postgres 8GB; Docling 2–4GB; Triton 1–2GB; Monitoring 2–4GB

## 2. Execution Plan (Load/Unload Sequence)
- Sequential phases:
  1) Load embed → run batch → unload
  2) Load rerank → run batch → unload
  3) Load generate → respond
- Pre-warm next model during CPU-side preprocessing (tokenisation) to hide latency
- LRU-based unload if co-existence required; cooldown between load/unload to avoid thrashing

## 3. Dynamic Batching & Concurrency
- Adaptive batch sizing:
  - Inputs: queue depth, VRAM headroom, recent p95 latency
  - Start small (e.g., embed 8, rerank 8, generate 1) then ramp
- Priority-aware: small batches for urgent queries; larger for background jobs
- Enable Triton dynamic batching with `preferred_batch_size` and `max_queue_delay_ms`

## 4. Perf Dashboards & Alerts
- Metrics: GPU VRAM utilisation, alloc failures, queue vs compute time, per-model latency (p50/p90/p99), throughput
- Alerts: VRAM > 80/90/95% (warning/major/critical), error rate > 2%, queue time > compute time sustained
- Tracing: add spans for embed/search/rerank/generate to identify hotspots

## 5. Tuning Playbook (Symptoms → Actions)
- Symptom: GPU OOM/alloc failure → Action: reduce batch size; unload idle model; lower context length
- Symptom: High queue time → Action: raise `max_queue_delay_ms` modestly; increase preferred batch; pre-warm next model
- Symptom: High generation latency → Action: reduce max tokens; tighten rerank threshold to fewer context chunks
- Symptom: Low throughput but low GPU util → Action: increase batch sizes; overlap CPU preproc; verify dynamic batching
- Symptom: Oscillation between loads/unloads → Action: add cooldown; predict workload to pre-load; cap concurrency

