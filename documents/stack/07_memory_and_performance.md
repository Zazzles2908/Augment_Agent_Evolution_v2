# Memory and Performance

GPU (16GB)
- GLM-4.5 Air (NVFP4): ~12GB
- Qwen3-4B Embedding (FP8): ~2–4GB (varies by kernel/sequence length)
- Qwen3-0.6B Reranker (NVFP4): ~0.15GB
- Triton overhead: ~0.5GB
- Buffer: ~2.35GB

System RAM (64GB)
- Redis: ~8GB
- Supabase/Postgres: ~8GB
- Docling: ~4GB
- Triton: ~4GB
- Monitoring: ~4GB
- OS: ~8GB
- Free: ~28GB


VRAM Pressure Strategy
- Use explicit model control to keep only required models loaded
- Prefer sequential phases (embed → unload → rerank → unload → generate) when VRAM is constrained
- Implement LRU-based unloading if multiple models must co-exist briefly
- Monitor Triton GPU metrics; set alerts on OOM/alloc failures

Throughput Guidance
- Tune dynamic batching per model; start small and increase gradually
- Measure queue/compute time in Triton metrics; optimize batch size and concurrency

Fallbacks
- If GLM-4.5 Air does not fit under certain workloads, consider:
  - Lower batch sizes / sequential execution
  - Alternate generator or CPU/offload
  - ORT/TensorRT-LLM optimizations if available

