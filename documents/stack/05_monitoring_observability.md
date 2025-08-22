# Monitoring & Observability

Prometheus
- Scrape Triton at http://localhost:8002/metrics (per NVIDIA docs)
- Collect Redis and DB metrics as available

Grafana
- Dashboards for: request rate, latency, GPU memory, Redis memory, DB query time

Loki + Alloy
- Centralize logs and optionally export Prometheus metrics via Alloy


Prometheus scrape config (example)
```yaml
scrape_configs:
  - job_name: triton
    static_configs:
      - targets: ["localhost:8002"]
```

Grafana dashboard checklist
- GPU VRAM usage, GPU utilization, GPU memory alloc failures
- p50/p95/p99 latency per model; throughput (RPS)
- Triton queue time, compute time; request/response sizes
- Redis memory/evictions; Postgres query time

Alert examples
- High GPU memory (>90% for 5m)
- Model load failure/retry loop
- Redis memory fragmentation > threshold
- Error rate > 2% for 10m

