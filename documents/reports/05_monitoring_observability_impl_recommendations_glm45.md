# GLM-4.5 Implementation Recommendations – 05_monitoring_observability.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Prometheus Configuration (Scrape + Relabel)
```yaml
scrape_configs:
  - job_name: 'triton'
    static_configs: [{ targets: ['localhost:8002'] }]
    metrics_path: '/metrics'
    scrape_interval: 15s
  - job_name: 'redis'
    static_configs: [{ targets: ['localhost:9121'] }]
    metrics_path: '/metrics'
  - job_name: 'postgres'
    static_configs: [{ targets: ['localhost:9187'] }]
    metrics_path: '/metrics'
```

## 2. Grafana Dashboards (Panels & Queries)
- GPU: VRAM usage (`triton_server_gpu_memory_bytes`), utilisation, alloc failures
- Models: p50/p95/p99 latency (`histogram_quantile` over `request_duration_seconds_bucket`), RPS, queue time
- Infra: Redis mem (`redis_memory_used_bytes`), evictions, Postgres query time

## 3. Alerting Rules (Thresholds & Routing)
```yaml
groups:
  - name: triton_alerts
    rules:
      - alert: HighGPUUsage
        expr: triton_server_gpu_memory_bytes / triton_server_gpu_memory_total_bytes > 0.9
        for: 5m
        labels: { severity: critical }
      - alert: ModelLoadFailure
        expr: increase(triton_server_model_load_failure_total[10m]) > 0
        labels: { severity: critical }
      - alert: HighErrorRate
        expr: rate(triton_server_request_error_count[10m]) / rate(triton_server_request_inference_count[10m]) > 0.02
        for: 10m
        labels: { severity: warning }
```

## 4. Logs: Loki + Alloy Pipelines
- Alloy tail: `/var/log/triton/*.log`, `/var/log/redis/*.log`, `/var/log/postgres/*.log`
- LogQL examples: `{container="triton"} |= "ERROR"`, `{container="redis"} |= "slowlog"`, `{container="postgres"} |= "duration"`

## 5. Capacity & Retention Planning
- Prometheus: ~30GB, 15d retention; TSDB compaction 2h
- Loki: ~50GB, 7d retention; 1GB chunk cache
- Single-node: co-locate Alertmanager; moderate scrape intervals to reduce load

