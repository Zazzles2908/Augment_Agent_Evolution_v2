# Monitoring & Observability

Prometheus
- Scrape Triton at http://localhost:8002/metrics (per NVIDIA docs)
- Collect Redis and DB metrics as available

Grafana
- Dashboards for: request rate, latency, GPU memory, Redis memory, DB query time

Loki + Alloy
- Centralize logs and optionally export Prometheus metrics via Alloy

