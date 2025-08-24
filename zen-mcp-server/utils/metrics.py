"""
Optional Prometheus metrics exposure for zen MCP.
Safe-by-default: no-ops unless PROMETHEUS_ENABLED=true and prometheus_client is installed.
"""
from __future__ import annotations
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROM_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
_METRICS_PORT = int(os.getenv("METRICS_PORT", "9108"))

try:
    if _PROM_ENABLED:
        from prometheus_client import start_http_server, Counter, Histogram, Gauge
    else:
        # Type stubs for editors
        Counter = Histogram = Gauge = object  # type: ignore
        start_http_server = None  # type: ignore
except Exception as e:
    logger.warning("Prometheus not available: %s", e)
    _PROM_ENABLED = False
    Counter = Histogram = Gauge = object  # type: ignore
    start_http_server = None  # type: ignore

# Metric handles (may be dummy objects)
_PROVIDER_REQ = None
_PROVIDER_LAT = None
_CIRCUIT_STATE = None


def init_metrics_server_if_enabled() -> bool:
    global _PROVIDER_REQ, _PROVIDER_LAT, _CIRCUIT_STATE
    if not _PROM_ENABLED or start_http_server is None:
        return False
    try:
        # Idempotent init
        if _PROVIDER_REQ is None:
            _PROVIDER_REQ = Counter(
                "provider_request_total",
                "Total provider requests by status",
                labelnames=["provider", "model", "status"],
            )
            _PROVIDER_LAT = Histogram(
                "provider_latency_ms",
                "Provider call latency (ms)",
                labelnames=["provider", "model"],
                buckets=(50, 100, 200, 400, 800, 1600, 3200, 6400, 12800),
            )
            _CIRCUIT_STATE = Gauge(
                "circuit_state",
                "Circuit state: 0=CLOSED, 1=HALF_OPEN, 2=OPEN",
                labelnames=["provider"],
            )
        start_http_server(_METRICS_PORT)
        logger.info("Prometheus metrics endpoint started on port %d", _METRICS_PORT)
        return True
    except Exception as e:
        logger.warning("Failed to start metrics endpoint: %s", e)
        return False


def record_provider_call(provider: str, model: str, ok: bool, latency_ms: Optional[float] = None) -> None:
    try:
        if _PROVIDER_REQ is not None:
            _PROVIDER_REQ.labels(provider=provider, model=model, status=("ok" if ok else "error")).inc()
        if _PROVIDER_LAT is not None and latency_ms is not None:
            _PROVIDER_LAT.labels(provider=provider, model=model).observe(float(latency_ms))
    except Exception:
        pass


def set_circuit_state(provider: str, state: str) -> None:
    try:
        if _CIRCUIT_STATE is None:
            return
        val = 0
        if state == "half_open":
            val = 1
        elif state == "open":
            val = 2
        _CIRCUIT_STATE.labels(provider=provider).set(val)
    except Exception:
        pass

