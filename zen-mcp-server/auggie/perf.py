"""
Performance tracking based on registry telemetry.
- Aggregates success rate, avg latency
- Basic recommendations utility
- Optional history persistence via session store (future)
"""
from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from providers.registry import ModelProviderRegistry as Registry


def aggregate_telemetry() -> Dict[str, Dict[str, Any]]:
    tel = Registry.get_telemetry() or {}
    out: Dict[str, Dict[str, Any]] = {}
    for m, t in tel.items():
        succ = int(t.get("success", 0)); fail = int(t.get("failure", 0))
        lat = [int(x) for x in (t.get("latency_ms") or [])]
        out[m] = {
            "calls": succ + fail,
            "success_rate": (succ / max(1, succ + fail)),
            "avg_latency_ms": int(mean(lat)) if lat else None,
            "input_tokens": int(t.get("input_tokens", 0)),
            "output_tokens": int(t.get("output_tokens", 0)),
        }
    return out


def recommend_for_category(category: str) -> List[str]:
    tel = aggregate_telemetry()
    # Prefer higher success rate, then lower latency
    ranked = sorted(tel.items(), key=lambda kv: (kv[1]["success_rate"], -1 * (kv[1]["avg_latency_ms"] or 10**9)), reverse=True)
    return [m for m, _ in ranked]

