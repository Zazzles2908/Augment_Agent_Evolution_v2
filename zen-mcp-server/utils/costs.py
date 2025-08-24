"""
Cost estimation utilities for Zen MCP.

- Reads per-model prices from environment JSON maps
- Computes estimated cost from usage tokens
- Safe-by-default: if pricing not configured, returns 0.0

ENV
- MODEL_INPUT_PRICE_JSON: {"model": dollars_per_MTok_input, ...}
- MODEL_OUTPUT_PRICE_JSON: {"model": dollars_per_MTok_output, ...}
- MODEL_COSTS_JSON: legacy single-number proxy (treated as output $/MTok)
"""
from __future__ import annotations
import json
import os
from typing import Optional, Tuple


def _load_json_map(env_key: str) -> dict[str, float]:
    raw = os.getenv(env_key, "{}")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def get_price_maps() -> Tuple[dict[str, float], dict[str, float]]:
    """Return (input_prices, output_prices) per model from env.
    Falls back to MODEL_COSTS_JSON as output prices when MODEL_OUTPUT_PRICE_JSON is empty.
    """
    inp = _load_json_map("MODEL_INPUT_PRICE_JSON")
    outp = _load_json_map("MODEL_OUTPUT_PRICE_JSON")
    if not outp:
        outp = _load_json_map("MODEL_COSTS_JSON")
    return inp, outp


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Estimate dollar cost for a call given token usage and env pricing.
    Returns 0.0 when prices unknown.
    """
    inp, outp = get_price_maps()
    in_rate = inp.get(model)
    out_rate = outp.get(model)
    cost = 0.0
    if in_rate is not None and input_tokens:
        cost += (input_tokens / 1_000_000.0) * in_rate
    if out_rate is not None and output_tokens:
        cost += (output_tokens / 1_000_000.0) * out_rate
    return round(cost, 6)

