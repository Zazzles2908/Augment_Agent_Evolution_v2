"""
Explainable model selector for Auggie.

Returns (model, reason) based on:
- tool category
- content hints
- performance telemetry
- user preferences/capabilities from auggie-config
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from providers.registry import ModelProviderRegistry as Registry
from auggie.config import get_auggie_settings


def _capability_score(name: str, caps: dict[str, Any], category: Optional[str], hints: dict[str, Any]) -> float:
    # Simple heuristic: category weight + hints
    speed = {"fast": 1.0, "medium": 0.6, "slow": 0.2}.get(str(caps.get("speed", "medium")).lower(), 0.6)
    reasoning = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(str(caps.get("reasoning", "medium")).lower(), 0.6)
    cost = {"low": 1.0, "medium": 0.7, "high": 0.3}.get(str(caps.get("cost", "medium")).lower(), 0.7)
    score = 0.0
    if category == "FAST_RESPONSE":
        score += 1.2 * speed + 0.5 * cost
    elif category == "EXTENDED_REASONING":
        score += 1.2 * reasoning + 0.4 * cost
    else:
        score += 0.9 * reasoning + 0.9 * speed
    if hints.get("code"):
        score += 0.1  # tiny boost for balanced models
    return score


def select_model(category: Optional[str], hints: Optional[dict[str, Any]] = None) -> Tuple[Optional[str], str]:
    s = get_auggie_settings() or {}
    models = (s.get("models") or {})
    available_cfg = models.get("available") or []
    caps_cfg = models.get("capabilities") or {}

    # Discover from registry
    chain = Registry._auggie_fallback_chain(None)

    # If config lists available models, filter chain to those
    if available_cfg:
        chain = [m for m in chain if m in available_cfg]

    # Build scores
    explanations = []
    scores = []
    for m in chain:
        caps = caps_cfg.get(m, {})
        sc = _capability_score(m, caps, category, hints or {})
        scores.append((sc, m, caps))
    if not scores:
        # fallback to preferred
        seed = Registry.get_preferred_fallback_model(None)
        return seed, f"Defaulted to registry preferred model: {seed}"
    scores.sort(reverse=True)
    best = scores[0]
    reason = f"Selected {best[1]} (score={best[0]:.2f}) based on caps={best[2]} and category={category}"

    # Incorporate simple telemetry nudge: prefer model with higher success rate if close
    tel = Registry.get_telemetry() or {}
    if len(scores) > 1 and tel:
        s0, m0, _ = scores[0]
        s1, m1, _ = scores[1]
        if abs(s0 - s1) < 0.15:
            def sr(m):
                t = tel.get(m, {})
                succ = t.get("success", 0); fail = t.get("failure", 0)
                return (succ / max(1, succ + fail))
            if sr(m1) > sr(m0) + 0.1:
                reason += f"; nudged toward {m1} by success-rate"
                return m1, reason
    return best[1], reason

