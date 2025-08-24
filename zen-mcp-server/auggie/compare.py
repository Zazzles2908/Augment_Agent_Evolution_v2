"""
Model comparison utilities for Auggie.
- Side-by-side generation across two models
- Basic scoring (relevance, coherence, helpfulness) heuristic
- Optional diff highlighting
- CLI-friendly output struct
"""
from __future__ import annotations

import difflib
from typing import Any, Optional

from providers.registry import ModelProviderRegistry as Registry


def _score_response(prompt: str, text: str) -> dict[str, float]:
    # Super-simple heuristic placeholders; can be improved with LLM-based judge later
    rel = min(1.0, len(set(prompt.lower().split()) & set(text.lower().split())) / max(1, len(set(prompt.lower().split()))))
    coh = 1.0 if text and text[-1] in ".!?" else 0.7
    helpf = 0.8 if len(text) > 30 else 0.5
    return {"relevance": rel, "coherence": coh, "helpfulness": helpf, "score": (rel + coh + helpf) / 3.0}


def _diff(a: str, b: str) -> str:
    return "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))


def compare_two_models(category, prompt: str, m1: str, m2: Optional[str] = None, with_diff: bool = True) -> dict[str, Any]:
    # Helper to call model via fallback but pinned to a name (telemetry recorded)
    def run(model_name: str):
        def call_fn(selected_model: str):
            prov = Registry.get_provider_for_model(selected_model)
            if not prov:
                raise RuntimeError(f"No provider for {selected_model}")
            return prov.generate_content(prompt=prompt, model_name=selected_model)
        # Pin single
        orig = Registry._auggie_fallback_chain
        try:
            Registry._auggie_fallback_chain = classmethod(lambda cls, c: [model_name])  # type: ignore
            resp = Registry.call_with_fallback(category, call_fn)
            return resp
        finally:
            Registry._auggie_fallback_chain = orig  # type: ignore

    r1 = run(m1)
    t1 = r1.content if r1 else ""

    r2 = None
    t2 = ""
    if m2:
        r2 = run(m2)
        t2 = r2.content if r2 else ""

    s1 = _score_response(prompt, t1)
    s2 = _score_response(prompt, t2) if m2 else None

    result = {
        "prompt": prompt,
        "models": [m for m in [m1, m2] if m],
        "results": [
            {"model": m1, "content": t1, "score": s1, "usage": getattr(r1, "usage", {})},
        ] + ([{"model": m2, "content": t2, "score": s2, "usage": getattr(r2, "usage", {})}] if m2 else []),
    }
    if with_diff and m2:
        result["diff"] = _diff(t1, t2)
    return result

