"""
Model discovery utility for Auggie.
- Queries providers for list_models()
- Merges with auggie.models.available if configured
"""
from __future__ import annotations

from typing import List

from providers.registry import ModelProviderRegistry as Registry
from auggie.config import get_auggie_settings


def discover_models() -> List[str]:
    chain = []
    # Merge all providers' models
    for ptype in Registry.PROVIDER_PRIORITY_ORDER:
        prov = Registry.get_provider(ptype)
        if not prov:
            continue
        try:
            names = prov.list_models(respect_restrictions=True)
            for n in names:
                if n not in chain:
                    chain.append(n)
        except Exception:
            continue
    # Merge config.available
    s = get_auggie_settings() or {}
    av = (s.get("models") or {}).get("available") or []
    for n in av:
        if n not in chain:
            chain.append(n)
    return chain

