"""
Template system for Auggie wrappers.

- Loads templates from configured directory (default: templates/auggie/)
- Variable substitution with str.format()
- Basic conditional blocks: {#if var}...{/if} treating truthy by Python truth rules
- Provides render_template(name, variables)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from auggie.config import get_auggie_settings

_IF_OPEN = re.compile(r"\{#if\s+([a-zA-Z0-9_\.]+)\}")
_IF_CLOSE = re.compile(r"\{/if\}")


def _get_templates_dir() -> Path:
    s = get_auggie_settings() or {}
    t = (s.get("templates") or {}).get("directory") or "templates/auggie/"
    p = Path(t)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p  # relative to repo root (zen-mcp-server)
    return p


def _load_template_text(name: str) -> str:
    base = _get_templates_dir()
    # Allow explicit filenames or bare names
    candidates = []
    p = base / name
    candidates.append(p)
    if not p.suffix:
        candidates.append(base / f"{name}.md")
        candidates.append(base / f"{name}.txt")
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Template not found: {p}")


def _apply_conditionals(text: str, variables: dict[str, Any]) -> str:
    # Supports simple, non-nested {#if var}...{/if}
    out = []
    i = 0
    while i < len(text):
        m = _IF_OPEN.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        var_name = m.group(1)
        m2 = _IF_CLOSE.search(text, m.end())
        if not m2:
            # No closing, keep literal
            out.append(text[m.start():])
            break
        block = text[m.end():m2.start()]
        cond = variables
        try:
            # Support dot lookup like a.b
            for part in var_name.split("."):
                if isinstance(cond, dict):
                    cond = cond.get(part)
                else:
                    cond = getattr(cond, part, None)
        except Exception:
            cond = None
        if cond:
            out.append(block)
        i = m2.end()
    return "".join(out)


def render_template(name: str, variables: dict[str, Any]) -> str:
    text = _load_template_text(name)
    text = _apply_conditionals(text, variables)
    try:
        return text.format(**variables)
    except Exception:
        return text  # fallback to raw on formatting error

