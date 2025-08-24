"""
Auggie configuration management for Zen MCP Server.

- Loads auggie-config.json (path from env AUGGIE_CONFIG or default next to server)
- Validates auggie-specific schema
- Caches settings with thread-safe access
- Supports hot-reload via mtime polling (2–5s)

Backwards-compatible: if file or section missing, returns empty/defaults.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

_AUG_CONF_LOCK = threading.RLock()
_AUG_CONF: Dict[str, Any] = {}
_AUG_CONF_PATH: Path | None = None
_AUG_CONF_MTIME: float | None = None
_POLL_INTERVAL_SEC = float(os.getenv("AUGGIE_CONFIG_POLL_INTERVAL", "3.0"))
_STOP_EVENT = threading.Event()

# Minimal JSON schema (manual) for speed and no extra deps
_ALLOWED_TOP_LEVEL_KEYS = {"mcpServers", "auggie"}
_AUGGIE_DEFAULTS = {
    "default_model": "auto",
    "session_persistence": True,
    "cli_output_format": "structured",
    "error_handling": "auggie_friendly",
    "fallback": {},  # e.g., {"chat": ["glm-4.5-air", ...], "reasoning": [ ... ], "coding": [ ... ]}
    "models": {"available": [], "capabilities": {}},
}


def _discover_config_path() -> Path | None:
    """Find auggie-config.json path.
    Priority:
      1) env AUGGIE_CONFIG
      2) zen-mcp-server/auggie-config.json (next to server.py)
    """
    env_path = os.getenv("AUGGIE_CONFIG")
    if env_path and os.path.exists(env_path):
        return Path(env_path)
    # Default next to server.py
    here = Path(__file__).resolve().parent.parent
    default_path = here / "auggie-config.json"
    return default_path if default_path.exists() else None


def _safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _validate_and_extract(raw: dict) -> dict:
    # Ensure only expected top-level keys; ignore extras to be resilient
    if not isinstance(raw, dict):
        return {}
    auggie_section = raw.get("auggie")
    if not isinstance(auggie_section, dict):
        return {}
    out = {}
    for k, v in auggie_section.items():
        if k == "default_model" and isinstance(v, str):
            out[k] = v
        elif k == "session_persistence" and isinstance(v, bool):
            out[k] = v
        elif k == "cli_output_format" and isinstance(v, str):
            out[k] = v
        elif k == "error_handling" and isinstance(v, str):
            out[k] = v
        elif k == "fallback" and isinstance(v, dict):
            # sanitize fallback chains to list[str]
            clean_fb = {}
            for cat, arr in v.items():
                if isinstance(arr, list):
                    clean_fb[cat] = [str(x) for x in arr if isinstance(x, (str, bytes))]
            out[k] = clean_fb
        elif k == "wrappers" and isinstance(v, dict):
            clean_wr = {}
            if "show_progress" in v:
                clean_wr["show_progress"] = bool(v.get("show_progress"))
            if "compact_output" in v:
                clean_wr["compact_output"] = bool(v.get("compact_output"))
            if "error_detail" in v and isinstance(v.get("error_detail"), str):
                clean_wr["error_detail"] = str(v.get("error_detail"))
            out[k] = clean_wr
        elif k == "models" and isinstance(v, dict):
            clean_models = {"available": [], "capabilities": {}}
            av = v.get("available")
            if isinstance(av, list):
                clean_models["available"] = [str(x) for x in av if isinstance(x, (str, bytes))]
            caps = v.get("capabilities")
            if isinstance(caps, dict):
                c2 = {}
                for name, mm in caps.items():
                    if isinstance(mm, dict):
                        c2[str(name)] = {k: str(mm.get(k)) for k in mm.keys()}
                clean_models["capabilities"] = c2
        elif k == "selector" and isinstance(v, dict):
            clean_sel = {}
            if "explanations" in v:
                clean_sel["explanations"] = bool(v.get("explanations"))
            if "auto_optimize" in v:
                clean_sel["auto_optimize"] = bool(v.get("auto_optimize"))
            out[k] = clean_sel
        elif k == "templates" and isinstance(v, dict):
            clean_t = {}
            if "directory" in v and isinstance(v.get("directory"), str):
                clean_t["directory"] = str(v.get("directory"))
            if "auto_use" in v:
                clean_t["auto_use"] = bool(v.get("auto_use"))
            out[k] = clean_t
        elif k == "auto_activate":
            out[k] = bool(v)

            out[k] = clean_models
        # ignore unknown keys for forward compatibility
    # Fill defaults for missing keys
    for k, dv in _AUGGIE_DEFAULTS.items():
        out.setdefault(k, dv)
    return out


def load_auggie_config(path: str | Path | None = None) -> None:
    """Load auggie config into memory (thread-safe)."""
    global _AUG_CONF_PATH, _AUG_CONF_MTIME
    cfg_path = Path(path) if path else _discover_config_path()
    if cfg_path is None:
        with _AUG_CONF_LOCK:
            _AUG_CONF.clear()
            _AUG_CONF_PATH = None
            _AUG_CONF_MTIME = None
        return
    raw = _safe_read_json(cfg_path)
    conf = _validate_and_extract(raw)
    with _AUG_CONF_LOCK:
        _AUG_CONF.clear()
        _AUG_CONF.update(conf)
        _AUG_CONF_PATH = cfg_path
        try:
            _AUG_CONF_MTIME = cfg_path.stat().st_mtime
        except Exception:
            _AUG_CONF_MTIME = None


def get_auggie_settings() -> dict:
    """Thread-safe shallow copy of current auggie settings."""
    with _AUG_CONF_LOCK:
        return dict(_AUG_CONF)


def _poller() -> None:
    global _AUG_CONF_MTIME
    while not _STOP_EVENT.is_set():
        time.sleep(_POLL_INTERVAL_SEC)
        with _AUG_CONF_LOCK:
            path = _AUG_CONF_PATH
        if not path:
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue
        if _AUG_CONF_MTIME is None or mtime <= _AUG_CONF_MTIME:
            continue
        # Reload on change
        load_auggie_config(path)


def start_config_watcher() -> None:
    """Start background watcher thread if not already running."""
    # Idempotent; safe to call more than once
    if getattr(start_config_watcher, "_started", False):
        return
    t = threading.Thread(target=_poller, name="auggie-config-watcher", daemon=True)
    t.start()
    start_config_watcher._started = True  # type: ignore[attr-defined]


def stop_config_watcher() -> None:
    _STOP_EVENT.set()

