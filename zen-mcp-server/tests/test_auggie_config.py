import json
import os
from pathlib import Path

from auggie.config import (
    get_auggie_settings,
    load_auggie_config,
    start_config_watcher,
)


def test_missing_config_is_safe(tmp_path: Path):
    os.environ.pop("AUGGIE_CONFIG", None)
    load_auggie_config(path=None)
    assert get_auggie_settings() == {}


def test_load_valid_config(tmp_path: Path):
    cfg = {
        "auggie": {
            "default_model": "auto",
            "session_persistence": True,
            "cli_output_format": "structured",
            "error_handling": "auggie_friendly",
        }
    }
    p = tmp_path / "auggie-config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    load_auggie_config(p)
    s = get_auggie_settings()
    assert s["default_model"] == "auto"
    assert s["session_persistence"] is True
    assert s["cli_output_format"] == "structured"
    assert s["error_handling"] == "auggie_friendly"


def test_reload_on_change(tmp_path: Path, monkeypatch):
    cfg1 = {"auggie": {"default_model": "auto"}}
    cfg2 = {"auggie": {"default_model": "glm-4.5"}}
    p = tmp_path / "auggie-config.json"
    p.write_text(json.dumps(cfg1), encoding="utf-8")
    load_auggie_config(p)
    s1 = get_auggie_settings()
    assert s1["default_model"] == "auto"

    # Start watcher and modify file
    start_config_watcher()
    p.write_text(json.dumps(cfg2), encoding="utf-8")

    # Poll up to ~6s for reload
    import time

    ok = False
    for _ in range(12):
        time.sleep(0.5)
        if get_auggie_settings().get("default_model") == "glm-4.5":
            ok = True
            break
    assert ok, "Auggie config did not hot-reload within timeout"

