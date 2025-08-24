import os
from pathlib import Path

import pytest

from auggie.templates import render_template
from auggie.config import load_auggie_config


def test_template_basic(tmp_path, monkeypatch):
    d = tmp_path / "templates/auggie"
    d.mkdir(parents=True)
    (d / "greet.txt").write_text("Hello {name}! {#if extra}Extra: {extra}{/if}", encoding="utf-8")

    cfg = {"auggie": {"templates": {"directory": str(d)}}}
    p = tmp_path / "auggie-config.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    load_auggie_config(p)

    out = render_template("greet", {"name": "Auggie", "extra": "Y"})
    assert "Hello Auggie!" in out and "Extra: Y" in out

    out2 = render_template("greet", {"name": "Auggie"})
    assert "Hello Auggie!" in out2 and "Extra:" not in out2

