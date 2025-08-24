import json
import os
from pathlib import Path
import types

import pytest

from providers.registry import ModelProviderRegistry as R
from providers.base import ModelProvider, ProviderType


class DummyProvider(ModelProvider):
    SUPPORTED_MODELS = {"modelA": None, "modelB": None}

    def __init__(self, api_key: str = "", base_url: str | None = None):
        self._api_key = api_key

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    def validate_model_name(self, model_name: str) -> bool:
        return model_name in self.SUPPORTED_MODELS

    def list_models(self, respect_restrictions: bool = True) -> list[str]:
        return list(self.SUPPORTED_MODELS.keys())

    def get_capabilities(self, model_name: str):
        return types.SimpleNamespace(model_name=model_name)

    def get_preferred_model(self, category, allowed_models):
        return None

    def generate_content(self, prompt, model_name, **kwargs):
        raise NotImplementedError

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text)


def test_fallback_chain_order_and_retry(monkeypatch):
    # Reset registry
    R.reset_for_testing()

    # Register two providers with overlapping models
    R.register_provider(ProviderType.CUSTOM, DummyProvider)

    # Monkeypatch provider instance to simulate one failure then success
    prov = R.get_provider(ProviderType.CUSTOM)

    calls = {"modelA": 0, "modelB": 0}

    def call_fn(model):
        calls[model] += 1
        if model == "modelA":
            raise RuntimeError("fail A")
        return types.SimpleNamespace(content="ok", usage={"input_tokens": 10, "output_tokens": 5})

    # Force fallback order to [modelA, modelB] by patching helper
    monkeypatch.setattr(R, "_auggie_fallback_chain", lambda c: ["modelA", "modelB"]) 

    resp = R.call_with_fallback(None, call_fn)
    assert resp.content == "ok"
    assert calls["modelA"] == 1 and calls["modelB"] == 1

    tel = R.get_telemetry()
    assert tel["modelA"]["failure"] >= 1
    assert tel["modelB"]["success"] >= 1
    assert tel["modelB"]["input_tokens"] == 10
    assert tel["modelB"]["output_tokens"] == 5


def test_fallback_chain_from_config(tmp_path, monkeypatch):
    # Prepare auggie-config.json with fallback order
    cfg = {"auggie": {"fallback": {"chat": ["m1", "m2"], "reasoning": ["m3"]}}}
    p = tmp_path / "auggie-config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    # Load config
    from auggie.config import load_auggie_config

    load_auggie_config(p)

    # Patch registry to believe providers support those models
    R.reset_for_testing()

    class Prov(ModelProvider):
        SUPPORTED_MODELS = {"m1": None, "m2": None, "m3": None}
        def __init__(self, api_key: str = "", base_url: str | None = None): pass
        def get_provider_type(self): return ProviderType.CUSTOM
        def validate_model_name(self, model_name): return model_name in self.SUPPORTED_MODELS
        def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
        def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
        def get_preferred_model(self, category, allowed_models): return None
        def generate_content(self, *a, **k): raise NotImplementedError
        def count_tokens(self, text, model_name): return len(text)

    R.register_provider(ProviderType.CUSTOM, Prov)

    # Request chain for FAST_RESPONSE (maps to "chat")
    from tools.models import ToolModelCategory as Cat
    chain = R._auggie_fallback_chain(Cat.FAST_RESPONSE)
    assert chain[:2] == ["m1", "m2"]


def test_noop_when_no_config(monkeypatch):
    # No config loaded
    from auggie.config import load_auggie_config
    load_auggie_config(path=None)

    R.reset_for_testing()

    class Prov(ModelProvider):
        SUPPORTED_MODELS = {"x": None}
        def __init__(self, api_key: str = "", base_url: str | None = None): pass
        def get_provider_type(self): return ProviderType.CUSTOM
        def validate_model_name(self, model_name): return model_name in self.SUPPORTED_MODELS
        def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
        def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
        def get_preferred_model(self, category, allowed_models): return None
        def generate_content(self, *a, **k): raise NotImplementedError
        def count_tokens(self, text, model_name): return len(text)

    R.register_provider(ProviderType.CUSTOM, Prov)

    # Chain should at least include a valid model name (fallback to heuristics)
    chain = R._auggie_fallback_chain(None)
    assert isinstance(chain, list) and len(chain) >= 1

