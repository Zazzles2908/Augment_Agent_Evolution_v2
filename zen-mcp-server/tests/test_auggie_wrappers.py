import asyncio
import json
import types

import pytest

from auggie.wrappers import aug_chat, aug_thinkdeep, aug_consensus
from providers.registry import ModelProviderRegistry as R
from providers.base import ModelProvider, ProviderType


class Prov(ModelProvider):
    SUPPORTED_MODELS = {"fast": None, "deep": None}
    def __init__(self, api_key: str = "", base_url: str | None = None): pass
    def get_provider_type(self): return ProviderType.CUSTOM
    def validate_model_name(self, name): return name in self.SUPPORTED_MODELS
    def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
    def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
    def get_preferred_model(self, category, allowed_models): return None
    def generate_content(self, prompt, model_name, **kwargs):
        return types.SimpleNamespace(content=f"ok:{model_name}", usage={"input_tokens": 1, "output_tokens": 2}, model_name=model_name)
    def count_tokens(self, text, model_name): return len(text)


@pytest.mark.asyncio
async def test_aug_chat_basic(monkeypatch):
    R.reset_for_testing()
    R.register_provider(ProviderType.CUSTOM, Prov)
    # Force chain
    monkeypatch.setattr(R, "_auggie_fallback_chain", lambda c: ["fast"])
    out = await aug_chat({"prompt": "hi", "model": "auto"})
    data = json.loads(out)
    assert data["tool"] == "aug_chat"
    assert data["model"] == "fast"
    assert data["content"].startswith("ok:")


@pytest.mark.asyncio
async def test_aug_thinkdeep_with_mode(monkeypatch):
    R.reset_for_testing()
    R.register_provider(ProviderType.CUSTOM, Prov)
    monkeypatch.setattr(R, "_auggie_fallback_chain", lambda c: ["deep"])
    out = await aug_thinkdeep({"step": "think", "thinking_mode": "medium"})
    data = json.loads(out)
    assert data["tool"] == "aug_thinkdeep"
    assert data["model"] == "deep"
    assert "progress" in data


@pytest.mark.asyncio
async def test_aug_consensus_two_models(monkeypatch):
    R.reset_for_testing()
    R.register_provider(ProviderType.CUSTOM, Prov)
    monkeypatch.setattr(R, "_auggie_fallback_chain", lambda c: ["deep", "fast"]) 
    out = await aug_consensus({"prompt": "compare"})
    data = json.loads(out)
    assert data["tool"] == "aug_consensus"
    assert len(data["results"]) >= 1

