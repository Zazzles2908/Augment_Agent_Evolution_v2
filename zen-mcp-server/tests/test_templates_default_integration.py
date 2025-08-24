import json
from pathlib import Path

import pytest

from auggie.wrappers import aug_chat
from auggie.config import load_auggie_config
from providers.base import ModelProvider, ProviderType
from providers.registry import ModelProviderRegistry as R
import types


class Prov(ModelProvider):
    SUPPORTED_MODELS = {"x": None}
    def __init__(self, api_key: str = "", base_url: str | None = None): pass
    def get_provider_type(self): return ProviderType.CUSTOM
    def validate_model_name(self, name): return name in self.SUPPORTED_MODELS
    def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
    def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
    def get_preferred_model(self, category, allowed_models): return None
    def generate_content(self, prompt, model_name, **kwargs):
        # Echo back prompt to observe template usage
        return types.SimpleNamespace(content=prompt[:10], usage={})
    def count_tokens(self, t, m): return len(t)


@pytest.mark.asyncio
async def test_default_template_applies(tmp_path):
    R.reset_for_testing(); R.register_provider(ProviderType.CUSTOM, Prov)
    cfg = {"auggie": {"templates": {"directory": str(tmp_path / "templates/auggie"), "auto_use": True}}}
    (tmp_path / "templates/auggie").mkdir(parents=True)
    (tmp_path / "templates/auggie/chat_basic.md").write_text("User:{prompt}", encoding="utf-8")
    p = tmp_path / "auggie-config.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
    load_auggie_config(p)
    out = await aug_chat({"prompt": "hello", "model": "x"})
    assert "User:" in out

