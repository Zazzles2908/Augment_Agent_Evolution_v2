import json
import types

from auggie.selector import select_model
from auggie.config import load_auggie_config
from providers.base import ModelProvider, ProviderType
from providers.registry import ModelProviderRegistry as R


class Prov(ModelProvider):
    SUPPORTED_MODELS = {"glm-4.5-air": None, "glm-4.5-pro": None, "kimi-k2-0711-preview": None}
    def __init__(self, api_key: str = "", base_url: str | None = None): pass
    def get_provider_type(self): return ProviderType.CUSTOM
    def validate_model_name(self, name): return name in self.SUPPORTED_MODELS
    def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
    def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
    def get_preferred_model(self, category, allowed_models): return None
    def generate_content(self, *a, **k): raise NotImplementedError
    def count_tokens(self, t, m): return len(t)


def test_selector_reasoning_prefers_pro(tmp_path):
    R.reset_for_testing(); R.register_provider(ProviderType.CUSTOM, Prov)
    cfg = {"auggie": {"models": {"available": ["glm-4.5-air","glm-4.5-pro"],
            "capabilities": {"glm-4.5-air": {"speed": "fast","reasoning":"medium","cost":"low"},
                              "glm-4.5-pro": {"speed": "medium","reasoning":"high","cost":"high"}}}}}
    p = tmp_path / "auggie-config.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
    load_auggie_config(p)
    m, reason = select_model("EXTENDED_REASONING", {})
    assert m == "glm-4.5-pro" and "reasoning" in reason

