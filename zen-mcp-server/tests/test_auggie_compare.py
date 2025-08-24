import json
import types

from auggie.compare import compare_two_models
from providers.base import ModelProvider, ProviderType
from providers.registry import ModelProviderRegistry as R


class Prov(ModelProvider):
    SUPPORTED_MODELS = {"m1": None, "m2": None}
    def __init__(self, api_key: str = "", base_url: str | None = None): pass
    def get_provider_type(self): return ProviderType.CUSTOM
    def validate_model_name(self, name): return name in self.SUPPORTED_MODELS
    def list_models(self, respect_restrictions=True): return list(self.SUPPORTED_MODELS.keys())
    def get_capabilities(self, model_name): return types.SimpleNamespace(model_name=model_name)
    def get_preferred_model(self, category, allowed_models): return None
    def generate_content(self, prompt, model_name, **kwargs):
        return types.SimpleNamespace(content=f"{model_name}:{prompt}", usage={"input_tokens":1,"output_tokens":2})
    def count_tokens(self, t, m): return len(t)


def test_compare_outputs_diff():
    R.reset_for_testing(); R.register_provider(ProviderType.CUSTOM, Prov)
    from tools.models import ToolModelCategory as Cat
    res = compare_two_models(Cat.EXTENDED_REASONING, "hello", "m1", "m2", with_diff=True)
    assert res["models"] == ["m1","m2"]
    assert isinstance(res.get("diff",""), str)
    assert len(res["results"]) == 2

