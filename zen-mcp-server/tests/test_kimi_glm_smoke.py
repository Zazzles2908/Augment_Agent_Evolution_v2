import os
import pytest

from providers.registry import ModelProviderRegistry
from providers.base import ProviderType
from providers.moonshot.provider import KimiModelProvider
from providers.zhipu.provider import GLMModelProvider


def setup_module(module):
    os.environ.setdefault("KIMI_API_KEY", "test-key")
    os.environ.setdefault("GLM_API_KEY", "test-key")
    ModelProviderRegistry.register_provider(ProviderType.KIMI, KimiModelProvider)
    ModelProviderRegistry.register_provider(ProviderType.GLM, GLMModelProvider)


def test_alias_resolution_glm():
    prov = GLMModelProvider(api_key="test-key")
    caps = prov.get_capabilities("glm-4.5-x")
    assert caps.model_name in ("glm-4.5-air", "glm-4.5")


def test_kimi_tools_stream_passthrough(monkeypatch):
    prov = KimiModelProvider(api_key="test-key")

    class DummyChoices:
        def __init__(self):
            self.message = type("msg", (), {"content": "ok"})
            self.finish_reason = "stop"

    class DummyStream:
        def __iter__(self):
            # Simulate one streaming event chunk with delta content
            Delta = type("Delta", (), {"content": "ok"})
            Choice = type("Choice", (), {"delta": Delta(), "message": None})
            Event = type(
                "Event",
                (),
                {
                    "choices": [Choice()],
                    "model": "kimi-k2-0711-preview",
                    "id": "id",
                    "created": 1,
                },
            )
            yield Event()

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert "tools" in kwargs
                    assert "tool_choice" in kwargs
                    assert kwargs.get("stream") is True
                    return DummyStream()

    # Patch underlying cached client used by the @property
    monkeypatch.setattr(prov, "_client", DummyClient(), raising=False)

    resp = prov.generate_content(
        prompt="test",
        model_name="kimi-k2-0711-preview",
        tools=[{"type": "function", "function": {"name": "foo", "parameters": {}}}],
        tool_choice="auto",
        stream=True,
    )
    assert resp.content == "ok"

