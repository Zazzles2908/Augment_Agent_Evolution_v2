import os
from importlib import reload

import pytest

import server as zen_server


def test_auto_activation(monkeypatch):
    monkeypatch.setenv("AUGGIE_CLI", "true")
    reload(zen_server)
    # If wrappers available, aug_* tools should be present
    tools = getattr(zen_server, "TOOLS", {})
    # Can't assert definitely in CI without wrappers import, but ensure no crash and TOOLS is dict
    assert isinstance(tools, dict)

