#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault('LOG_LEVEL', 'INFO')

# Configure environment for free-first
os.environ['FREE_TIER_PREFERENCE_ENABLED'] = 'true'
# Include both canonical and alias names to maximize matches
os.environ['FREE_MODEL_LIST'] = os.getenv('FREE_MODEL_LIST', 'glm-4.5-flash,glm-4.5-air,GLM-4-Flash-250414,GLM-4-Air-250414')

# Providers
# Require GLM key to be set in environment; KIMI optional

from providers.registry import ModelProviderRegistry
from providers.base import ProviderType

# Ensure providers initialized similar to server
from server import configure_providers
configure_providers()

# Import ToolModelCategory lazily to avoid circulars
from tools.models import ToolModelCategory

preferred = ModelProviderRegistry.get_preferred_fallback_model(ToolModelCategory.FAST_RESPONSE)
print('FREE_FIRST_CHECK_START')
print({'preferred_fast_response': preferred})
print('FREE_FIRST_CHECK_END')

