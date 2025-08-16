#!/usr/bin/env python3
"""
Test Brain1 imports after standardization.
"""

try:
    from brains.embedding_service.embedding_manager import Brain1Manager
    print("✅ Brain1Manager import (modular) successful")
except Exception as e:
    print(f"❌ Brain1Manager import failed: {e}")

try:
    from brains.embedding_service.config.settings import brain1_settings
    print("✅ Brain1Settings import successful")
except Exception as e:
    print(f"❌ Brain1Settings import failed: {e}")

try:
    from brains.embedding_service.api.models import EmbeddingRequest
    print("✅ API models import successful")
except Exception as e:
    print(f"❌ API models import failed: {e}")

print("Brain1 standardization test complete")
