#!/usr/bin/env python3
"""
Simple Brain1 import test.
"""

try:
    print("Testing Brain1Manager import (modular)...")
    from brains.embedding_service.embedding_manager import Brain1Manager
    print("✅ Brain1Manager import successful")

    print("Testing Brain1Manager instantiation (no-start)...")
    mgr = Brain1Manager()
    print("✅ Brain1Manager instantiation successful")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete")
