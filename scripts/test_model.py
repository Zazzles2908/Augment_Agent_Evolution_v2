#!/usr/bin/env python3
import torch
import sys

if len(sys.argv) != 2:
    print("Usage: python3 test_model.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]
try:
    # Try with weights_only=False for older PyTorch models
    m = torch.load(model_path, weights_only=False)
    print(f"Model loaded successfully: {model_path}")
    print(f"Precision: {m.get('precision', 'unknown')}")
    print(f"Quantized: {m.get('quantized', False)}")
    print(f"Batch size: {m.get('batch_size', 'unknown')}")
    print(f"Seq length: {m.get('seq_len', 'unknown')}")
    print(f"Model data size: {len(str(m))} characters")

    # Try to access the actual model if it exists
    if 'model' in m:
        print(f"Model object type: {type(m['model'])}")
        print(f"Model has parameters: {hasattr(m['model'], 'parameters')}")
    elif 'model_state_dict' in m:
        print(f"Model state dict keys: {len(m['model_state_dict'])}")

except Exception as e:
    print(f"Error loading model: {e}")
    # Try with weights_only=True as fallback
    try:
        m = torch.load(model_path, weights_only=True)
        print(f"Model loaded with weights_only=True: {model_path}")
        print(f"Keys: {list(m.keys()) if isinstance(m, dict) else 'Not a dict'}")
    except Exception as e2:
        print(f"Also failed with weights_only=True: {e2}")
