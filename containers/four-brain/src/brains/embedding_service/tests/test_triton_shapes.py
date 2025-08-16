#!/usr/bin/env python3
import os
import numpy as np
import pytest

# Minimal integration test for Triton client shapes

def test_triton_batch_shapes(monkeypatch):
    try:
        from brains.embedding_service.modules.triton_client import TritonEmbeddingClient
    except Exception as e:
        pytest.skip(f"tritonclient not available: {e}")

    url = os.getenv("TRITON_URL", "localhost:8000")
    client = TritonEmbeddingClient(url=url, model_name="qwen3_embedding_trt", timeout_s=10)

    # Skip if model not ready (CI tolerance)
    if not client.is_ready():
        pytest.skip("Triton model not ready")

    # Create a small batch [N,S]
    input_ids = np.array([[1,2,3,4,5],[6,7,8,0,0]], dtype=np.int64)
    attention = np.array([[1,1,1,1,1],[1,1,1,0,0]], dtype=np.int64)

    out = client.infer_batch(input_ids, attention)
    assert out is not None, "No output from Triton"
    assert out.ndim == 2, f"Expected 2D embedding batch, got {out.shape}"
    assert out.shape[0] == input_ids.shape[0], "Batch size mismatch"
    # Embedding dim is implementation-defined; just require > 0
    assert out.shape[1] > 0, "Embedding dimension must be > 0"

