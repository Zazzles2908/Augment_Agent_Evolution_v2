#!/usr/bin/env python3
import os
import pytest


def test_docling_gpu_smoke():
    # Basic smoke: ensure GPU exposure is configured when requested
    nvd = os.getenv('NVIDIA_VISIBLE_DEVICES', '')
    if not nvd or nvd == 'none':
        pytest.skip('GPU not exposed to document-processor container')
    # We cannot import torch/docling reliably in host test; rely on env-based smoke
    assert nvd in ('all','0','1','0,1')

