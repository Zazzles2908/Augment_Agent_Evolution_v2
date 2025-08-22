import os
from pathlib import Path

def test_config_files_exist():
    base = Path('scripts/tensorrt/config')
    for name in ['qwen3_4b_embedding.yaml','qwen3_0_6b_reranking.yaml','glm45_air.yaml']:
        assert (base / name).exists()

