#!/usr/bin/env python3
"""
Phase 1 smoke test for Triton-centric stack (no HRM)
- Loads/unloads qwen3_embedding_trt, qwen3_reranker_trt, docling_gpu
- Validates readiness endpoints
- Exercises ResourceManager LRU with small VRAM budget to force evictions
"""
from __future__ import annotations
import os
import json
import time
import urllib.request
import urllib.error
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir))
SHARED_PATH = os.path.join(REPO_ROOT, 'containers', 'four-brain', 'src')
if SHARED_PATH not in sys.path:
    sys.path.insert(0, SHARED_PATH)

from shared.triton_repository_client import TritonRepositoryClient
from shared.resource_manager.triton_resource_manager import TritonResourceManager, ResourceManagerConfig

TRITON_URL = os.environ.get('TRITON_URL', 'http://localhost:8000')
MODELS = ['qwen3_embedding_trt','qwen3_reranker_trt','docling_gpu','glm45_air']


def post(path: str, body: dict | None = None):
    url = f"{TRITON_URL}{path}"
    data = json.dumps(body or {}).encode('utf-8')
    req = urllib.request.Request(url=url, data=data, method='POST', headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.getcode(), resp.read().decode('utf-8')


def get(path: str):
    url = f"{TRITON_URL}{path}"
    req = urllib.request.Request(url=url, method='GET')
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.getcode(), resp.read().decode('utf-8')


def repo_index():
    code, txt = post('/v2/repository/index', {})
    try:
        arr = json.loads(txt)
    except Exception:
        arr = []
    return code, arr


def main():
    results = {"triton": {}, "rm": {}}
    # Health
    code, _ = get('/v2/health/ready')
    results['triton']['health_ready'] = code

    # Load and ready checks
    for m in MODELS:
        try:
            code, _ = post(f"/v2/repository/models/{m}/load", {})
            results['triton'][f'load_{m}'] = code
        except Exception as e:
            results['triton'][f'load_{m}'] = str(e)
        time.sleep(0.1)
        try:
            code, _ = get(f"/v2/models/{m}/ready")
            results['triton'][f'ready_{m}'] = code
        except Exception as e:
            results['triton'][f'ready_{m}'] = str(e)

    # Repository index snapshot
    code, idx = repo_index()
    results['triton']['repo_index_code'] = code
    results['triton']['repo_index'] = idx

    # ResourceManager LRU test
    client = TritonRepositoryClient(base_url=TRITON_URL)
    cfg = ResourceManagerConfig(total_vram_gb=4.0, reserved_gb=0.5, always_loaded={}, registry={
        'hrm_l_trt': 1.5,
        'qwen3_reranker_trt': 1.5,
        'docling_gpu': 1.5,
    })
    rm = TritonResourceManager(client, cfg)
    # ensure_loaded should evict as needed
    rm.ensure_loaded(['qwen3_embedding_trt', 'qwen3_reranker_trt'])
    rm.ensure_loaded(['docling_gpu'])
    results['rm']['status'] = rm.status()

    # Final unload
    for m in MODELS:
        try:
            post(f"/v2/repository/models/{m}/unload", {})
        except Exception:
            pass

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

