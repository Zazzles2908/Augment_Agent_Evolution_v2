#!/usr/bin/env python3
"""
Proof-of-Concept: Qwen3 Embeddings (2000-dim) → HRM Input → TensorRT L-Module compile probe
- Attempts to load Qwen model from HF or local path
- Generates an embedding and truncates to 2000 dims (aligns with project MRL target)
- Attempts to import HRM input processor if available
- Optionally compiles an ONNX L-Module with TensorRT (NVFP4/FP16 fallback)
- Measures VRAM before/after using pynvml or nvidia-smi

Environment variables (optional):
  QWEN_MODEL_PATH       # local path or HF repo id (e.g., Qwen/Qwen3-4B-Embedding)
  TEXT_SAMPLE           # text to embed; default: "Hello from the HRM PoC"
  HRM_INPUT_MODULE      # python import path to HRM input processor (e.g., hrm_phase0.input_adapter)
  HRM_L_ONNX            # path to L-Module ONNX for TensorRT compile probe
  TRT_PRECISION         # fp8|nvfp4|fp16 (default: nvfp4)
  CUDA_VISIBLE_DEVICES  # which GPU
"""
import os
import subprocess
import sys
from typing import Optional


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def err(msg: str):
    print(f"[ERROR] {msg}")


def get_vram_usage_mb() -> Optional[int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(mem.used / (1024 * 1024))
    except Exception:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
            return int(out.splitlines()[0])
        except Exception:
            return None


def load_qwen_and_embed(text: str):
    model_id = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen3-4B-Embedding")
    info(f"Loading embedding model: {model_id}")
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError as e:
        err("Missing transformers/torch. Install with: pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cuXXX")
        raise

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        # Use pooled or CLS representation depending on model
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vec = outputs.pooler_output.squeeze(0).cpu().numpy()
        else:
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    info(f"Native embedding dim: {vec.shape[0]}")
    # Project/Truncate to 2000 dims to align with Supabase cap and project target
    target_dim = 2000
    if vec.shape[0] >= target_dim:
        vec2000 = vec[:target_dim]
    else:
        # Zero-pad if smaller (rare)
        import numpy as np
        vec2000 = np.zeros((target_dim,), dtype=vec.dtype)
        vec2000[: vec.shape[0]] = vec
    info(f"Prepared 2000-dim embedding (len={len(vec2000)})")
    return vec2000


def hrm_input_process(vec2000):
    mod_path = os.getenv("HRM_INPUT_MODULE", "")
    if not mod_path:
        warn("HRM_INPUT_MODULE not set; skipping HRM input processing step. Set to your adapter import path.")
        return True
    info(f"Attempting HRM input processing via {mod_path}")
    try:
        mod = __import__(mod_path, fromlist=["*"])
        if hasattr(mod, "process_embedding_input"):
            out = mod.process_embedding_input(vec2000)
            info("HRM input processing completed.")
            return out is not None
        else:
            warn("Module has no 'process_embedding_input' function; skipping.")
            return True
    except Exception as e:
        err(f"HRM input processing failed: {e}")
        return False


def trt_compile_probe():
    onnx_path = os.getenv("HRM_L_ONNX", "")
    if not onnx_path:
        warn("HRM_L_ONNX not set; skipping TensorRT compile probe.")
        return True
    precision = os.getenv("TRT_PRECISION", "nvfp4").lower()
    flag = "--nvfp4" if precision == "nvfp4" else ("--fp8" if precision == "fp8" else "--fp16")
    engine_out = os.path.splitext(onnx_path)[0] + ".engine"
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        flag,
        f"--saveEngine={engine_out}",
        "--device=0",
        "--memPoolSize=workspace:4096",
        "--buildOnly",
    ]
    info("Running TensorRT compile probe: " + " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        info(f"TensorRT engine built: {engine_out}")
        return True
    except subprocess.CalledProcessError as e:
        err(f"TensorRT compile failed with exit code {e.returncode}")
        return False


def main():
    text = os.getenv("TEXT_SAMPLE", "Hello from the HRM PoC")
    vram_before = get_vram_usage_mb()
    if vram_before is not None:
        info(f"VRAM before: {vram_before} MiB")

    vec2000 = load_qwen_and_embed(text)
    hrm_ok = hrm_input_process(vec2000)

    trt_ok = trt_compile_probe()

    vram_after = get_vram_usage_mb()
    if vram_after is not None:
        info(f"VRAM after: {vram_after} MiB (Δ {vram_after - vram_before if vram_before is not None else 'n/a'} MiB)")

    if hrm_ok and trt_ok:
        info("PoC successful: External embeddings accepted and TensorRT probe passed (or skipped by env)")
        sys.exit(0)
    else:
        err("PoC incomplete: Check logs above for failures.")
        sys.exit(1)


if __name__ == "__main__":
    main()

