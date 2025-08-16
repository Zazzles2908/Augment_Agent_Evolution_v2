#!/usr/bin/env python3
import os
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def run_ref(model_id: str, text: str) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()
    t = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**t)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output
    else:
        emb = (out.last_hidden_state * t["attention_mask"].unsqueeze(-1)).sum(1) / t["attention_mask"].sum(1, keepdim=True)
    return emb.squeeze(0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--text", default="hello world")
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()

    ref = run_ref(args.model_id, args.text)
    print("ref_dim=", ref.shape)

    try:
        import onnxruntime as ort
    except Exception as e:
        print("onnxruntime not installed:", e)
        return 1

    sess = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) if os.getenv("ORT_CUDA", "1") == "1" else ort.InferenceSession(args.onnx)
    # crude tokenization again
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    t = tok(args.text, return_tensors="np")
    out = sess.run(["embedding"], {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"]})[0]
    onnx_vec = out.squeeze(0).astype(np.float32)

    print("onnx_dim=", onnx_vec.shape)
    print("cosine(ref, onnx)=", cosine(ref, onnx_vec))


if __name__ == "__main__":
    main()

