#!/usr/bin/env python3
"""
End-to-end smoke test:
- Embed query via Triton (qwen3_4b_embedding)
- Query Supabase/pgvector using augment_agent.match_documents
- Rerank via Triton (qwen3_0_6b_reranking) on top-K
- Generate via Triton (glm45_air) [placeholder request/shape]

Requires env:
  TRITON_URL=http://localhost:8000
  SUPABASE_URL=...
  SUPABASE_ANON_KEY=...

Note: This is a smoke test; adjust tokenization and shapes per your deployed plans.
"""
from __future__ import annotations
import os, sys, json, time
import argparse
import numpy as np

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
except Exception:
    httpclient = None

import requests

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000").replace("http://","")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3_4b_embedding")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3_0_6b_reranking")
GEN_MODEL = os.getenv("GEN_MODEL", "glm45_air")
TOPK = int(os.getenv("TOPK", "5"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def triton_infer(model: str, inputs: dict[str, np.ndarray], outputs: list[str]) -> dict[str, np.ndarray] | None:
    if httpclient is None:
        print("tritonclient not available; skipping")
        return None
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    infer_inputs = []
    for name, arr in inputs.items():
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.dtype == np.int32:
            arr = arr.astype(np.int64)
        ii = httpclient.InferInput(name, arr.shape, np_to_triton_dtype(arr.dtype))
        ii.set_data_from_numpy(arr)
        infer_inputs.append(ii)
    req_outputs = [httpclient.InferRequestedOutput(o, binary_data=True) for o in outputs]
    result = client.infer(model_name=model, inputs=infer_inputs, outputs=req_outputs, timeout=30)
    out = {o: result.as_numpy(o) for o in outputs}
    return out


def embed_query(text: str) -> np.ndarray:
    # Placeholder tokenizer: map chars->ids for smoke only
    ids = np.array([min(ord(c), 255) for c in text], dtype=np.int64)
    mask = np.ones_like(ids)
    out = triton_infer(EMBED_MODEL, {"input_ids": ids, "attention_mask": mask}, ["embedding"]) or {}
    emb = out.get("embedding")
    if emb is None:
        raise RuntimeError("Embedding failed")
    if emb.ndim == 1:
        emb = emb[None, :]
    return emb.astype(np.float32)


def supabase_match(embedding: np.ndarray, topk: int = 5):
    # Uses PostgREST RPC
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query_embedding": embedding[0].tolist(),
        "match_count": topk,
        "similarity_threshold": 0.0,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()


def rerank(query: str, docs: list[dict]) -> list[tuple[float, dict]]:
    if not docs:
        return []
    # Simple char-tokenizer again
    q_ids = np.array([min(ord(c), 255) for c in query], dtype=np.int64)[None, :]
    q_mask = np.ones_like(q_ids)
    scores = []
    for d in docs:
        text = d.get("text_excerpt") or d.get("content") or ""
        x_ids = np.array([min(ord(c), 255) for c in text], dtype=np.int64)[None, :]
        x_mask = np.ones_like(x_ids)
        out = triton_infer(RERANK_MODEL,
                           {"query_ids": q_ids, "query_mask": q_mask, "doc_ids": x_ids, "doc_mask": x_mask},
                           ["score"]) or {}
        sc = out.get("score")
        val = float(sc.squeeze()) if sc is not None else -1.0
        scores.append((val, d))
    scores.sort(key=lambda t: t[0], reverse=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description="E2E smoke test: embed -> match -> rerank -> (generate placeholder)")
    parser.add_argument("query", nargs="?", default="What is the system architecture?", help="Query text")
    parser.add_argument("--dry-run", action="store_true", help="Do not call external services; print planned steps only")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Would embed query with model:", EMBED_MODEL)
        print("[DRY-RUN] Would call Supabase RPC match_documents (if configured)")
        print("[DRY-RUN] Would rerank with model:", RERANK_MODEL)
        print("[DRY-RUN] Would generate with model:", GEN_MODEL)
        print("[DRY-RUN] Done")
        return

    query = args.query

    print("[1/4] Embedding query...")
    emb = embed_query(query)

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("SUPABASE_URL/ANON_KEY not set; skipping DB step")
        docs = []
    else:
        print("[2/4] Matching documents via Supabase...")
        docs = supabase_match(emb, TOPK)

    print("[3/4] Reranking...")
    ranked = rerank(query, docs)[:TOPK]

    print("[4/4] Generate (placeholder)")
    # Generation placeholder; depends on your GLM plan I/O
    print(json.dumps({
        "query": query,
        "matches": [
            {"score": s, "doc": d} for s, d in ranked
        ]
    }, indent=2))


if __name__ == "__main__":
    main()

