#!/usr/bin/env python3
"""
Docling-based ingestion CLI
- Discover files (PDF, TXT)
- Extract + chunk via Docling (placeholder splitter; integrate real docling when available)
- Embed via Triton qwen3_4b_embedding
- Insert into Supabase (augment_agent.document_vectors)

Safety: --dry-run does not call external services. Idempotency via (doc_id, chunk_id) upsert.
"""
from __future__ import annotations
import os, sys, json, argparse, hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
except Exception:
    httpclient = None

try:
    import requests
except Exception:
    requests = None

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000").replace("http://","")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3_4b_embedding")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def discover_files(root: Path) -> List[Path]:
    exts = {".pdf", ".txt", ".md"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def naive_chunk(text: str, max_len: int = 800) -> List[str]:
    chunks = []
    cur = []
    cur_len = 0
    for token in text.split():
        cur.append(token)
        cur_len += len(token) + 1
        if cur_len >= max_len:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # Placeholder: expect docling integration
        return path.stem
    return path.read_text(encoding="utf-8", errors="ignore")


def triton_embed(batch_texts: List[str]) -> np.ndarray:
    if httpclient is None:
        raise RuntimeError("tritonclient not available")
    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Toy tokenizer: map chars to ids
    max_len = max(len(t) for t in batch_texts)
    ids = np.zeros((len(batch_texts), max_len), dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, t in enumerate(batch_texts):
        arr = np.array([min(ord(c), 255) for c in t], dtype=np.int64)
        ids[i, :arr.shape[0]] = arr
        mask[i, :arr.shape[0]] = 1

    in_ids = httpclient.InferInput("input_ids", ids.shape, np_to_triton_dtype(ids.dtype))
    in_ids.set_data_from_numpy(ids)
    in_mask = httpclient.InferInput("attention_mask", mask.shape, np_to_triton_dtype(mask.dtype))
    in_mask.set_data_from_numpy(mask)

    req_out = [httpclient.InferRequestedOutput("embedding", binary_data=True)]
    result = client.infer(model_name=EMBED_MODEL, inputs=[in_ids, in_mask], outputs=req_out, timeout=60)
    emb = result.as_numpy("embedding")
    if emb is None:
        raise RuntimeError("No embedding output")
    if emb.ndim == 1:
        emb = emb[None, :]
    return emb.astype(np.float32)


def supabase_upsert_vectors(rows: List[Dict[str, Any]]) -> None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL/ANON_KEY not configured")
    if requests is None:
        raise RuntimeError("requests not available")

    url = f"{SUPABASE_URL}/rest/v1/augment_agent.document_vectors"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    r = requests.post(url, headers=headers, data=json.dumps(rows))
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert failed: {r.status_code} {r.text}")


def run_ingest(input_dir: Path, batch_size: int = 16, dry_run: bool = False) -> None:
    files = discover_files(input_dir)
    print(f"Discovered {len(files)} files")

    for fp in files:
        text = read_text(fp)
        chunks = naive_chunk(text)
        doc_id = hashlib.sha1(str(fp).encode("utf-8")).hexdigest()
        if dry_run:
            print(f"[DRY] {fp} -> {len(chunks)} chunks")
            continue

        # Batch embed
        rows: List[Dict[str, Any]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            emb = triton_embed(batch)  # [B,2000]
            for j, ch in enumerate(batch):
                chunk_id = f"{i+j}"
                rows.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": fp.name,
                    "text_excerpt": ch[:512],
                    "embedding": emb[j].tolist(),
                    "metadata": {"path": str(fp)}
                })
        supabase_upsert_vectors(rows)
        print(f"Ingested {fp} -> {len(rows)} vectors")


def main():
    ap = argparse.ArgumentParser(description="Docling ingestion -> Triton embed -> Supabase vectors")
    ap.add_argument("input", help="Input directory of documents")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_ingest(Path(args.input), batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

