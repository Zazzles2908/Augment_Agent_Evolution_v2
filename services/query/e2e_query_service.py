#!/usr/bin/env python3
"""
Minimal E2E query service (Phase 1)
- Embed query via Triton (qwen3_4b_embedding)
- Retrieve via placeholder local store (JSONL/SQLite TODO) with cosine
- Rerank via Triton (qwen3_0_6b_reranking)
- Optional: compress context (TODO)
- Generate via Triton (glm45_air) [stubbed tokens output]
- Redis caching for query hash (optional)
"""
from __future__ import annotations
import os, json, math
from typing import List, Dict, Any
import numpy as np
import tritonclient.http as http
from tritonclient.utils import np_to_triton_dtype

TRITON_URL=os.getenv("TRITON_URL","http://localhost:8000").replace("http://","")
EMBED_MODEL=os.getenv("EMBED_MODEL","qwen3_4b_embedding")
RERANK_MODEL=os.getenv("RERANK_MODEL","qwen3_0_6b_reranking")
GEN_MODEL=os.getenv("GEN_MODEL","glm45_air")
REDIS_URL=os.getenv("REDIS_URL")

try:
    import redis
    R = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    R = None

client=http.InferenceServerClient(url=TRITON_URL)

def encode(texts: List[str]):
    max_len=max(len(t) for t in texts)
    ids=np.zeros((len(texts),max_len),dtype=np.int64)
    mask=np.zeros_like(ids)
    for i,t in enumerate(texts):
        arr=np.array([min(ord(c),255) for c in t],dtype=np.int64)
        ids[i,:arr.shape[0]]=arr
        mask[i,:arr.shape[0]]=1
    return ids,mask

def embed_query(q: str) -> np.ndarray:
    ids,mask=encode([q])
    i1=http.InferInput("input_ids",ids.shape,np_to_triton_dtype(ids.dtype))
    i1.set_data_from_numpy(ids)
    i2=http.InferInput("attention_mask",mask.shape,np_to_triton_dtype(mask.dtype))
    i2.set_data_from_numpy(mask)
    out=[http.InferRequestedOutput("embedding",binary_data=True)]
    res=client.infer(EMBED_MODEL,[i1,i2],outputs=out,timeout=30)
    return res.as_numpy("embedding")[0]

def cosine(a,b):
    d=(a*b).sum(); na=math.sqrt((a*a).sum()); nb=math.sqrt((b*b).sum());
    return d/(na*nb+1e-9)

# Placeholder: in-memory candidates (replace with local store)
CAND=[{"text":"Redis cache improves latency","emb":np.random.rand(2000).astype(np.float32)}]

def retrieve(q_emb: np.ndarray, k=10):
    scored=[(cosine(q_emb.astype(np.float32),c["emb"]),c) for c in CAND]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _,c in scored[:k]]

def rerank(q: str, cands: List[Dict[str,Any]]):
    # Stub scores aligned with candidates; integrate proper tensor format later
    return cands

def generate(prompt: str):
    # Stub: call GEN_MODEL later; return placeholder
    return "[GEN] " + prompt[:64]

def query(q: str) -> Dict[str,Any]:
    key=None
    if R:
        import hashlib
        key=f"rr:v1:{hashlib.sha256(q.encode()).hexdigest()}"
        v=R.get(key)
        if v:
            try:
                return json.loads(v)
            except Exception:
                pass
    q_emb=embed_query(q)
    cands=retrieve(q_emb)
    reranked=rerank(q,cands)
    ctx="\n".join([c["text"] for c in reranked[:3]])
    answer=generate(f"Question: {q}\nContext:\n{ctx}")
    resp={"answer":answer, "sources":[{"text":c["text"]} for c in reranked[:3]]}
    if R and key:
        R.setex(key, 1800, json.dumps(resp))
    return resp

if __name__=='__main__':
    import sys
    q=sys.argv[1] if len(sys.argv)>1 else "what is redis?"
    print(json.dumps(query(q), indent=2))

