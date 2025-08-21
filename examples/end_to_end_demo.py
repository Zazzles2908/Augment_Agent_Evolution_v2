#!/usr/bin/env python3
import argparse, yaml, numpy as np
from pathlib import Path
from examples.utils.triton_client import TritonHelper
from examples.utils.supabase_client import SupabaseHelper
from examples.utils.redis_client import RedisHelper

# Uses HuggingFace tokenizers for Qwen and GLM-4.5 Air; requires `pip install transformers`

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--document", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    th = TritonHelper(cfg["triton"]["url"]) 
    sb = SupabaseHelper(cfg["supabase"]["url"], cfg["supabase"]["key"]) 
    rh = RedisHelper(cfg["redis"]["url"]) 

    # 1) Docling conversion placeholder (you can call your Docling pipeline here)
    # Secure file reading with proper validation
    doc_path = Path(args.document)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {args.document}")
    if doc_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError(f"Document too large: {doc_path.stat().st_size / 1024 / 1024:.1f}MB")
    doc_text = doc_path.read_text(encoding="utf-8")[:4000]

    # 2) Embedding (with Redis cache) using HF tokenizer
    from transformers import AutoTokenizer
    emb_tok_name = cfg["hf"]["embedding_tokenizer"]
    emb_tok = AutoTokenizer.from_pretrained(emb_tok_name)
    max_len = int(cfg["max_lengths"]["embedding"])

    cached = rh.get_embedding(doc_text)
    if cached is None:
        emb_inputs = emb_tok(doc_text, max_length=max_len, truncation=True, padding="max_length", return_tensors="np")
        input_ids = emb_inputs["input_ids"].astype(np.int64)
        attention_mask = emb_inputs["attention_mask"].astype(np.int64)
        emb = th.embed(cfg["triton"]["embedding_model"], input_ids, attention_mask)[0].tolist()
        rh.set_embedding(doc_text, emb)
    else:
        emb = cached

    # 3) Vector search via Supabase RPC
    rows = sb.match_documents(emb, match_count=5, similarity_threshold=0.3)
    print("Similar documents:", rows)

    # 4) Rerank top candidate using HF tokenizer
    rer_tok_name = cfg["hf"]["reranker_tokenizer"]
    rer_tok = AutoTokenizer.from_pretrained(rer_tok_name)
    qmax = int(cfg["max_lengths"]["reranker_query"])
    dmax = int(cfg["max_lengths"]["reranker_doc"])
    q_inputs = rer_tok(args.question, max_length=qmax, truncation=True, padding="max_length", return_tensors="np")
    # For demo, use document text; in practice, use top doc from RPC results
    d_inputs = rer_tok(doc_text, max_length=dmax, truncation=True, padding="max_length", return_tensors="np")
    q_ids = q_inputs["input_ids"].astype(np.int64)
    q_mask = q_inputs["attention_mask"].astype(np.int64)
    d_ids = d_inputs["input_ids"].astype(np.int64)
    d_mask = d_inputs["attention_mask"].astype(np.int64)
    scores = th.rerank(cfg["triton"]["reranker_model"], q_ids, q_mask, d_ids, d_mask)
    print("Reranker scores:", scores)

    # 5) Generate answer with GLM tokenizer
    gen_tok_name = cfg["hf"]["generator_tokenizer"]
    gen_tok = AutoTokenizer.from_pretrained(gen_tok_name)
    gmax = int(cfg["max_lengths"]["generator"])
    prompt = f"Question: {args.question}\nContext: {doc_text[:1500]}"
    g_inputs = gen_tok(prompt, max_length=gmax, truncation=True, padding="max_length", return_tensors="np")
    prompt_ids = g_inputs["input_ids"].astype(np.int64)
    prompt_mask = g_inputs["attention_mask"].astype(np.int64)
    tokens = th.generate(cfg["triton"]["generator_model"], prompt_ids, prompt_mask)
    print("Generated tokens:", tokens)

if __name__ == "__main__":
    main()

