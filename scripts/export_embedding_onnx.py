#!/usr/bin/env python3
"""
Reproducible ONNX export for embedding model with dynamic shapes.
Zero-fabrication: exits with clear error if model not available.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("export_embedding_onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except Exception as e:
        log.error(f"transformers/torch not available: {e}")
        sys.exit(2)

    log.info(f"Loading model {args.model} on {args.device}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    # Dummy inputs for tracing
    bs = 4
    seq = args.max_len
    input_ids = torch.randint(low=0, high=tok.vocab_size, size=(bs, seq), dtype=torch.long)
    attn = torch.ones((bs, seq), dtype=torch.long)
    if args.device == "cuda" and torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attn = attn.cuda()

    # Forward to get outputs
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            y = out.pooler_output
        else:
            y = out.last_hidden_state.mean(dim=1)

    # Build dynamic axes mapping for 2D inputs and 2D output [N,D]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "embedding": {0: "batch"}
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting ONNX to {out_path}")
    torch.onnx.export(
        model,
        (input_ids, attn),
        f=str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True
    )

    log.info("ONNX export complete")


if __name__ == "__main__":
    main()

