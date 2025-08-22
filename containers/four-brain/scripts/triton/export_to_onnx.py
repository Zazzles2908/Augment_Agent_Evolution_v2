#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Try to reduce SDPA/FlashAttention paths that break tracing/ONNX export
os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT", "1")
os.environ.setdefault("PYTORCH_SDP_DISABLE_HEURISTIC", "1")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    SentenceTransformer = None
    _HAS_SBERT = False


def export(model_id: str, out_path: str, max_len: int = 8192):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Detect SentenceTransformer packaging (preferred for Qwen3 embedding local folder)
    is_sbert = _HAS_SBERT and os.path.exists(os.path.join(model_id, "config_sentence_transformers.json"))

    if is_sbert:
        sbert = SentenceTransformer(model_id, device="cpu")
        # Expect first module to be Transformer (AutoModel) and second to be Pooling
        transformer = sbert[0].auto_model
        pooling = sbert[1]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = transformer
    else:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.use_cache = False
        # Force eager attention to avoid SDPA masking/vmap path during tracing
        try:
            config.attn_implementation = "eager"
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, config=config)

    model.eval().to("cpu")

    dummy = tokenizer("hello world", return_tensors="pt")

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "embedding": {0: "batch", 1: "dim"}
    }

    class Wrapper(torch.nn.Module):
        def __init__(self, m, pooling_layer=None):
            super().__init__()
            self.m = m
            self.pooling = pooling_layer
        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if self.pooling is not None:
                features = {
                    'token_embeddings': out.last_hidden_state,
                    'attention_mask': attention_mask
                }
                emb = self.pooling(features)['sentence_embedding']
            else:
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    emb = (out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
                else:
                    raise RuntimeError("Model outputs do not provide a known embedding; adjust wrapper.")
            return emb

    wrapper = Wrapper(model, pooling_layer=pooling if is_sbert else None)

    # Use legacy exporter to avoid onnxscript requirement in dynamo_export
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True
    )
    print(f"ONNX written to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export(args.model_id, args.out)


if __name__ == "__main__":
    main()

