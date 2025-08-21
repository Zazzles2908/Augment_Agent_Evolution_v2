#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse, os, sys, subprocess, json, yaml
from pathlib import Path

# Minimal TensorRT conversion orchestrator for NVFP4
# - Assumes ONNX input present; generates TensorRT plan under model/1/model.plan
# - Keeps config.pbtxt in place; only replaces engine

SUPPORTED = {"qwen3_4b_embedding", "qwen3_0_6b_reranking", "glm45_air"}


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def build_engine(model_name: str, onnx_path: Path, plan_path: Path, precision: str, opt_cfg: dict):
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    trtexec = os.environ.get("TRTEXEC", "trtexec")
    # Base args
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={plan_path}"]
    # Precision flags
    if precision.lower() == "nvfp4":
        # Prefer FP8 path under NVFP4 on Blackwell; do not combine with INT8
        cmd += ["--precisionConstraints=prefer", "--fp8"]
        # If you explicitly want INT8 instead of FP8, set builder.int8=true in YAML and precision=nvfp4 will honor INT8 only
        if opt_cfg.get("int8", False):
            cmd = [x for x in cmd if x != "--fp8"]
            cmd += ["--int8"]
            calib = opt_cfg.get("calib_cache")
            if calib:
                cmd += [f"--calibCache={calib}"]
    elif precision.lower() == "fp8":
        cmd += ["--fp8"]
    elif precision.lower() == "fp16":
        cmd += ["--fp16"]
    # Common builder options
    if opt_cfg.get("workspace_gb"):
        mibs = int(float(opt_cfg['workspace_gb']) * 1024)
        cmd += [f"--memPoolSize=workspace:{mibs}M"]
    # Shapes
    if model_name == "qwen3_0_6b_reranking":
        # Reranker has four inputs; reuse the same seq shape for both query and doc
        if opt_cfg.get("opt_batch"):
            shape = opt_cfg['opt_batch']
            cmd += [f"--optShapes=query_ids:{shape},query_mask:{shape},doc_ids:{shape},doc_mask:{shape}"]
        if opt_cfg.get("min_batch") and opt_cfg.get("max_batch"):
            min_s = opt_cfg['min_batch']
            max_s = opt_cfg['max_batch']
            cmd += [f"--minShapes=query_ids:{min_s},query_mask:{min_s},doc_ids:{min_s},doc_mask:{min_s}"]
            cmd += [f"--maxShapes=query_ids:{max_s},query_mask:{max_s},doc_ids:{max_s},doc_mask:{max_s}"]
    else:
        if opt_cfg.get("opt_batch"):
            cmd += [f"--optShapes=input_ids:{opt_cfg['opt_batch']},attention_mask:{opt_cfg['opt_batch']}"]
        if opt_cfg.get("min_batch") and opt_cfg.get("max_batch"):
            cmd += [f"--minShapes=input_ids:{opt_cfg['min_batch']},attention_mask:{opt_cfg['min_batch']}"]
            cmd += [f"--maxShapes=input_ids:{opt_cfg['max_batch']},attention_mask:{opt_cfg['max_batch']}"]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--precision", default="nvfp4")
    args = ap.parse_args()

    if args.model not in SUPPORTED:
        print(f"Unsupported model: {args.model}", file=sys.stderr)
        sys.exit(2)

    repo = Path(args.repo)
    model_dir = repo / args.model
    if not model_dir.exists():
        print(f"Model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)

    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Locate or export ONNX input
    onnx_path = model_dir / "1" / "model.onnx"
    if not onnx_path.exists():
        alt = model_dir / "onnx" / "model.onnx"
        if alt.exists():
            onnx_path = alt
        else:
            # Download HF model and export to ONNX
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
            hf_id = cfg.get("hf", {}).get("model_id")
            if not hf_id:
                print("Missing hf.model_id in config", file=sys.stderr)
                sys.exit(2)
            seq = int(cfg.get("export", {}).get("seq_length", 512))
            print(f"Downloading model {hf_id} and exporting to ONNX (seq_length={seq})")
            from scripts.tensorrt.exporters import EmbeddingWrapper, RerankerWrapper, GeneratorWrapper
            tok = AutoTokenizer.from_pretrained(hf_id)
            try:
                tok("sanity", return_tensors="pt")
            except Exception as e:
                fb = cfg.get("hf", {}).get("fallback")
                if fb:
                    print(f"Tokenizer load failed for {hf_id}, falling back to {fb}: {e}")
                    hf_id = fb
                    tok = AutoTokenizer.from_pretrained(hf_id)
            if args.model == "glm45_air":

                model = GeneratorWrapper(hf_id)
                dummy = tok("hello", return_tensors="pt", padding="max_length", truncation=True, max_length=seq)
                inputs = (dummy["input_ids"], dummy["attention_mask"])
                outputs = ["logits"]
            elif args.model == "qwen3_0_6b_reranking":
                model = RerankerWrapper(hf_id)
                q = tok("hello", return_tensors="pt", padding="max_length", truncation=True, max_length=seq)
                d = tok("world", return_tensors="pt", padding="max_length", truncation=True, max_length=seq)
                inputs = (q["input_ids"], q["attention_mask"], d["input_ids"], d["attention_mask"])
                outputs = ["score"]
            else:
                model = EmbeddingWrapper(hf_id, target_dim=2000)
                dummy = tok("hello", return_tensors="pt", padding="max_length", truncation=True, max_length=seq)
                inputs = (dummy["input_ids"], dummy["attention_mask"])
                outputs = ["embedding"]
            import torch
            # Prepare for ONNX export
            model.eval()
            model.to("cpu")
            # Force math attention to avoid SDPA tracer issues
            try:
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)  # no-op on CPU
            except Exception:
                pass
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            in_names = ["input_ids","attention_mask"] if args.model != "qwen3_0_6b_reranking" else [
                "query_ids","query_mask","doc_ids","doc_mask"
            ]
            # Legacy ONNX exporter with fallback
            def legacy_export(current_model, current_tok, seq_len:int):
                nonlocal inputs
                # Recreate inputs at desired seq length
                if args.model == "qwen3_0_6b_reranking":
                    q = current_tok("hello", return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)
                    d = current_tok("world", return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)
                    inputs = (q["input_ids"], q["attention_mask"], d["input_ids"], d["attention_mask"])
                    dyn_axes = {
                        "query_ids": {0: "batch", 1: "seq"},
                        "query_mask": {0: "batch", 1: "seq"},
                        "doc_ids": {0: "batch", 1: "seq"},
                        "doc_mask": {0: "batch", 1: "seq"},
                    }
                else:
                    dummy = current_tok("hello", return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)
                    inputs = (dummy["input_ids"], dummy["attention_mask"])
                    dyn_axes = {
                        "input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                    }
                with torch.no_grad():
                    torch.onnx.export(
                        current_model, inputs, str(onnx_path),
                        input_names=in_names,
                        output_names=outputs,
                        opset_version=17,
                        dynamic_axes=dyn_axes,
                    )

            try:
                legacy_export(model, tok, seq)
            except Exception as e1:
                print(f"Primary ONNX export failed (seq={seq}): {e1}. Retrying with seq={max(128, seq//2)}")
                try:
                    legacy_export(model, tok, max(128, seq//2))
                except Exception as e2:
                    fb = cfg.get("hf", {}).get("fallback")
                    if fb:
                        print(f"Exporter still failing. Falling back to simpler model {fb}")
                        from scripts.tensorrt.exporters import EmbeddingWrapper, RerankerWrapper, GeneratorWrapper
                        # Rebuild model+tokenizer with fallback
                        tok_fb = AutoTokenizer.from_pretrained(fb)
                        if args.model == "glm45_air":
                            model_fb = GeneratorWrapper(fb)
                        elif args.model == "qwen3_0_6b_reranking":
                            model_fb = RerankerWrapper(fb)
                        else:
                            model_fb = EmbeddingWrapper(fb, target_dim=2000)
                        model_fb.eval(); model_fb.to("cpu")
                        legacy_export(model_fb, tok_fb, 128)
                    else:
                        raise
            print(f"Exported ONNX to {onnx_path}")
            # Optional: run ONNX shape inference
            try:
                import onnx
                m = onnx.load(str(onnx_path))
                m = onnx.shape_inference.infer_shapes(m)
                onnx.save(m, str(onnx_path))
                print("ONNX shape inference complete")
            except Exception as e:
                print(f"ONNX shape inference skipped: {e}")

    plan_path = model_dir / "1" / "model.plan"

    # Apply template adjustments (placeholder)
    opt_cfg = cfg.get("builder", {})

    # Build
    build_engine(args.model, onnx_path, plan_path, args.precision, opt_cfg)

    print(f"Engine built: {plan_path}")


if __name__ == "__main__":
    main()

