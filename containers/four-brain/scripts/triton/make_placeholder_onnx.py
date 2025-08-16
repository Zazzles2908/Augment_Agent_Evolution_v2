#!/usr/bin/env python3
"""
Generate tiny placeholder ONNX models matching specified I/O for Phase 1 validation.
- All models accept INT64 token inputs and produce simple FP32 outputs with correct dims.
- No external dependencies beyond onnx and numpy.
"""
import argparse
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto as tp
import numpy as np
import os


def make_h_like(out_path: str):
    # inputs: input_ids [N,S] int64, attention_mask [N,S] int64
    # output: logits [N,S] fp32 (dummy passthrough of mask expanded)
    n = oh.make_tensor_value_info("input_ids", tp.INT64, ["N", "S"])
    m = oh.make_tensor_value_info("attention_mask", tp.INT64, ["N", "S"])
    y = oh.make_tensor_value_info("logits", tp.FLOAT, ["N", "S"])

    # Cast mask to float and output as logits
    cast = oh.make_node("Cast", inputs=["attention_mask"], outputs=["logits"], to=tp.FLOAT)
    graph = oh.make_graph([cast], name="hrm_stub", inputs=[n, m], outputs=[y])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)


def make_l_like(out_path: str):
    return make_h_like(out_path)


def make_reranker_like(out_path: str):
    # inputs: query_ids [N,S], query_mask [N,S], doc_ids [N,S], doc_mask [N,S] (all int64)
    # output: score [N,1] float
    qi = oh.make_tensor_value_info("query_ids", tp.INT64, ["N", "S"])
    qm = oh.make_tensor_value_info("query_mask", tp.INT64, ["N", "S"])
    di = oh.make_tensor_value_info("doc_ids", tp.INT64, ["N", "S"])
    dm = oh.make_tensor_value_info("doc_mask", tp.INT64, ["N", "S"])
    y = oh.make_tensor_value_info("score", tp.FLOAT, ["N", 1])

    cast = oh.make_node("Cast", inputs=["query_mask"], outputs=["qf"], to=tp.FLOAT)
    reduce = oh.make_node("ReduceMean", inputs=["qf"], outputs=["score"], keepdims=1, axes=[1])
    graph = oh.make_graph([cast, reduce], name="reranker_stub", inputs=[qi, qm, di, dm], outputs=[y])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)


def make_docling_like(out_path: str):
    # input: input_bytes [K] uint8, output: blocks [1] float
    ib = oh.make_tensor_value_info("input_bytes", tp.UINT8, ["K"])
    out = oh.make_tensor_value_info("blocks", tp.FLOAT, [1])
    # Provide a constant scalar float output
    s = onh.from_array(np.array([0.0], dtype=np.float32), name="c0")
    const = oh.make_node("Constant", inputs=[], outputs=["blocks"], value=s)
    graph = oh.make_graph([const], name="docling_stub", inputs=[ib], outputs=[out])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["hrm_h", "hrm_l", "reranker", "docling"])
    ap.add_argument("out", help="Output ONNX path")
    args = ap.parse_args()

    if args.kind == "hrm_h":
        make_h_like(args.out)
    elif args.kind == "hrm_l":
        make_l_like(args.out)
    elif args.kind == "reranker":
        make_reranker_like(args.out)
    else:
        make_docling_like(args.out)


if __name__ == "__main__":
    main()

