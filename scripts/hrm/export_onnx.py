#!/usr/bin/env python3
import argparse, torch

# TODO: replace with actual HRM model import + load
# from HRM.models import build_model, load_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--onnx', required=True)
    args = p.parse_args()

    # model = build_model(); load_checkpoint(model, args.checkpoint)
    model = torch.nn.Identity()
    model.eval().half()

    dummy = torch.zeros(1,128, dtype=torch.int64)  # adjust to HRM expected input(s)
    torch.onnx.export(
        model, (dummy,), args.onnx,
        input_names=['input'], output_names=['output'],
        opset_version=19, dynamic_axes={'input': {0: 'B', 1: 'S'}, 'output': {0: 'B'}},
    )
    print(f"Saved ONNX to {args.onnx}")

if __name__ == '__main__':
    main()

