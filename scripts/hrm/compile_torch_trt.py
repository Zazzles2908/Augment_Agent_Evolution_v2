#!/usr/bin/env python3
import argparse, torch

# TODO: replace with actual HRM model import + load
# from HRM.models import build_model, load_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--precision', default='fp16', choices=['fp16','fp8'])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    # model = build_model()
    # load_checkpoint(model, args.checkpoint)
    model = torch.nn.Identity()  # placeholder
    model.eval()
    model = model.half()

    import torch_tensorrt as trt
    compiled = trt.compile(
        model,
        inputs=[trt.Input((1,128), dtype=torch.half)],
        enabled_precisions={torch.half},
        require_full_compilation=False,
    )
    torch.jit.save(compiled, args.out)
    print(f"Saved Torch-TRT module to {args.out}")

if __name__ == '__main__':
    main()

