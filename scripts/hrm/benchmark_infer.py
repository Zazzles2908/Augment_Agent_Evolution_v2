#!/usr/bin/env python3
import argparse, time, torch

def bench_pytorch(checkpoint):
    # TODO: load HRM model and checkpoint
    model = torch.nn.Identity().eval().half()
    x = torch.zeros(1,128, dtype=torch.int64).cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        # warmup
        for _ in range(10): model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model(x)
        torch.cuda.synchronize()
        t1 = time.time()
    print(f"PyTorch FP16 avg latency: {(t1-t0)/100*1000:.3f} ms")

def bench_trt(engine):
    # For simplicity, use trtexec externally; this stub is a placeholder.
    print("Use trtexec or a proper TRT runtime benchmark; stub here.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', choices=['pytorch','trt'], required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--engine')
    args = ap.parse_args()
    if args.impl=='pytorch': bench_pytorch(args.checkpoint)
    else: bench_trt(args.engine)

