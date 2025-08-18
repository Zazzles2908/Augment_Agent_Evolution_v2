# HRM Phase 0 Runbook — CUDA 13 + TensorRT 10.13.x (Blackwell, Ubuntu 24.04)

Objective
- Train HRM on Sudoku-Extreme (single GPU) and validate TensorRT conversion and performance with FP16 baseline and exploratory FP8 under accuracy gates.

Prereqs
- NVIDIA driver supporting Blackwell, Docker with NVIDIA runtime
- Images:
  - Training: nvcr.io/nvidia/pytorch:25.06-py3
  - Conversion/serving: nvcr.io/nvidia/tritonserver:25.06-py3

1) Build training container
```
docker build -f containers/hrm/Dockerfile.pytorch-cuda13 -t hrm-train:25.06 .
```

2) Launch dev shell
```
docker run --rm -it --gpus all -v $PWD:/workspace hrm-train:25.06 bash
```

3) Get HRM repo and prepare data (inside container)
```
# Clone official repo
cd /workspace && git clone https://github.com/sapientinc/HRM.git && cd HRM

# Build Sudoku-Extreme dataset (1k)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

4) Train HRM (single GPU)
```
wandb login  # optional
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

5) Evaluate baseline
```
OMP_NUM_THREADS=8 python evaluate.py checkpoint=<CHECKPOINT_PATH>
```

6) Torch-TensorRT compile (partial graph)
```
python /workspace/scripts/hrm/compile_torch_trt.py --checkpoint <CHECKPOINT_PATH> --precision fp16 --out /workspace/models/hrm_sudoku_trt_fp16.ts
```

7) ONNX export (SDPA path) and TRT engine build
```
python /workspace/scripts/hrm/export_onnx.py --checkpoint <CHECKPOINT_PATH> --onnx /workspace/models/hrm_sudoku.onnx
trtexec --onnx=/workspace/models/hrm_sudoku.onnx --saveEngine=/workspace/models/hrm_sudoku_fp16.plan \
  --minShapes=input:1x128 --optShapes=input:4x256 --maxShapes=input:8x512 --fp16 --builderOptimizationLevel=5 --skipInference
```

8) Benchmark
```
python /workspace/scripts/hrm/benchmark_infer.py --impl pytorch --checkpoint <CHECKPOINT_PATH>
python /workspace/scripts/hrm/benchmark_infer.py --impl trt --engine /workspace/models/hrm_sudoku_fp16.plan
```

9) Validate accuracy
```
python /workspace/scripts/hrm/validate_accuracy.py --impl pytorch --checkpoint <CHECKPOINT_PATH> --data data/sudoku-extreme-1k-aug-1000
python /workspace/scripts/hrm/validate_accuracy.py --impl trt --engine /workspace/models/hrm_sudoku_fp16.plan --data data/sudoku-extreme-1k-aug-1000
```

Notes
- FP8 requires calibration or QAT (TensorRT Model Optimizer). Only enable when ≤1% accuracy regression.
- If ONNX export fails on custom ops, pursue Torch-TensorRT partial compile; keep FP16 outputs.
- Consider building inside Triton image for ONNX→TRT to ensure version alignment.

