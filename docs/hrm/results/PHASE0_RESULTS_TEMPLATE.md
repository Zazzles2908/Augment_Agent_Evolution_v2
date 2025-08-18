# HRM Phase 0 — Results (Template)

Environment
- Host: Windows 11 + WSL2 (Ubuntu 24.04)
- GPU: RTX 5070 Ti (Blackwell, SM_120), 16GB
- Containers: PyTorch 25.06, Triton 25.06 (TRT 10.13.x)

Training
- Dataset: Sudoku-Extreme 1k (aug)
- Command: pretrain.py (see runbook)
- W&B run: <link>

Evaluation (PyTorch FP16)
- exact_accuracy: <value>
- latency (per sample): <ms>

Torch‑TensorRT (FP16)
- exact_accuracy: <value>
- latency: <ms>

ONNX→TRT (FP16)
- exact_accuracy: <value>
- latency: <ms>

FP8 (if attempted)
- quantization: PTQ/QAT (specify)
- exact_accuracy: <value>
- latency: <ms>
- regression vs FP16: <delta%> (must be ≤1% to pass)

Findings
- Summary of performance gains/losses
- Any export gaps and mitigations

Next
- Proceed/not proceed to FP8
- Expand to HRM Low or other tasks

