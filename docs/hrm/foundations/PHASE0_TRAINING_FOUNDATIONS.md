# HRM Training Foundations — How training makes the model better

This page explains, in plain language, what happens during training, how weights update, and how this relates to TensorRT/Triton deployment.

## 1) What is “training” in practice?
- You have a model with parameters (weights). Initially they’re random or partially trained.
- You feed batches of examples (inputs and targets) to the model.
- The model produces predictions; a loss function measures error vs the targets.
- An optimizer (e.g., AdamW) nudges the weights to reduce the loss.
- Repeating this over many batches/epochs gradually improves the model’s ability to solve tasks similar to the training data.

## 2) Do the weights change with each run?
- Yes. Each training step updates the weights in memory.
- If you save a checkpoint, the updated weights are written to disk (e.g., `checkpoints/…pt`).
- If you run training again from scratch, weights start fresh (unless you load a checkpoint).
- If you resume from a checkpoint, training continues improving from that point.

## 3) Optimizers: Adam-ATAN2 vs AdamW (and why we used AdamW now)
- `Adam-ATAN2` is a custom optimizer that can improve stability, but its CUDA kernel must support your GPU’s architecture (SM 12.0 for Blackwell).
- On RTX 5070 Ti, we observed a runtime error: “no kernel image available,” meaning the shipped kernel wasn’t compiled for SM_120.
- We switched to `AdamW` for the smoke run to ensure compatibility.
- Plan for next runs: we can attempt to build or obtain an SM_120-compatible `adam-atan2` (or keep AdamW if it’s performing well).

## 4) Does using TensorRT later “reset” our training?
- No. Training produces weights. TensorRT is an inference accelerator that compiles your trained model into a highly optimized runtime engine.
- Workflow:
  1) Train in PyTorch. Save checkpoint(s).
  2) Export to ONNX or TorchScript, or use Torch-TensorRT to compile directly.
  3) TensorRT builds an engine using the frozen weights from your trained model.
  4) Triton (NVIDIA inference server) serves that TensorRT engine.
- You do not lose your training work; you’re packaging the learned weights for fast inference.

## 5) Compatibility checklist (so you don’t block future TRT/Triton use)
- Keep the model exportable:
  - Prefer standard PyTorch ops and SDPA attention path.
  - Avoid dynamic control flow that blocks ONNX export (or be ready to fall back to TorchScript/Torch-TensorRT flows).
- Capture and save trained weights (checkpoints) during/after training.
- For TRT, we’ll pick one of these paths:
  - Torch-TensorRT (partial graph compile) to a TorchScript .ts with TRT segments
  - ONNX -> trtexec to build a .plan TensorRT engine
- Triton can serve either a TorchScript (.pt/.ts) or TensorRT (.plan) backend.

## 6) What “improvement” means and how to measure it
- Lower loss over time during training.
- Higher accuracy on a validation/evaluation set.
- For HRM: ACT metrics improving (e.g., q_halt, q_continue) and task-specific metrics.
- Performance (throughput): steps/sec, tokens/sec increasing as we tune batch size and kernels.
- We track these in W&B and in on-disk logs.

## 7) Putting it together
- Training updates weights -> better model behavior on tasks.
- Checkpoints preserve that progress -> export/compile later for deployment.
- TensorRT/Triton are downstream of training and use the trained weights; they don’t erase learning.
- You can iterate: train more -> export new engine -> serve newer/better model.

See also:
- guides/PHASE0_USER_GUIDE.md (how to run and find outputs)
- monitoring/PHASE0_VISUAL_MONITORING.md (how to watch live)
- 22_hrm_phase0_runbook.md (end-to-end training/convert/benchmark steps)

