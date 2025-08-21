# Training & Fine-Tuning Procedures

This document covers data preparation, supervised fine-tuning (SFT), and evaluation for HRM modules and retrieval components.

## Scope
- HRM L-Module: instruction-tuned for tool-use and short-hop reasoning
- Embedding/Reranker: optional domain adaptation via contrastive/LLM-as-judge feedback
- Docling: rely on vendor model; fine-tuning if permitted

## Data Sources
- Household tasks, calendars, emails (consent-based)
- Public cooking/fitness/how-to datasets
- Redact PII and store only necessary features

## Pipeline
1) Collect raw data → anonymize
2) Build instruction datasets (input, tools, expected outputs)
3) Split train/val/test; balance per task type
4) Train with Hydra configs and W&B tracking

## Example Hydra Config (YAML)
```yaml
trainer:
  max_steps: 20000
  gradient_accumulation_steps: 4
  lr: 2e-5
  batch_size: 4
  precision: bf16
model:
  base: Qwen3-8B
  target: L-Module
  quant: nvfp4
data:
  path: data/sft/
  val_ratio: 0.05
logging:
  wandb_project: hrm-lmodule
```

## Retrieval Adaptation
- Hard negatives: use reranker scores to mine
- Pairwise ranking loss to improve ordering
- Validate Top-K recall @5/10 and MRR

## Evaluation
- Task success rate under HRM loop budget
- Latency and VRAM under concurrent load
- Human eval for helpfulness and safety adherence

## Export & Deployment
- Export ONNX and build TensorRT engines (see inference_stack.md)
- Version artifacts; store engines in model repository

## Safety
- Disallow training on sensitive household data unless both users consent
- Maintain audit logs for dataset lineage



## Blackwell (SM_120) Training Optimization
- Mixed precision: prefer bf16 autocast for stability/perf
- Use torch.compile(mode="max-autotune") for HRM modules when supported
- Gradient checkpointing for long sequences; tune micro-batch sizes
- Fused optimizers (AdamW fused) if available; enable channels-last memory format
- Dataloader: pin_memory=True, non_blocking transfers, persistent workers
- Profiling: Nsight Systems/Compute for kernel-level; nvidia-smi dmon for VRAM/SM utilization; PyTorch profiler for hotspots
- Validate convergence vs precision by ablation runs (bf16 vs fp16)
