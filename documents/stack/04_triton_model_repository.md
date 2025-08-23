# Triton Model Repository

Layout

```
models/
├── glm45_air/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── qwen3_4b_embedding/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── qwen3_0_6b_reranking/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
└── docling/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

Notes
- Use explicit model-control-mode and load/unload via /v2/repository APIs
- Keep consistent input dtypes: TRT usually INT32; ORT often INT64
- Compatibility: As of 2025-08-22, TensorRT supports CUDA 12.x and 13.x; prefer Triton 25.08-py3 (security fixes). Ensure your plan files are built with TensorRT 10.13.2 on CUDA 13 for best results.
- Naming: Use canonical names only — qwen3_4b_embedding (embedding), qwen3_0_6b_reranking (reranker), glm45_air (generation). Keep these consistent across code/scripts/configs.


Explicit Model Control (examples)
- Start Triton with explicit mode (example):
  - tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --http-port=8000 \
    --metrics-port=8002
- Load/unload via HTTP (abridged):
  - POST /v2/repository/models/{model}/load
  - POST /v2/repository/models/{model}/unload

Input/Output Normalization
- Prefer INT32 input IDs for TensorRT backends; ensure clients cast inputs accordingly
- Align tokenization and pad/truncation strategy across embedding/reranking/generation
- Validate shapes/dtypes with a smoke client before production use

Config Tips (config.pbtxt)
- Set max_batch_size and dynamic batching parameters appropriate to your GPU
- Document input names and dtypes; keep consistent across clients and tests
- Keep model versioning under 1/ by default; increment only with validated changes

