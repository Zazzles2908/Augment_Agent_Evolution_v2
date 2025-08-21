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
└── qwen3_0_6b_reranking/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

Notes
- Use explicit model-control-mode and load/unload via /v2/repository APIs
- Keep consistent input dtypes: TRT usually INT32; ORT often INT64
- Compatibility: As of 2025-08-21, TensorRT supports CUDA 12.x and 13.x; prefer Triton 25.07+ (security fixes). Ensure your plan files are built with matching TensorRT/CUDA.
- Naming: Current repository uses qwen3_embedding_trt (embedding), qwen3_reranker_trt (reranker), glm45_air (generation). Keep these consistent across code/scripts/configs.

