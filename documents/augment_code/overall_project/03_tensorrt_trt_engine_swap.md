Goal
- Build TensorRT engines optimized for Blackwell (NVFP4/FP16) and swap Triton to use *_trt variants as needed

Facts in repo
- ONNX export script: scripts/export_embedding_onnx.py
- TRT build script: scripts/build_trt_engine.ps1 (runs nvcr.io/nvidia/tensorrt:25.06-py3 trtexec)
- Triton model repo has *_trt entries expecting explicit-batch INT32 inputs for TRT

Targets
- hrm_h_trt (FP16, ~0.5GB) — preload at startup
- hrm_l_trt (NVFP4, ~0.3GB)
- qwen3_embedding_trt (NVFP4, ~2.0GB)
- qwen3_reranker_trt (NVFP4, ~2.0GB)

Plan (staged)
1) Export (if building from ONNX)
- python scripts/export_embedding_onnx.py --model <hf_id_or_local> --out containers/four-brain/triton/model_repository/<base_name>/1/model.onnx --opset 17 --max_len 512

2) Build TRT engines (examples)
- PowerShell (Option B: Weight Streaming + Strongly Typed, recommended on Blackwell):
  ./scripts/build_trt_engine.ps1 -Repo "C:\Project\Augment_Agent_Evolution\containers\four-brain\triton\model_repository" -Model qwen3_embedding_trt -MinShape 'input_ids:1x128,attention_mask:1x128' -OptShape 'input_ids:4x256,attention_mask:4x256' -MaxShape 'input_ids:4x512,attention_mask:4x512' -WorkspaceMB 4096 -BuilderOpt 3 -ExtraArgs '--allowWeightStreaming --stronglyTyped --weightStreamingBudget=6G'
  Note: --fp8 cannot be combined with --stronglyTyped via trtexec.
- PowerShell (FP16 example):
  ./scripts/build_trt_engine.ps1 -Repo "C:\Project\Augment_Agent_Evolution\containers\four-brain\triton\model_repository" -Model qwen3_embedding_trt -MinShape 'input_ids:1x128,attention_mask:1x128' -OptShape 'input_ids:4x256,attention_mask:4x256' -MaxShape 'input_ids:4x512,attention_mask:4x512' -WorkspaceMB 4096 -BuilderOpt 4 -ExtraArgs '--fp16'
- PowerShell (NVFP4 example for Blackwell):
  ./scripts/build_trt_engine.ps1 -Repo "C:\Project\Augment_Agent_Evolution\containers\four-brain\triton\model_repository" -Model qwen3_embedding_trt -MinShape 'input_ids:1x128,attention_mask:1x128' -OptShape 'input_ids:4x256,attention_mask:4x256' -MaxShape 'input_ids:4x512,attention_mask:4x512' -WorkspaceMB 4096 -BuilderOpt 4 -ExtraArgs '--nvfp4'
- Outputs: <model_name>/1/model.plan

3) Client dtype/shape
- TRT models use INT32 inputs with explicit batch; cast and shape payloads accordingly

4) Load and test
- POST /v2/repository/models/<model_name>/load
- POST /v2/models/<model_name>/infer (INT32 payload for TRT)

Notes
- Do not mix ONNX and TRT dtype expectations; keep separate model dirs
- Prefer NVFP4 on Blackwell for throughput; validate accuracy vs FP16
- .gitignore excludes *.plan and numeric version dirs already
