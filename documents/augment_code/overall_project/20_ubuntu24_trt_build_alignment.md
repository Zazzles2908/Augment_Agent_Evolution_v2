# Ubuntu 24.04 — TensorRT Build Alignment (Blackwell SM_120)

Objective
- Build TensorRT engines on Ubuntu 24.04 with matching TensorRT (10.13.x) to Triton image

Recommended approach
- Build inside Triton image or a matching base to guarantee ABI

Option A: Build inside Triton container (preferred)
```
docker run --rm -it --gpus all \
  -v $(pwd)/containers/four-brain/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 bash -lc '
  apt-get update && apt-get install -y python3-pip && \
  cd /models/brain1_embedding_fp8/1 && \
  trtexec --onnx=model.onnx \
    --saveEngine=/models/brain1_embedding_fp8/1/model.plan \
    --minShapes=input_ids:1x128,attention_mask:1x128 \
    --optShapes=input_ids:4x256,attention_mask:4x256 \
    --maxShapes=input_ids:8x512,attention_mask:8x512 \
    --builderOptimizationLevel=5 --fp8 --skipInference'
```

Option B: Build in Ubuntu 24.04 (WSL or host)
- Ensure TensorRT 10.13.x installed
- Use scripts/trt/build_qwen3_embedding_fp8.ps1 (WSL) or an equivalent bash script

Notes
- Match input dtypes with engine (TRT LLM often expects INT32). If needed, adjust config.pbtxt and client inputs
- Keep outputs FP16 for downstream stability even when using FP8/NVFP4 engines
- Persist tactic cache to reduce rebuild time

