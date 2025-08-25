# Triton Explicit Mode — Minimal Smoke Validation (Ubuntu 24.04 / Blackwell SM_120)

Purpose
- Verify Triton explicit model control works with our configs and engines
- Validate FP16 outputs, batching, and basic latency on Ubuntu 24.04

Prereqs
- Docker with NVIDIA runtime; Ubuntu 24.04 host or WSL2
- Triton image: nvcr.io/nvidia/tritonserver:25.06-py3 (TensorRT 10.13.x)
- Model repository mounted at /models with config.pbtxt only versioned
- Engines (.plan) and ONNX staged locally (not in git)

1) Start Triton (explicit mode)
- Compose service already configured. Or run directly:
```
docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/containers/four-brain/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit \
  --log-verbose=1 --strict-readiness=false
```

2) Repository index and load a model
```
# List repository (empty or with configs only)
curl -s http://localhost:8000/v2/repository/index | jq .

# Load Qwen3-4B Embedding
curl -s -X POST http://localhost:8000/v2/repository/models/qwen3_4b_embedding/load

# Check readiness
curl -s http://localhost:8000/v2/models/qwen3_4b_embedding/ready
```

3) Run a tiny inference and validate dtype/latency
```
# Example using curl HTTP/JSON inference
cat > /tmp/embedding_input.json <<'JSON'
{
  "id": "1",
  "inputs": [
    {"name": "input_ids", "shape": [1, 128], "datatype": "INT64", "data": [$(python - <<PY
print(",".join(["1"]*128))
PY
)]},
    {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64", "data": [$(python - <<PY
print(",".join(["1"]*128))
PY
)]}
  ],
  "outputs": [{"name": "embedding"}]
}
JSON

# Send request
/usr/bin/time -f '%E real' curl -s -X POST http://localhost:8000/v2/models/qwen3_4b_embedding/infer \
  -H 'Content-Type: application/json' -d @/tmp/embedding_input.json | jq '.outputs[0].datatype'
```
Expected
- Output datatype is "FP32"
- End-to-end latency acceptable for [1,128] (this is a smoke test, not a benchmark)

4) Notes
- For FP8/NVFP4 models, ensure the corresponding model.plan is present in /models/<name>/1
- Inputs must match engine dtypes (some TensorRT engines expect INT32); adjust config.pbtxt and inputs accordingly
- To unload: POST /v2/repository/models/<name>/unload

5) Next
- Add simple Python tritonclient examples for embedding/reranker showing [2,4,8] batching and basic timing
- Integrate checks into CI to run HTTP readiness + repository index validation

