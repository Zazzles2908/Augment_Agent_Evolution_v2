Goal
- Ensure Triton (explicit model-control-mode) loads qwen3_embedding and returns embeddings

Key facts from repo
- Triton runs with: --model-control-mode=explicit and model repo mounted at containers/four-brain/triton/model_repository
- qwen3_embedding config expects:
  - platform: onnxruntime_onnx
  - inputs: input_ids:int64 [ -1 ], attention_mask:int64 [ -1 ] (runtime interprets with batch dim => [N,S])
  - output: embedding:fp32 [ -1 ] (runtime => [N,D])
- qwen3_embedding_trt has different dtypes/shapes (INT32, explicit-batch). Use only if calling that model.

Likely pitfalls the other AI may hit
1) Model not loaded (explicit mode) — Triton requires a load call
2) Input dtype mismatch — curl payload defaulting to INT32 while config requires INT64
3) Shape mismatch — sending 1D arrays without batch dimension when needed
4) Wrong model name — calling qwen3_embedding but engine exists only for qwen3_embedding_trt (or vice versa)

Actions (safe, idempotent)
1) Verify server & repository
- GET http://localhost:8000/v2/health/ready
- GET http://localhost:8000/v2/repository/index — confirm both qwen3_embedding and qwen3_embedding_trt listed

2) Explicitly load model (if state is UNAVAILABLE)
- POST http://localhost:8000/v2/repository/models/qwen3_embedding/load
  Body: {}
- GET http://localhost:8000/v2/models/qwen3_embedding — status should be READY

3) Minimal working inference (curl)
- POST http://localhost:8000/v2/models/qwen3_embedding/infer
  Header: Content-Type: application/json
  Body example (single sentence, batch=1):
  {
    "inputs": [
      {"name": "input_ids", "datatype": "INT64", "shape": [1, 8], "data": [101, 2009, 2001, 1037, 2204, 2154, 102, 0]},
      {"name": "attention_mask", "datatype": "INT64", "shape": [1, 8], "data": [1,1,1,1,1,1,1,0]}
    ],
    "outputs": [{"name": "embedding", "binary_data": false}]
  }
- Expect outputs[0].data length ≈ embedding dimension (2560), or truncated later in client to 2000

4) Using tritonclient (Python) with correct dtypes
- Ensure inputs are numpy.int64 and shaped [N,S]
- Use containers/four-brain/src/brains/embedding_service/modules/triton_client.py as reference

Troubleshooting
- 404/UNAVAILABLE: load the model via repository load API
- INVALID_ARG: check datatype INT64 vs INT32 and shape [N,S]
- READY but slow: use dynamic_batching (already configured) and prefer batch sizes 2–8

