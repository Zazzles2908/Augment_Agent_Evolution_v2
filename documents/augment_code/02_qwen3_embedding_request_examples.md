Known-good requests for qwen3_embedding (ONNXRuntime)

Curl inference (batch=1)
POST http://localhost:8000/v2/models/qwen3_embedding/infer
Headers: Content-Type: application/json
Body
{
  "inputs": [
    {"name": "input_ids", "datatype": "INT64", "shape": [1, 8], "data": [101, 2009, 2001, 1037, 2204, 2154, 102, 0]},
    {"name": "attention_mask", "datatype": "INT64", "shape": [1, 8], "data": [1,1,1,1,1,1,1,0]}
  ],
  "outputs": [{"name": "embedding", "binary_data": false}]
}

Python (tritonclient.http)
from containers.four_brain.src.brains.embedding_service.modules.triton_client import TritonEmbeddingClient
import numpy as np
client = TritonEmbeddingClient(url="http://localhost:8000", model_name="qwen3_embedding")
ids = np.array([[101,2009,2001,1037,2204,2154,102,0]], dtype=np.int64)
am = np.array([[1,1,1,1,1,1,1,0]], dtype=np.int64)
emb = client.infer_batch(ids, am)
print(emb.shape)

If calling qwen3_embedding_trt (TensorRT)
- Inputs are INT32, explicit batch dims in config are [-1,-1]
- Curl payload must use "datatype": "INT32" and int32 data
- The model name changes to qwen3_embedding_trt

Repository control (explicit mode)
# List models
GET  http://localhost:8000/v2/repository/index
# Load model
POST http://localhost:8000/v2/repository/models/qwen3_embedding/load
# Unload model
POST http://localhost:8000/v2/repository/models/qwen3_embedding/unload

