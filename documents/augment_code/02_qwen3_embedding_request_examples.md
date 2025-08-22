Known-good requests for qwen3_4b_embedding (TensorRT plan)

Curl inference (batch=1)
POST http://localhost:8000/v2/models/qwen3_4b_embedding/infer
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
client = TritonEmbeddingClient(url="http://localhost:8000", model_name="qwen3_4b_embedding")
ids = np.array([[101,2009,2001,1037,2204,2154,102,0]], dtype=np.int64)
am = np.array([[1,1,1,1,1,1,1,0]], dtype=np.int64)
emb = client.infer_batch(ids, am)
print(emb.shape)

Embedding output dims = [ -1, 2000 ] (Supabase pgvector)
- Inputs are INT32, explicit batch dims in config are [-1,-1]
- Curl payload must use "datatype": "INT32" and int32 data
- Ensure /models repository has qwen3_4b_embedding with config.pbtxt and plan file

Repository control (explicit mode)
# List models
GET  http://localhost:8000/v2/repository/index
# Load model
POST http://localhost:8000/v2/repository/models/qwen3_4b_embedding/load
# Unload model
POST http://localhost:8000/v2/repository/models/qwen3_4b_embedding/unload

