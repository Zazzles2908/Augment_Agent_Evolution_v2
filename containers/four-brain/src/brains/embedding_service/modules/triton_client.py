"""
Triton Client for Brain-1 Embeddings

Minimal HTTP client for calling Triton Inference Server with ONNX model
that expects inputs:
- input_ids: int64 [N, S]
- attention_mask: int64 [N, S]
Outputs:
- embedding: fp32 [N, D] or [D]

Assumes a local tokenizer for batching; no model weights are loaded locally.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
except Exception as e:  # pragma: no cover
    httpclient = None

logger = logging.getLogger(__name__)


class TritonEmbeddingClient:
    def __init__(self,
                 url: str,
                 model_name: str,
                 timeout_s: int = 30,
                 input_ids_name: str = "input_ids",
                 attention_mask_name: str = "attention_mask",
                 output_name: str = "embedding"):
        if httpclient is None:
            raise RuntimeError("tritonclient is not available in this environment")
        # Some tritonclient versions require scheme-less host:port
        safe_url = url.replace("http://", "").replace("https://", "") if isinstance(url, str) else url
        self.url = safe_url
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.input_ids_name = input_ids_name
        self.attention_mask_name = attention_mask_name
        self.output_name = output_name
        self._client = httpclient.InferenceServerClient(url=safe_url, verbose=False)

    def is_ready(self) -> bool:
        try:
            return self._client.is_model_ready(self.model_name)
        except Exception as e:
            logger.warning(f"Triton readiness check failed: {e}")
            return False

    def infer_batch(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform a single Triton inference for a batch of tokenized inputs.
        Shapes: input_ids [N,S] int64, attention_mask [N,S] int64
        Returns: embeddings [N,D] float32
        """
        try:
            if input_ids.ndim == 1:
                input_ids = np.expand_dims(input_ids, 0)
            if attention_mask.ndim == 1:
                attention_mask = np.expand_dims(attention_mask, 0)

            if input_ids.dtype != np.int64:
                input_ids = input_ids.astype(np.int64)
            if attention_mask.dtype != np.int64:
                attention_mask = attention_mask.astype(np.int64)

            in0 = httpclient.InferInput(self.input_ids_name, input_ids.shape, np_to_triton_dtype(input_ids.dtype))
            in0.set_data_from_numpy(input_ids)

            in1 = httpclient.InferInput(self.attention_mask_name, attention_mask.shape, np_to_triton_dtype(attention_mask.dtype))
            in1.set_data_from_numpy(attention_mask)

            outputs = [httpclient.InferRequestedOutput(self.output_name, binary_data=True)]

            result = self._client.infer(model_name=self.model_name, inputs=[in0, in1], outputs=outputs, timeout=self.timeout_s)
            out = result.as_numpy(self.output_name)
            if out is None:
                return None
            if out.ndim == 1:
                out = out.reshape(1, -1)
            return out.astype(np.float32)
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            return None

