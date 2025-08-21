import logging
from typing import Optional

import numpy as np

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
except Exception:  # pragma: no cover
    httpclient = None
    np_to_triton_dtype = lambda dt: ""  # type: ignore

logger = logging.getLogger(__name__)


class TritonRerankerClient:
    """
    Minimal Triton HTTP client for Qwen3-0.6B reranker served via TensorRT on Triton.
    Expects inputs named: query_ids, query_mask, doc_ids, doc_mask
    Output: score [N,1] float32
    """

    def __init__(
        self,
        url: str,
        model_name: str = "qwen3_0_6b_reranking",
        timeout_s: int = 30,
        query_ids_name: str = "query_ids",
        query_mask_name: str = "query_mask",
        doc_ids_name: str = "doc_ids",
        doc_mask_name: str = "doc_mask",
        output_name: str = "score",
    ) -> None:
        if httpclient is None:
            raise RuntimeError("tritonclient is not available in this environment")
        safe_url = url.replace("http://", "").replace("https://", "") if isinstance(url, str) else url
        self._client = httpclient.InferenceServerClient(url=safe_url, verbose=False)
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.query_ids_name = query_ids_name
        self.query_mask_name = query_mask_name
        self.doc_ids_name = doc_ids_name
        self.doc_mask_name = doc_mask_name
        self.output_name = output_name

    def is_ready(self) -> bool:
        try:
            return self._client.is_model_ready(self.model_name)
        except Exception as e:
            logger.warning(f"Triton reranker readiness check failed: {e}")
            return False

    def infer_batch(
        self,
        query_ids: np.ndarray,
        query_mask: np.ndarray,
        doc_ids: np.ndarray,
        doc_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            # Ensure 2D shapes [N,S]
            if query_ids.ndim == 1:
                query_ids = np.expand_dims(query_ids, 0)
            if query_mask.ndim == 1:
                query_mask = np.expand_dims(query_mask, 0)
            if doc_ids.ndim == 1:
                doc_ids = np.expand_dims(doc_ids, 0)
            if doc_mask.ndim == 1:
                doc_mask = np.expand_dims(doc_mask, 0)

            # dtypes INT64 per config.pbtxt
            if query_ids.dtype != np.int64:
                query_ids = query_ids.astype(np.int64)
            if query_mask.dtype != np.int64:
                query_mask = query_mask.astype(np.int64)
            if doc_ids.dtype != np.int64:
                doc_ids = doc_ids.astype(np.int64)
            if doc_mask.dtype != np.int64:
                doc_mask = doc_mask.astype(np.int64)

            qi = httpclient.InferInput(self.query_ids_name, query_ids.shape, np_to_triton_dtype(query_ids.dtype))
            qi.set_data_from_numpy(query_ids)
            qm = httpclient.InferInput(self.query_mask_name, query_mask.shape, np_to_triton_dtype(query_mask.dtype))
            qm.set_data_from_numpy(query_mask)
            di = httpclient.InferInput(self.doc_ids_name, doc_ids.shape, np_to_triton_dtype(doc_ids.dtype))
            di.set_data_from_numpy(doc_ids)
            dm = httpclient.InferInput(self.doc_mask_name, doc_mask.shape, np_to_triton_dtype(doc_mask.dtype))
            dm.set_data_from_numpy(doc_mask)

            outputs = [httpclient.InferRequestedOutput(self.output_name, binary_data=True)]
            result = self._client.infer(
                model_name=self.model_name,
                inputs=[qi, qm, di, dm],
                outputs=outputs,
                timeout=self.timeout_s,
            )
            out = result.as_numpy(self.output_name)
            if out is None:
                return None
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            return out.astype(np.float32)
        except Exception as e:
            logger.error(f"Triton reranker inference failed: {e}")
            return None

