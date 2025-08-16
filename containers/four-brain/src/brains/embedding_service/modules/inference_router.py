"""
InferenceRouter
- Decides whether to route inference to Triton or local engine
- Keeps embedding_manager.py thin by isolating decision + execution
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

class InferenceRouter:
    def __init__(self, *, triton_enabled: bool, triton_client=None, embedding_engine=None,
                 tokenizer=None, config=None):
        self.triton_enabled = triton_enabled
        self.triton_client = triton_client
        self.embedding_engine = embedding_engine
        self.tokenizer = tokenizer
        self.config = config or {}

    def embed_one(self, text: str, truncate_to_2000: bool = True) -> Optional[np.ndarray]:
        if self.triton_enabled and self.triton_client is not None and self.tokenizer is not None:
            enc = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
            ids = enc["input_ids"].astype(np.int64)
            am = enc["attention_mask"].astype(np.int64)
            out = self.triton_client.infer_batch(ids, am)
            if out is None:
                return None
            vec = out[0] if out.ndim > 1 else out
            if truncate_to_2000 and vec.shape[0] > int(self.config.get("target_dimensions", 2000)):
                vec = vec[: int(self.config.get("target_dimensions", 2000))]
            return vec
        else:
            if self.embedding_engine is None:
                return None
            return self.embedding_engine.generate_embedding(text=text, truncate_to_2000=truncate_to_2000)

    def embed_batch(self, texts: List[str], truncate_to_2000: bool = True) -> List[Optional[np.ndarray]]:
        if self.triton_enabled and self.triton_client is not None and self.tokenizer is not None:
            enc = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
            ids = enc["input_ids"].astype(np.int64)
            am = enc["attention_mask"].astype(np.int64)
            out = self.triton_client.infer_batch(ids, am)
            if out is None:
                return [None for _ in texts]
            if out.ndim == 1:
                out = out.reshape(1, -1)
            # optional truncation
            td = int(self.config.get("target_dimensions", 2000))
            if truncate_to_2000 and out.shape[1] > td:
                out = out[:, :td]
            return [row for row in out]
        else:
            if self.embedding_engine is None:
                return [None for _ in texts]
            return self.embedding_engine.generate_batch_embeddings(texts=texts, truncate_to_2000=truncate_to_2000)

