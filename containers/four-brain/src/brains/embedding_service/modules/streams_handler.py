"""
StreamsHandler for Brain‑1 Embedding Service

- Registers Redis Streams handlers
- Delegates embedding work to a provided function
- Persists results via PersistenceClient
- Keeps embedding_manager thin
"""
from __future__ import annotations
from typing import List, Callable, Dict, Any
import numpy as np

from shared.streams import StreamNames, EmbeddingResult


class StreamsHandler:
    def __init__(self,
                 redis_client,
                 persistence_client,
                 embed_batch_func: Callable[[List[str], bool], List[np.ndarray] | List[None]],
                 target_dimensions: int = 2000):
        self.redis_client = redis_client
        self.persistence = persistence_client
        self.embed_batch = embed_batch_func
        self.target_dimensions = target_dimensions

    async def register_embedding_batch_handler(self) -> None:
        if not self.redis_client:
            return

        async def handle_batch(msg):
            data: Dict[str, Any] = msg.data
            doc_id = data.get('doc_id')
            chunk_refs = data.get('chunk_refs', [])
            target_dim = int(data.get('target_dim', self.target_dimensions))
            texts: List[str] = []
            chunk_ids: List[str] = []

            # Phase 1: use provided text_excerpt directly; later fetch from DB/storage
            for ref in chunk_refs:
                txt = ref.get('text_excerpt') or ''
                texts.append(txt)
                chunk_ids.append(ref.get('chunk_id'))

            # Adaptive small batches for stability
            batch_size = min(32, max(4, 4096 // max(8, int(np.mean([len(t) for t in texts]) or 1)))) if texts else 8

            vectors_pairs = []
            for i in range(0, len(texts), batch_size):
                sub = texts[i:i + batch_size]
                embs = await self.embed_batch(sub, truncate_to_2000=(target_dim == self.target_dimensions))
                for j, v in enumerate(embs):
                    if v is None:
                        continue
                    vectors_pairs.append((chunk_ids[i + j], v))

            # Persist vectors
            if vectors_pairs:
                records = []
                for cid, vec in vectors_pairs:
                    records.append((doc_id, str(cid), None, None, None, list(vec)))
                await self.persistence.insert_document_vectors(records)

            # Publish result
            vectors_meta = [{"chunk_id": cid, "vector_ref": None, "dim": target_dim} for cid, _ in vectors_pairs]
            res = EmbeddingResult(doc_id=doc_id, chunk_batch_id=data.get('chunk_batch_id'), vectors=vectors_meta, stats={"count": len(vectors_pairs)})
            await self.redis_client.send_message(StreamNames.EMBEDDING_RESULTS, res)

        self.redis_client.register_handler(StreamNames.EMBEDDING_REQUESTS, handle_batch)

    async def begin_consuming(self) -> None:
        if self.redis_client and self.redis_client.is_connected:
            await self.redis_client.start_consuming()

