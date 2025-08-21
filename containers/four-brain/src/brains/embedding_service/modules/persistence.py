"""
Persistence utilities for Brain‑1 Embedding Service

- Encapsulates DB writes so the manager and stream handlers remain thin
- Uses asyncpg; expects a Postgres URL in the form postgresql://user:pass@host:port/db
"""
from __future__ import annotations
from typing import Iterable, Tuple, Optional
import asyncio

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None

Record = Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[str], list]


class PersistenceClient:
    def __init__(self, database_url: str, connect_timeout: float = 10.0) -> None:
        self.database_url = database_url
        self.connect_timeout = connect_timeout

    async def insert_document_vectors(self, records: Iterable[Record]) -> None:
        """
        Insert embeddings into augment_agent.document_vectors.
        Each record is a tuple: (doc_id, chunk_id, page_no, title, text_excerpt, embedding_list)
        """
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        conn = await asyncpg.connect(self.database_url, timeout=self.connect_timeout)
        try:
            # executemany could be used; keep simple, explicit loop for clarity
            query = (
                """
                INSERT INTO augment_agent.document_vectors
                    (doc_id, chunk_id, page_no, title, text_excerpt, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
            )
            for (doc_id, chunk_id, page_no, title, excerpt, vec_list) in records:
                await conn.execute(query, doc_id, chunk_id, page_no, title, excerpt, vec_list)
        finally:
            await conn.close()

