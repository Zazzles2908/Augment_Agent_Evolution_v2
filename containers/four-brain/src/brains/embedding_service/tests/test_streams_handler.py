import numpy as np
import pytest

from brains.embedding_service.modules.streams_handler import StreamsHandler


class DummyRedis:
    def __init__(self):
        self.handlers = {}
        self.sent = []
        self.is_connected = True
    def register_handler(self, stream_name, handler):
        self.handlers[stream_name] = handler
    async def send_message(self, stream_name, message):
        self.sent.append((stream_name, message))
    async def start_consuming(self):
        # noop for test
        return

class DummyPersistence:
    def __init__(self):
        self.records = []
    async def insert_document_vectors(self, recs):
        self.records.extend(recs)

class DummyMessage:
    def __init__(self, data):
        self.data = data

@pytest.mark.asyncio
async def test_streams_handler_batch_flow():
    redis = DummyRedis()
    persistence = DummyPersistence()

    async def embed_batch(texts, truncate_to_2000=True):
        return [np.ones((2000,), dtype=np.float32) for _ in texts]

    handler = StreamsHandler(redis_client=redis, persistence_client=persistence, embed_batch_func=embed_batch, target_dimensions=2000)
    await handler.register_embedding_batch_handler()

    # simulate a message
    data = {
        "doc_id": "doc-1",
        "chunk_batch_id": "batch-1",
        "chunk_refs": [
            {"chunk_id": "c1", "text_excerpt": "aaa"},
            {"chunk_id": "c2", "text_excerpt": "bbb"}
        ],
        "target_dim": 2000
    }
    await redis.handlers.get("embedding_requests")(DummyMessage(data))

    assert len(persistence.records) == 2
    assert len(redis.sent) == 1

