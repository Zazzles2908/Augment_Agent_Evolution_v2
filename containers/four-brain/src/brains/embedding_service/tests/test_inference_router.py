import numpy as np
import pytest

from brains.embedding_service.modules.inference_router import InferenceRouter


class DummyEngine:
    def __init__(self):
        self.calls = []
    def generate_embedding(self, text: str, truncate_to_2000: bool = True):
        self.calls.append((text, truncate_to_2000))
        return np.ones((2560,), dtype=np.float32)
    def generate_batch_embeddings(self, texts, truncate_to_2000=True):
        self.calls.append((tuple(texts), truncate_to_2000))
        return [np.ones((2560,), dtype=np.float32) for _ in texts]

class DummyTriton:
    def __init__(self):
        self.calls = []
    def infer_batch(self, ids, am):
        self.calls.append((ids.shape, am.shape))
        return np.ones((len(ids), 2560), dtype=np.float32)

class DummyTok:
    def __call__(self, inp, return_tensors="np", truncation=True, max_length=512, padding=False):
        import numpy as _np
        if isinstance(inp, str):
            return {"input_ids": _np.ones((1, max_length), dtype=_np.int64), "attention_mask": _np.ones((1, max_length), dtype=_np.int64)}
        else:
            n = len(inp)
            return {"input_ids": _np.ones((n, max_length), dtype=_np.int64), "attention_mask": _np.ones((n, max_length), dtype=_np.int64)}


def test_router_local_path_single_and_batch():
    eng = DummyEngine()
    router = InferenceRouter(triton_enabled=False, embedding_engine=eng)
    v = router.embed_one("hello")
    assert v is not None and v.shape[0] in (2000, 2560)
    vs = router.embed_batch(["a","b"])
    assert len(vs) == 2 and all(x is not None for x in vs)


def test_router_triton_path_single_and_batch():
    triton = DummyTriton()
    tok = DummyTok()
    router = InferenceRouter(triton_enabled=True, triton_client=triton, tokenizer=tok, config={"target_dimensions":2000})
    v = router.embed_one("hello")
    assert v is not None and v.shape[0] in (2000, 2560)
    vs = router.embed_batch(["a","b","c"], truncate_to_2000=False)
    assert len(vs) == 3 and all(x is not None for x in vs)

