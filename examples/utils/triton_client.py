import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

class TritonHelper:
    def __init__(self, url: str):
        self.client = InferenceServerClient(url=url)

    def embed(self, model: str, input_ids: np.ndarray, attention_mask: np.ndarray):
        ii = InferInput("input_ids", input_ids.shape, "INT64"); ii.set_data_from_numpy(input_ids)
        am = InferInput("attention_mask", attention_mask.shape, "INT64"); am.set_data_from_numpy(attention_mask)
        res = self.client.infer(model, [ii, am])
        return res.as_numpy("embedding")

    def rerank(self, model: str, q_ids: np.ndarray, q_mask: np.ndarray, d_ids: np.ndarray, d_mask: np.ndarray):
        qi = InferInput("query_ids", q_ids.shape, "INT64"); qi.set_data_from_numpy(q_ids)
        qm = InferInput("query_mask", q_mask.shape, "INT64"); qm.set_data_from_numpy(q_mask)
        di = InferInput("doc_ids", d_ids.shape, "INT64"); di.set_data_from_numpy(d_ids)
        dm = InferInput("doc_mask", d_mask.shape, "INT64"); dm.set_data_from_numpy(d_mask)
        res = self.client.infer(model, [qi, qm, di, dm])
        return res.as_numpy("score")

    def generate(self, model: str, prompt_ids: np.ndarray, prompt_mask: np.ndarray):
        ii = InferInput("input_ids", prompt_ids.shape, "INT64"); ii.set_data_from_numpy(prompt_ids)
        am = InferInput("attention_mask", prompt_mask.shape, "INT64"); am.set_data_from_numpy(prompt_mask)
        res = self.client.infer(model, [ii, am])
        return res.as_numpy("logits")

