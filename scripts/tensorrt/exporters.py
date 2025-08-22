import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class MeanPool(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        # Mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

class EmbeddingWrapper(nn.Module):
    def __init__(self, base_model_name: str, target_dim: int = 2000):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        hidden = self.model.config.hidden_size
        self.pool = MeanPool()
        # Linear projection to required output dimension (2000)
        self.proj = nn.Linear(hidden, target_dim)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        emb = self.proj(pooled)
        return emb

class RerankerWrapper(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        hidden = self.model.config.hidden_size
        self.pool = MeanPool()
        # Simple bilinear scorer over pooled representations
        self.scorer = nn.Bilinear(hidden, hidden, 1)

    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        q = self.model(input_ids=query_ids, attention_mask=query_mask)
        d = self.model(input_ids=doc_ids, attention_mask=doc_mask)
        qv = self.pool(q.last_hidden_state, query_mask)
        dv = self.pool(d.last_hidden_state, doc_mask)
        score = self.scorer(qv, dv)
        return score

class GeneratorWrapper(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

