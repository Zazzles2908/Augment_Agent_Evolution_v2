import os
import json
import typing as t
import requests

MCP_SUPABASE_URL = os.getenv("MCP_SUPABASE_URL", "http://localhost:9108")
MCP_SUPABASE_KEY = os.getenv("MCP_SUPABASE_KEY", "")

class MCPClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or MCP_SUPABASE_URL
        self.api_key = api_key or MCP_SUPABASE_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        r = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        if r.text:
            try:
                return r.json()
            except Exception:
                return {"status": "ok"}
        return {"status": "ok"}

    # vectors
    def upsert_document_vectors(self, rows: list[dict]) -> dict:
        return self._post("/vectors/upsert_document_vectors", {"rows": rows})

    def match_documents(self, query_embedding: list[float], match_count: int = 10, similarity_threshold: float = 0.3) -> dict:
        return self._post("/vectors/match_documents", {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "similarity_threshold": similarity_threshold,
        })

    # agents
    def upsert_message(self, agent_id: str, role: str, content: str, embedding: list[float], metadata: dict | None = None) -> dict:
        return self._post("/agents/upsert_message", {
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
        })

    def match_messages(self, query_embedding: list[float], match_count: int = 10, similarity_threshold: float = 0.3) -> dict:
        return self._post("/agents/match_messages", {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "similarity_threshold": similarity_threshold,
        })

