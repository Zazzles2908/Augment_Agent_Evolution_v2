import hashlib
import json
import redis

class RedisHelper:
    def __init__(self, url: str):
        self.r = redis.from_url(url)

    def key_for(self, text: str) -> str:
        return f"emb:{hashlib.sha1(text.encode()).hexdigest()}"

    def get_embedding(self, text: str):
        v = self.r.get(self.key_for(text))
        return json.loads(v) if v else None

    def set_embedding(self, text: str, emb):
        self.r.set(self.key_for(text), json.dumps(emb))

