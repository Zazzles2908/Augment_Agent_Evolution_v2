"""
Session persistence for Auggie CLI workflows.

Backends:
- Redis (if REDIS_URL present)
- File JSON store fallback

Public API:
- get_store()
- create_session(session_id: str, meta: dict | None = None)
- get_session(session_id: str) -> dict
- update_session(session_id: str, **kv)
- append_history(session_id: str, entry: dict)
- delete_session(session_id: str)

Thread-safe for single-process MCP via RLock.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


class _BaseStore:
    def create(self, sid: str, meta: Optional[dict] = None) -> dict: raise NotImplementedError
    def get(self, sid: str) -> dict: raise NotImplementedError
    def update(self, sid: str, **kv) -> dict: raise NotImplementedError
    def append_history(self, sid: str, entry: dict) -> dict: raise NotImplementedError
    def delete(self, sid: str) -> None: raise NotImplementedError


class _FileStore(_BaseStore):
    def __init__(self, path: Path, ttl: Optional[int] = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._ttl = ttl
        # Ensure file exists
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def _prune(self, data: dict) -> None:
        if not self._ttl:
            return
        now = int(time.time())
        to_del = []
        for sid, obj in data.items():
            ts = int(obj.get("_ts", now))
            if now - ts > self._ttl:
                to_del.append(sid)
        for sid in to_del:
            data.pop(sid, None)

    def create(self, sid: str, meta: Optional[dict] = None) -> dict:
        with self._lock:
            data = self._load()
            if sid in data:
                return data[sid]
            obj = {"session_id": sid, "history": [], "meta": meta or {}, "_ts": int(time.time())}
            data[sid] = obj
            self._prune(data)
            self._save(data)
            return obj

    def get(self, sid: str) -> dict:
        with self._lock:
            data = self._load()
            obj = data.get(sid, None)
            if obj is None:
                return {}
            # touch timestamp
            obj["_ts"] = int(time.time())
            data[sid] = obj
            self._save(data)
            return obj

    def update(self, sid: str, **kv) -> dict:
        with self._lock:
            data = self._load()
            obj = data.get(sid) or {"session_id": sid, "history": [], "meta": {}, "_ts": int(time.time())}
            obj.update(kv)
            obj["_ts"] = int(time.time())
            data[sid] = obj
            self._prune(data)
            self._save(data)
            return obj

    def append_history(self, sid: str, entry: dict) -> dict:
        with self._lock:
            data = self._load()
            obj = data.get(sid) or {"session_id": sid, "history": [], "meta": {}, "_ts": int(time.time())}
            obj["history"].append(entry)
            obj["_ts"] = int(time.time())
            data[sid] = obj
            self._prune(data)
            self._save(data)
            return obj

    def delete(self, sid: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(sid, None)
            self._save(data)


class _RedisStore(_BaseStore):
    def __init__(self, url: str, ttl: Optional[int] = None) -> None:
        import redis  # type: ignore
        self._r = redis.Redis.from_url(url)
        self._ttl = ttl

    def _key(self, sid: str) -> str:
        return f"auggie:session:{sid}"

    def create(self, sid: str, meta: Optional[dict] = None) -> dict:
        obj = {"session_id": sid, "history": [], "meta": meta or {}, "_ts": int(time.time())}
        self._r.set(self._key(sid), json.dumps(obj), ex=self._ttl)
        return obj

    def get(self, sid: str) -> dict:
        raw = self._r.get(self._key(sid))
        if not raw:
            return {}
        obj = json.loads(raw)
        if self._ttl:
            self._r.expire(self._key(sid), self._ttl)
        return obj

    def update(self, sid: str, **kv) -> dict:
        obj = self.get(sid) or {"session_id": sid, "history": [], "meta": {}, "_ts": int(time.time())}
        obj.update(kv)
        obj["_ts"] = int(time.time())
        self._r.set(self._key(sid), json.dumps(obj), ex=self._ttl)
        return obj

    def append_history(self, sid: str, entry: dict) -> dict:
        obj = self.get(sid) or {"session_id": sid, "history": [], "meta": {}, "_ts": int(time.time())}
        obj["history"].append(entry)
        obj["_ts"] = int(time.time())
        self._r.set(self._key(sid), json.dumps(obj), ex=self._ttl)
        return obj

    def delete(self, sid: str) -> None:
        self._r.delete(self._key(sid))


_store_cache: _BaseStore | None = None
_store_lock = threading.RLock()


def get_store() -> _BaseStore:
    global _store_cache
    with _store_lock:
        if _store_cache is not None:
            return _store_cache
        # Discover settings
        from auggie.config import get_auggie_settings
        s = get_auggie_settings() or {}
        sess = s.get("session") or {}
        ttl = sess.get("ttl") if isinstance(sess, dict) else None
        # Env first for Redis URL
        redis_url = os.getenv("REDIS_URL") or os.getenv("AUGGIE_REDIS_URL")
        if (sess.get("persistence") or "").lower() == "redis" and redis_url:
            try:
                _store_cache = _RedisStore(redis_url, ttl=ttl)
                return _store_cache
            except Exception:
                pass
        # File fallback
        base = Path(os.getenv("AUGGIE_SESSION_FILE", Path(__file__).parent / ".auggie_sessions.json"))
        _store_cache = _FileStore(base if isinstance(base, Path) else Path(str(base)), ttl=ttl)
        return _store_cache


# Convenience façade

def create_session(session_id: str, meta: Optional[dict] = None) -> dict:
    return get_store().create(session_id, meta)


def get_session(session_id: str) -> dict:
    return get_store().get(session_id)


def update_session(session_id: str, **kv) -> dict:
    return get_store().update(session_id, **kv)


def append_history(session_id: str, entry: dict) -> dict:
    return get_store().append_history(session_id, entry)


def delete_session(session_id: str) -> None:
    return get_store().delete(session_id)

