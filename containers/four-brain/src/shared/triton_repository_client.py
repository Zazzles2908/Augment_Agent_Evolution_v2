#!/usr/bin/env python3
"""
Triton Repository Client (HTTP)
- Explicit model control (list, load, unload)
- Model readiness/status queries
- Dependency-free HTTP using urllib (avoids adding requests/httpx)

Usage:
    client = TritonRepositoryClient(base_url="http://triton:8000")
    ok = client.load_model("qwen3_embedding_trt")
    ready = client.is_model_ready("qwen3_embedding_trt")
"""
from __future__ import annotations
import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TritonHTTPError(RuntimeError):
    pass


class TritonRepositoryClient:
    def __init__(self, base_url: str = "http://triton:8000", timeout_s: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # --------------- Low-level HTTP helpers ---------------
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                code = resp.getcode()
                raw = resp.read()
                if not raw:
                    return code, {}
                try:
                    return code, json.loads(raw.decode("utf-8"))
                except Exception:
                    return code, {"raw": raw.decode("utf-8", errors="ignore")}
        except urllib.error.HTTPError as e:
            try:
                payload = e.read().decode("utf-8")
            except Exception:
                payload = str(e)
            logger.error(f"Triton HTTP {method} {path} failed: {e.code} {payload}")
            raise TritonHTTPError(f"{e.code}: {payload}")
        except urllib.error.URLError as e:
            logger.error(f"Triton HTTP {method} {path} connection error: {e}")
            raise TritonHTTPError(str(e))

    def _get(self, path: str) -> Dict[str, Any]:
        _, j = self._request("GET", path)
        return j

    def _post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _, j = self._request("POST", path, body)
        return j

    # --------------- Repository / Model APIs ---------------
    def health_ready(self) -> bool:
        try:
            self._get("/v2/health/ready")
            return True
        except Exception:
            return False

    def repository_index(self) -> List[Dict[str, Any]]:
        j = self._get("/v2/repository/index")
        # Triton returns a JSON array
        if isinstance(j, list):
            return j
        return j.get("models", []) if isinstance(j, dict) else []

    def list_models(self) -> List[str]:
        return [m.get("name", "") for m in self.repository_index()]

    def load_model(self, name: str) -> bool:
        try:
            self._post(f"/v2/repository/models/{name}/load", {})
            return True
        except Exception as e:
            logger.warning(f"Load model '{name}' failed: {e}")
            return False

    def unload_model(self, name: str) -> bool:
        try:
            self._post(f"/v2/repository/models/{name}/unload", {})
            return True
        except Exception as e:
            logger.warning(f"Unload model '{name}' failed: {e}")
            return False

    def model_metadata(self, name: str) -> Dict[str, Any]:
        try:
            return self._get(f"/v2/models/{name}")
        except Exception:
            return {}

    def is_model_ready(self, name: str) -> bool:
        try:
            meta = self.model_metadata(name)
            # Some Triton versions include state under 'state' or 'ready'
            state = meta.get("state") or meta.get("status") or meta.get("ready")
            if isinstance(state, bool):
                return state
            if isinstance(state, str):
                return state.upper() == "READY"
            # Fallback check via explicit ready endpoint
            self._get(f"/v2/models/{name}/ready")
            return True
        except Exception:
            return False

    # Convenience util
    def ensure_loaded(self, name: str) -> bool:
        if self.is_model_ready(name):
            return True
        return self.load_model(name)

