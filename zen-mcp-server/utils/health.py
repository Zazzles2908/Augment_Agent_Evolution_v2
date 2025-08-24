"""
Provider health and circuit breaker (opt-in).
Safe-by-default: logging only until explicitly enabled.
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class CircuitConfig:
    failure_threshold: int = 3
    half_open_probe_sec: float = 30.0

class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, cfg: CircuitConfig | None = None):
        self.cfg = cfg or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failures = 0
        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        # Allow in CLOSED or HALF_OPEN; deny in OPEN
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.cfg.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning("Circuit opened after %d failures", self.failures)

    async def half_open_after_delay(self):
        await asyncio.sleep(self.cfg.half_open_probe_sec)
        async with self._lock:
            if self.state == CircuitState.OPEN:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit moving to HALF_OPEN for probe")

class ProviderHealth:
    def __init__(self, name: str, breaker: CircuitBreaker | None = None):
        self.name = name
        self.breaker = breaker or CircuitBreaker()

    async def is_available(self) -> bool:
        return await self.breaker.allow_request()

    async def _half_open_wrapper(self):
        try:
            await self.breaker.half_open_after_delay()
        finally:
            try:
                from utils.metrics import set_circuit_state
                set_circuit_state(self.name, self.breaker.state)
            except Exception:
                pass

    async def record_result(self, ok: bool) -> None:
        if ok:
            await self.breaker.record_success()
            try:
                from utils.metrics import set_circuit_state
                set_circuit_state(self.name, "closed")
            except Exception:
                pass
        else:
            prev_state = self.breaker.state
            await self.breaker.record_failure()
            try:
                from utils.metrics import set_circuit_state
                set_circuit_state(self.name, "open")
            except Exception:
                pass
            # If we transitioned to OPEN just now, schedule HALF_OPEN after delay
            if prev_state != CircuitState.OPEN and self.breaker.state == CircuitState.OPEN:
                try:
                    asyncio.create_task(self._half_open_wrapper())
                except Exception:
                    pass

class HealthManager:
    def __init__(self):
        self._providers: Dict[str, ProviderHealth] = {}

    def get(self, name: str) -> ProviderHealth:
        if name not in self._providers:
            self._providers[name] = ProviderHealth(name)
        return self._providers[name]

