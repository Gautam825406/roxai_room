"""Circuit breaker: after `failure_threshold` consecutive failures, stop even trying a
provider for `reset_timeout_s` (fail fast instead of piling up slow timeouts on a
provider that's clearly down), then allow one half-open probe -- success closes the
circuit again, failure re-opens it for another cooldown window.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger("roxroom.reliability.circuit_breaker")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"circuit {name!r} is open -- refusing to call")
        self.name = name


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self._reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
                logger.info("circuit %r half-open, allowing one probe", self.name)
        return self._state

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        current = self.state
        if current == CircuitState.OPEN:
            raise CircuitOpenError(self.name)
        try:
            result = await fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        if self._state != CircuitState.CLOSED:
            logger.info("circuit %r closed after a successful call", self.name)
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        should_open = self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self._failure_threshold
        if should_open:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "circuit %r opening after %d consecutive failure(s)", self.name, self._consecutive_failures
                )
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
