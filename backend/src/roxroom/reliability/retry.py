"""Retry-with-jitter for transient provider failures: exponential backoff with full
jitter (delay = uniform(0, min(max_delay, base * 2**(attempt-1)))) so multiple
concurrent retries against the same flaky endpoint don't all resync onto the same
retry cadence.

`sleep`/`rand` are injectable so tests can run this at full speed and deterministically
instead of doing real backoff sleeps.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger("roxroom.reliability.retry")

T = TypeVar("T")


class RetryError(Exception):
    """All retry attempts were exhausted; wraps the last underlying exception."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        super().__init__(f"gave up after {attempts} attempt(s): {last_exception!r}")
        self.attempts = attempts
        self.last_exception = last_exception


async def retry_with_jitter(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exception: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            last_exception = exc
            if attempt == max_attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            jittered = delay * rand()
            logger.warning("attempt %d/%d failed (%r), retrying in %.2fs", attempt, max_attempts, exc, jittered)
            if on_retry:
                on_retry(attempt, exc)
            await sleep(jittered)

    assert last_exception is not None
    raise RetryError(max_attempts, last_exception)
