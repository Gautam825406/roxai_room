"""The floor: one asyncio.Lock shared across both BotAgents. A bot may only synthesize
speech while holding it -- this is what the "never both speaking at once without an
explicit plan" constraint is built on. Releasing it emits `floor_released(bot,
outcome)` to any registered listener (the Orchestrator uses this to drive the
suppression window and alternation fallback via `on_bot_reply_completed`).
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Awaitable, Callable, Union

logger = logging.getLogger("roxroom.orchestrator.speak_lock")


class FloorOutcome(str, Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


FloorReleasedCallback = Union[
    Callable[[str, FloorOutcome], Awaitable[None]],
    Callable[[str, FloorOutcome], None],
]


class SpeakLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._holder: str | None = None
        self._listeners: list[FloorReleasedCallback] = []

    @property
    def holder(self) -> str | None:
        return self._holder

    def on_floor_released(self, callback: FloorReleasedCallback) -> None:
        self._listeners.append(callback)

    async def acquire(self, bot: str) -> None:
        await self._lock.acquire()
        self._holder = bot

    def release(self, bot: str, outcome: FloorOutcome) -> None:
        if self._holder != bot:
            logger.warning("release() called by %r but floor is held by %r", bot, self._holder)
        self._holder = None
        self._lock.release()
        logger.info("floor released by %s: %s", bot, outcome.value)
        for listener in self._listeners:
            result = listener(bot, outcome)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)

    async def speak_turn(self, bot: str, fn: Callable[[], Awaitable[FloorOutcome | None]]) -> FloorOutcome:
        """Acquire, run fn(), and always release with the right outcome -- even if
        fn() raises or is cancelled (real task cancellation).

        `fn` may return a `FloorOutcome` explicitly (BotAgent's flag-based barge-in
        cancellation isn't a real `asyncio.CancelledError` -- it exits its loops
        cleanly and needs to report INTERRUPTED itself); returning None keeps the
        default exception-based inference (COMPLETED, or FAILED/INTERRUPTED on an
        exception)."""
        await self.acquire(bot)
        outcome = FloorOutcome.COMPLETED
        try:
            result = await fn()
            if isinstance(result, FloorOutcome):
                outcome = result
        except asyncio.CancelledError:
            outcome = FloorOutcome.INTERRUPTED
            raise
        except Exception:
            outcome = FloorOutcome.FAILED
            logger.exception("%s: speak turn failed", bot)
        finally:
            self.release(bot, outcome)
        return outcome
