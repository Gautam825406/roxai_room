"""Generic fallback chain: try named providers in order, each attempt going through
retry-with-jitter and its own circuit breaker (so a provider that's been failing
repeatedly gets skipped fast via its open circuit instead of burning a full retry
budget on every single call), until one succeeds.

Not yet wired up anywhere with a real second link -- we only have one working
implementation per stage today (Deepgram for STT, ElevenLabs for TTS, Anthropic for
LLM; see DECISIONS.md for why Sarvam's adapters are stubs). This module exists and is
tested so that adding a second real provider later is a matter of appending a
`FallbackLink`, not building the mechanism from scratch. BotAgent's LLM-down ->
degradation-line and TTS-down -> chat-only-reply fallbacks are bespoke instead of using
this, because they degrade to a *different capability* (text instead of voice), not
another same-shaped provider -- see bot_agent.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Generic, TypeVar

from roxroom.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError
from roxroom.reliability.retry import RetryError, retry_with_jitter

logger = logging.getLogger("roxroom.reliability.fallback_chain")

T = TypeVar("T")


@dataclass
class FallbackLink(Generic[T]):
    name: str
    call: Callable[[], Awaitable[T]]
    breaker: CircuitBreaker | None = None
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if self.breaker is None:
            self.breaker = CircuitBreaker(self.name)


class AllProvidersFailedError(Exception):
    def __init__(self, attempted: list[str]) -> None:
        super().__init__(f"all providers failed: {', '.join(attempted)}")
        self.attempted = attempted


async def call_with_fallback(links: list[FallbackLink[T]]) -> T:
    attempted: list[str] = []
    for link in links:
        attempted.append(link.name)
        assert link.breaker is not None
        try:
            return await link.breaker.call(lambda link=link: retry_with_jitter(link.call, max_attempts=link.max_attempts))
        except CircuitOpenError:
            logger.warning("skipping %r: circuit open", link.name)
        except RetryError as exc:
            logger.warning("%r exhausted retries: %s", link.name, exc)
    raise AllProvidersFailedError(attempted)
