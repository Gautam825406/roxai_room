"""TTSProvider interface. Implemented per-vendor in bots/providers/*.py (M4).

Streaming in both directions: takes a text stream (sentence chunks from the LLM) and
yields PCM audio frames as they're synthesized, so first audio can go out before the
full reply text exists. `cancel()` must stop synthesis and any in-flight network call
promptly -- it backs barge-in's <200ms stop requirement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(frozen=True)
class AudioChunk:
    pcm16_bytes: bytes
    sample_rate: int
    num_channels: int


class TTSStream(ABC):
    @abstractmethod
    async def push_text(self, text_chunk: str) -> None:
        ...

    @abstractmethod
    async def end_input(self) -> None:
        """Signal no more text is coming; stream drains and closes after final audio."""

    @abstractmethod
    def audio(self) -> AsyncIterator[AudioChunk]:
        ...

    @abstractmethod
    async def cancel(self, reason: str) -> None:
        ...


class TTSProvider(ABC):
    @abstractmethod
    async def open_stream(self, *, voice_id: str, sample_rate: int = 48_000) -> TTSStream:
        ...
