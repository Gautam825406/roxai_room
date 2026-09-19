"""STTProvider interface. Implemented per-vendor in transcriber/providers/*.py (M1).

Design constraint from ARCHITECTURE: one dedicated streaming session per participant
track (this is how speaker attribution is obtained for free) -- never a single mixed
stream fed to a diarizer. `open_stream` is therefore called once per subscribed human
audio track, not once per room.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class STTResult:
    text: str
    is_final: bool
    lang: str  # BCP-47ish, e.g. "hi-en" for code-mixed, "hi", "en"
    confidence: float | None = None


class AudioFrameSource(Protocol):
    """Whatever the transcriber feeds in -- kept abstract so providers don't need to
    depend on a specific LiveKit type."""

    def __call__(self) -> AsyncIterator[bytes]: ...


class STTStream(ABC):
    """One streaming session bound to a single participant's audio track."""

    @abstractmethod
    async def push_frame(self, pcm16_bytes: bytes) -> None:
        """Feed one frame of 16-bit PCM audio."""

    @abstractmethod
    def results(self) -> AsyncIterator[STTResult]:
        """Yields partials as they arrive, then a final STTResult per utterance."""

    @abstractmethod
    async def close(self) -> None:
        ...


class STTProvider(ABC):
    """Vendor-agnostic factory for per-participant streaming STT sessions."""

    @abstractmethod
    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        ...
