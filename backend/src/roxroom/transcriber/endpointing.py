"""Pure endpointing decision logic -- no audio, no I/O, so it's unit-testable with
synthetic timestamps.

Rule: finalize an utterance after ~600-800ms of silence, but extend the wait if the
latest partial transcript ends in a word that signals the speaker isn't done
(a trailing question/discourse marker like "kya", "kaise", "kyun", "matlab").
"""
from __future__ import annotations

TRAILING_EXTEND_WORDS = {"kya", "kaise", "kyun", "kyu", "matlab", "toh", "ki"}

DEFAULT_BASE_SILENCE_MS = 700
DEFAULT_EXTENDED_SILENCE_MS = 1400


class EndpointPolicy:
    """Feed it speech/silence frame events (with monotonic timestamps) and the latest
    partial STT text; ask `should_finalize(now)` on each tick to decide whether the
    current utterance has ended.
    """

    def __init__(
        self,
        base_silence_ms: int = DEFAULT_BASE_SILENCE_MS,
        extended_silence_ms: int = DEFAULT_EXTENDED_SILENCE_MS,
    ) -> None:
        self.base_silence_ms = base_silence_ms
        self.extended_silence_ms = extended_silence_ms
        self._silence_started_at: float | None = None
        self._latest_partial_text = ""

    def on_speech_frame(self, t: float) -> None:
        self._silence_started_at = None

    def on_silence_frame(self, t: float) -> None:
        if self._silence_started_at is None:
            self._silence_started_at = t

    def on_partial_text(self, text: str) -> None:
        self._latest_partial_text = text

    def should_finalize(self, now: float) -> bool:
        if self._silence_started_at is None:
            return False
        elapsed_ms = (now - self._silence_started_at) * 1000
        return elapsed_ms >= self._required_silence_ms()

    def reset(self) -> None:
        self._silence_started_at = None
        self._latest_partial_text = ""

    def _required_silence_ms(self) -> int:
        if self._last_word(self._latest_partial_text) in TRAILING_EXTEND_WORDS:
            return self.extended_silence_ms
        return self.base_silence_ms

    @staticmethod
    def _last_word(text: str) -> str:
        words = text.strip().lower().split()
        return words[-1] if words else ""
