"""Scripted Extractor/Summarizer for tests -- deterministic, no LLM call, so M2's test
suite can exercise profile/entity/summary wiring without a live provider (same pattern
as transcriber/providers/fake.py for STT).
"""
from __future__ import annotations

from roxroom.context.extractor import ExtractionResult, Extractor
from roxroom.context.models import Turn
from roxroom.context.summarizer import Summarizer


class FakeExtractor(Extractor):
    """`by_text` maps a Turn's exact text -> the ExtractionResult it should produce
    (keyed by text rather than utt_id since utt_id is generated inside RoomContext.add_turn,
    before a test could know it). Turns with no matching entry yield an empty result."""

    def __init__(self, by_text: dict[str, ExtractionResult] | None = None) -> None:
        self._by_text = by_text or {}
        self.calls: list[Turn] = []

    async def extract(self, turn: Turn, recent_turns: list[Turn]) -> ExtractionResult:
        self.calls.append(turn)
        return self._by_text.get(turn.text, ExtractionResult())


class FailingExtractor(Extractor):
    """Always raises -- for testing that a flaky extractor degrades gracefully
    (RoomContext.add_turn must still complete) instead of taking down turn
    processing."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or ConnectionError("extractor unreachable")
        self.calls: list[Turn] = []

    async def extract(self, turn: Turn, recent_turns: list[Turn]) -> ExtractionResult:
        self.calls.append(turn)
        raise self._exc


class FakeSummarizer(Summarizer):
    """Records calls and returns a deterministic, easily-asserted-on string."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Turn], str]] = []

    async def summarize(self, turns: list[Turn], previous_summary: str) -> str:
        self.calls.append((list(turns), previous_summary))
        joined = "|".join(t.text for t in turns)
        return f"{previous_summary}+[{joined}]" if previous_summary else f"[{joined}]"
