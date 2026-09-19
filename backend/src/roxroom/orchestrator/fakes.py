"""Scripted StageBGate for tests -- no LLM call, deterministic verdicts keyed by the
utterance's exact text (same pattern as the fakes in transcriber/ and context/)."""
from __future__ import annotations

from roxroom.context.models import Turn, Utterance
from roxroom.orchestrator.gate import GateVerdict, StageBGate


class FakeStageBGate(StageBGate):
    def __init__(self, by_text: dict[str, GateVerdict] | None = None) -> None:
        self._by_text = by_text or {}
        self.calls: list[Utterance] = []

    async def classify(self, utterance: Utterance, recent_turns: list[Turn]) -> GateVerdict:
        self.calls.append(utterance)
        return self._by_text.get(
            utterance.text, GateVerdict(respond=False, addressed_bot="none", reason="unscripted", confidence=0.0)
        )
