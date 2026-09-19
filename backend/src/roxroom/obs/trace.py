"""Trace: one bot reply's lifecycle timestamps, from STT-final through to the first
published audio frame.

`trace_id` is just the triggering utterance's own `utt_id` -- it's already a unique
per-utterance identifier, no need to invent a second one. Spans recorded, in the order
the spec asks for: `stt_final` -> `gate_route` (Orchestrator's should_respond +
routing decision, measured as one span since they execute as one synchronous-ish call,
not separately awaited/streamed like the LLM/TTS steps) -> `llm_first_token` ->
`tts_first_byte` -> `audio_published`.

`mark()` is idempotent per stage: only the first call for a given stage is kept, since
"first token"/"first byte" is what the spec and the latency target care about, not
every token. For a multi-bot plan, only the first step gets a Trace (see
plan_executor.py) -- the <1.5s end-of-speech-to-first-audio target is about the room's
first response, not every subsequent bot's turn.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("roxroom.obs.trace")

STAGES = ("stt_final", "gate_route", "llm_first_token", "tts_first_byte", "audio_published")


@dataclass
class Trace:
    trace_id: str
    participant_id: str
    bot_id: str | None = None
    clock: Callable[[], float] = time.monotonic
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, at: float | None = None) -> None:
        if stage in self.marks:
            return
        self.marks[stage] = at if at is not None else self.clock()

    def duration_ms(self, from_stage: str, to_stage: str) -> float | None:
        start = self.marks.get(from_stage)
        end = self.marks.get(to_stage)
        if start is None or end is None:
            return None
        return (end - start) * 1000

    @property
    def end_of_speech_to_first_audio_ms(self) -> float | None:
        return self.duration_ms("stt_final", "audio_published")

    def breakdown(self) -> dict[str, float | None]:
        """Per-stage elapsed time since stt_final (ms), plus the headline
        end-of-speech -> first-audio figure. A stage that never happened (e.g. no
        audio because the reply degraded to chat-only) is None, not zero."""
        base = self.marks.get("stt_final")
        result: dict[str, float | None] = {}
        for stage in STAGES:
            t = self.marks.get(stage)
            result[f"{stage}_ms"] = (t - base) * 1000 if (t is not None and base is not None) else None
        result["total_ms"] = self.end_of_speech_to_first_audio_ms
        return result

    def log(self) -> None:
        logger.info(
            "latency breakdown for %s",
            self.trace_id,
            extra={
                "trace_id": self.trace_id,
                "participant_id": self.participant_id,
                "bot_id": self.bot_id,
                **self.breakdown(),
            },
        )
