"""Scripted STTProvider for tests -- ignores actual audio content, replays a fixed
sequence of STTResult per participant_id. Lets us unit-test the track pipeline's
finalize/dedupe/timing logic without a live vendor connection or real speech audio.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from roxroom.transcriber.stt_provider import STTProvider, STTResult, STTStream


class ScriptedSTTStream(STTStream):
    def __init__(self, script: list[STTResult], result_delay: float = 0.0) -> None:
        self._queue: asyncio.Queue[STTResult] = asyncio.Queue()
        self._emit_task = asyncio.create_task(self._emit(script, result_delay))

    async def _emit(self, script: list[STTResult], delay: float) -> None:
        for result in script:
            if delay:
                await asyncio.sleep(delay)
            await self._queue.put(result)

    async def push_frame(self, pcm16_bytes: bytes) -> None:
        pass  # scripted provider doesn't look at audio

    def results(self) -> AsyncIterator[STTResult]:
        async def gen() -> AsyncIterator[STTResult]:
            while True:
                yield await self._queue.get()

        return gen()

    async def close(self) -> None:
        self._emit_task.cancel()


class FakeSTTProvider(STTProvider):
    """`scripts` maps participant_id -> the STTResult sequence that participant's
    stream should replay."""

    def __init__(self, scripts: dict[str, list[STTResult]], result_delay: float = 0.0) -> None:
        self._scripts = scripts
        self._result_delay = result_delay

    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        return ScriptedSTTStream(self._scripts.get(participant_id, []), self._result_delay)


class FailingSTTProvider(STTProvider):
    """Every open_stream() call raises -- simulates a fully unreachable STT provider
    (M7 reliability tests: run_track_pipeline should degrade to giving up on the track,
    not crash the worker)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or ConnectionError("stt unreachable")
        self.call_count = 0

    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        self.call_count += 1
        raise self._exc
