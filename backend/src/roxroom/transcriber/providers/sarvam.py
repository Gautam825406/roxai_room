"""Sarvam AI Saarika streaming STT provider -- our documented PRIMARY pick
(DECISIONS.md) for Hindi/Hinglish code-mixing quality.

NOT YET IMPLEMENTED. Unlike Deepgram's long-stable public protocol, Sarvam's
streaming websocket contract (endpoint URL, auth header shape, and the exact JSON
message schema for partial/final results) is the kind of detail that's worth pulling
straight from https://docs.sarvam.ai (Speech-to-Text streaming) with a live API key
in hand rather than guessing at from memory -- getting it wrong would silently look
"done" while producing nothing at runtime, which is worse than an explicit gap.

Wire up against `STTProvider`/`STTStream` in transcriber/stt_provider.py, following
the same shape as `DeepgramSTTProvider` in this package. Until then, `DeepgramSTTProvider`
is what `main.py`/`agent.py` should be configured with for a working M1 demo.
"""
from __future__ import annotations

from roxroom.transcriber.stt_provider import STTProvider, STTStream


class SarvamSTTProvider(STTProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        raise NotImplementedError(
            "SarvamSTTProvider needs its streaming protocol implemented against "
            "current Sarvam docs + a live API key before use -- see module docstring."
        )
