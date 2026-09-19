"""Sarvam Bulbul streaming TTS provider -- our documented PRIMARY pick (DECISIONS.md)
for Indian-accent authenticity and correct Hinglish pronunciation.

NOT YET IMPLEMENTED, for the same reason as transcriber/providers/sarvam.py: the exact
streaming wire protocol (endpoint, auth, message schema) is the kind of detail worth
pulling from https://docs.sarvam.ai with a live API key in hand, not guessed at from
memory -- getting it wrong would silently produce no audio at runtime.

Wire up against `TTSProvider`/`TTSStream` in bots/tts_provider.py, following the same
shape as `ElevenLabsTTSProvider` in this package. Until then, `ElevenLabsTTSProvider`
is what `run_bot.py`/`bot_agent.py` should be configured with for a working M4 demo.
"""
from __future__ import annotations

from roxroom.bots.tts_provider import TTSProvider, TTSStream


class SarvamTTSProvider(TTSProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def open_stream(self, *, voice_id: str, sample_rate: int = 48_000) -> TTSStream:
        raise NotImplementedError(
            "SarvamTTSProvider needs its streaming protocol implemented against "
            "current Sarvam docs + a live API key before use -- see module docstring."
        )
