"""ElevenLabs Flash v2.5 streaming TTS provider.

Per DECISIONS.md, Sarvam Bulbul is the primary pick for Indian-accent authenticity,
but its adapter isn't implemented yet for the same reason as the STT side (see
transcriber/providers/sarvam.py's docstring: the wire protocol should be pulled from
live docs with a real key in hand, not guessed at). ElevenLabs is what's actually wired
up and runnable for this milestone, on a well-documented, stable protocol.

Protocol: one WebSocket connection per TTSStream, at
wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input. The first message
carries voice_settings + the API key; each `push_text` call sends `{"text": "..."}`;
`end_input` sends `{"text": ""}` to flush and close the generation. The server sends
back `{"audio": "<base64 PCM>", "isFinal": bool}` messages. Verify field names against
https://elevenlabs.io/docs/api-reference/websockets before pinning a model/output-format
combination that differs from what's used here.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

from roxroom.bots.tts_provider import AudioChunk, TTSProvider, TTSStream

logger = logging.getLogger("roxroom.bots.elevenlabs")

ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

# Opus (what LiveKit/WebRTC publishes) supports 8k/12k/16k/24k/48k natively -- 24kHz is
# a good balance of quality vs bandwidth for speech and is one of ElevenLabs' supported
# pcm_* output rates, unlike the TTSProvider interface's generic 48000 default.
DEFAULT_SAMPLE_RATE = 24_000


class ElevenLabsTTSStream(TTSStream):
    def __init__(self, ws, sample_rate: int) -> None:
        self._ws = ws
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def push_text(self, text_chunk: str) -> None:
        await self._ws.send(json.dumps({"text": text_chunk, "try_trigger_generation": True}))

    async def end_input(self) -> None:
        await self._ws.send(json.dumps({"text": ""}))

    def audio(self) -> AsyncIterator[AudioChunk]:
        async def gen() -> AsyncIterator[AudioChunk]:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                yield item

        return gen()

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                audio_b64 = data.get("audio")
                if audio_b64:
                    pcm = base64.b64decode(audio_b64)
                    await self._queue.put(
                        AudioChunk(pcm16_bytes=pcm, sample_rate=self._sample_rate, num_channels=1)
                    )
                if data.get("isFinal"):
                    break
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            logger.info("elevenlabs connection closed")
        except Exception:
            logger.exception("elevenlabs recv loop crashed")
        finally:
            await self._queue.put(None)

    async def cancel(self, reason: str) -> None:
        logger.info("cancelling elevenlabs stream: %s", reason)
        self._recv_task.cancel()
        await self._queue.put(None)
        try:
            await self._ws.close()
        except Exception:
            pass


class ElevenLabsTTSProvider(TTSProvider):
    def __init__(self, api_key: str, *, model_id: str = "eleven_flash_v2_5") -> None:
        self._api_key = api_key
        self._model_id = model_id

    async def open_stream(self, *, voice_id: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> TTSStream:
        params = {"model_id": self._model_id, "output_format": f"pcm_{sample_rate}"}
        url = f"{ELEVENLABS_WS_URL.format(voice_id=voice_id)}?{urlencode(params)}"
        ws = await websockets.connect(url)
        await ws.send(
            json.dumps(
                {
                    "text": " ",
                    # Lower stability than the 0.5 default trades a little consistency
                    # for natural pitch/pace variation -- at 0.5+ ElevenLabs voices read
                    # noticeably flatter/more robotic, which fights the personas' whole
                    # "sounds like an actual person talking" goal. style/speaker_boost
                    # push further the same direction (more expressive, fuller-sounding
                    # voice) and are supported by eleven_flash_v2_5, not just non-Flash
                    # models -- see https://elevenlabs.io/docs/api-reference/websockets.
                    "voice_settings": {
                        "stability": 0.35,
                        "similarity_boost": 0.8,
                        "style": 0.25,
                        "use_speaker_boost": True,
                    },
                    "xi_api_key": self._api_key,
                }
            )
        )
        logger.info("elevenlabs stream opened for voice %s", voice_id)
        return ElevenLabsTTSStream(ws, sample_rate)
