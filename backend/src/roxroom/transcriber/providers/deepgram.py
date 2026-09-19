"""Deepgram Nova-3 streaming STT provider (fallback per DECISIONS.md -- Sarvam
Saarika is primary for accent/code-mixing quality, this is the well-documented,
standardized-protocol fallback).

Protocol: one WebSocket connection per participant track, raw PCM16 binary frames
sent as they arrive, JSON `Results` messages received back. This is Deepgram's
stable streaming contract as of Nova-3; verify query params against
https://developers.deepgram.com/reference/listen-live if pinning a specific
`model`/`language` combination differs from what's used here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

from roxroom.transcriber.stt_provider import STTProvider, STTResult, STTStream

logger = logging.getLogger("roxroom.transcriber.deepgram")

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTTStream(STTStream):
    def __init__(self, ws) -> None:
        self._ws = ws
        self._queue: asyncio.Queue[STTResult] = asyncio.Queue()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def push_frame(self, pcm16_bytes: bytes) -> None:
        await self._ws.send(pcm16_bytes)

    def results(self) -> AsyncIterator[STTResult]:
        async def gen() -> AsyncIterator[STTResult]:
            while True:
                yield await self._queue.get()

        return gen()

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                if data.get("type") != "Results":
                    continue
                alternatives = data.get("channel", {}).get("alternatives", [])
                if not alternatives:
                    continue
                text = alternatives[0].get("transcript", "")
                if not text:
                    continue
                is_final = bool(data.get("is_final") or data.get("speech_final"))
                await self._queue.put(
                    STTResult(
                        text=text,
                        is_final=is_final,
                        lang="hi-en",
                        confidence=alternatives[0].get("confidence"),
                    )
                )
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            logger.info("deepgram connection closed")
        except Exception:
            logger.exception("deepgram recv loop crashed")

    async def close(self) -> None:
        self._recv_task.cancel()
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        await self._ws.close()


class DeepgramSTTProvider(STTProvider):
    def __init__(self, api_key: str, *, model: str = "nova-3", language: str = "multi") -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        params = {
            "model": self._model,
            "language": self._language,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
        }
        url = f"{DEEPGRAM_WS_URL}?{urlencode(params)}"
        ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )
        logger.info("deepgram stream opened for %s", participant_id)
        return DeepgramSTTStream(ws)
