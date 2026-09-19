"""Scripted LLM/TTS providers for tests -- deterministic, no network (same pattern as
the fakes in transcriber/, context/, orchestrator/)."""
from __future__ import annotations

from typing import AsyncIterator, Literal

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.bots.tts_provider import AudioChunk, TTSProvider, TTSStream


class FakeLLMProvider(LLMProvider):
    """`on_chunk`, if given, is called right after each chunk is handed to the
    consumer -- useful for tests that need to trigger something (like a barge-in
    cancel()) partway through a reply. `aborted` flips to True if the stream was
    ever `.aclose()`d before running to completion (i.e. GeneratorExit reached us
    mid-loop), which is how tests verify barge-in actually aborts the LLM stream
    rather than just stopping reading from it."""

    def __init__(self, reply_chunks: list[str], on_chunk=None) -> None:
        self._reply_chunks = reply_chunks
        self._on_chunk = on_chunk
        self.calls: list[list[ChatMessage]] = []
        self.aborted = False

    async def stream_reply(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        try:
            for chunk in self._reply_chunks:
                yield chunk
                if self._on_chunk:
                    self._on_chunk(chunk)
        except GeneratorExit:
            self.aborted = True
            raise


class FailingLLMProvider(LLMProvider):
    """Every call raises -- simulates a fully unreachable LLM provider (M7 reliability
    tests: BotAgent should fall back to a degradation line, not crash)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or ConnectionError("llm unreachable")
        self.call_count = 0

    async def stream_reply(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[str]:
        self.call_count += 1
        raise self._exc
        yield  # pragma: no cover -- unreachable; makes this an async generator function


class FailingTTSProvider(TTSProvider):
    """Every call raises -- simulates a fully unreachable TTS provider (M7 reliability
    tests: BotAgent should fall back to a chat-only reply, not crash)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or ConnectionError("tts unreachable")
        self.call_count = 0

    async def open_stream(self, *, voice_id: str, sample_rate: int = 24_000) -> TTSStream:
        self.call_count += 1
        raise self._exc


class FakeTTSStream(TTSStream):
    def __init__(self, sample_rate: int) -> None:
        self.pushed_text: list[str] = []
        self._sample_rate = sample_rate
        self._chunks: list[AudioChunk] = []
        self.cancelled_reason: str | None = None
        self.ended = False

    async def push_text(self, text_chunk: str) -> None:
        self.pushed_text.append(text_chunk)
        # emit a tiny fake audio chunk per pushed text so play_audio has something to consume
        self._chunks.append(AudioChunk(pcm16_bytes=b"\x00\x00" * 4, sample_rate=self._sample_rate, num_channels=1))

    async def end_input(self) -> None:
        self.ended = True

    def audio(self) -> AsyncIterator[AudioChunk]:
        async def gen() -> AsyncIterator[AudioChunk]:
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def cancel(self, reason: str) -> None:
        self.cancelled_reason = reason


class FakeTTSProvider(TTSProvider):
    def __init__(self) -> None:
        self.streams: list[FakeTTSStream] = []

    async def open_stream(self, *, voice_id: str, sample_rate: int = 24_000) -> TTSStream:
        stream = FakeTTSStream(sample_rate)
        self.streams.append(stream)
        return stream
