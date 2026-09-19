"""BotAgent: one persona's LiveKit connection, identity, LLM, TTS voice, and published
audio track.

`speak(text_stream)` is the literal spec surface -- it takes a stream of text chunks
(already sentence-chunked) and doesn't care where they came from: normalizes each
chunk for TTS, feeds it into the TTS stream while holding the shared floor (SpeakLock),
publishes the resulting audio to the room, mirrors the full text into room chat and
RoomContext. `reply_to(instruction)` is the practical entrypoint the Orchestrator
actually calls: builds this persona's prompt fresh from the shared RoomContext (which
is why a multi-bot plan's second step automatically sees the first bot's reply -- by
the time it runs, RoomContext already has that turn in it) + the routed instruction,
streams a reply from the LLM, and hands it to `speak()`.

Barge-in (M6): `cancel(reason)` does four things, in order, synchronously where it
can be: (1) flips a flag both the LLM-feed and TTS-playback loops check between
chunks/audio segments so they stop pushing/playing more; (2) calls
`AudioSource.clear_queue()` immediately -- LiveKit's AudioSource buffers up to
`queue_size_ms` (default 1000ms) of already-published audio, so just stopping the feed
isn't enough, already-queued audio needs to be actively flushed or it keeps playing out;
(3) asks the TTS stream to cancel (closes its connection, stops more audio arriving);
(4) closes the LLM's async-generator text stream (`aclose()`), which aborts the
in-flight completion rather than just letting the consumer stop reading from it.
`speak()` then reports `FloorOutcome.INTERRUPTED` and writes whatever text chunks had
already been pushed to TTS before the flag was noticed into RoomContext prefixed
`[interrupted]`, so a follow-up like "ruko, simple example se samjhao" has something
coherent to resume from.

Caveat: "already pushed to TTS" is an approximation of "actually spoken" -- we don't
track sample-accurate playback position, so a chunk that was queued but whose audio
hadn't played yet (now flushed by clear_queue()) still counts as spoken here. Good
enough for a prototype's resume-coherently goal; not audio-timeline-accurate.

The <200ms target depends on the TTS provider streaming audio in small enough segments
that the playback loop's between-chunk check is actually tight -- this is written
against ElevenLabs' streaming contract (which does stream incrementally) but was never
measured against a live connection in this environment (see DECISIONS.md).

Reliability (M7), two independent fallback paths, both "degrade to a different
capability" rather than "try another same-shaped provider" (see
reliability/fallback_chain.py's docstring for why that's a deliberate choice):
- LLM unreachable (retry+circuit-breaker around fetching the first chunk, in
  `reply_to`): fall back to speaking one of the persona's own `degradation_lines`
  ("ek second, network thoda slow hai" style) instead of going silent.
- TTS unreachable (retry+circuit-breaker around opening the stream, in `speak`):
  fall back to delivering the reply as text-only in room chat -- the human still gets
  an answer, just without a voice.
Both still go through RoomContext/chat exactly like a normal reply; the only branching
is *how* the audio gets produced (or doesn't).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from livekit import rtc

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.bots.persona_loader import Persona
from roxroom.bots.text_normalizer import normalize_for_tts
from roxroom.bots.tts_provider import TTSProvider, TTSStream
from roxroom.config import BotIdentity, RoxRoomConfig
from roxroom.constants import CHAT_TOPIC
from roxroom.context.room_context import RoomContext
from roxroom.livekit_token import mint_token
from roxroom.obs.trace import Trace
from roxroom.orchestrator.speak_lock import FloorOutcome, SpeakLock
from roxroom.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError
from roxroom.reliability.retry import RetryError, retry_with_jitter

logger = logging.getLogger("roxroom.bots.bot_agent")

DEFAULT_DEGRADATION_LINE = "Ek second, thoda technical dikkat aa rahi hai."


async def _aclose_quietly(stream: AsyncIterator[str]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return  # not an async generator (e.g. a plain async iterator) -- nothing to close
    try:
        await aclose()
    except Exception:
        logger.exception("failed to close LLM text stream during barge-in")


@dataclass(frozen=True)
class SpeakResult:
    text: str
    outcome: FloorOutcome


class BotAgent:
    def __init__(
        self,
        *,
        config: RoxRoomConfig,
        identity: BotIdentity,
        persona: Persona,
        llm: LLMProvider,
        tts: TTSProvider,
        room_context: RoomContext,
        speak_lock: SpeakLock,
        sample_rate: int = 24_000,
        clock=time.monotonic,
        llm_breaker: CircuitBreaker | None = None,
        tts_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._config = config
        self._identity = identity
        self._persona = persona
        self._llm = llm
        self._tts = tts
        self._room_context = room_context
        self._speak_lock = speak_lock
        self._sample_rate = sample_rate
        self._clock = clock
        self._llm_breaker = llm_breaker or CircuitBreaker(f"{identity.identity}-llm")
        self._tts_breaker = tts_breaker or CircuitBreaker(f"{identity.identity}-tts")
        self._room = rtc.Room()
        self._audio_source: rtc.AudioSource | None = None
        self._current_tts_stream: TTSStream | None = None
        self._cancel_requested = False

    @property
    def bot_id(self) -> str:
        return self._persona.bot_id

    async def connect(self) -> None:
        token = mint_token(self._config, identity=self._identity.identity, name=self._identity.name)
        await self._room.connect(self._config.livekit_url, token)

        self._audio_source = rtc.AudioSource(self._sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(f"{self._identity.identity}-voice", self._audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self._room.local_participant.publish_track(track, options)
        logger.info("%s connected and publishing audio", self._identity.identity)

    async def aclose(self) -> None:
        await self._room.disconnect()

    async def reply_to(self, instruction: str, trace: Trace | None = None) -> SpeakResult:
        messages = self._build_messages(instruction)

        async def open_and_get_first() -> tuple[AsyncIterator[str], str | None]:
            gen = self._llm.stream_reply(messages, response_format="text")
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                first = None
            return gen, first

        try:
            gen, first_chunk = await self._llm_breaker.call(
                lambda: retry_with_jitter(open_and_get_first, max_attempts=2)
            )
        except (RetryError, CircuitOpenError) as exc:
            logger.warning("%s: LLM unavailable (%s) -- falling back to a degradation line", self.bot_id, exc)
            return await self._speak_degradation_line(trace)

        async def rest_of_stream() -> AsyncIterator[str]:
            # `finally` matters here: if THIS generator is torn down via aclose()
            # (barge-in mid-reply), Python does NOT automatically propagate that to
            # `gen` just because we stop iterating it -- without this, cancel()'s
            # "abort the LLM stream" would silently leave the real Anthropic stream
            # open. Safe to call on an already-exhausted gen too (aclose() is a no-op
            # then).
            try:
                if first_chunk is not None:
                    yield first_chunk
                async for chunk in gen:
                    yield chunk
            finally:
                await _aclose_quietly(gen)

        return await self.speak(rest_of_stream(), trace)

    async def _speak_degradation_line(self, trace: Trace | None = None) -> SpeakResult:
        line = self._persona.degradation_lines[0] if self._persona.degradation_lines else DEFAULT_DEGRADATION_LINE

        async def one_chunk() -> AsyncIterator[str]:
            yield line

        return await self.speak(one_chunk(), trace)

    async def speak(self, text_stream: AsyncIterator[str], trace: Trace | None = None) -> SpeakResult:
        self._cancel_requested = False
        full_text_parts: list[str] = []

        async def _do_speak() -> FloorOutcome | None:
            try:
                tts_stream = await self._tts_breaker.call(
                    lambda: retry_with_jitter(
                        lambda: self._tts.open_stream(
                            voice_id=self._persona.voice.voice_id, sample_rate=self._sample_rate
                        ),
                        max_attempts=2,
                    )
                )
            except (RetryError, CircuitOpenError) as exc:
                logger.warning("%s: TTS unavailable (%s) -- falling back to chat-only reply", self.bot_id, exc)
                try:
                    async for chunk in text_stream:
                        if trace is not None:
                            trace.mark("llm_first_token")
                        full_text_parts.append(chunk)
                except Exception:
                    logger.exception("%s: LLM stream also failed during chat-only fallback", self.bot_id)
                    return FloorOutcome.FAILED
                return None

            self._current_tts_stream = tts_stream
            feed_failed = False

            async def feed_text() -> None:
                nonlocal feed_failed
                try:
                    async for chunk in text_stream:
                        if trace is not None:
                            trace.mark("llm_first_token")
                        if self._cancel_requested:
                            break
                        full_text_parts.append(chunk)
                        await tts_stream.push_text(normalize_for_tts(chunk))
                    if self._cancel_requested:
                        await _aclose_quietly(text_stream)  # abort the LLM stream, don't just stop reading it
                    else:
                        await tts_stream.end_input()
                except Exception:
                    # A mid-stream LLM/TTS-push failure (not just "unreachable at the
                    # start", which the retry/breaker above already handles): stop
                    # play_audio too via the same flag barge-in uses, so it doesn't
                    # keep waiting on a stream that's never getting more input.
                    feed_failed = True
                    self._cancel_requested = True
                    logger.exception("%s: reply failed mid-stream", self.bot_id)
                    await _aclose_quietly(text_stream)

            async def play_audio() -> None:
                async for audio_chunk in tts_stream.audio():
                    if trace is not None:
                        trace.mark("tts_first_byte")
                    if self._cancel_requested:
                        break
                    await self._publish_audio_chunk(audio_chunk, trace)

            await asyncio.gather(feed_text(), play_audio())
            self._current_tts_stream = None
            if feed_failed:
                return FloorOutcome.FAILED
            return FloorOutcome.INTERRUPTED if self._cancel_requested else None

        outcome = await self._speak_lock.speak_turn(self.bot_id, _do_speak)
        full_text = "".join(full_text_parts).strip()

        if full_text:
            if outcome == FloorOutcome.INTERRUPTED:
                stored_text = f"[interrupted] {full_text}"
            elif outcome == FloorOutcome.FAILED:
                # Distinct from [interrupted] (a human cut it off on purpose) -- this
                # was a system failure mid-reply, so the LLM shouldn't treat it as a
                # deliberately-brief answer to build on, just an incomplete one.
                stored_text = f"[error] {full_text}"
            else:
                stored_text = full_text
            await self._room_context.add_turn(
                speaker_id=self._identity.identity,
                display_name=self._identity.name,
                role="bot",
                text=stored_text,
                ts=self._clock(),
                modality="voice",
            )
            try:
                await self._room.local_participant.send_text(full_text, topic=CHAT_TOPIC)
            except Exception:
                logger.exception("%s: failed to publish reply to room chat", self.bot_id)

        if trace is not None:
            trace.bot_id = self.bot_id
            trace.log()

        return SpeakResult(text=full_text, outcome=outcome)

    def cancel(self, reason: str) -> None:
        """Barge-in entrypoint: called the moment human VAD fires while this bot holds
        the floor. Everything here is either synchronous or fire-and-forget so this
        returns immediately -- it does not wait for the in-flight speak() to actually
        unwind."""
        if self._speak_lock.holder != self.bot_id:
            return  # nothing in flight to cancel
        logger.info("%s: cancel requested (%s)", self.bot_id, reason)
        self._cancel_requested = True
        if self._audio_source is not None:
            self._audio_source.clear_queue()  # flush already-queued audio, not just stop feeding more
        if self._current_tts_stream is not None:
            asyncio.ensure_future(self._current_tts_stream.cancel(reason))

    async def _publish_audio_chunk(self, chunk, trace: Trace | None = None) -> None:
        assert self._audio_source is not None
        samples_per_channel = len(chunk.pcm16_bytes) // 2 // chunk.num_channels
        if samples_per_channel == 0:
            return
        frame = rtc.AudioFrame.create(chunk.sample_rate, chunk.num_channels, samples_per_channel)
        dest = np.frombuffer(frame.data, dtype=np.int16)
        src = np.frombuffer(chunk.pcm16_bytes, dtype=np.int16)
        dest[: len(src)] = src
        await self._audio_source.capture_frame(frame)
        if trace is not None:
            trace.mark("audio_published")

    def _build_messages(self, instruction: str) -> list[ChatMessage]:
        prompt_ctx = self._room_context.build_prompt_context()
        lines: list[str] = []
        if prompt_ctx.rolling_summary:
            lines.append(f"SUMMARY SO FAR: {prompt_ctx.rolling_summary}")
        entities_line = prompt_ctx.recent_entities_line()
        if entities_line:
            lines.append(entities_line)
        for turn in prompt_ctx.recent_turns:
            lines.append(f"{turn.display_name}: {turn.text}")

        context_block = "\n".join(lines)
        user_content = f"{context_block}\n\nRespond to: {instruction}" if context_block else instruction

        return [
            ChatMessage(role="system", content=self._persona.system_prompt),
            ChatMessage(role="user", content=user_content),
        ]
