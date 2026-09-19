"""Per-track pipeline: audio frames + STT results -> Utterance emissions.

Deliberately decoupled from LiveKit types (`pcm16_frames` is a plain
AsyncIterator[bytes]) so it can be unit-tested with a fake VAD and a scripted STT
provider, independent of a live room or a real vendor connection. `transcriber/agent.py`
is the thin LiveKit-specific glue that feeds this from a real subscribed track.

Finalization authority: our own EndpointPolicy (local VAD silence + trailing-word
extension) decides when an utterance ends, not the vendor's own auto-endpointing --
that's the only way to get the "kya/kaise/kyun/matlab -> keep listening" behavior the
spec asks for. A provider-reported `is_final` is treated as an earlier candidate
finalize point (most providers commit a segment when *they* think the speaker paused),
whichever fires first wins.

Barge-in signal (M6): `on_speech_started`, if given, fires once per silence->speech
transition (the rising edge, not every speech frame) -- this is deliberately decoupled
from utterance finalization, since a full utterance can take 700ms-1400s+ to finalize
(the endpoint policy's silence window) but a barge-in reaction needs to start the moment
speech *begins*, not once it's over. `transcriber/agent.py` wires this to whatever
currently holds the speak-lock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from roxroom.context.models import Utterance
from roxroom.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError
from roxroom.reliability.retry import RetryError, retry_with_jitter
from roxroom.transcriber.endpointing import EndpointPolicy
from roxroom.transcriber.stt_provider import STTProvider, STTStream
from roxroom.transcriber.vad import VADLike

logger = logging.getLogger("roxroom.transcriber.pipeline")

OnUtterance = Callable[[Utterance], Awaitable[None]]
OnSpeechStarted = Callable[[], None]


async def run_track_pipeline(
    *,
    participant_id: str,
    display_name: str,
    pcm16_frames,  # AsyncIterator[bytes], 16kHz mono PCM16 frames
    stt_provider: STTProvider,
    vad: VADLike,
    on_utterance: OnUtterance,
    sample_rate: int = 16_000,
    endpoint_policy: EndpointPolicy | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_speech_started: OnSpeechStarted | None = None,
    stt_breaker: CircuitBreaker | None = None,
) -> None:
    policy = endpoint_policy or EndpointPolicy()
    breaker = stt_breaker or CircuitBreaker(f"stt-{participant_id}")

    async def open_stt() -> STTStream:
        return await stt_provider.open_stream(participant_id=participant_id, sample_rate=sample_rate)

    try:
        stt_stream = await breaker.call(lambda: retry_with_jitter(open_stt, max_attempts=3))
    except (RetryError, CircuitOpenError) as exc:
        # No second real STT provider exists to fall back to today (see
        # transcriber/providers/sarvam.py) -- the honest degrade is "this
        # participant isn't transcribed until they reconnect," logged clearly,
        # rather than crashing the worker over one bad connection.
        logger.error("could not open STT session for %s (%s) -- giving up on this track", participant_id, exc)
        return

    state: dict = {"text": "", "lang": "hi-en", "start": None}

    async def finalize(reason: str) -> None:
        text = state["text"].strip()
        if text:
            now = clock()
            utt = Utterance(
                participant_id=participant_id,
                display_name=display_name,
                text=text,
                is_final=True,
                lang=state["lang"],
                t_start=state["start"] if state["start"] is not None else now,
                t_end=now,
            )
            logger.info(
                "STT final (%s): %r",
                reason,
                text,
                extra={"trace_id": utt.utt_id, "participant_id": participant_id, "stage": "stt_final"},
            )
            try:
                await on_utterance(utt)
            except Exception:
                # Whatever on_utterance does (orchestrator, LLM, bots...) has its own
                # resilience, but this is the last line of defense: one bad utterance
                # must not kill this participant's whole transcription session.
                logger.exception("on_utterance failed for %s, continuing", utt.utt_id)
        policy.reset()
        state["text"] = ""
        state["start"] = None

    async def consume_stt_results() -> None:
        try:
            async for result in stt_stream.results():
                if not result.text.strip():
                    continue
                state["text"] = result.text
                state["lang"] = result.lang
                if state["start"] is None:
                    state["start"] = clock()
                policy.on_partial_text(result.text)
                if result.is_final:
                    await finalize("provider_final")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Mid-stream disconnect (e.g. the provider's websocket dropped). We don't
            # have a live reconnect-and-resume here, only retry-on-open (see the
            # breaker.call above) -- so the honest move is to stop this track cleanly
            # (flush whatever was already captured, below) rather than keep feeding
            # frames into a connection that's never going to answer again.
            logger.error("STT stream for %s disconnected mid-utterance, ending this track", participant_id)
            state["stt_dead"] = True

    async def drive_audio() -> None:
        was_speaking = False
        async for frame in pcm16_frames:
            if state.get("stt_dead"):
                break
            now = clock()
            try:
                await stt_stream.push_frame(frame)
            except Exception:
                logger.error("STT stream for %s failed pushing audio, ending this track", participant_id)
                state["stt_dead"] = True
                break
            if vad.is_speech(frame):
                if not was_speaking and on_speech_started is not None:
                    on_speech_started()
                was_speaking = True
                policy.on_speech_frame(now)
            else:
                was_speaking = False
                policy.on_silence_frame(now)
                if state["text"].strip() and policy.should_finalize(now):
                    await finalize("local_silence_endpoint")

    consumer = asyncio.create_task(consume_stt_results())
    try:
        await drive_audio()
    finally:
        consumer.cancel()
        await stt_stream.close()
        if state["text"].strip():
            await finalize("track_ended")
