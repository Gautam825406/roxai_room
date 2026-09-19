import asyncio

import pytest

from roxroom.reliability.circuit_breaker import CircuitBreaker
from roxroom.transcriber.endpointing import EndpointPolicy
from roxroom.transcriber.providers.fake import FailingSTTProvider, FakeSTTProvider
from roxroom.transcriber.stt_provider import STTResult
from roxroom.transcriber.track_pipeline import run_track_pipeline


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeVAD:
    """Replays a scripted is_speech() sequence; True for as long as the script lasts,
    then defaults to `tail`."""

    def __init__(self, script: list[bool], tail: bool = False) -> None:
        self._script = list(script)
        self._tail = tail

    def is_speech(self, pcm16_bytes: bytes) -> bool:
        if self._script:
            return self._script.pop(0)
        return self._tail

    def reset(self) -> None:
        pass


async def _frames(n: int, clock: FakeClock, step: float = 0.032):
    for _ in range(n):
        clock.advance(step)
        yield b"\x00\x00"
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_provider_final_emits_utterance_immediately():
    clock = FakeClock()
    provider = FakeSTTProvider(
        {
            "p1": [
                STTResult(text="mujhe cricket pasand hai", is_final=False, lang="hi-en"),
                STTResult(text="mujhe cricket pasand hai", is_final=True, lang="hi-en"),
            ]
        }
    )
    vad = FakeVAD(script=[], tail=True)  # constant speech; local endpoint should never fire
    emitted = []

    async def on_utt(u):
        emitted.append(u)

    await run_track_pipeline(
        participant_id="p1",
        display_name="Priya",
        pcm16_frames=_frames(10, clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        endpoint_policy=EndpointPolicy(base_silence_ms=700),
        clock=clock,
    )

    assert len(emitted) == 1
    assert emitted[0].text == "mujhe cricket pasand hai"
    assert emitted[0].is_final is True
    assert emitted[0].participant_id == "p1"


@pytest.mark.asyncio
async def test_local_silence_endpoint_finalizes_when_provider_never_declares_final():
    clock = FakeClock()
    # Provider only ever sends partials -- local VAD-driven endpointing must finalize.
    provider = FakeSTTProvider(
        {"p1": [STTResult(text="haan wahi topic", is_final=False, lang="hi-en")]}
    )
    # 3 speech frames, then silence for the rest (well past the 700ms base window at 32ms/frame).
    vad = FakeVAD(script=[True, True, True], tail=False)
    emitted = []

    async def on_utt(u):
        emitted.append(u)

    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_frames(40, clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        endpoint_policy=EndpointPolicy(base_silence_ms=700),
        clock=clock,
    )

    assert len(emitted) == 1
    assert emitted[0].text == "haan wahi topic"


@pytest.mark.asyncio
async def test_trailing_question_word_delays_local_finalize():
    clock = FakeClock()
    provider = FakeSTTProvider(
        {"p1": [STTResult(text="tum kal aa rahe ho kya", is_final=False, lang="hi-en")]}
    )
    vad = FakeVAD(script=[True], tail=False)
    emitted = []
    emitted_at_clock = []

    async def on_utt(u):
        emitted_at_clock.append(clock.t)
        emitted.append(u)

    # 30 silence frames * 32ms = ~960ms: past the 700ms base window, short of the
    # 1400ms extended window. If the trailing-word extension weren't applied, the
    # local endpoint would fire at ~700ms (frame ~22); with it, nothing fires until
    # the frame generator is exhausted and teardown flushes it at ~960ms.
    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_frames(30, clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        endpoint_policy=EndpointPolicy(base_silence_ms=700, extended_silence_ms=1400),
        clock=clock,
    )

    assert len(emitted) == 1
    assert emitted[0].text == "tum kal aa rahe ho kya"
    assert emitted_at_clock[0] > 0.9  # proves it did NOT finalize at the ~700ms base mark


@pytest.mark.asyncio
async def test_on_speech_started_fires_once_per_onset_not_per_frame():
    clock = FakeClock()
    provider = FakeSTTProvider({"p1": []})
    # two separate speech "runs" (onset, hold, offset) separated by silence
    vad_script = [True, True, True, False, False, False, True, True, False, False]
    vad = FakeVAD(script=vad_script, tail=False)
    onsets: list[None] = []

    async def on_utt(u):
        pass

    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_frames(len(vad_script), clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        on_speech_started=lambda: onsets.append(None),
        clock=clock,
    )

    assert len(onsets) == 2  # one per silence->speech transition, not one per speech frame


@pytest.mark.asyncio
async def test_on_speech_started_never_fires_when_track_is_always_silent():
    clock = FakeClock()
    provider = FakeSTTProvider({"p1": []})
    vad = FakeVAD(script=[], tail=False)
    onsets: list[None] = []

    async def on_utt(u):
        pass

    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_frames(10, clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        on_speech_started=lambda: onsets.append(None),
        clock=clock,
    )

    assert onsets == []


@pytest.mark.asyncio
async def test_pipeline_works_without_on_speech_started_configured():
    """M1-M5 callers don't pass on_speech_started -- must stay a no-op default, not a
    required argument."""
    clock = FakeClock()
    provider = FakeSTTProvider({"p1": [STTResult(text="haan bilkul", is_final=True, lang="hi-en")]})
    vad = FakeVAD(script=[True], tail=False)
    emitted = []

    async def on_utt(u):
        emitted.append(u)

    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_frames(5, clock),
        stt_provider=provider,
        vad=vad,
        on_utterance=on_utt,
        clock=clock,
    )

    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_stt_open_failure_degrades_to_giving_up_on_the_track_without_raising():
    """No second real STT provider exists to fall back to (see
    transcriber/providers/sarvam.py) -- the honest degrade is logging and returning,
    not crashing the worker over one participant's bad connection."""
    provider = FailingSTTProvider()
    breaker = CircuitBreaker("test-stt", failure_threshold=5)  # don't trip open mid-retry
    emitted = []

    async def on_utt(u):
        emitted.append(u)

    async def frames_that_should_never_be_consumed():
        raise AssertionError("drive_audio must not run if the STT stream never opened")
        yield b""  # pragma: no cover -- unreachable, makes this an async generator

    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=frames_that_should_never_be_consumed(),
        stt_provider=provider,
        vad=None,  # never touched if drive_audio never starts
        on_utterance=on_utt,
        stt_breaker=breaker,
    )

    assert provider.call_count == 3  # retried twice before giving up (default max_attempts=3)
    assert emitted == []


@pytest.mark.asyncio
async def test_stt_open_circuit_opens_and_skips_retry_on_next_track():
    provider = FailingSTTProvider()
    breaker = CircuitBreaker("test-stt", failure_threshold=1)

    async def on_utt(u):
        pass

    async def no_frames():
        return
        yield b""  # pragma: no cover

    await run_track_pipeline(
        participant_id="p1", display_name="Rahul", pcm16_frames=no_frames(),
        stt_provider=provider, vad=None, on_utterance=on_utt, stt_breaker=breaker,
    )
    calls_after_first = provider.call_count
    # failure_threshold=1 counts against the BREAKER, which only sees one failure per
    # call (the whole retry_with_jitter sequence) -- so this is 3 raw attempts (the
    # default max_attempts) before that single counted failure opens the circuit.
    assert calls_after_first == 3

    await run_track_pipeline(
        participant_id="p2", display_name="Priya", pcm16_frames=no_frames(),
        stt_provider=provider, vad=None, on_utterance=on_utt, stt_breaker=breaker,
    )
    assert provider.call_count == calls_after_first  # circuit open -- never even tried
