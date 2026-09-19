"""Chaos tests: inject provider failures (mid-stream disconnects, not just "down from
the start") and assert nothing crashes and a user-visible fallback happens.

This complements, not duplicates, the failure-path tests already in
tests/unit/test_bot_agent.py (LLM/TTS unreachable from the start -> degradation
line / chat-only), tests/unit/test_gate.py (Stage B LLM unreachable -> silent), and
tests/unit/test_track_pipeline.py (STT open failure -> give up on the track). Those
cover "provider never worked this call." What's new here: failures that happen
*partway through* an otherwise-working stream, and a couple of full
Orchestrator->BotAgent integration checks that the human-facing artifact (a chat
message, a RoomContext turn) is actually produced, not just that no exception occurred.
"""
from __future__ import annotations

from typing import AsyncIterator

import pytest
from livekit import rtc

from roxroom.bots.bot_agent import BotAgent
from roxroom.bots.fakes import FailingLLMProvider, FailingTTSProvider, FakeLLMProvider, FakeTTSProvider
from roxroom.bots.persona_loader import load_persona
from roxroom.config import BotIdentity, RoxRoomConfig
from roxroom.context.models import Utterance
from roxroom.context.room_context import RoomContext
from roxroom.orchestrator.orchestrator import Orchestrator
from roxroom.orchestrator.speak_lock import FloorOutcome, SpeakLock
from roxroom.reliability.circuit_breaker import CircuitBreaker
from roxroom.transcriber.stt_provider import STTProvider, STTResult, STTStream
from roxroom.transcriber.track_pipeline import run_track_pipeline


# ---------------------------------------------------------------------------
# STT: mid-stream disconnect
# ---------------------------------------------------------------------------


class DisconnectingSTTStream(STTStream):
    """Yields one partial result, then the results() stream itself blows up --
    simulating a websocket dropping mid-utterance, not a failure to connect at all."""

    def __init__(self) -> None:
        self.pushed_frames = 0

    async def push_frame(self, pcm16_bytes: bytes) -> None:
        self.pushed_frames += 1

    def results(self) -> AsyncIterator[STTResult]:
        async def gen() -> AsyncIterator[STTResult]:
            yield STTResult(text="mujhe cricket", is_final=False, lang="hi-en")
            raise ConnectionError("websocket dropped mid-utterance")

        return gen()

    async def close(self) -> None:
        pass


class DisconnectingSTTProvider(STTProvider):
    async def open_stream(self, *, participant_id: str, sample_rate: int) -> STTStream:
        return DisconnectingSTTStream()


class AlwaysSilentVAD:
    def is_speech(self, pcm16_bytes: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


async def _endless_silence(n: int = 200):
    for _ in range(n):
        yield b"\x00\x00"


@pytest.mark.asyncio
async def test_stt_mid_stream_disconnect_does_not_crash_the_track_pipeline():
    emitted = []

    async def on_utt(u: Utterance) -> None:
        emitted.append(u)

    # Should return cleanly (no exception escaping) once the results() stream dies,
    # instead of spinning forever pushing frames into a dead connection.
    await run_track_pipeline(
        participant_id="p1",
        display_name="Rahul",
        pcm16_frames=_endless_silence(),
        stt_provider=DisconnectingSTTProvider(),
        vad=AlwaysSilentVAD(),
        on_utterance=on_utt,
        stt_breaker=CircuitBreaker("test-stt", failure_threshold=5),
    )
    # The partial text never got a chance to finalize (no is_final, no local silence
    # window elapsed before the disconnect) -- the point is that nothing crashed and
    # we ended cleanly, not that we recovered the partial utterance.
    assert emitted == []


# ---------------------------------------------------------------------------
# LLM: mid-stream disconnect during an active reply
# ---------------------------------------------------------------------------


def _bot_config() -> RoxRoomConfig:
    return RoxRoomConfig(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="k",
        livekit_api_secret="s",
        room_name="roxroom-test",
        dost=BotIdentity(identity="roxstar-ai-dost", name="Roxstar AI Dost"),
        sathi=BotIdentity(identity="roxstar-ai-sathi", name="Roxstar AI Sathi"),
    )


class MidStreamFailureLLM:
    """Yields two good chunks, then the underlying generator raises -- unlike
    FailingLLMProvider (down from the first call), this simulates a connection that
    was working and then dropped."""

    async def stream_reply(self, messages, *, max_output_tokens=None, response_format="text"):
        yield "Cricket ek"
        yield " team sport hai."
        raise ConnectionError("connection reset mid-completion")


@pytest.mark.asyncio
async def test_llm_mid_stream_disconnect_marks_partial_reply_as_error_not_silence():
    config = _bot_config()
    persona = load_persona("dost")
    room_context = RoomContext()
    speak_lock = SpeakLock()
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=MidStreamFailureLLM(),
        tts=FakeTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)

    result = await agent.reply_to("cricket kya hota hai")

    assert result.outcome == FloorOutcome.FAILED
    assert result.text == "Cricket ek team sport hai."  # whatever arrived before the drop
    pc = room_context.build_prompt_context()
    assert pc.recent_turns[0].text == "[error] Cricket ek team sport hai."
    assert speak_lock.holder is None  # floor released, not stuck


# ---------------------------------------------------------------------------
# End-to-end: Orchestrator routes to a bot whose LLM is down -> human still gets
# something coherent (a degradation line), not silence and not a crash.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_llm_outage_still_produces_a_human_visible_reply():
    config = _bot_config()
    room_context = RoomContext()
    speak_lock = SpeakLock()
    dost_persona = load_persona("dost")
    dost_agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=dost_persona,
        llm=FailingLLMProvider(),
        tts=FakeTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        llm_breaker=CircuitBreaker("e2e-llm", failure_threshold=5),
    )
    dost_agent._audio_source = rtc.AudioSource(16_000, 1)

    orchestrator = Orchestrator()  # Stage A alone is enough for an explicit address
    utt = Utterance(
        participant_id="priya",
        display_name="Priya",
        text="AI Dost, tumhara naam kya hai?",
        is_final=True,
        lang="hi-en",
        t_start=0.0,
        t_end=0.1,
    )
    decision = await orchestrator.handle_utterance(utt, [])
    assert decision.respond is True
    assert decision.chosen_bot == "dost"

    result = await dost_agent.reply_to(decision.plan[0].instruction)

    assert result.text == dost_persona.degradation_lines[0]
    pc = room_context.build_prompt_context()
    assert pc.recent_turns[0].text == dost_persona.degradation_lines[0]


@pytest.mark.asyncio
async def test_end_to_end_tts_outage_still_delivers_the_real_answer_via_chat():
    config = _bot_config()
    room_context = RoomContext()
    speak_lock = SpeakLock()
    persona = load_persona("dost")
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=FakeLLMProvider(["Mera naam Roxstar AI Dost hai."]),
        tts=FailingTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        tts_breaker=CircuitBreaker("e2e-tts", failure_threshold=5),
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)

    orchestrator = Orchestrator()
    utt = Utterance(
        participant_id="priya", display_name="Priya", text="AI Dost, tumhara naam kya hai?",
        is_final=True, lang="hi-en", t_start=0.0, t_end=0.1,
    )
    decision = await orchestrator.handle_utterance(utt, [])
    result = await agent.reply_to(decision.plan[0].instruction)

    assert result.outcome == FloorOutcome.COMPLETED
    assert result.text == "Mera naam Roxstar AI Dost hai."  # the real answer, TTS just couldn't speak it
    assert speak_lock.holder is None


# ---------------------------------------------------------------------------
# Stage B gate outage -> Orchestrator stays silent, never raises
# ---------------------------------------------------------------------------


class MisbehavingGate:
    """A StageBGate implementation that forgets to guard its own provider call --
    unlike LLMStageBGate (which has its own retry+circuit-breaker->silent fallback,
    see test_gate.py), this one just raises. The Orchestrator itself should still not
    propagate it -- defense in depth, not reliance on every StageBGate implementation
    getting its own resilience right."""

    async def classify(self, utterance, recent_turns):
        raise TimeoutError("stage B timed out")


@pytest.mark.asyncio
async def test_orchestrator_survives_a_stage_b_gate_that_forgot_to_handle_its_own_errors():
    orch = Orchestrator(stage_b_gate=MisbehavingGate())
    utt = Utterance(
        participant_id="p1", display_name="Priya", text="Rahul, tumhe kya lagta hai?",
        is_final=True, lang="hi-en", t_start=0.0, t_end=0.1,
    )
    decision = await orch.handle_utterance(utt, [])  # must not raise
    assert decision.respond is False
    assert decision.rule_fired == "stage_b_crashed"
