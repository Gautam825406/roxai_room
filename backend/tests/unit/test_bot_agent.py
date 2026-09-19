import pytest
from livekit import rtc

from roxroom.bots.bot_agent import BotAgent
from roxroom.bots.fakes import FailingLLMProvider, FailingTTSProvider, FakeLLMProvider, FakeTTSProvider
from roxroom.bots.persona_loader import load_persona
from roxroom.config import BotIdentity, RoxRoomConfig
from roxroom.context.room_context import RoomContext
from roxroom.obs.trace import Trace
from roxroom.orchestrator.speak_lock import FloorOutcome, SpeakLock
from roxroom.reliability.circuit_breaker import CircuitBreaker


def _config() -> RoxRoomConfig:
    return RoxRoomConfig(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="k",
        livekit_api_secret="s",
        room_name="roxroom-test",
        dost=BotIdentity(identity="roxstar-ai-dost", name="Roxstar AI Dost"),
        sathi=BotIdentity(identity="roxstar-ai-sathi", name="Roxstar AI Sathi"),
    )


def _make_agent(llm_chunks: list[str]):
    config = _config()
    persona = load_persona("dost")
    llm = FakeLLMProvider(llm_chunks)
    tts = FakeTTSProvider()
    room_context = RoomContext()
    speak_lock = SpeakLock()
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=llm,
        tts=tts,
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)  # bypass connect() for the unit test
    return agent, llm, tts, room_context, speak_lock


@pytest.mark.asyncio
async def test_reply_to_streams_llm_output_through_tts_and_publishes():
    agent, llm, tts, room_context, _ = _make_agent(["haan bilkul, ", "yeh accha idea hai."])

    result = await agent.reply_to("kya tum madad kar sakte ho?")

    assert result.outcome == FloorOutcome.COMPLETED
    assert result.text == "haan bilkul, yeh accha idea hai."
    assert len(llm.calls) == 1
    assert len(tts.streams) == 1
    assert tts.streams[0].ended is True
    # text pushed to TTS should be normalized (accha -> Devanagari) before synthesis
    assert any("अच्छा" in pushed for pushed in tts.streams[0].pushed_text)


@pytest.mark.asyncio
async def test_reply_feeds_back_into_room_context_as_a_bot_turn():
    agent, llm, tts, room_context, _ = _make_agent(["theek hai, chalte hain."])
    await agent.reply_to("chalna hai?")

    pc = room_context.build_prompt_context()
    assert len(pc.recent_turns) == 1
    assert pc.recent_turns[0].role == "bot"
    assert pc.recent_turns[0].speaker_id == "roxstar-ai-dost"
    assert pc.recent_turns[0].text == "theek hai, chalte hain."


@pytest.mark.asyncio
async def test_speak_lock_is_released_after_reply():
    agent, llm, tts, room_context, speak_lock = _make_agent(["ek chota sa jawab."])
    await agent.reply_to("kuch bhi")
    assert speak_lock.holder is None


@pytest.mark.asyncio
async def test_build_messages_includes_persona_prompt_and_instruction():
    agent, llm, tts, room_context, _ = _make_agent(["ok"])
    await agent.reply_to("yeh kya hai?")

    messages = llm.calls[0]
    assert messages[0].role == "system"
    assert "warm, practical" in messages[0].content  # dost.yaml persona_description
    assert messages[1].role == "user"
    assert "yeh kya hai?" in messages[1].content


@pytest.mark.asyncio
async def test_prompt_context_is_included_when_room_has_prior_turns():
    agent, llm, tts, room_context, _ = _make_agent(["ok"])
    await room_context.add_turn(
        speaker_id="rahul", display_name="Rahul", role="human", text="mujhe cricket pasand hai", ts=0.0
    )
    await agent.reply_to("kya tumhe pata hai?")

    user_message = llm.calls[0][1].content
    assert "Rahul: mujhe cricket pasand hai" in user_message


@pytest.mark.asyncio
async def test_barge_in_stops_mid_reply_flushes_audio_and_aborts_llm():
    config = _config()
    persona = load_persona("dost")
    room_context = RoomContext()
    speak_lock = SpeakLock()
    tts = FakeTTSProvider()

    agent_holder: dict[str, BotAgent] = {}

    def cancel_after_second_chunk(chunk: str) -> None:
        if chunk == " team sport hai.":
            agent_holder["agent"].cancel("barge_in")

    llm = FakeLLMProvider(
        ["Cricket ek", " team sport hai.", " Bahut popular hai India mein.", " Aur bhi kuch hai."],
        on_chunk=cancel_after_second_chunk,
    )
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=llm,
        tts=tts,
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)
    clear_queue_calls = []
    agent._audio_source.clear_queue = lambda: clear_queue_calls.append(True)
    agent_holder["agent"] = agent

    result = await agent.reply_to("cricket kya hota hai")

    assert result.outcome == FloorOutcome.INTERRUPTED
    assert result.text == "Cricket ek team sport hai."  # stopped before the 3rd/4th chunks
    assert llm.aborted is True  # LLM stream was aclose()'d, not just abandoned
    assert clear_queue_calls == [True]  # already-queued audio was flushed
    assert tts.streams[0].cancelled_reason == "barge_in"

    pc = room_context.build_prompt_context()
    assert pc.recent_turns[0].text == "[interrupted] Cricket ek team sport hai."
    assert speak_lock.holder is None  # floor still released correctly


@pytest.mark.asyncio
async def test_cancel_is_a_noop_when_this_bot_does_not_hold_the_floor():
    agent, llm, tts, room_context, speak_lock = _make_agent(["ok"])
    agent.cancel("stray barge-in signal")  # nothing in flight -- must not raise
    result = await agent.reply_to("kuch bhi")
    assert result.outcome == FloorOutcome.COMPLETED  # unaffected


@pytest.mark.asyncio
async def test_llm_unreachable_falls_back_to_a_degradation_line():
    config = _config()
    persona = load_persona("dost")  # has real degradation_lines from dost.yaml
    room_context = RoomContext()
    speak_lock = SpeakLock()
    llm = FailingLLMProvider()
    tts = FakeTTSProvider()
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=llm,
        tts=tts,
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        llm_breaker=CircuitBreaker("test-llm", failure_threshold=5),  # don't trip open mid-retry
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)

    result = await agent.reply_to("cricket kya hota hai")

    assert result.outcome == FloorOutcome.COMPLETED
    assert result.text == persona.degradation_lines[0]
    assert llm.call_count == 2  # retried once before giving up (max_attempts=2)
    # the degradation line still went through TTS and landed in RoomContext, same as
    # any normal reply -- degrading is not the same as staying silent.
    assert tts.streams[0].pushed_text  # something was actually synthesized
    pc = room_context.build_prompt_context()
    assert pc.recent_turns[0].text == persona.degradation_lines[0]


@pytest.mark.asyncio
async def test_llm_circuit_opens_and_skips_llm_entirely_on_subsequent_replies():
    config = _config()
    persona = load_persona("dost")
    room_context = RoomContext()
    speak_lock = SpeakLock()
    llm = FailingLLMProvider()
    breaker = CircuitBreaker("test-llm", failure_threshold=1)
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=llm,
        tts=FakeTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        llm_breaker=breaker,
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)

    await agent.reply_to("first question")
    calls_after_first = llm.call_count
    assert calls_after_first >= 1

    result = await agent.reply_to("second question")
    assert result.text == persona.degradation_lines[0]
    assert llm.call_count == calls_after_first  # circuit was open -- LLM never retried


@pytest.mark.asyncio
async def test_tts_unreachable_falls_back_to_chat_only_reply():
    config = _config()
    persona = load_persona("dost")
    room_context = RoomContext()
    speak_lock = SpeakLock()
    llm = FakeLLMProvider(["Cricket ek team sport hai."])
    tts = FailingTTSProvider()
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=llm,
        tts=tts,
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        tts_breaker=CircuitBreaker("test-tts", failure_threshold=5),
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)

    result = await agent.reply_to("cricket kya hota hai")

    assert result.outcome == FloorOutcome.COMPLETED
    assert result.text == "Cricket ek team sport hai."  # the REAL answer, not a degradation line
    assert tts.call_count == 2  # retried once before giving up
    pc = room_context.build_prompt_context()
    assert pc.recent_turns[0].text == "Cricket ek team sport hai."


@pytest.mark.asyncio
async def test_reply_to_marks_the_full_trace_in_order():
    agent, llm, tts, room_context, _ = _make_agent(["Cricket ek", " team sport hai."])
    trace = Trace(trace_id="u1", participant_id="priya")
    trace.mark("stt_final")
    trace.mark("gate_route")

    await agent.reply_to("cricket kya hota hai", trace=trace)

    for stage in ("stt_final", "gate_route", "llm_first_token", "tts_first_byte", "audio_published"):
        assert stage in trace.marks, f"{stage} was never marked"
    assert trace.marks["stt_final"] <= trace.marks["gate_route"] <= trace.marks["llm_first_token"]
    assert trace.marks["llm_first_token"] <= trace.marks["tts_first_byte"] <= trace.marks["audio_published"]
    assert trace.bot_id == "dost"
    assert trace.end_of_speech_to_first_audio_ms is not None


@pytest.mark.asyncio
async def test_chat_only_fallback_never_marks_audio_stages():
    """When TTS is down, there's no audio at all -- audio_published/tts_first_byte
    must stay unmarked so this reply is correctly excluded from the latency summary
    (see obs/latency_report.py), not counted as a 0ms or bogus sample."""
    config = _config()
    persona = load_persona("dost")
    room_context = RoomContext()
    speak_lock = SpeakLock()
    agent = BotAgent(
        config=config,
        identity=config.dost,
        persona=persona,
        llm=FakeLLMProvider(["Mera naam Dost hai."]),
        tts=FailingTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
        tts_breaker=CircuitBreaker("trace-test-tts", failure_threshold=5),
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)
    trace = Trace(trace_id="u1", participant_id="priya")
    trace.mark("stt_final")

    await agent.reply_to("naam kya hai", trace=trace)

    assert "llm_first_token" in trace.marks
    assert "tts_first_byte" not in trace.marks
    assert "audio_published" not in trace.marks
    assert trace.end_of_speech_to_first_audio_ms is None
