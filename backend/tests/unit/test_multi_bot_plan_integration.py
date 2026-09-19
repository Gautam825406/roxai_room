"""End-to-end (but fully offline) proof that a multi-bot plan actually executes
correctly: Orchestrator routes "AI Dost tum answer karo, phir AI Sathi example dena"
into a 2-step plan, PlanExecutor runs it through two real BotAgent instances sharing
one RoomContext, and Sathi's turn genuinely sees Dost's reply in its own prompt."""
from __future__ import annotations

import pytest
from livekit import rtc

from roxroom.bots.bot_agent import BotAgent
from roxroom.bots.fakes import FakeLLMProvider, FakeTTSProvider
from roxroom.bots.persona_loader import load_persona
from roxroom.config import BotIdentity, RoxRoomConfig
from roxroom.context.models import Utterance
from roxroom.context.room_context import RoomContext
from roxroom.orchestrator.orchestrator import Orchestrator
from roxroom.orchestrator.plan_executor import PlanExecutor
from roxroom.orchestrator.speak_lock import SpeakLock


def _config() -> RoxRoomConfig:
    return RoxRoomConfig(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="k",
        livekit_api_secret="s",
        room_name="roxroom-test",
        dost=BotIdentity(identity="roxstar-ai-dost", name="Roxstar AI Dost"),
        sathi=BotIdentity(identity="roxstar-ai-sathi", name="Roxstar AI Sathi"),
    )


def _make_bot_agent(config, identity, bot_id, llm_chunks, room_context, speak_lock) -> tuple[BotAgent, FakeLLMProvider]:
    persona = load_persona(bot_id)
    llm = FakeLLMProvider(llm_chunks)
    agent = BotAgent(
        config=config,
        identity=identity,
        persona=persona,
        llm=llm,
        tts=FakeTTSProvider(),
        room_context=room_context,
        speak_lock=speak_lock,
        sample_rate=16_000,
    )
    agent._audio_source = rtc.AudioSource(16_000, 1)  # bypass connect() for the test
    return agent, llm


def _utt(text: str) -> Utterance:
    return Utterance(
        participant_id="priya",
        display_name="Priya",
        text=text,
        is_final=True,
        lang="hi-en",
        t_start=0.0,
        t_end=0.1,
    )


@pytest.mark.asyncio
async def test_multi_bot_plan_executes_both_bots_in_order_with_shared_context():
    config = _config()
    room_context = RoomContext()
    speak_lock = SpeakLock()

    dost_agent, dost_llm = _make_bot_agent(
        config, config.dost, "dost", ["Cricket ek team sport hai."], room_context, speak_lock
    )
    sathi_agent, sathi_llm = _make_bot_agent(
        config, config.sathi, "sathi", ["Jaise football, do teams khelti hain."], room_context, speak_lock
    )

    orchestrator = Orchestrator()
    speak_lock.on_floor_released(lambda bot, outcome: orchestrator.on_bot_reply_completed(bot))
    plan_executor = PlanExecutor({"dost": dost_agent, "sathi": sathi_agent})

    text = "AI Dost tum answer karo, phir AI Sathi example dena"
    decision = await orchestrator.handle_utterance(_utt(text), [])

    assert decision.respond is True
    assert decision.rule_fired == "multi_bot_plan"
    assert [step.bot for step in decision.plan] == ["dost", "sathi"]

    results = await plan_executor.execute(decision.plan)

    assert [r.text for r in results] == ["Cricket ek team sport hai.", "Jaise football, do teams khelti hain."]

    # Both replies landed in the shared RoomContext, in order.
    turns = list(room_context.turns)
    assert [t.speaker_id for t in turns] == ["roxstar-ai-dost", "roxstar-ai-sathi"]
    assert [t.text for t in turns] == [r.text for r in results]

    # The key integration guarantee: Sathi's own prompt included Dost's reply, because
    # BotAgent reads RoomContext fresh each time rather than from a stale snapshot.
    sathi_user_message = sathi_llm.calls[0][1].content
    assert "Cricket ek team sport hai." in sathi_user_message

    # And Dost's prompt did NOT include Sathi's reply (it hadn't happened yet).
    dost_user_message = dost_llm.calls[0][1].content
    assert "football" not in dost_user_message

    assert speak_lock.holder is None  # floor released after both turns
