"""Scenario tests replaying conversation flows against the Orchestrator.

NOTE ON PROVENANCE: the assignment references "the PDF's Scenarios 1-7" as the source
of truth for these flows, but no such PDF was ever attached to this conversation. These
7 scenarios are reconstructed from the explicit rules stated in the prompt itself
(backchannel dropping, explicit address with ASR noise, human-to-human silence,
persona affinity, multi-bot plans, alternation) rather than transcribed from a source
document. If you have the actual PDF, the fixtures below should be swapped for the real
exchanges -- the Orchestrator plumbing they exercise won't need to change.
"""
from __future__ import annotations

import pytest

from roxroom.context.models import Turn, Utterance
from roxroom.orchestrator.fakes import FakeStageBGate
from roxroom.orchestrator.gate import GateVerdict
from roxroom.orchestrator.orchestrator import Orchestrator


def _utt(text: str, utt_id: str, participant_id: str, display_name: str) -> Utterance:
    return Utterance(
        participant_id=participant_id,
        display_name=display_name,
        text=text,
        is_final=True,
        lang="hi-en",
        t_start=0.0,
        t_end=0.1,
        utt_id=utt_id,
    )


def _turn(text: str, speaker_id: str, display_name: str, role: str = "human") -> Turn:
    return Turn(speaker_id=speaker_id, display_name=display_name, role=role, text=text, ts=0.0)


@pytest.mark.asyncio
async def test_scenario_1_backchannel_produces_silence():
    orch = Orchestrator()
    await orch.handle_utterance(
        _utt("aaj weather bahut accha hai", "s1-1", "priya", "Priya"), []
    )
    decision = await orch.handle_utterance(_utt("haan", "s1-2", "rahul", "Rahul"), [])
    assert decision.respond is False
    assert decision.rule_fired == "backchannel"


@pytest.mark.asyncio
async def test_scenario_2_explicit_address_clean_pronunciation():
    orch = Orchestrator()
    decision = await orch.handle_utterance(
        _utt("AI Dost, tumhara naam kya hai?", "s2-1", "priya", "Priya"), []
    )
    assert decision.respond is True
    assert decision.chosen_bot == "dost"
    assert decision.rule_fired == "explicit_address"


@pytest.mark.asyncio
async def test_scenario_3_explicit_address_survives_asr_noise():
    orch = Orchestrator()
    decision = await orch.handle_utterance(
        _utt("sathee, ek example do na", "s3-1", "rahul", "Rahul"), []
    )
    assert decision.respond is True
    assert decision.chosen_bot == "sathi"
    assert decision.rule_fired == "explicit_address"


@pytest.mark.asyncio
async def test_scenario_4_humans_talking_to_each_other_is_silent():
    text = "Rahul, tumhe kya lagta hai iske baare mein?"
    gate = FakeStageBGate(
        by_text={text: GateVerdict(respond=False, addressed_bot="none", reason="addressed to Rahul, a human", confidence=0.85)}
    )
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt(text, "s4-1", "priya", "Priya"), [])
    assert decision.respond is False
    assert decision.rule_fired == "stage_b_silent"


@pytest.mark.asyncio
async def test_scenario_5_profile_facts_survive_and_stay_isolated_then_a_later_question_routes_correctly():
    """Rahul states a fact early; several human-to-human turns pass (no bot replies);
    a later generic factual question (no bot named) should still get routed via the
    gate/router without the pipeline losing track of who's who. Full profile-aware
    prompt content is a BotAgent (M4) concern -- here we only assert the Orchestrator's
    routing behaves correctly across a longer, mixed conversation."""
    orch = Orchestrator()
    recent_turns: list[Turn] = []

    d1 = await orch.handle_utterance(
        _utt("mera naam Rahul hai aur mujhe cricket bahut pasand hai", "s5-1", "rahul", "Rahul"),
        recent_turns,
    )
    recent_turns.append(_turn("mera naam Rahul hai aur mujhe cricket bahut pasand hai", "rahul", "Rahul"))
    assert d1.respond is False  # plain statement, no question/imperative/address signal

    d2 = await orch.handle_utterance(
        _utt("mujhe toh painting pasand hai", "s5-2", "priya", "Priya"), recent_turns
    )
    recent_turns.append(_turn("mujhe toh painting pasand hai", "priya", "Priya"))
    assert d2.respond is False

    gate = FakeStageBGate(
        by_text={
            "cricket world cup kab hota hai": GateVerdict(
                respond=True, addressed_bot="dost", reason="factual question", confidence=0.75
            )
        }
    )
    orch2 = Orchestrator(stage_b_gate=gate)
    d3 = await orch2.handle_utterance(
        _utt("cricket world cup kab hota hai", "s5-3", "rahul", "Rahul"), recent_turns
    )
    assert d3.respond is True
    assert d3.chosen_bot == "dost"


@pytest.mark.asyncio
async def test_scenario_6_persona_affinity_routes_elaboration_request_to_sathi():
    text = "mujhe samajh nahi aa raha, ek simple example samjhao"
    gate = FakeStageBGate(
        by_text={text: GateVerdict(respond=True, addressed_bot="none", reason="elaboration request", confidence=0.7)}
    )
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt(text, "s6-1", "priya", "Priya"), [])
    assert decision.respond is True
    assert decision.chosen_bot == "sathi"
    assert decision.rule_fired == "persona_affinity"


@pytest.mark.asyncio
async def test_scenario_7_multi_bot_plan_assigns_ordered_work_to_both_bots():
    text = "AI Dost tum pehle answer karo, phir AI Sathi ek example dena"
    orch = Orchestrator()
    decision = await orch.handle_utterance(_utt(text, "s7-1", "priya", "Priya"), [])
    assert decision.respond is True
    assert decision.rule_fired == "multi_bot_plan"
    assert [step.bot for step in decision.plan] == ["dost", "sathi"]
    assert decision.chosen_bot is None  # not a single-bot decision -- see decision.plan
