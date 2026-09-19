import pytest

from roxroom.context.models import Utterance
from roxroom.orchestrator.fakes import FakeStageBGate
from roxroom.orchestrator.gate import GateVerdict
from roxroom.orchestrator.orchestrator import Orchestrator


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _utt(
    text: str,
    utt_id: str = "u1",
    participant_id: str = "p1",
    display_name: str = "Priya",
    modality: str = "voice",
) -> Utterance:
    return Utterance(
        participant_id=participant_id,
        display_name=display_name,
        text=text,
        is_final=True,
        lang="hi-en",
        t_start=0.0,
        t_end=0.1,
        utt_id=utt_id,
        modality=modality,
    )


@pytest.mark.asyncio
async def test_duplicate_utt_id_is_silent():
    orch = Orchestrator()
    utt = _utt("AI Dost tumhara naam kya hai", utt_id="dup-1")
    first = await orch.handle_utterance(utt, [])
    second = await orch.handle_utterance(utt, [])
    assert first.respond is True
    assert second.respond is False
    assert second.rule_fired == "duplicate"


@pytest.mark.asyncio
async def test_backchannel_is_silent():
    orch = Orchestrator()
    decision = await orch.handle_utterance(_utt("haan"), [])
    assert decision.respond is False
    assert decision.rule_fired == "backchannel"


@pytest.mark.asyncio
async def test_no_signal_statement_is_silent_without_calling_stage_b():
    gate = FakeStageBGate()
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt("mujhe cricket bahut pasand hai"), [])
    assert decision.respond is False
    assert decision.rule_fired == "stage_a_no_signal"
    assert gate.calls == []  # cheap heuristics resolved it, no LLM call needed


@pytest.mark.asyncio
async def test_explicit_address_responds_without_calling_stage_b():
    gate = FakeStageBGate()
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt("AI Dost, tumhara naam kya hai?"), [])
    assert decision.respond is True
    assert decision.chosen_bot == "dost"
    assert gate.calls == []


@pytest.mark.asyncio
async def test_ambiguous_utterance_uses_stage_b_and_can_stay_silent():
    text = "Rahul, tumhe kya lagta hai iske baare mein?"
    gate = FakeStageBGate(
        by_text={text: GateVerdict(respond=False, addressed_bot="none", reason="addressed to Rahul", confidence=0.9)}
    )
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt(text), [])
    assert decision.respond is False
    assert decision.rule_fired == "stage_b_silent"
    assert len(gate.calls) == 1


@pytest.mark.asyncio
async def test_ambiguous_utterance_stage_b_says_respond_and_gets_routed():
    text = "yeh kaise kaam karta hai"
    gate = FakeStageBGate(
        by_text={text: GateVerdict(respond=True, addressed_bot="dost", reason="factual question", confidence=0.8)}
    )
    orch = Orchestrator(stage_b_gate=gate)
    decision = await orch.handle_utterance(_utt(text), [])
    assert decision.respond is True
    assert decision.chosen_bot == "dost"
    assert decision.rule_fired == "stage_b_addressed"


@pytest.mark.asyncio
async def test_suppression_window_blocks_reply_shortly_after_a_bot_finished():
    clock = FakeClock()
    gate = FakeStageBGate(
        by_text={"yeh kaise kaam karta hai": GateVerdict(respond=True, addressed_bot="dost", reason="", confidence=0.8)}
    )
    orch = Orchestrator(stage_b_gate=gate, clock=clock)
    orch.on_bot_reply_completed("dost", at=0.0)
    clock.advance(0.5)  # well within the 1.5s suppression window

    decision = await orch.handle_utterance(_utt("yeh kaise kaam karta hai"), [])
    assert decision.respond is False
    assert decision.rule_fired == "suppressed_recent_reply"


@pytest.mark.asyncio
async def test_suppression_window_is_overridden_by_explicit_bot_address():
    clock = FakeClock()
    orch = Orchestrator(clock=clock)
    orch.on_bot_reply_completed("dost", at=0.0)
    clock.advance(0.2)

    decision = await orch.handle_utterance(_utt("Sathi, ek aur sawaal hai"), [])
    assert decision.respond is True
    assert decision.chosen_bot == "sathi"


@pytest.mark.asyncio
async def test_suppression_window_expires_after_1_5_seconds():
    clock = FakeClock()
    gate = FakeStageBGate(
        by_text={"yeh kaise kaam karta hai": GateVerdict(respond=True, addressed_bot="dost", reason="", confidence=0.8)}
    )
    orch = Orchestrator(stage_b_gate=gate, clock=clock)
    orch.on_bot_reply_completed("dost", at=0.0)
    clock.advance(1.6)

    decision = await orch.handle_utterance(_utt("yeh kaise kaam karta hai"), [])
    assert decision.respond is True


@pytest.mark.asyncio
async def test_decision_log_dict_has_the_spec_fields():
    orch = Orchestrator()
    decision = await orch.handle_utterance(_utt("AI Dost, tumhara naam kya hai?"), [])
    log_dict = decision.to_log_dict()
    assert set(log_dict.keys()) >= {"utt_id", "stage_a", "stage_b", "chosen_bot", "rule_fired", "latency_ms"}
    assert orch.decision_log == [decision]


@pytest.mark.asyncio
async def test_text_chat_utterance_is_routed_identically_to_voice():
    """The gate/router never look at modality (M5: voice and text chat share one
    pipeline) -- a text-chat utterance with the same content routes the same way."""
    orch_voice = Orchestrator()
    orch_text = Orchestrator()

    voice_decision = await orch_voice.handle_utterance(
        _utt("AI Dost, tumhara naam kya hai?", utt_id="v1", modality="voice"), []
    )
    text_decision = await orch_text.handle_utterance(
        _utt("AI Dost, tumhara naam kya hai?", utt_id="t1", modality="text"), []
    )

    assert voice_decision.respond == text_decision.respond is True
    assert voice_decision.chosen_bot == text_decision.chosen_bot == "dost"
    assert voice_decision.rule_fired == text_decision.rule_fired
