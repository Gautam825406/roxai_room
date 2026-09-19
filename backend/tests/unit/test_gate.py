import pytest

from roxroom.context.models import Utterance
from roxroom.orchestrator.gate import (
    LLMStageBGate,
    evaluate_stage_a,
    is_backchannel,
    is_direct_address,
    is_imperative,
    is_interrogative,
)
from roxroom.reliability.circuit_breaker import CircuitBreaker


def test_backchannels_are_dropped():
    for text in ["haan", "Hmm", "achha.", "ok", "theek hai", "haan haan", ""]:
        assert is_backchannel(text) is True, text


def test_non_backchannel_statement_is_not_dropped():
    assert is_backchannel("mujhe cricket bahut pasand hai") is False


def test_interrogative_detection():
    assert is_interrogative("tumhara naam kya hai") is True
    assert is_interrogative("what time is it") is True
    assert is_interrogative("is this correct?") is True
    assert is_interrogative("mujhe cricket pasand hai") is False


def test_imperative_detection():
    assert is_imperative("ek example batao") is True
    assert is_imperative("please explain this") is True
    assert is_imperative("mujhe cricket pasand hai") is False


def test_direct_address_detection():
    assert is_direct_address("tumhe kya lagta hai") is True
    assert is_direct_address("can you help me") is True
    assert is_direct_address("mujhe cricket pasand hai") is False


def test_stage_a_explicit_single_bot_mention():
    result = evaluate_stage_a("AI Dost, tumhara naam kya hai?")
    assert result.explicit_bot == "dost"
    assert result.has_explicit_mention is True
    assert result.has_any_signal is True


def test_stage_a_explicit_bot_mention_tolerates_asr_noise():
    result = evaluate_stage_a("sathee ek example do na")
    assert result.explicit_bot == "sathi"


def test_stage_a_multi_bot_mention_has_no_single_explicit_bot():
    result = evaluate_stage_a("Dost tum answer karo, Sathi tum example dena")
    assert result.explicit_bot is None
    assert result.mentioned_bots == frozenset({"dost", "sathi"})
    assert result.has_explicit_mention is True


def test_stage_a_no_signal_for_plain_statement():
    result = evaluate_stage_a("mujhe cricket bahut pasand hai")
    assert result.has_any_signal is False


def test_stage_a_ambiguous_when_question_but_no_bot_named():
    result = evaluate_stage_a("Rahul, tumhe kya lagta hai iske baare mein?")
    assert result.has_explicit_mention is False
    assert result.has_any_signal is True  # direct address + interrogative present


class _AlwaysDownLLM:
    """Every call raises -- simulates a fully unreachable LLM provider."""

    def __init__(self) -> None:
        self.call_count = 0

    async def stream_reply(self, messages, *, max_output_tokens=None, response_format="text"):
        self.call_count += 1
        raise ConnectionError("stage B LLM unreachable")
        yield  # pragma: no cover -- makes this a generator function, never reached


def _utt(text: str) -> Utterance:
    return Utterance(
        participant_id="p1", display_name="Priya", text=text, is_final=True, lang="hi-en", t_start=0.0, t_end=0.1
    )


@pytest.mark.asyncio
async def test_llm_stage_b_gate_degrades_to_silent_when_llm_is_unreachable():
    llm = _AlwaysDownLLM()
    gate = LLMStageBGate(llm, max_attempts=2)

    verdict = await gate.classify(_utt("Rahul, tumhe kya lagta hai?"), [])

    assert verdict.respond is False
    assert verdict.addressed_bot == "none"
    assert verdict.reason == "stage_b_unavailable"
    assert llm.call_count == 2  # retried once before giving up


@pytest.mark.asyncio
async def test_llm_stage_b_gate_circuit_opens_and_skips_llm_entirely():
    llm = _AlwaysDownLLM()
    breaker = CircuitBreaker("stage_b_gate", failure_threshold=1)
    gate = LLMStageBGate(llm, breaker=breaker, max_attempts=1)

    await gate.classify(_utt("first call trips the breaker"), [])
    assert llm.call_count == 1

    verdict = await gate.classify(_utt("second call should be skipped by the open circuit"), [])
    assert verdict.reason == "stage_b_unavailable"
    assert llm.call_count == 1  # circuit was open -- LLM never called again
