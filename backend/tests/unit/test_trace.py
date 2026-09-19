import pytest

from roxroom.obs.trace import Trace


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_mark_records_elapsed_time_from_clock():
    clock = FakeClock()
    trace = Trace(trace_id="u1", participant_id="p1", clock=clock)

    trace.mark("stt_final")
    clock.advance(0.1)
    trace.mark("gate_route")
    clock.advance(0.4)
    trace.mark("llm_first_token")
    clock.advance(0.2)
    trace.mark("tts_first_byte")
    clock.advance(0.1)
    trace.mark("audio_published")

    assert trace.duration_ms("stt_final", "gate_route") == pytest.approx(100.0)
    assert trace.duration_ms("stt_final", "llm_first_token") == pytest.approx(500.0)
    assert trace.end_of_speech_to_first_audio_ms == pytest.approx(800.0)


def test_mark_is_idempotent_keeps_first_occurrence():
    clock = FakeClock()
    trace = Trace(trace_id="u1", participant_id="p1", clock=clock)

    trace.mark("stt_final")
    clock.advance(0.05)
    trace.mark("llm_first_token")
    clock.advance(0.5)
    trace.mark("llm_first_token")  # a later chunk -- must not overwrite the first

    assert trace.duration_ms("stt_final", "llm_first_token") == pytest.approx(50.0)


def test_duration_ms_is_none_when_a_stage_never_happened():
    trace = Trace(trace_id="u1", participant_id="p1")
    trace.mark("stt_final")
    assert trace.end_of_speech_to_first_audio_ms is None


def test_breakdown_has_none_for_unreached_stages_not_zero():
    clock = FakeClock()
    trace = Trace(trace_id="u1", participant_id="p1", clock=clock)
    trace.mark("stt_final")
    clock.advance(0.1)
    trace.mark("gate_route")
    # LLM/TTS/audio never happened (e.g. degraded to chat-only, no voice at all)

    breakdown = trace.breakdown()
    assert breakdown["stt_final_ms"] == pytest.approx(0.0)
    assert breakdown["gate_route_ms"] == pytest.approx(100.0)
    assert breakdown["llm_first_token_ms"] is None
    assert breakdown["tts_first_byte_ms"] is None
    assert breakdown["audio_published_ms"] is None
    assert breakdown["total_ms"] is None


def test_explicit_at_timestamp_is_honored_over_clock():
    trace = Trace(trace_id="u1", participant_id="p1", clock=lambda: 999.0)
    trace.mark("stt_final", at=10.0)
    trace.mark("audio_published", at=11.5)
    assert trace.end_of_speech_to_first_audio_ms == pytest.approx(1500.0)
