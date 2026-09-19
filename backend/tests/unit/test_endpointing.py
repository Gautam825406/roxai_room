from roxroom.transcriber.endpointing import EndpointPolicy


def test_no_finalize_without_any_silence():
    policy = EndpointPolicy(base_silence_ms=700)
    assert policy.should_finalize(now=100.0) is False


def test_finalize_after_base_silence_elapsed():
    policy = EndpointPolicy(base_silence_ms=700)
    policy.on_silence_frame(t=10.0)
    assert policy.should_finalize(now=10.6) is False  # 600ms < 700ms
    assert policy.should_finalize(now=10.71) is True  # 710ms >= 700ms


def test_speech_frame_resets_silence_timer():
    policy = EndpointPolicy(base_silence_ms=700)
    policy.on_silence_frame(t=10.0)
    policy.on_speech_frame(t=10.5)  # speaker resumed before threshold
    assert policy.should_finalize(now=10.71) is False  # timer was reset


def test_trailing_question_word_extends_silence_window():
    policy = EndpointPolicy(base_silence_ms=700, extended_silence_ms=1400)
    policy.on_partial_text("tum kal aa rahe ho kya")
    policy.on_silence_frame(t=0.0)
    assert policy.should_finalize(now=0.71) is False  # would've fired without extension
    assert policy.should_finalize(now=1.41) is True


def test_non_trailing_word_uses_base_window():
    policy = EndpointPolicy(base_silence_ms=700, extended_silence_ms=1400)
    policy.on_partial_text("mujhe cricket pasand hai")
    policy.on_silence_frame(t=0.0)
    assert policy.should_finalize(now=0.71) is True


def test_reset_clears_state():
    policy = EndpointPolicy(base_silence_ms=700)
    policy.on_partial_text("kya")
    policy.on_silence_frame(t=0.0)
    policy.reset()
    assert policy.should_finalize(now=100.0) is False
    # after reset, extension from stale partial text shouldn't linger
    policy.on_silence_frame(t=200.0)
    assert policy.should_finalize(now=200.71) is True
