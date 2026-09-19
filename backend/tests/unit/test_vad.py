import numpy as np

from roxroom.transcriber.vad import SILERO_FRAME_SAMPLES, SileroVAD


def test_silero_vad_classifies_digital_silence_as_not_speech():
    vad = SileroVAD()
    silent_frame = np.zeros(SILERO_FRAME_SAMPLES, dtype=np.int16).tobytes()
    assert vad.is_speech(silent_frame) is False


def test_silero_vad_reset_does_not_raise():
    vad = SileroVAD()
    frame = np.zeros(SILERO_FRAME_SAMPLES, dtype=np.int16).tobytes()
    vad.is_speech(frame)
    vad.reset()
    assert vad.is_speech(frame) is False
