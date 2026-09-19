"""Silero VAD wrapper: per-frame speech/silence classification.

One `SileroVAD` instance must be used by exactly one audio stream -- the underlying
model is stateful (RNN hidden state across calls), so sharing an instance across two
participants' tracks would corrupt both. `transcriber/agent.py` creates one per
subscribed track, matching the "one dedicated session per participant track" rule
used for STT.

Frames must be 16kHz mono PCM16, 512 samples (32ms) each -- the shape Silero expects.
`transcriber/agent.py` requests exactly this from LiveKit's `AudioStream` via
`sample_rate=16000, frame_size_ms=32`, so no resampling code lives here.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from silero_vad import load_silero_vad

SILERO_SAMPLE_RATE = 16_000
SILERO_FRAME_MS = 32
SILERO_FRAME_SAMPLES = 512  # 16000 * 0.032

DEFAULT_THRESHOLD = 0.5


class VADLike(Protocol):
    def is_speech(self, pcm16_bytes: bytes) -> bool: ...
    def reset(self) -> None: ...


class SileroVAD:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._model = load_silero_vad(onnx=True)

    def is_speech(self, pcm16_bytes: bytes) -> bool:
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with torch.no_grad():
            prob = self._model(torch.from_numpy(samples), SILERO_SAMPLE_RATE).item()
        return prob >= self.threshold

    def reset(self) -> None:
        self._model.reset_states()
