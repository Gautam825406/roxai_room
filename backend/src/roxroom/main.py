"""M0 entrypoint: connects one bot identity to a LiveKit room and publishes an audible
test tone, while logging room lifecycle events (humans joining/leaving/reconnecting).

This exists to prove the transport layer end-to-end before any STT/LLM/TTS logic
exists. It will be replaced in M4/M5 by real worker wiring that starts the
Transcriber, Orchestrator, and both BotAgents together.

Run: `python -m roxroom.main` (see scripts/run_dev.ps1 / run_dev.sh)
"""
from __future__ import annotations

import asyncio
import logging
import math

import numpy as np
from livekit import rtc

from roxroom.config import load_config
from roxroom.livekit_token import mint_token
from roxroom.obs.json_logger import setup_logging

logger = logging.getLogger("roxroom.m0")

SAMPLE_RATE = 48_000
NUM_CHANNELS = 1
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
TONE_HZ = 440.0
TONE_AMPLITUDE = 3000  # int16 headroom, keep well under 32767 to avoid clipping


def _register_room_logging(room: rtc.Room) -> None:
    @room.on("participant_connected")
    def _on_join(participant: rtc.RemoteParticipant) -> None:
        logger.info("participant joined: %s (%s)", participant.identity, participant.name)

    @room.on("participant_disconnected")
    def _on_leave(participant: rtc.RemoteParticipant) -> None:
        logger.info("participant left: %s (%s)", participant.identity, participant.name)

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        logger.info(
            "subscribed to %s track from %s", track.kind.name.lower(), participant.identity
        )

    @room.on("disconnected")
    def _on_disconnected(reason: rtc.DisconnectReason | None = None) -> None:
        logger.warning("room disconnected: %s", reason)


async def _publish_test_tone(room: rtc.Room, track_name: str) -> None:
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)
    logger.info("publishing %.0fHz test tone as track %r", TONE_HZ, track_name)

    phase = 0.0
    phase_step = 2 * math.pi * TONE_HZ / SAMPLE_RATE
    while True:
        frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_FRAME)
        samples = np.frombuffer(frame.data, dtype=np.int16)
        t = phase + phase_step * np.arange(SAMPLES_PER_FRAME)
        samples[:] = (np.sin(t) * TONE_AMPLITUDE).astype(np.int16)
        phase = float(t[-1] + phase_step)
        await source.capture_frame(frame)
        await asyncio.sleep(FRAME_MS / 1000)


async def run() -> None:
    config = load_config()
    room = rtc.Room()
    _register_room_logging(room)

    token = mint_token(config, config.dost.identity, config.dost.name)
    await room.connect(config.livekit_url, token)
    logger.info(
        "connected to room %r as %s (%s)", config.room_name, config.dost.name, config.dost.identity
    )

    try:
        await _publish_test_tone(room, track_name=f"{config.dost.identity}-test-tone")
    finally:
        await room.disconnect()


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
