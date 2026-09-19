"""Transcriber: joins the room as a hidden participant, opens one dedicated STT+VAD
session per subscribed *human* audio track (bot identities are skipped -- bots don't
transcribe themselves), and forwards finalized Utterances to `on_utterance`.

Also listens on the room's chat text-stream topic (M5): a human's chat message is
wrapped into the same `Utterance` type (modality="text") and forwarded through the
same `on_utterance` callback, so voice and text share one pipeline all the way through
gating/routing/context, per the spec. A bot's own chat message (published when it
speaks) is filtered out here -- it's already added to RoomContext directly by
BotAgent.speak(), so re-ingesting it would double-count that turn.

Barge-in signal (M6): if `on_speech_started` is given, it fires (with the speaking
participant's identity) the moment any human track's VAD detects a silence->speech
transition -- independent of and much earlier than utterance finalization, since
barge-in needs to react to speech *starting*, not to a full utterance completing. The
caller (run_bot.py) is expected to check whether a bot currently holds the speak-lock
and, if so, cancel it -- this module doesn't know about bots or the speak-lock at all.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from livekit import rtc

from roxroom.config import RoxRoomConfig
from roxroom.constants import CHAT_TOPIC
from roxroom.context.models import Utterance
from roxroom.livekit_token import mint_token
from roxroom.transcriber.stt_provider import STTProvider
from roxroom.transcriber.track_pipeline import run_track_pipeline
from roxroom.transcriber.vad import SILERO_FRAME_MS, SILERO_SAMPLE_RATE, SileroVAD

logger = logging.getLogger("roxroom.transcriber.agent")

TRANSCRIBER_IDENTITY = "roxroom-transcriber"
TRANSCRIBER_NAME = "RoxRoom Transcriber"


class TranscriberAgent:
    def __init__(
        self,
        config: RoxRoomConfig,
        stt_provider: STTProvider,
        on_utterance: Callable[[Utterance], Awaitable[None]],
        on_speech_started: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._stt_provider = stt_provider
        self._on_utterance = on_utterance
        self._on_speech_started = on_speech_started
        self._bot_identities = {config.dost.identity, config.sathi.identity}
        self._room = rtc.Room()
        self._track_tasks: dict[str, asyncio.Task] = {}

    async def run(self) -> None:
        self._room.on("track_subscribed", self._handle_track_subscribed)
        self._room.on("track_unsubscribed", self._handle_track_unsubscribed)
        self._room.on("participant_connected", self._handle_participant_connected)
        self._room.on("participant_disconnected", self._handle_participant_disconnected)
        self._room.register_text_stream_handler(CHAT_TOPIC, self._handle_chat_stream)

        token = mint_token(
            self._config,
            identity=TRANSCRIBER_IDENTITY,
            name=TRANSCRIBER_NAME,
            hidden=True,
        )
        await self._room.connect(self._config.livekit_url, token)
        logger.info("transcriber connected to room %r", self._config.room_name)

    async def aclose(self) -> None:
        for task in list(self._track_tasks.values()):
            task.cancel()
        await self._room.disconnect()

    def _handle_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity not in self._bot_identities:
            logger.info("human joined: %s (%s)", participant.identity, participant.name)

    def _handle_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity not in self._bot_identities:
            logger.info("human left: %s (%s)", participant.identity, participant.name)

    def _handle_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity in self._bot_identities:
            return
        logger.info("opening STT session for %s", participant.identity)
        task = asyncio.create_task(self._run_track(track, participant))
        self._track_tasks[publication.sid] = task

    def _handle_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        task = self._track_tasks.pop(publication.sid, None)
        if task is not None:
            task.cancel()

    def _handle_chat_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity in self._bot_identities:
            return
        asyncio.ensure_future(self._consume_chat_stream(reader, participant_identity))

    async def _consume_chat_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        try:
            text = await reader.read_all()
        except Exception:
            logger.exception("failed to read chat text stream from %s", participant_identity)
            return
        if not text.strip():
            return

        participant = self._room.remote_participants.get(participant_identity)
        display_name = (participant.name if participant else None) or participant_identity
        now = time.monotonic()
        utt = Utterance(
            participant_id=participant_identity,
            display_name=display_name,
            text=text,
            is_final=True,
            lang="hi-en",
            t_start=now,
            t_end=now,
            modality="text",
        )
        logger.info("chat message from %s: %r", participant_identity, text)
        try:
            await self._on_utterance(utt)
        except Exception:
            # This runs as a fire-and-forget task (asyncio.ensure_future in
            # _handle_chat_stream) -- an uncaught exception here would otherwise just
            # vanish into asyncio's "exception was never retrieved" warning instead of
            # being logged clearly, and would silently drop the chat message.
            logger.exception("on_utterance failed for chat message from %s, continuing", participant_identity)

    async def _run_track(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        audio_stream = rtc.AudioStream(
            track,
            sample_rate=SILERO_SAMPLE_RATE,
            num_channels=1,
            frame_size_ms=SILERO_FRAME_MS,
        )

        async def pcm16_frames():
            async for event in audio_stream:
                yield bytes(event.frame.data)

        vad = SileroVAD()
        on_speech_started = None
        if self._on_speech_started is not None:
            on_speech_started = lambda: self._on_speech_started(participant.identity)  # noqa: E731

        try:
            await run_track_pipeline(
                participant_id=participant.identity,
                display_name=participant.name or participant.identity,
                pcm16_frames=pcm16_frames(),
                stt_provider=self._stt_provider,
                vad=vad,
                on_utterance=self._on_utterance,
                sample_rate=SILERO_SAMPLE_RATE,
                on_speech_started=on_speech_started,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("track pipeline crashed for %s", participant.identity)
        finally:
            await audio_stream.aclose()
