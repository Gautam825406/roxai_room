"""M1 entrypoint: runs the Transcriber alone against a live room, logging every
finalized Utterance to the console. No orchestrator/bots yet -- this exists to
verify per-participant streaming STT + Silero endpointing end-to-end with >=2 real
humans joining/leaving/reconnecting.

Run: `python -m roxroom.run_transcriber` (needs DEEPGRAM_API_KEY in .env -- Sarvam
support lands once transcriber/providers/sarvam.py is implemented, see its docstring).
"""
from __future__ import annotations

import asyncio
import logging
import os

from roxroom.config import load_config
from roxroom.context.models import Utterance
from roxroom.obs.json_logger import setup_logging
from roxroom.transcriber.agent import TranscriberAgent
from roxroom.transcriber.providers.deepgram import DeepgramSTTProvider

logger = logging.getLogger("roxroom.m1")


async def _on_utterance(utt: Utterance) -> None:
    logger.info("[%s] %s: %r (lang=%s)", utt.utt_id, utt.display_name, utt.text, utt.lang)


async def run() -> None:
    config = load_config()
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPGRAM_API_KEY is required for M1 -- set it in .env")

    stt_provider = DeepgramSTTProvider(api_key)
    agent = TranscriberAgent(config, stt_provider, _on_utterance)
    await agent.run()
    logger.info("transcriber live in room %r -- join as >=2 humans and talk (Ctrl+C to stop)", config.room_name)
    try:
        await asyncio.Event().wait()
    finally:
        await agent.aclose()


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
