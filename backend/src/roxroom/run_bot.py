"""M5/M6 entrypoint: the two-bot worker -- Transcriber (voice + room text chat) ->
RoomContext -> Orchestrator -> PlanExecutor -> Dost and/or Sathi speaking back into
the room, including real multi-bot plans ("AI Dost tum answer karo, phir AI Sathi
example dena") and real barge-in.

Two independent interrupt paths, both driven off the Transcriber's per-track VAD:
- On every finalized utterance: cancel remaining not-yet-started PlanExecutor steps
  (M5) -- e.g. Sathi's queued turn is dropped if a human starts a new utterance before
  it begins.
- On speech *starting* (M6, much faster than waiting for an utterance to finalize):
  if a bot currently holds the speak-lock, call its cancel() -- see bot_agent.py's
  docstring for exactly what that does (flush queued audio, abort the TTS stream, abort
  the LLM stream, mark the partial reply `[interrupted]` in context).

M8 adds tracing: each utterance gets a Trace (trace_id = utt_id) marked at STT-final and
after the Orchestrator's gate+route decision; it's handed to the first bot to speak
(see plan_executor.py for why only the first step of a multi-bot plan gets it), which
marks LLM-first-token, TTS-first-byte, and first-audio-published itself. Completed
end-to-end latencies feed a LatencyTracker; its p50/p95 summary prints on shutdown.

LLM: OpenAI (gpt-4o for replies, gpt-4o-mini for the gate/extraction/summary calls) --
see bots/providers/openai_llm.py. Unlike Groq's free tier, OpenAI's rate limits are
generous enough that the gate and RoomContext's extractor/summarizer can safely share
one "fast" model instance without one starving the other (see DECISIONS.md for the
Groq-specific contention bug that made splitting those necessary there). Groq's
GroqLLMProvider and Anthropic's AnthropicLLMProvider are still available behind the
same LLMProvider interface if you'd rather switch; swapping is a one-line change here.

Run: `python -m roxroom.run_bot` (needs LIVEKIT_*, DEEPGRAM_API_KEY, OPENAI_API_KEY,
ELEVENLABS_API_KEY, ELEVENLABS_DOST_VOICE_ID, ELEVENLABS_SATHI_VOICE_ID in .env; set
ROXROOM_JSON_LOGS=1 for structured JSON logs instead of the human-readable default).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os

from roxroom.bots.bot_agent import BotAgent
from roxroom.bots.persona_loader import load_persona
from roxroom.bots.providers.elevenlabs_tts import ElevenLabsTTSProvider
from roxroom.bots.providers.openai_llm import DEFAULT_FAST_MODEL, OpenAILLMProvider
from roxroom.config import load_config
from roxroom.context.extractor import LLMExtractor
from roxroom.context.models import Utterance
from roxroom.context.room_context import RoomContext
from roxroom.context.summarizer import LLMSummarizer
from roxroom.obs.json_logger import setup_logging
from roxroom.obs.latency_report import LatencyTracker
from roxroom.obs.trace import Trace
from roxroom.orchestrator.gate import LLMStageBGate
from roxroom.orchestrator.orchestrator import Orchestrator
from roxroom.orchestrator.plan_executor import PlanExecutor
from roxroom.orchestrator.speak_lock import SpeakLock
from roxroom.transcriber.agent import TranscriberAgent
from roxroom.transcriber.providers.deepgram import DeepgramSTTProvider

logger = logging.getLogger("roxroom.m5")


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required for M5 -- set it in .env")
    return value


def _make_agent(config, identity, bot_id: str, voice_id: str, llm, tts, room_context, speak_lock) -> BotAgent:
    persona = load_persona(bot_id)
    persona = dataclasses.replace(persona, voice=dataclasses.replace(persona.voice, voice_id=voice_id))
    return BotAgent(
        config=config,
        identity=identity,
        persona=persona,
        llm=llm,
        tts=tts,
        room_context=room_context,
        speak_lock=speak_lock,
    )


async def run() -> None:
    config = load_config()
    deepgram_key = _require_env("DEEPGRAM_API_KEY")
    openai_key = _require_env("OPENAI_API_KEY")
    elevenlabs_key = _require_env("ELEVENLABS_API_KEY")
    dost_voice_id = _require_env("ELEVENLABS_DOST_VOICE_ID")
    sathi_voice_id = _require_env("ELEVENLABS_SATHI_VOICE_ID")

    llm_fast = OpenAILLMProvider(openai_key, model=DEFAULT_FAST_MODEL)  # gate + extraction + summary
    llm_reply = OpenAILLMProvider(openai_key)  # persona replies (gpt-4o, the default model)
    tts = ElevenLabsTTSProvider(elevenlabs_key)

    room_context = RoomContext(extractor=LLMExtractor(llm_fast), summarizer=LLMSummarizer(llm_fast))
    speak_lock = SpeakLock()
    orchestrator = Orchestrator(stage_b_gate=LLMStageBGate(llm_fast))
    speak_lock.on_floor_released(lambda bot, outcome: orchestrator.on_bot_reply_completed(bot))

    dost_agent = _make_agent(config, config.dost, "dost", dost_voice_id, llm_reply, tts, room_context, speak_lock)
    sathi_agent = _make_agent(config, config.sathi, "sathi", sathi_voice_id, llm_reply, tts, room_context, speak_lock)
    await asyncio.gather(dost_agent.connect(), sathi_agent.connect())
    logger.info("Dost and Sathi connected and ready")

    bot_agents_by_id = {"dost": dost_agent, "sathi": sathi_agent}
    plan_executor = PlanExecutor(bot_agents_by_id)
    latency_tracker = LatencyTracker()

    def on_speech_started(participant_id: str) -> None:
        holder = speak_lock.holder
        if holder is None:
            return
        logger.info("barge-in: %s started speaking while %s held the floor", participant_id, holder)
        plan_executor.cancel_current("barge_in")
        agent = bot_agents_by_id.get(holder)
        if agent is not None:
            agent.cancel("barge_in")

    async def on_utterance(utt: Utterance) -> None:
        # Any new utterance means a human spoke -- stop before starting whatever step
        # of a currently-running plan hasn't started yet (see plan_executor.py).
        plan_executor.cancel_current("new utterance arrived")

        await room_context.add_turn(
            speaker_id=utt.participant_id,
            display_name=utt.display_name,
            role="human",
            text=utt.text,
            ts=utt.t_end,
            modality=utt.modality,
            lang=utt.lang,
        )
        trace = Trace(trace_id=utt.utt_id, participant_id=utt.participant_id)
        trace.mark("stt_final", at=utt.t_end)

        decision = await orchestrator.handle_utterance(utt, list(room_context.turns))
        trace.mark("gate_route")
        if not decision.respond:
            return
        await plan_executor.execute(decision.plan, trace=trace)
        latency_tracker.record(trace.end_of_speech_to_first_audio_ms)

    transcriber = TranscriberAgent(
        config, DeepgramSTTProvider(deepgram_key), on_utterance, on_speech_started=on_speech_started
    )
    await transcriber.run()
    logger.info(
        "transcriber live in room %r -- talk (voice or chat) to Dost/Sathi (Ctrl+C to stop)",
        config.room_name,
    )

    try:
        await asyncio.Event().wait()
    finally:
        await transcriber.aclose()
        await asyncio.gather(dost_agent.aclose(), sathi_agent.aclose())
        print(latency_tracker.summary_table())


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
