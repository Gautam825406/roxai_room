# Decisions

## LiveKit

Given as a hard constraint. Three separate connections in one process (Transcriber +
2x BotAgent) using the raw `livekit` Python SDK (`rtc.Room`) directly rather than the
higher-level `livekit-agents` job-per-room framework -- we need fine control over
several simultaneous connections and custom audio publishing in one process, which is
what the raw SDK is for.

## STT: Sarvam AI Saarika (primary, not yet implemented) / Deepgram Nova-3 (implemented, running)

Saarika is purpose-built for Hindi/Hinglish code-mixing and should give the cleanest
per-track transcripts for this use case -- the whole routing/gate design depends on
clean text, so this is the highest-leverage provider pick in the system. **It is not
implemented** (`transcriber/providers/sarvam.py` raises `NotImplementedError`): its
streaming websocket contract wasn't confidently known from memory, and guessing at it
would produce code that looks finished but silently fails at runtime. That's worse than
an explicit gap.

Deepgram Nova-3 (`transcriber/providers/deepgram.py`) is what's actually implemented
and running -- a well-documented, stable, standard websocket protocol (raw PCM16 in,
JSON `Results` messages out). Weaker Hindi/English code-switch accuracy than Saarika,
but real and testable.

Rejected: ElevenLabs Scribe (streaming API less mature at the time of writing, pricier),
Google STT v2 hi-IN (mature/cheap but weaker code-mix handling, needs juggling
alternate-language hints).

Rough cost: Saarika ≈ $0.006-0.010/min, Nova-3 ≈ $0.0043-0.0059/min. Verify current
published rates before committing -- these move.

## VAD: Silero (real, running)

ONNX runtime build (`silero-vad` package with `onnx=True`), one model instance per
subscribed track (the model is stateful across calls, so instances can't be shared
across concurrent speakers). Verified locally against real audio frames (not mocked) --
see `tests/unit/test_vad.py`. Drives our own endpointing (`transcriber/endpointing.py`)
rather than relying on the STT vendor's built-in auto-endpointing, which is what makes
the trailing-question-word extension (`kya`/`kaise`/`kyun`/`matlab` -> keep listening)
possible.

## LLM: Groq -- openai/gpt-oss-120b (replies) / openai/gpt-oss-20b (gate) / qwen/qwen3.8-27b (extraction + summary)

**Originally built against Claude Sonnet 5 / Haiku 4.5, then switched to OpenAI gpt-4o
/ gpt-4o-mini, then switched again to Groq** -- each time at the project owner's
explicit request (account/access preference), not a quality judgment against the
previous pick. Because `LLMProvider` is an interface with the LLM-specific logic
(system-prompt handling, JSON mode, sentence-chunking) isolated behind it, each swap
only touched `run_bot.py` (which model(s) to construct) and added one new file
(`bots/providers/groq_llm.py`); the Orchestrator, RoomContext, and BotAgent code that
*use* an `LLMProvider` needed zero changes across any of these swaps. `bots/providers/
openai_llm.py` and `bots/providers/anthropic_llm.py` are both kept in the codebase,
fully implemented and tested, as documented one-line-swap-back alternatives -- see
their docstrings.

Unlike every other provider in this document, this one **was** live-tested end to end
against real API keys (LiveKit, Deepgram, Groq, ElevenLabs) in a real room with a real
human -- not just verified at the code/SDK level. That surfaced three real bugs no
amount of reading the SDK would have caught:

1. **Model IDs drift fast on Groq.** The first pick (`llama-3.3-70b-versatile` /
   `llama-3.1-8b-instant`) 404'd -- removed from the catalog. Replaced with
   `openai/gpt-oss-120b` / `openai/gpt-oss-20b`, confirmed against a live account's
   `client.models.list()` call, not documentation or training-time memory. If this
   provider ever 404s with "model ... does not exist", the catalog moved again --
   check `models.list()` yourself rather than trusting a name written down here.
2. **Reasoning models silently eat the JSON-mode token budget.** gpt-oss and Qwen3 are
   both reasoning models that spend part of `max_tokens` on hidden chain-of-thought
   before the actual answer. `gate.py`'s Stage B call asks for only 40 output tokens
   (correctly sized for a tiny `{"respond": ...}` verdict on a non-reasoning model) --
   against a reasoning model that intermittently 400'd with `json_validate_failed` and
   an empty `failed_generation`, because reasoning alone consumed the budget before any
   JSON was emitted. Fixed two ways in `groq_llm.py`, both Groq-specific and
   deliberately kept out of the provider-agnostic call sites: a `_JSON_MIN_TOKENS`
   floor on every JSON-mode call regardless of what the caller asked for, plus
   `reasoning_effort` tuned per model family (`"none"` on Qwen3, which supports fully
   disabling it; `"low"` on gpt-oss, which doesn't).
3. **One shared model = one shared rate-limit bucket, and that bucket was too small
   for three features.** `run_bot.py` originally pointed the Stage B gate, RoomContext's
   extractor, and its summarizer at the same `GroqLLMProvider(model=DEFAULT_FAST_MODEL)`
   instance. Extraction fires on *every* human turn; the gate only fires on the
   ambiguous middle case (no explicit bot mention, but a real signal). On Groq's free
   tier, extraction traffic alone exhausted `openai/gpt-oss-20b`'s ~8000 TPM limit
   during ordinary back-and-forth conversation, tripping the gate's circuit breaker --
   which then did exactly what DECISIONS.md's reliability table says it should
   (`respond: false`, stay silent) but looked, from the outside, like "the bot stopped
   responding to anything not explicitly addressed to it." Fixed by giving the gate its
   own model (`DEFAULT_FAST_MODEL`) and moving extraction/summarization onto a
   *different* model (`DEFAULT_BACKGROUND_MODEL = qwen/qwen3.8-27b`) -- Groq rate-limits
   per model, so the two no longer compete. Both are wired separately in `run_bot.py`
   (`llm_gate` vs `llm_background`); persona replies (`openai/gpt-oss-120b`) were
   already on a third, separate model and were never part of this contention.

Groq's free tier covers this project's usage pattern without needing published
per-token pricing to estimate a cost, unlike the OpenAI/Anthropic alternatives below --
but its per-model rate limits are real and easy to exhaust once more than one feature
shares a model, as above. If extraction/summarization still hits its own rate limit
under heavy load, that degrades gracefully (skipped profile update / naive-concatenation
summary, per the reliability table) rather than affecting whether the bots can speak.

Rejected: one model for everything (wastes latency/cost on the cheap calls, and was the
direct cause of bug 3 above); keeping OpenAI as the default (works fine technically,
but the project owner specifically wanted Groq, likely for its free tier and low
latency).

## TTS: Sarvam Bulbul (primary, not yet implemented) / ElevenLabs Flash v2.5 (implemented, running)

Same situation as STT: Bulbul has the most authentically Indian accent and native
handling of code-mixed script, which is the actual rubric-relevant quality bar here --
but `bots/providers/sarvam_tts.py` is a stub for the same reason as the STT side
(protocol should come from live docs + a real key, not memory).

ElevenLabs Flash v2.5 (`bots/providers/elevenlabs_tts.py`) is what's implemented:
WebSocket streaming, well-documented protocol, real (though not live-tested here --
no API key available in this environment; the code was written against the documented
message shapes, not verified against a live connection).

Rejected as primary: Smallest.ai Waves (good Indian-market fit and cost, but we could
only actually implement one "real" provider with confidence in the time available, and
ElevenLabs' protocol was the one we were most sure of), Cartesia (best-in-class latency,
but weaker Indian-language voice quality at the time of writing).

Rough cost: Bulbul ≈ $0.015-0.02/min synthesized audio, ElevenLabs Flash ≈
$0.02-0.03/min. These are rough, character-rate-derived estimates, not measured.

## Script handling: LLM-native mixed script (primary) + a small safety-net lexicon (secondary)

Full writeup in `bots/text_normalizer.py`'s docstring. Summary: the persona system
prompts (`personas/_shared.yaml`) instruct the LLM to write Hindi in Devanagari and
English/loanwords in Latin within the same sentence -- the LLM has full sentence
context, so it can disambiguate things a text-only pass can't (e.g. "main" as Hindi "I"
vs. English "main" as in "main point").

We also tried a generic phonetic transliterator (`indic-transliteration`'s ITRANS
engine) as a possible safety net and **measured it against real words instead of
assuming it would work**:

| Input | ITRANS output | Correct |
| --- | --- | --- |
| kya | क्य | क्या |
| accha | अच्च | अच्छा |
| batao | बतओ | बताओ |

It's wrong because ITRANS expects strict long-vowel/doubled-consonant marking that
casual Hinglish typing doesn't follow. Shipping that would make pronunciation *worse*.
We rejected it and shipped a small, hand-verified lexicon instead (`HINGLISH_LEXICON`
in `text_normalizer.py`) covering only high-frequency discourse words with **zero
English-homograph collision** -- words like "do" (Hindi "two" / English verb "do"),
"main" (Hindi "I" / English "main"), and "the" (Hindi "were" / English article) are
deliberately excluded, since converting the wrong sense would corrupt an English
sentence. This runs as a safety net on the LLM's output, not as the primary mechanism.

Acronyms (`ACRONYM_MAP`) get a small phonetic-Devanagari respelling (`AI` -> `ए आई`).
Arbitrary number-to-Hindi-words (e.g. "2024" -> "दो हज़ार चौबीस") is an explicit **known
gap**: no reliable library exists (checked `num2words` -- no Hindi support) and Hindi's
irregular 1-99 forms are easy to get wrong without native-speaker verification. Bare
digits pass through unchanged.

## Reliability (M7)

Every provider sits behind an interface (`STTProvider`, `LLMProvider`, `TTSProvider`,
`Extractor`, `Summarizer`, `StageBGate`) with a scripted Fake for tests, and every
*live* call site into a provider now goes through `reliability/retry.py`
(exponential backoff with full jitter) and `reliability/circuit_breaker.py` (fail
fast after repeated failures instead of piling up timeouts, half-open probe after a
cooldown). `reliability/fallback_chain.py` is the generic "try provider A, then B"
mechanism -- built and tested, but not wired to a real second link anywhere yet since
we only have one working implementation per stage (see the Sarvam gaps below); it's
ready for the day a second STT/TTS provider is implemented.

Where a stage genuinely has nowhere further to fall back to, it degrades to a
*different capability* instead:

| Failure | Degrades to |
| --- | --- |
| Stage B gate LLM call fails (retries + circuit exhausted) | `respond: false` (same as `AlwaysNoGate` -- stay silent, don't guess) |
| Orchestrator's own defense-in-depth, if a `StageBGate` implementation raises anyway | same silent fallback, `rule_fired="stage_b_crashed"` |
| RoomContext's per-turn fact/entity extractor fails | skip that turn's profile update, log it, keep going (non-critical background enrichment) |
| RoomContext's rolling summarizer fails | fall back to `NaiveSummarizer`'s concatenation (built in M2, reused here) |
| A BotAgent's LLM is unreachable (fetching the first chunk of a reply) | speak one of the persona's own `degradation_lines` ("ek second, network thoda slow hai" style) instead of going silent |
| A BotAgent's TTS is unreachable (opening the stream) | delivered as a text-only reply in room chat -- the human still gets the real answer, just without a voice |
| A BotAgent's LLM stream dies *mid-reply* (not just unreachable at the start) | whatever arrived is kept, marked `[error]` in RoomContext (distinct from `[interrupted]` -- a system failure, not an intentional human cut-off) so the persona doesn't treat it as a deliberately brief answer to build on |
| An STT provider's stream dies *mid-utterance* | that track stops cleanly (logged loudly) rather than spinning forever feeding frames into a dead connection -- no second real STT provider exists to fail over to yet |

"Nothing crashes the worker" is enforced at multiple layers, not just inside
providers: `track_pipeline.py`'s call into `on_utterance` and
`TranscriberAgent._consume_chat_stream`'s call into it are both wrapped, so a bug
anywhere downstream (orchestrator, bots, whatever) can't take out a participant's
whole transcription session or vanish as an unretrieved-exception warning in a
fire-and-forget task.

One thing found *by* testing this, not designed upfront: the first version of the
LLM-retry wrapping in `BotAgent.reply_to` broke barge-in's "abort the LLM stream"
guarantee -- wrapping the stream in a passthrough generator for the eager-first-chunk
fetch meant `aclose()` on the wrapper no longer propagated to the real generator
(Python doesn't do that automatically). The existing M6 barge-in test caught it
immediately when M7's change was layered on. Fixed with an explicit
`try/finally: await gen.aclose()`.

## Observability (M8)

`obs/trace.py`'s `Trace` carries one utterance's lifecycle timestamps end to end:
`stt_final` -> `gate_route` (Orchestrator's should_respond + routing, measured as one
span since they run as one synchronous-ish call) -> `llm_first_token` ->
`tts_first_byte` -> `audio_published`. `trace_id` is just the triggering utterance's
own `utt_id` -- already a unique per-utterance identifier, no need for a second one, and
it's what every log line along the way (`STT final`, `routing decision`, `latency
breakdown`) carries so they can be correlated by grepping/filtering on one value.
`mark()` is idempotent per stage (first call wins), matching "first token"/"first
byte"/"first audio", not every token or chunk.

For a multi-bot plan, only the **first** step gets the Trace (`plan_executor.py`) --
the <1.5s target is about the room's first response to an utterance, not every
subsequent bot's turn. A reply that degrades to chat-only (TTS down) never reaches
`audio_published`, so it's correctly excluded from the latency summary rather than
counted as a bogus 0ms or missing sample.

`obs/latency_report.py`'s `LatencyTracker` collects every completed
`end_of_speech_to_first_audio_ms` and reports p50/p95 (linear interpolation between
closest ranks, the same convention numpy's default `percentile` uses) against the
<1.5s target; `run_bot.py` prints `summary_table()` on shutdown.

`obs/json_logger.py` defaults to human-readable console logs (unchanged from M0) --
set `ROXROOM_JSON_LOGS=1` (or pass `json_format=True` to `setup_logging`) for
one-JSON-object-per-line output, with every `extra={...}` field a call site attaches
(trace_id, participant_id, per-stage latencies, the full routing decision) promoted to
top-level JSON keys rather than buried in a formatted message string -- verified end to
end against a real `Orchestrator.handle_utterance` call, not just unit-tested against a
synthetic LogRecord.

**Not done**: real measured p50/p95 from an actual live session -- no live provider
keys were available in this environment (same limitation noted throughout this
document), so the latency numbers this milestone can report are 0 (no session has ever
run end-to-end here). The mechanism is built, tested with synthetic timestamps, and
wired into `run_bot.py`; what's missing is a real run to point it at.

## Known limitations (read this before demoing)

- **No PDF was ever attached to this conversation.** The "PDF's Scenario 1-5" few-shot
  examples in `personas/dost.yaml`/`sathi.yaml` and the 7 scenario tests in
  `tests/scenario/test_scenarios.py` are reconstructed from the rules stated explicitly
  in the assignment prompt, not transcribed from the actual source document. If you
  have it, swap the fixtures/few-shots for the real content -- the code they exercise
  doesn't need to change.
- **Sarvam (STT and TTS) are stubs.** Deepgram and ElevenLabs are what's actually
  wired up and used by `run_bot.py`. Swapping to Sarvam once a key is available is a
  contained task behind the existing interfaces.
- **No live provider testing was possible in this environment** -- no API keys were
  available for Deepgram, Groq, or ElevenLabs. Every provider was verified at the
  code level (SDK method signatures checked against the installed package, protocol
  shapes cross-checked against documentation from training) but never run against a
  real endpoint. Confirm live behavior before treating any of this as production-ready.
- **Barge-in (M6) is implemented but its <200ms target is unverified.** The mechanism
  is real and tested: `BotAgent.cancel()` calls `AudioSource.clear_queue()` to flush
  already-queued audio, cancels the TTS stream, and `aclose()`s the LLM's
  async-generator stream (aborting the completion, not just abandoning it) --
  see `tests/unit/test_bot_agent.py::test_barge_in_stops_mid_reply_flushes_audio_and_aborts_llm`.
  What's unverified is actual wall-clock latency: it depends on the TTS provider
  streaming audio in small enough segments that the playback loop's between-chunk check
  is tight, which was never measured against a live connection (no TTS key available
  here). VAD-onset detection also has no false-positive tuning -- any speech-like sound
  on any human track triggers it, even a cough or a false start.
- **Barge-in only interrupts a bot already speaking.** Skipping a not-yet-started
  multi-bot plan step (M5) and interrupting an in-progress reply (M6) are two different
  mechanisms triggered from the same place (`run_bot.py`'s `on_speech_started` /
  `on_utterance`) -- the speech-onset path is just far faster to fire since it doesn't
  wait for a full utterance to finalize.
- **Cost figures throughout this document are rough, not measured** -- derived from
  published per-unit pricing and assumed usage patterns, not from an actual running
  session (which would need the live keys noted above).

## Next steps

All nine milestones are done (see README.md's status line). What's left is explicitly
about live verification, not architecture:

1. **A real end-to-end run with live provider keys.** Nothing in this system has been
   run against a real Deepgram/Groq/ElevenLabs connection in this environment --
   every provider is verified at the code/SDK level (method signatures, protocol shapes
   cross-checked against docs) but not live-tested. This is the single highest-value
   next step: it would surface any remaining integration issues and produce the actual
   measured p50/p95 latency numbers the Observability section can only describe the
   mechanism for right now.
2. **Implement the Sarvam STT and TTS adapters** (`transcriber/providers/sarvam.py`,
   `bots/providers/sarvam_tts.py`) against live docs + a real key -- our documented
   primary picks for Hindi/Hinglish accuracy and accent, currently stubs.
3. **Real before/after TTS audio** for the demo checklist (`demo/audio_samples/`
   currently has the before/after as text, not audio, for the same reason as #1).
4. **Swap in the actual assignment PDF's content** if it becomes available -- the few-
   shot examples in `personas/*.yaml`, the 7 scenario tests, and the demo script's 10
   beats are all reconstructed from the prompt's stated rules, not transcribed from a
   source document that was never attached to this conversation. The code they exercise
   doesn't need to change, only the fixtures/copy.
5. **A shared circuit breaker per LLM connection** instead of one per component
   (Stage B gate, extractor, summarizer, and each BotAgent's reply call currently each
   get their own breaker) -- would let a sustained outage trip faster across the board.
   Minor, not currently a correctness issue.
