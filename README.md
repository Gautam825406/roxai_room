# RoxRoom AI

A LiveKit voice room where two AI participants — **Roxstar AI Dost** (male) and
**Roxstar AI Sathi** (female) — listen to multiple humans speaking
Hindi/Hinglish/English, decide when and which bot should reply, and answer in
natural conversational Hinglish with an Indian accent.

Status: **M9 — feature-complete prototype.** All nine milestones are done: transcriber,
context/memory, orchestrator routing, a live voice bot, two bots with full multi-bot
routing, real barge-in, reliability fallbacks, and tracing/latency observability. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the component + sequence diagrams,
[DECISIONS.md](DECISIONS.md) for provider choices and every known gap, and
[ROUTING.md](ROUTING.md) for the routing decision table. `demo/DEMO_SCRIPT.md` has a
10-beat walkthrough covering the whole system end to end.

**Read before treating this as done:** no live provider keys (Deepgram/Groq/
ElevenLabs) were available in this environment, so nothing here has been run end-to-end
against real services -- every provider is verified at the code/SDK level, not
live-tested. Sarvam's STT and TTS adapters are stubs (ElevenLabs/Deepgram are what
actually run). See DECISIONS.md's "Known limitations" section for the full list before
demoing.

## Layout

Two independent top-level folders:

- **`backend/`** — the Python LiveKit worker (transcriber, orchestrator, bots,
  providers) plus the token server the frontend talks to. Everything below runs
  `cd backend` first, then a `.venv\Scripts\python.exe ...` command.
- **`frontend/`** — the React app humans actually join the room through. See
  [`frontend/README.md`](frontend/README.md).

## Prereqs

- Python 3.11+
- Node.js (for the frontend, see `frontend/README.md`)
- A LiveKit server or [LiveKit Cloud](https://livekit.io/cloud) project (free tier is fine)
  — you need its `wss://` URL, API key, and API secret.

**New here?** [`SETUP.md`](SETUP.md) is a complete step-by-step guide to every API key
this project needs (LiveKit, Deepgram, Groq, ElevenLabs), where to get each one,
what it costs, and what you can run at each stage. The quick version below assumes
you've already got LiveKit credentials.

## One-command start (M0)

```powershell
cd backend
cp .env.example .env   # then fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
./scripts/run_dev.ps1
```

(`./scripts/run_dev.sh` on macOS/Linux/WSL.)

This creates a venv, installs the package, and connects `Roxstar AI Dost` to the
`ROXROOM_ROOM_NAME` room (default `roxroom-dev`), publishing a 440Hz test tone.
Console logs show connection status and any human participants joining/leaving.

## Joining as a human (to verify the tone)

```powershell
cd backend
.venv\Scripts\python.exe scripts\gen_human_token.py "Priya"
```

Run it twice with different names to join as two humans. Each invocation prints a
`https://meet.livekit.io/custom?...` link — open it in a browser tab to join the room
and confirm you can hear the bot's tone, and that the worker's console logs show your
participant connecting/disconnecting.

**Or use the React frontend** (`frontend/`) instead of the CLI script + hosted meet
client — a proper join screen, participant list, mic control, chat panel, and live
audio-reactive avatars. See [`frontend/README.md`](frontend/README.md); it talks to a
small token-minting HTTP server run from `backend/`
(`.venv\Scripts\python.exe -m roxroom.token_server`, needs `pip install -e ".[server]"`).

## Transcriber (M1)

Requires a [Deepgram](https://deepgram.com) API key (Sarvam is the documented primary
pick for accent/code-mixing quality, but its provider adapter isn't implemented yet —
see `transcriber/providers/sarvam.py`'s docstring). Add `DEEPGRAM_API_KEY` to `.env`,
then:

```powershell
cd backend
.venv\Scripts\python.exe -m roxroom.run_transcriber
```

Join as two humans (see above) and talk — one at a time, then overlapping. The
console logs one `Utterance` per finalized turn, tagged with the speaking
participant's identity/name, e.g.:

```
[a1b2c3d4e5f6] Priya: 'mujhe cricket bahut pasand hai' (lang=hi-en)
```

Things to verify: each human gets attributed correctly (no cross-talk mixing between
two simultaneous speakers, since each track has its own STT+VAD session), a trailing
"...kya"/"...kaise" doesn't get cut into two utterances, and a human leaving mid-sentence
still flushes whatever was said as a final utterance instead of dropping it.

## Talking to Dost and Sathi live (M5)

Needs `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `ELEVENLABS_API_KEY`,
`ELEVENLABS_DOST_VOICE_ID`, and `ELEVENLABS_SATHI_VOICE_ID` (real voice ids from
[elevenlabs.io/app/voice-library](https://elevenlabs.io/app/voice-library) -- pick a
male and a female voice) in `.env`, in addition to the `LIVEKIT_*` values from M0. Then:

```powershell
cd backend
.venv\Scripts\python.exe -m roxroom.run_bot
```

Join as a human (`gen_human_token.py`, see above, or the frontend). Things to try:
- *"AI Dost, tumhara naam kya hai?"* -- Dost answers, out loud and in room chat.
- *"Sathi, ek example do na"* -- Sathi answers (ASR-noise-tolerant name matching too:
  "sathee" works the same way).
- *"AI Dost tum pehle answer karo, phir AI Sathi ek example dena"* -- both bots speak in
  order, sequentially through the shared floor; Sathi's answer should sound like it
  actually heard Dost's answer (it does -- see ROUTING.md's multi-bot plan section).
- Type a message in the room's text chat instead of speaking -- same gating/routing
  applies, and the bot's reply comes back as both voice and chat.
- Start talking again while a two-step plan is still running -- the not-yet-started
  step should be skipped (check the logs for "cancelling remaining steps").
- Start talking *while a bot is actively speaking* -- it should stop within roughly a
  TTS-chunk's worth of latency (see DECISIONS.md for the honest caveat on the <200ms
  target), and the room chat / RoomContext should show its reply prefixed
  `[interrupted]`. Then say something like "ruko, phir se simple example do" and the
  bot should pick up from where it left off instead of restarting from scratch.

**Script-mixing note:** the LLM is instructed to emit Hindi in Devanagari + English in
Latin directly; `bots/text_normalizer.py` is a small safety-net lexicon layered on top
(only for words with zero English-homograph collision -- see its docstring for why a
generic transliteration library was tried and rejected). `demo/audio_samples/` has a
before/after **text** sample; producing the actual before/after **audio** pair the demo
checklist wants needs a live TTS key, which isn't available in this environment -- see
DECISIONS.md's known-limitations section.

## Observability (M8)

Console logs are human-readable by default; run with `ROXROOM_JSON_LOGS=1` set for
structured one-JSON-object-per-line logs instead:

```powershell
cd backend
$env:ROXROOM_JSON_LOGS = "1"
.venv\Scripts\python.exe -m roxroom.run_bot
```

Every utterance gets a `trace_id` (its own `utt_id`) that shows up on the `STT final`,
`routing decision`, and `latency breakdown` log lines, so `grep`/`jq`-filtering by one
`trace_id` reconstructs that utterance's whole path through the system. On Ctrl+C,
`run_bot.py` prints a latency summary table (end-of-speech -> first bot audio, p50/p95
against the <1.5s target) for whatever replies happened during that session.

## RoomContext & prompt token budget (M2)

Every human turn (voice or text-chat -- they share one buffer) and every bot reply is
appended to `RoomContext.turns`. Three knobs control what actually reaches an LLM
prompt:

| Knob | Value | Why |
| --- | --- | --- |
| Compression cadence | every 10 turns added | matches the spec's "every 10 turns" cadence |
| Compression batch | oldest 5 turns per cycle | "compress the oldest half" of that 10-turn window |
| Hard buffer cap | 20 turns | safety net if the summarizer is failing -- degrade by dropping detail, not by growing memory unbounded |
| Prompt window | summary + last 12 turns | fixed prompt size regardless of how many turns are currently buffered (5-20) |
| Summary length | ~110 words (~150 tokens) | approximated by word count, not a real tokenizer -- English/Hinglish text runs a bit under 1.4 tokens/word, so 110 words is a conservative stand-in for a 150-token ceiling |

So a bot's prompt is always: `rolling_summary` (≤~150 tokens, itself an ongoing
compression of everything older) + the last 12 raw turns + a `RECENT ENTITIES: ...`
line (last 3 salient entities, most-recent-last) so the model doesn't have to guess
what "uski"/"wahi topic"/"ye" refers to.

Per-speaker profiles (`{name, stated_facts, preferences, language_style}`) are updated
by an async `Extractor` call after every *human* turn only -- bot turns never trigger
extraction, and facts are attached strictly to the turn's own speaker, so one person's
stated facts can never leak into another's profile.

Both the extractor and the summarizer are swappable (`Extractor`/`Summarizer`
interfaces in `context/extractor.py` / `context/summarizer.py`). The default
(`NullExtractor` / `NaiveSummarizer`) needs no LLM and degrades gracefully; the real
`LLMExtractor` / `LLMSummarizer` need a working `LLMProvider`, which lands in M4.

## Repo layout

```
backend/
  src/roxroom/
    config.py            environment/config loading
    livekit_token.py      access token minting
    token_server.py        HTTP token server the frontend calls (FastAPI)
    main.py                 M0 entrypoint (bot connects + publishes test tone)
    transcriber/             per-participant streaming STT (M1)
    context/                  RoomContext, turn buffer, summaries, profiles, coref (M2)
    orchestrator/              turn detection, relevance gate, routing (M3); speak_lock.py (M4)
    bots/                      BotAgent, persona loader, text normalizer, LLM/TTS providers (M4)
    reliability/                circuit breakers, retries, fallback chains (M7)
    obs/                         structured logging, latency tracing (M8)
  personas/                dost.yaml / sathi.yaml persona prompts (M4)
  tests/                    unit / scenario / language_quality / chaos suites
  scripts/                  dev run scripts, human-join token helper
  demo/                     demo script + before/after TTS audio samples
  pyproject.toml
  .env / .env.example
frontend/
  src/                    React app (join screen, room UI, chat, audio visualizers)
  README.md               frontend-specific setup
```

## Architecture, decisions, routing

See [ARCHITECTURE.md](ARCHITECTURE.md), [DECISIONS.md](DECISIONS.md),
[ROUTING.md](ROUTING.md) — filled in as their corresponding milestones land.
