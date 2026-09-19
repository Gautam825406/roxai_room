# Architecture

## Component diagram

Three LiveKit connections in one worker process: the Transcriber (hidden participant),
and one connection per BotAgent. The Orchestrator makes no LiveKit connection of its
own -- it's pure decision logic, fed `Utterance`s and returning `Decision`s. RoomContext
and the SpeakLock are shared, in-process state, not services.

```mermaid
graph TB
    subgraph Room["LiveKit Room"]
        Humans["Humans (audio + chat)"]
        DostTrack["Dost's published audio track"]
        SathiTrack["Sathi's published audio track"]
        ChatTopic["Room chat (lk.chat text stream)"]
    end

    subgraph Worker["Worker process (one asyncio event loop)"]
        Transcriber["Transcriber<br/>(hidden participant)"]
        Orchestrator["Orchestrator<br/>(gate + router, pure logic)"]
        PlanExecutor["PlanExecutor"]
        RoomContext[("RoomContext<br/>(shared)")]
        SpeakLock{{"SpeakLock<br/>(the floor, shared)"}}
        DostAgent["BotAgent: Dost"]
        SathiAgent["BotAgent: Sathi"]
    end

    subgraph Providers["Providers (behind interfaces, each with retry + circuit breaker)"]
        STT["STTProvider<br/>Deepgram (real) / Sarvam (stub)"]
        LLMGate["LLMProvider (gate)<br/>openai/gpt-oss-20b"]
        LLMBackground["LLMProvider (background)<br/>qwen/qwen3.8-27b"]
        LLMReply["LLMProvider (reply)<br/>openai/gpt-oss-120b"]
        TTS["TTSProvider<br/>ElevenLabs (real) / Sarvam Bulbul (stub)"]
    end

    Humans -- audio track --> Transcriber
    Humans -- chat message --> Transcriber
    Transcriber -- per-track session --> STT
    Transcriber -- Utterance --> Orchestrator
    Orchestrator -- Stage B gate call --> LLMGate
    RoomContext -- extraction / summary --> LLMBackground
    Orchestrator -- Decision --> PlanExecutor
    PlanExecutor --> DostAgent
    PlanExecutor --> SathiAgent
    DostAgent --> LLMReply
    SathiAgent --> LLMReply
    DostAgent --> TTS
    SathiAgent --> TTS
    DostAgent -- publish --> DostTrack
    SathiAgent -- publish --> SathiTrack
    DostAgent -- reply --> ChatTopic
    SathiAgent -- reply --> ChatTopic
    DostAgent <--> SpeakLock
    SathiAgent <--> SpeakLock
    DostAgent <--> RoomContext
    SathiAgent <--> RoomContext
    Orchestrator -.->|reads| RoomContext
```

Notes on the boundaries:
- **Transcriber** owns one STT+VAD session per subscribed *human* track (bot audio is
  excluded) and also listens on the chat topic -- both become the same `Utterance`
  type, so voice and text share one pipeline from the very first hop.
- **Orchestrator** never touches a BotAgent, TTS, or audio directly. Its only I/O is the
  Stage B gate's small/fast LLM call, and that call is wrapped in retry + circuit
  breaker with a silent fallback (see DECISIONS.md). `LLMGate` and `LLMBackground` are
  deliberately different models/providers -- on Groq, RoomContext's per-turn
  extraction traffic was starving the gate's shared rate-limit budget when both used
  the same model (DECISIONS.md has the full story).
- **RoomContext** is read by the Orchestrator (Stage B context, persona-affinity input)
  and by both BotAgents (prompt building) and written to by the Transcriber (human
  turns) and both BotAgents (bot turns) -- one shared object, not a service with its own
  connection.
- **SpeakLock** is a plain `asyncio.Lock` plus outcome tracking; both BotAgents hold a
  reference to the same instance so only one can synthesize audio at a time.

## Sequence: voice question -> reply

```mermaid
sequenceDiagram
    participant H as Human
    participant LK as LiveKit Room
    participant T as Transcriber
    participant STT as STTProvider
    participant O as Orchestrator
    participant PE as PlanExecutor
    participant B as BotAgent (Dost)
    participant LLM as LLMProvider
    participant TTS as TTSProvider
    participant RC as RoomContext

    H->>LK: speaks "AI Dost, tumhara naam kya hai?"
    LK->>T: subscribed audio frames
    T->>STT: per-track streaming session
    STT-->>T: partial transcripts, then final
    T->>T: EndpointPolicy finalizes the utterance
    T->>RC: add_turn(human turn)
    T->>O: handle_utterance(Utterance) [trace: stt_final]
    O->>O: Stage A: explicit "Dost" mention -> respond, skip Stage B
    O-->>T: Decision{chosen_bot: dost} [trace: gate_route]
    T->>PE: execute(plan, trace)
    PE->>B: reply_to(instruction, trace)
    B->>RC: build_prompt_context() (summary + last 12 turns + RECENT ENTITIES)
    B->>LLM: stream_reply(messages)
    LLM-->>B: first sentence chunk [trace: llm_first_token]
    B->>TTS: open_stream() + push_text(normalized chunk)
    TTS-->>B: first audio chunk [trace: tts_first_byte]
    B->>LK: capture_frame() [trace: audio_published]
    B->>LK: send_text(reply) on the chat topic
    B->>RC: add_turn(bot turn)
    H->>LK: hears Dost's reply (and sees it in chat)
```

## Sequence: barge-in

```mermaid
sequenceDiagram
    participant H as Human
    participant T as Transcriber (per-track VAD)
    participant Run as run_bot.py wiring
    participant SL as SpeakLock
    participant B as BotAgent (currently speaking)
    participant TTS as TTSProvider
    participant LLM as LLMProvider
    participant AS as AudioSource

    Note over B: Dost is mid-reply, holds the floor
    H->>T: starts speaking (interrupts)
    T->>T: VAD: silence -> speech transition
    T->>Run: on_speech_started(participant_id)
    Run->>SL: who holds the floor? -> "dost"
    Run->>B: cancel("barge_in")
    B->>AS: clear_queue() (flush already-queued audio)
    B->>TTS: cancel() (stop synthesis, close connection)
    B->>LLM: text_stream.aclose() (abort the completion)
    B->>SL: release(dost, INTERRUPTED)
    B->>B: RoomContext.add_turn("[interrupted] " + partial text)
    Note over H,B: human's new utterance now flows through<br/>the normal voice-question-to-reply sequence
```

## Sequence: two-bot plan

```mermaid
sequenceDiagram
    participant H as Human
    participant O as Orchestrator
    participant R as Router
    participant PE as PlanExecutor
    participant D as BotAgent (Dost)
    participant S as BotAgent (Sathi)
    participant RC as RoomContext

    H->>O: "AI Dost tum answer karo, phir AI Sathi example dena"
    O->>O: Stage A: BOTH bots mentioned -> explicit, skip Stage B
    O->>R: route(text, ...)
    R->>R: find_bot_mentions() -> [dost@pos1, sathi@pos2]
    R-->>O: RouteDecision{plan: [dost:"answer karo", sathi:"example dena"]}
    O->>PE: execute(plan, trace)
    PE->>D: reply_to("answer karo", trace)
    D->>RC: build prompt (no prior bot turn yet this exchange)
    D->>D: LLM stream -> TTS -> publish
    D->>RC: add_turn(Dost's reply)
    PE->>S: reply_to("example dena", trace=None)
    S->>RC: build prompt (RoomContext NOW includes Dost's reply)
    S->>S: LLM stream -> TTS -> publish
    S->>RC: add_turn(Sathi's reply)
    Note over D,S: both hold the SAME SpeakLock in turn --<br/>never speaking simultaneously
```

## Sequence: provider failure

```mermaid
sequenceDiagram
    participant B as BotAgent
    participant Br as CircuitBreaker
    participant LLM as LLMProvider
    participant TTS as TTSProvider
    participant Chat as Room chat
    participant RC as RoomContext

    B->>Br: call(fetch first LLM chunk)
    Br->>LLM: attempt 1
    LLM--xBr: ConnectionError
    Br->>LLM: attempt 2 (jittered backoff)
    LLM--xBr: ConnectionError
    Br-->>B: RetryError (circuit opens once threshold is hit)
    B->>B: fall back to this persona's degradation_line
    B->>Br: call(open TTS stream)
    Br->>TTS: attempt 1
    TTS-->>Br: success
    B->>Chat: send_text(degradation line)
    B->>RC: add_turn(degradation line)
    Note over B,TTS: If TTS were down INSTEAD of the LLM:<br/>the real LLM answer is consumed directly,<br/>audio is skipped, delivered as chat-only text
```

## Where each spec requirement lives

| Requirement | Component |
| --- | --- |
| Bots as real, named participants | `BotAgent.connect()`, `config.py`'s `BotIdentity` |
| Per-participant speaker attribution | `transcriber/agent.py` (one STT+VAD session per track) |
| Turn detection / relevance gate | `orchestrator/gate.py` |
| Routing (explicit/multi-bot/affinity/alternation) | `orchestrator/router.py`, `ROUTING.md` |
| Rolling summary + speaker profiles + coreference | `context/room_context.py`, `context/coref.py` |
| Never-both-speaking | `orchestrator/speak_lock.py` |
| Multi-bot sequential execution | `orchestrator/plan_executor.py` |
| Barge-in | `bot_agent.py::cancel()`, `track_pipeline.py`'s `on_speech_started` |
| Script-mixing (Hinglish TTS pronunciation) | `bots/text_normalizer.py`, `personas/_shared.yaml` |
| Reliability (retry/circuit-breaker/fallback) | `reliability/` |
| Tracing + latency | `obs/trace.py`, `obs/latency_report.py` |
