# Demo script

**Provenance note:** the assignment references a "10-point demo checklist" from its
source PDF, which was never attached to this conversation (same gap noted throughout
ROUTING.md/DECISIONS.md/the scenario tests). The 10 beats below are reconstructed from
every capability explicitly named in the assignment prompt itself -- turn-taking,
persona-affinity routing, multi-bot plans, memory/context, barge-in, text-chat parity,
and reliability -- rather than transcribed from that document. If you have the real
checklist, swap the specific lines for the real ones; the system underneath doesn't
change.

## Setup

1. `.env` filled in with `LIVEKIT_*`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`,
   `ELEVENLABS_API_KEY`, `ELEVENLABS_DOST_VOICE_ID`, `ELEVENLABS_SATHI_VOICE_ID`.
2. Terminal 1: `.venv\Scripts\python.exe -m roxroom.run_bot` -- wait for "Dost and
   Sathi connected and ready" and "transcriber live in room...".
3. Two more terminals or browser tabs, each running
   `.venv\Scripts\python.exe scripts\gen_human_token.py "<Name>"` and opening the
   printed `meet.livekit.io` link -- this is your two humans (call them Priya and
   Rahul below; use whatever names you actually joined as).
4. Keep Terminal 1 visible during the demo -- the routing decision, trace, and
   barge-in log lines are part of what you're demonstrating, not just internal noise.

## The 10 beats

### 1. Both bots are real, named participants
Just look at the room's participant list once everyone's joined: `Roxstar AI Dost` and
`Roxstar AI Sathi` should be listed like any human, each with their own audio track.
*Proves:* the hard constraint that bots aren't a backend trick, they're room members.

### 2. Direct question to Dost
**Priya says:** *"AI Dost, tumhara naam kya hai?"*
Dost should answer promptly, in Hinglish, concise (2-4 sentences), and the same text
should land in room chat. *Proves:* explicit-address routing, persona reply, TTS
pronunciation of Hindi words (not English-phonetic "kai-ya" for "kya").

### 3. Direct question to Sathi, mispronounced on purpose
**Rahul says:** *"Sathee, ek example do na"* (deliberately using the ASR-noise
spelling "Sathee" instead of "Sathi").
Sathi should still answer -- warmer, more example-driven than Dost's style.
*Proves:* fuzzy alias matching survives ASR misrecognition; the two personas are
distinguishable by tone alone.

### 4. Persona-affinity routing without naming a bot
**Priya says:** *"5G kya hota hai?"* (no bot named -- expect Dost, factual/definitional).
**Rahul says:** *"Mujhe samajh nahi aaya, ek simple example do."* (no bot named --
expect Sathi, elaboration/example request).
*Proves:* intent-based routing (Stage B + persona-affinity) works without an explicit
address.

### 5. Humans talking to each other -> silence
**Priya says to Rahul (not a bot):** *"Rahul, tumhe kal ka match dekha?"*
**Rahul replies:** *"haan"* (a bare backchannel).
Neither should get a bot reply. *Proves:* the should_respond gate correctly
distinguishes human-to-human conversation, and backchannels are dropped outright.

### 6. Memory survives across turns, without leaking between speakers
**Rahul says:** *"Mera naam Rahul hai aur mujhe cricket bahut pasand hai."*
**Priya says (unrelated):** *"Mujhe painting pasand hai."*
*(continue with 2-3 more unrelated exchanges)*
**Rahul says, later:** *"AI Dost, mujhe kaunsa sport pasand hai?"*
Dost's answer should correctly reference cricket -- and if you ask what Priya likes, it
should say painting, never cricket. *Proves:* per-speaker profiles in RoomContext are
isolated and survive the rolling-summary compression.

### 7. Multi-bot plan
**Priya says:** *"AI Dost tum pehle answer karo, phir AI Sathi ek example dena."*
Dost answers first, then Sathi answers second -- and Sathi's answer should sound like
it heard Dost's (not a repeat, not contradictory). *Proves:* ordered multi-bot
`ResponsePlan` execution through the shared speak-lock, with live context handoff
between bots.

### 8. Barge-in
**Rahul asks a question that prompts a longer answer**, e.g. *"AI Sathi, network kaise
kaam karta hai, poora samjhao."*
**While Sathi is still talking, Priya interrupts:** *"ruko, ruko!"*
Sathi should stop within about a beat (watch Terminal 1 for the "barge-in" log line),
not finish her sentence. Then **Priya says:** *"phir se, simple example se samjhao."*
Sathi's next answer should pick up the thread, not restart from "network kaise kaam
karta hai" as if nothing happened. *Proves:* real barge-in (audio flush, LLM abort,
`[interrupted]`-marked context) and the resume-coherently persona instruction.

### 9. Text chat and voice share one context
**Rahul types in room chat** (not speaking): *"AI Dost, aaj weather kaisa hai?"*
Dost should reply in **both** voice and chat. Then **Priya asks by voice:** *"usne
abhi kya bola?"* ("what did he just say?") -- a bot's answer should correctly
reference the text-chat exchange. *Proves:* voice and text share one Utterance/
RoomContext pipeline end to end, not two separate systems.

### 10. Reliability and observability
Two options, pick based on what you want to show:
- **Narrated failure:** temporarily point `OPENAI_API_KEY` at an invalid value,
  restart, and ask a question -- the bot should still say something in-persona
  ("ek second, network thoda slow hai" style) instead of going silent or crashing.
  Restore the real key afterward.
- **Observability tour:** run with `ROXROOM_JSON_LOGS=1` set, ask one question, and
  grep the terminal output for that utterance's `trace_id` -- show the `STT final`,
  `routing decision`, and `latency breakdown` log lines all sharing it. Then Ctrl+C
  and show the latency summary table (p50/p95 against the <1.5s target) that prints
  on shutdown.
*Proves:* the reliability fallback chain and the tracing/latency infrastructure are
real, not just described in DECISIONS.md.

## If something doesn't go as scripted

- No reply at all: check Terminal 1 for a routing decision log line -- was
  `respond: false`? That's probably correct behavior (see beat 5), not a bug.
- Wrong bot answered: check the `rule_fired` field in that log line against
  ROUTING.md's priority table.
- A reply came back but sounded odd/mispronounced: check whether the LLM emitted
  Romanized Hindi instead of Devanagari (see DECISIONS.md's script-handling section)
  -- `bots/text_normalizer.py`'s safety net only covers a specific word list, not
  everything.
