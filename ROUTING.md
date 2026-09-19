# Routing

How the Orchestrator decides, for one finalized `Utterance`, whether a bot replies and
which one. Implementation: `src/roxroom/orchestrator/`.

## The funnel (must pass all, in order)

| # | Check | Module | On failure |
| - | --- | --- | --- |
| 1 | Not a duplicate `utt_id` | `dedupe.py` | silence, `rule_fired="duplicate"` |
| 2 | Not a bare backchannel (`haan`, `hmm`, `achha`, `ok`, ...) | `gate.py::is_backchannel` | silence, `rule_fired="backchannel"` |
| 3 | Not suppressed (a bot finished replying <1.5s ago) *unless* this utterance explicitly names a bot | `orchestrator.py::_is_suppressed` | silence, `rule_fired="suppressed_recent_reply"` |
| 4 | `should_respond` (Stage A / Stage B) | `gate.py` | silence, `rule_fired` one of `stage_a_no_signal` / `stage_b_silent` |
| 5 | Routing picks the bot(s) | `router.py` | n/a -- this is the success path |

## should_respond (Stage A / Stage B)

Stage A is four cheap, deterministic checks over the utterance text:

- **Explicit bot mention** (`bot_aliases.py`) -- exact match against a curated alias
  list per bot (`dost`/`dosth`/`dast`/`dosht`/`dosdt`, `sathi`/`saathi`/`sathee`/...).
  We deliberately did **not** use a generic edit-distance/ratio threshold here: `"dast"`
  vs `"dost"` scores identically to `"cost"` vs `"dost"` (0.75) under `difflib`'s
  `SequenceMatcher`, so any cutoff that catches the intended ASR error also fires on
  ordinary English words. A curated list for a closed set of two names is the safer
  choice.
- **Interrogative markers** -- question words (`kya`, `kaise`, `kyun`, `kab`, `kaun`,
  English `what/how/why/...`) or a literal `?`.
- **Imperative verbs** -- `batao`, `samjhao`, `bolo`, `explain`, `tell me`, ...
- **Direct 2nd-person address** -- `tum`, `aap`, `tumhe`, `you`, `your`, ...

Decision from Stage A alone:

| Stage A result | Outcome |
| --- | --- |
| Exactly one bot named | Respond=true, skip Stage B (`rule_fired` becomes whatever the router assigns, typically `explicit_address`) |
| Both bots named | Respond=true, skip Stage B, routed to `multi_bot_plan` |
| No bot named, but interrogative / imperative / direct-address present | **Ambiguous** -> Stage B |
| None of the above signals at all | Respond=false, `rule_fired="stage_a_no_signal"` (no LLM call -- this is the case that keeps cost down: plain statements and humans just chatting) |

Stage B (`LLMStageBGate`, needs a real `LLMProvider` from M4) is a small/fast model
call, ~40 output tokens, JSON-only: `{respond, addressed_bot, reason, confidence}`,
given the last 6 turns. This is what resolves the genuinely hard case Stage A can't:
*"Rahul, tumhe kya lagta hai?"* has a direct address + a question word -- identical
Stage-A signature to a question aimed at a bot -- only the LLM, with context, can tell
it's addressed to a human named Rahul and return `respond: false`. Until a real
`LLMProvider` exists, `AlwaysNoGate` is the default: ambiguous utterances stay silent
rather than guess (see DECISIONS.md's reliability section).

## Routing priority (once should_respond says yes)

1. **Explicit address** -- one bot named -> that bot. Both named -> multi-bot plan
   (below).
2. **Multi-bot plan** -- when both bots are named in one utterance (e.g. *"AI Dost tum
   pehle answer karo, phir AI Sathi ek example dena"*), build an ordered
   `ResponsePlan`: bots in the order they're mentioned, each step's `instruction` is the
   text between that mention and the next (or end of utterance). `orchestrator/
   plan_executor.py` (M5) executes it sequentially through the shared speak-lock -- each
   step's BotAgent reads RoomContext fresh, so the second step genuinely sees the first
   bot's reply. A new utterance cancels remaining *not-yet-started* steps (see that
   module's docstring for exactly what that does and doesn't interrupt -- stopping a bot
   already mid-reply is real barge-in, M6).
3. **Stage B's own `addressed_bot`** -- if Stage A found no explicit name but Stage B
   did resolve one from context/tone, use it.
4. **Persona affinity** -- keyword-scored: definitional/factual/concise language
   (`define`, `kya hota hai`, `fact`, `short mein`) -> Dost; example/analogy/elaboration/
   emotional language (`example`, `misal`, `simple mein`, `confuse`, `pareshan`) ->
   Sathi. Ties or no keywords on either side fall through.
5. **Alternation fallback** -- whoever spoke less recently (`Router.last_reply_at`,
   updated via `Orchestrator.on_bot_reply_completed`).

## Suppression window

After any bot finishes a reply, new utterances are suppressed for 1.5s **unless** the
new utterance explicitly names a bot -- explicit address always overrides suppression.
This exists to stop a burst of near-simultaneous human speech right after a bot
finishes from triggering a second, unwanted reply.

## Logging

Every call to `Orchestrator.handle_utterance` appends a `Decision` to
`orchestrator.decision_log` and logs it via `Decision.to_log_dict()`:
`{utt_id, stage_a, stage_b, chosen_bot, plan, rule_fired, respond, latency_ms}`.
`chosen_bot` is `None` for a multi-bot plan -- check `plan` for the full ordered list
in that case.

## Known gap

The 7 scenario tests in `tests/scenario/test_scenarios.py` are reconstructed from the
rules stated in the assignment prompt, not transcribed from the assignment's actual PDF
(it was never attached to this conversation). If you have the source PDF, swap the
fixtures for the real exchanges -- the Orchestrator plumbing they exercise doesn't need
to change.
