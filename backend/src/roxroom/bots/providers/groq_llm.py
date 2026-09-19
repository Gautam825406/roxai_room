"""Groq implementation of LLMProvider -- what run_bot.py wires up by default.

Model choice: **`openai/gpt-oss-120b`** for persona replies, **`openai/gpt-oss-20b`**
for the cheap/fast calls (the Stage B gate and RoomContext's extraction/summarization)
-- construct a second instance with that model rather than sharing one model for
everything; see run_bot.py for how the two are wired up separately.

Groq's hosted-model catalog changes over time (llama-3.3-70b-versatile and
llama-3.1-8b-instant, an earlier pick here, were removed from it) -- these two were
confirmed against a live account's `client.models.list()` as the current active
chat-capable models. If a call to this provider 404s with "model ... does not exist",
your account's catalog has moved on again; pass `model=` explicitly with a value from
https://console.groq.com/docs/models or your own `models.list()` call.

The Groq API is Chat-Completions-compatible (same shape OpenAI's endpoint uses), so
this mirrors `openai_llm.py`'s structure closely -- streaming re-chunked into sentences
for `response_format="text"`, buffered into one JSON payload for `response_format=
"json"`. One real difference: Groq's endpoint takes `max_tokens`, not OpenAI's newer
`max_completion_tokens`.

`reasoning_effort` (see `_reasoning_effort()`) is passed on every call: gpt-oss and
Qwen3 are both reasoning models that spend part of their `max_tokens` budget on hidden
chain-of-thought before the actual answer -- confirmed live, the JSON-mode gate/
extraction calls (small `max_output_tokens` budgets, e.g. 40-200) reliably 400'd with
`json_validate_failed` and an empty `failed_generation` at the default reasoning
effort, because reasoning alone exhausted the budget before any JSON was emitted.
Qwen3 supports disabling reasoning outright (`"none"`); gpt-oss doesn't, so it gets the
lowest effort level (`"low"`) instead.

`_JSON_MIN_TOKENS` floors the token budget on every JSON-mode call regardless of what
the caller asked for. Callers size `max_output_tokens` for the *answer* (e.g. gate.py
asks for 40 -- plenty for its tiny `{"respond": ..., "addressed_bot": ...}` verdict on
a non-reasoning model), with no way to know a specific provider also needs headroom for
hidden reasoning it never sees. Confirmed live: even with `reasoning_effort="low"`,
gate.py's 40-token budget still 400'd intermittently, because reasoning length varies
with prompt complexity and can still exceed a budget that tight. This is a Groq/gpt-oss
quirk, so it's absorbed here rather than pushed onto provider-agnostic call sites.
"""
from __future__ import annotations

from typing import AsyncIterator, Literal

from groq import AsyncGroq

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.bots.sentence_chunking import sentence_chunk

DEFAULT_REPLY_MODEL = "openai/gpt-oss-120b"
DEFAULT_FAST_MODEL = "openai/gpt-oss-20b"

# A *different* model from DEFAULT_FAST_MODEL, deliberately -- Groq's rate limits are
# per-model, so RoomContext's extractor/summarizer (fired on every human turn, not on
# the reply-critical path) get their own token budget here instead of competing with
# the Stage B gate for DEFAULT_FAST_MODEL's. Confirmed live: on the free tier,
# DEFAULT_FAST_MODEL's ~8000 TPM limit was getting exhausted by extraction alone during
# a normal back-and-forth, tripping the gate's circuit breaker and making the bots go
# silent on anything not explicitly addressed to them -- see run_bot.py for the wiring.
DEFAULT_BACKGROUND_MODEL = "qwen/qwen3.8-27b"

_JSON_MIN_TOKENS = 200

# Qwen3 models support fully disabling reasoning ("none"); gpt-oss models don't and
# need a nonzero effort level, so this picks the cheaper/more-reliable option per
# model family rather than hardcoding one value for every Groq model this project uses.
def _reasoning_effort(model: str) -> str:
    return "none" if "qwen" in model else "low"


class GroqLLMProvider(LLMProvider):
    def __init__(self, api_key: str, *, model: str = DEFAULT_REPLY_MODEL, max_tokens: int = 400) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def stream_reply(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[str]:
        conversation = [{"role": m.role, "content": m.content} for m in messages]
        max_tokens = max_output_tokens or self._max_tokens
        reasoning_effort = _reasoning_effort(self._model)

        if response_format == "json":
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                max_tokens=max(max_tokens, _JSON_MIN_TOKENS),
                response_format={"type": "json_object"},
                reasoning_effort=reasoning_effort,
            )
            yield completion.choices[0].message.content or ""
            return

        async def raw_tokens() -> AsyncIterator[str]:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                max_tokens=max_tokens,
                stream=True,
                reasoning_effort=reasoning_effort,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        async for sentence in sentence_chunk(raw_tokens()):
            yield sentence
