"""Anthropic Claude implementation of LLMProvider.

Model choice (DECISIONS.md): Claude Sonnet 5 (`claude-sonnet-5`) for persona replies --
best instruction-following for the tone/register constraints that matter here (banned
literary-Hindi vocabulary, keeping Dost/Sathi distinguishable, Devanagari/Latin script
mixing). Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) is for the cheap/fast calls --
the Stage B gate and RoomContext's extraction/summarization -- construct a second
instance with that model rather than sharing one model for everything; see
run_bot.py / DECISIONS.md for how the two are wired up separately.

Streaming: `response_format="text"` re-chunks the raw token stream into sentences (so
downstream TTS can start on the first sentence before generation finishes) instead of
yielding raw token fragments. `response_format="json"` (the gate/extraction calls)
instead buffers the full response and yields it once, since those need a single valid
JSON payload, not sentence chunks.

Not the LLM run_bot.py wires up by default (see openai_llm.py / DECISIONS.md for why
the project switched to OpenAI) -- kept available and tested as a documented
alternative behind the same LLMProvider interface, swappable back in one line.
"""
from __future__ import annotations

from typing import AsyncIterator, Literal

from anthropic import AsyncAnthropic

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.bots.sentence_chunking import sentence_chunk

DEFAULT_REPLY_MODEL = "claude-sonnet-5"
DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"

__all__ = ["AnthropicLLMProvider", "DEFAULT_REPLY_MODEL", "DEFAULT_FAST_MODEL", "sentence_chunk"]


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, *, model: str = DEFAULT_REPLY_MODEL, max_tokens: int = 400) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def stream_reply(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[str]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        conversation = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        async def raw_tokens() -> AsyncIterator[str]:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_output_tokens or self._max_tokens,
                system=system,
                messages=conversation,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        if response_format == "json":
            chunks = [chunk async for chunk in raw_tokens()]
            yield "".join(chunks)
            return

        async for sentence in sentence_chunk(raw_tokens()):
            yield sentence
