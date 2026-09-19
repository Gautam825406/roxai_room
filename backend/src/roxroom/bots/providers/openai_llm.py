"""OpenAI implementation of LLMProvider -- what run_bot.py wires up by default.

Model choice: **`gpt-4o`** for persona replies, **`gpt-4o-mini`** for the cheap/fast
calls (the Stage B gate and RoomContext's extraction/summarization) -- construct a
second instance with that model rather than sharing one model for everything; see
run_bot.py for how the two are wired up separately.

These two model IDs are deliberately conservative: they're exact, currently-correct
OpenAI model identifiers I'm confident about. If your account has access to a newer
GPT generation, pass `model=` explicitly when constructing this provider -- verify the
exact model ID string against https://platform.openai.com/docs/models first, the same
caution this project applies to every provider integration it can't live-test (see
DECISIONS.md).

Streaming: `response_format="text"` re-chunks the raw token stream into sentences (see
bots/sentence_chunking.py) so downstream TTS can start on the first sentence before
generation finishes. `response_format="json"` (the gate/extraction calls) uses the
Chat Completions API's native JSON mode (`response_format={"type": "json_object"}`)
and buffers the full response into one chunk, since those need a single valid JSON
payload, not sentence pieces. Note OpenAI's JSON mode additionally requires the word
"json" to appear somewhere in the prompt -- our gate/extractor system prompts already
say "Return valid JSON only", which satisfies that.
"""
from __future__ import annotations

from typing import AsyncIterator, Literal

from openai import AsyncOpenAI

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.bots.sentence_chunking import sentence_chunk

DEFAULT_REPLY_MODEL = "gpt-4o"
DEFAULT_FAST_MODEL = "gpt-4o-mini"


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, *, model: str = DEFAULT_REPLY_MODEL, max_tokens: int = 400) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
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

        if response_format == "json":
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            yield completion.choices[0].message.content or ""
            return

        async def raw_tokens() -> AsyncIterator[str]:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                max_completion_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        async for sentence in sentence_chunk(raw_tokens()):
            yield sentence
