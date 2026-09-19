"""Rolling-summary compression: folds a batch of turns into a running Hinglish summary,
merged with whatever summary already existed.

Token budget note (documented per the spec's request): we cap the summary at
~110 words as a proxy for "~150 tokens". We don't ship a tokenizer dependency just for
this -- English/Hinglish text averages a bit under 1.4 tokens/word, so 110 words is a
conservative stand-in for a 150-token ceiling. Good enough for a prompt-size budget;
not meant to be exact.

`LLMSummarizer` needs a real `LLMProvider` (M4) to produce a genuinely compressed
summary. `NaiveSummarizer` is the dependency-free default: it doesn't compress so much
as concatenate-and-truncate, which is honest about being a placeholder/fallback rather
than a real summarizer -- it's also what RoomContext falls back to if the LLM call
fails (see RELIABILITY fallback-chain approach in DECISIONS.md).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.context.models import Turn

APPROX_WORDS_FOR_150_TOKENS = 110


class Summarizer(ABC):
    @abstractmethod
    async def summarize(self, turns: list[Turn], previous_summary: str) -> str: ...


class NaiveSummarizer(Summarizer):
    async def summarize(self, turns: list[Turn], previous_summary: str) -> str:
        new_bits = "; ".join(f"{t.display_name}: {t.text}" for t in turns if t.text.strip())
        combined = f"{previous_summary} {new_bits}".strip()
        words = combined.split()
        if len(words) > APPROX_WORDS_FOR_150_TOKENS:
            words = words[-APPROX_WORDS_FOR_150_TOKENS:]
        return " ".join(words)


_SUMMARY_SYSTEM_PROMPT = """Compress these turns into an updated running summary of a \
group voice conversation, written in conversational Hinglish. Merge with the previous \
summary where relevant; keep names, stated facts, and open topics/questions; drop small \
talk and backchannels. Output ONLY the summary text, at most ~150 tokens (~110 words), \
no preamble, no labels, no markdown."""


class LLMSummarizer(Summarizer):
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def summarize(self, turns: list[Turn], previous_summary: str) -> str:
        new_bits = "\n".join(f"{t.display_name}: {t.text}" for t in turns)
        messages = [
            ChatMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Previous summary:\n{previous_summary or '(none yet)'}\n\nNew turns:\n{new_bits}",
            ),
        ]
        chunks = [chunk async for chunk in self._llm.stream_reply(messages, max_output_tokens=220)]
        return "".join(chunks).strip()
