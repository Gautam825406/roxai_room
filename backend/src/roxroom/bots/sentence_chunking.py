"""Shared by every streaming LLMProvider implementation: re-buffers a raw token/delta
stream into sentence-sized chunks (so downstream TTS can start on the first sentence
before generation finishes), splitting on the first sentence-ending punctuation found.

Provider-agnostic on purpose -- Anthropic, OpenAI, or anything else streaming raw
text deltas all need the exact same re-chunking, so it lives here rather than being
duplicated (or one importing it from the other, which would create an odd dependency
between two sibling provider modules).
"""
from __future__ import annotations

from typing import AsyncIterator

SENTENCE_END_CHARS = frozenset({".", "!", "?", "।"})


def _find_sentence_break(buffer: str) -> int | None:
    for i, ch in enumerate(buffer):
        if ch in SENTENCE_END_CHARS:
            return i + 1
    return None


async def sentence_chunk(token_iter: AsyncIterator[str]) -> AsyncIterator[str]:
    """Splits on the first sentence-ending punctuation found (., !, ?, the Hindi ।
    danda), flushing whatever's left (even without terminal punctuation) once the
    source stream ends."""
    buffer = ""
    async for token in token_iter:
        buffer += token
        cut = _find_sentence_break(buffer)
        while cut is not None:
            yield buffer[:cut]
            buffer = buffer[cut:]
            cut = _find_sentence_break(buffer)
    if buffer.strip():
        yield buffer
