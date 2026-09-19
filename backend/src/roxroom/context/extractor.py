"""Per-human-turn extraction: pulls stated facts, preferences, salient entities, and a
language-style hint out of one Turn, run as an async call after every human turn so
speaker profiles and RECENT ENTITIES stay current.

`LLMExtractor` needs a real `LLMProvider` (wired in M4 -- see bots/llm_provider.py and
its Anthropic implementation) to actually work; it can't be exercised live yet, so M2's
tests use `FakeExtractor` (roxroom.context.fakes) instead. `NullExtractor` is the safe
default RoomContext falls back to when no extractor is configured: profiles/entities
just won't populate, which is a graceful degradation, not a crash.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.context.models import Turn

logger = logging.getLogger("roxroom.context.extractor")


@dataclass
class ExtractionResult:
    facts: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    language_style: str | None = None


class Extractor(ABC):
    @abstractmethod
    async def extract(self, turn: Turn, recent_turns: list[Turn]) -> ExtractionResult: ...


class NullExtractor(Extractor):
    async def extract(self, turn: Turn, recent_turns: list[Turn]) -> ExtractionResult:
        return ExtractionResult()


_EXTRACTION_SYSTEM_PROMPT = """You extract structured memory from one turn of a group \
voice conversation. Given a few turns of context and the latest turn, return ONLY a \
JSON object of this shape:
{"facts": [...], "preferences": [...], "entities": [...], "language_style": "..."}

- facts: short factual statements the speaker made about themselves (e.g. "works as a designer").
- preferences: likes/dislikes/opinions the speaker expressed.
- entities: salient nouns/names/topics mentioned (people, places, topics), short strings.
- language_style: one short phrase describing how they speak (e.g. "casual Hinglish").

Only extract what the speaker of the LATEST turn said -- never attribute a fact to
someone else in the context. If nothing qualifies for a field, use an empty list.
Return valid JSON only, no prose, no markdown fences."""


class LLMExtractor(Extractor):
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def extract(self, turn: Turn, recent_turns: list[Turn]) -> ExtractionResult:
        context_lines = "\n".join(f"{t.display_name}: {t.text}" for t in recent_turns)
        messages = [
            ChatMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Context:\n{context_lines}\n\nLatest turn ({turn.display_name}): {turn.text}",
            ),
        ]
        chunks = [
            chunk
            async for chunk in self._llm.stream_reply(
                messages, max_output_tokens=200, response_format="json"
            )
        ]
        raw = "".join(chunks)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("extractor returned non-JSON, dropping: %r", raw[:200])
            return ExtractionResult()
        return ExtractionResult(
            facts=list(data.get("facts") or []),
            preferences=list(data.get("preferences") or []),
            entities=list(data.get("entities") or []),
            language_style=data.get("language_style") or None,
        )
