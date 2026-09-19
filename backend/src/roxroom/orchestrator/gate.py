"""should_respond gate: Stage A cheap heuristics, Stage B small-LLM adjudication for
the ambiguous middle ground Stage A can't resolve on its own (e.g. "Rahul, tumhe kya
lagta hai?" has a direct address + a question word, exactly like a question aimed at a
bot would -- only an LLM with the last few turns of context can tell those apart).
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from roxroom.bots.llm_provider import ChatMessage, LLMProvider
from roxroom.context.models import Turn, Utterance
from roxroom.orchestrator.bot_aliases import mentioned_bots
from roxroom.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError
from roxroom.reliability.retry import RetryError, retry_with_jitter

logger = logging.getLogger("roxroom.orchestrator.gate")

BACKCHANNELS = frozenset(
    {
        "haan", "haanji", "haan ji", "haa", "hm", "hmm", "hmmm",
        "achha", "acha", "accha", "theek hai", "thik hai", "ok", "okay",
        "ok ok", "yeah", "yep", "right", "sahi hai",
    }
)

QUESTION_WORDS = frozenset(
    {
        "kya", "kaise", "kyun", "kyu", "kab", "kaun", "kahan", "kidhar",
        "kitna", "kitne", "kitni", "konsa", "konsi", "kaunsa", "kaunsi",
        "what", "how", "why", "when", "who", "where", "which",
    }
)

IMPERATIVE_VERBS = frozenset(
    {
        "batao", "bata", "batado", "bataiye", "samjhao", "samjha", "samjhaiye",
        "bolo", "bol", "boliye", "explain", "tell", "batana", "samjhana",
    }
)

DIRECT_ADDRESS_WORDS = frozenset(
    {"tum", "tumhe", "tumko", "tumhara", "tumhari", "aap", "aapko", "aapka", "aapki", "you", "your"}
)


def _tokens(text: str) -> list[str]:
    return [w.strip(".,?!\"'").lower() for w in text.split()]


def is_backchannel(text: str) -> bool:
    normalized = text.strip().strip(".,!").lower()
    if not normalized:
        return True
    if normalized in BACKCHANNELS:
        return True
    tokens = normalized.split()
    return bool(tokens) and all(t in BACKCHANNELS for t in tokens)


def is_interrogative(text: str) -> bool:
    if "?" in text:
        return True
    return bool(set(_tokens(text)) & QUESTION_WORDS)


def is_imperative(text: str) -> bool:
    if "tell me" in text.lower():
        return True
    return bool(set(_tokens(text)) & IMPERATIVE_VERBS)


def is_direct_address(text: str) -> bool:
    return bool(set(_tokens(text)) & DIRECT_ADDRESS_WORDS)


@dataclass(frozen=True)
class StageAResult:
    mentioned_bots: frozenset[str]
    is_interrogative: bool
    is_imperative: bool
    is_direct_address: bool

    @property
    def explicit_bot(self) -> str | None:
        return next(iter(self.mentioned_bots)) if len(self.mentioned_bots) == 1 else None

    @property
    def has_explicit_mention(self) -> bool:
        return bool(self.mentioned_bots)

    @property
    def has_any_signal(self) -> bool:
        return bool(self.mentioned_bots or self.is_interrogative or self.is_imperative or self.is_direct_address)

    def to_log_dict(self) -> dict:
        return {
            "mentioned_bots": sorted(self.mentioned_bots),
            "is_interrogative": self.is_interrogative,
            "is_imperative": self.is_imperative,
            "is_direct_address": self.is_direct_address,
        }


def evaluate_stage_a(text: str) -> StageAResult:
    return StageAResult(
        mentioned_bots=mentioned_bots(text),
        is_interrogative=is_interrogative(text),
        is_imperative=is_imperative(text),
        is_direct_address=is_direct_address(text),
    )


AddressedBot = Literal["dost", "sathi", "none"]


@dataclass(frozen=True)
class GateVerdict:
    respond: bool
    addressed_bot: AddressedBot
    reason: str
    confidence: float

    def to_log_dict(self) -> dict:
        return {
            "respond": self.respond,
            "addressed_bot": self.addressed_bot,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class StageBGate(ABC):
    @abstractmethod
    async def classify(self, utterance: Utterance, recent_turns: list[Turn]) -> GateVerdict: ...


class AlwaysNoGate(StageBGate):
    """Safe default when no Stage B LLM is configured: stay silent on ambiguous
    utterances rather than guess. Matches the reliability principle of degrading by
    doing less, not by doing something wrong."""

    async def classify(self, utterance: Utterance, recent_turns: list[Turn]) -> GateVerdict:
        return GateVerdict(respond=False, addressed_bot="none", reason="no_stage_b_configured", confidence=0.0)


_STAGE_B_SYSTEM_PROMPT = """You decide whether an AI voice assistant should reply in a \
group conversation with two AI bots (Dost, Sathi) and multiple humans. You'll see the \
last few turns and the latest utterance. Return ONLY a JSON object of this exact shape:
{"respond": true/false, "addressed_bot": "dost"|"sathi"|"none", "reason": "...", "confidence": 0.0-1.0}

Rules:
- If the utterance is clearly one human addressing another human (by name, or in a
  private back-and-forth), respond MUST be false, addressed_bot "none".
- If it's a genuine question/request that isn't addressed to a specific human and would
  naturally be answered by an assistant in the room, respond true.
- addressed_bot is "dost" or "sathi" only if you can tell which one fits better from
  content/tone; otherwise "none" is fine even when respond is true.
Return valid JSON only, no prose, no markdown fences."""


_SILENT_FALLBACK = GateVerdict(respond=False, addressed_bot="none", reason="stage_b_unavailable", confidence=0.0)


class LLMStageBGate(StageBGate):
    """Retry-with-jitter + circuit breaker around the underlying LLM call: if it's
    down (timeouts, connection errors, rate limits), degrade to `_SILENT_FALLBACK`
    rather than let the exception propagate into the Orchestrator -- same "silent
    instead of wrong" principle as `AlwaysNoGate`, just reached via a failure path
    instead of "no LLM configured" at all."""

    def __init__(self, llm: LLMProvider, *, breaker: CircuitBreaker | None = None, max_attempts: int = 2) -> None:
        self._llm = llm
        self._breaker = breaker or CircuitBreaker("stage_b_gate")
        self._max_attempts = max_attempts

    async def classify(self, utterance: Utterance, recent_turns: list[Turn]) -> GateVerdict:
        context_lines = "\n".join(f"{t.display_name}: {t.text}" for t in recent_turns[-6:])
        messages = [
            ChatMessage(role="system", content=_STAGE_B_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Last turns:\n{context_lines}\n\nLatest utterance ({utterance.display_name}): {utterance.text}",
            ),
        ]

        async def call_llm() -> str:
            chunks = [
                chunk
                async for chunk in self._llm.stream_reply(messages, max_output_tokens=100, response_format="json")
            ]
            return "".join(chunks)

        try:
            raw = await self._breaker.call(lambda: retry_with_jitter(call_llm, max_attempts=self._max_attempts))
        except (RetryError, CircuitOpenError) as exc:
            logger.warning("stage B gate unavailable (%s), defaulting to silent", exc)
            return _SILENT_FALLBACK

        try:
            data = json.loads(raw)
            return GateVerdict(
                respond=bool(data.get("respond", False)),
                addressed_bot=data.get("addressed_bot", "none"),
                reason=str(data.get("reason", "")),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("stage B gate returned unparseable output, defaulting to silent: %r", raw[:200])
            return GateVerdict(respond=False, addressed_bot="none", reason="unparseable_stage_b_output", confidence=0.0)
