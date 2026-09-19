"""Routing: given an utterance that already passed the should_respond gate, decide
which bot(s) answer and in what order. Priority order (ROUTING.md has the full table):

1. Explicit address (single bot named) / multi-bot plan (both named)
2. Stage B's own addressed_bot call, when Stage A found no explicit name
3. Persona affinity (definitional/factual -> Dost, example/analogy/elaboration -> Sathi)
4. Alternation fallback (whoever spoke less recently)
"""
from __future__ import annotations

from dataclasses import dataclass

from roxroom.orchestrator.bot_aliases import BotMention, find_bot_mentions

DOST = "dost"
SATHI = "sathi"

DOST_AFFINITY_KEYWORDS = frozenset(
    {
        "define", "definition", "matlab kya hai", "kya hota hai", "kya hai",
        "fact", "concise", "short mein", "jaldi bata", "quickly",
    }
)
SATHI_AFFINITY_KEYWORDS = frozenset(
    {
        "example", "misal", "analogy", "simple mein", "asaan bhasha", "asaan",
        "samjha do", "confuse", "samajh nahi", "feel", "pareshan", "elaborate", "detail mein",
    }
)


@dataclass(frozen=True)
class ResponsePlanStep:
    bot: str  # "dost" | "sathi"
    instruction: str


@dataclass(frozen=True)
class RouteDecision:
    plan: list[ResponsePlanStep]
    rule_fired: str

    @property
    def chosen_bot(self) -> str | None:
        """The single bot for a one-step plan; None for a multi-bot plan (there isn't
        one "the" bot -- see `plan` for the full ordered list)."""
        return self.plan[0].bot if len(self.plan) == 1 else None


def classify_persona_affinity(text: str) -> str | None:
    lowered = text.lower()
    dost_score = sum(1 for kw in DOST_AFFINITY_KEYWORDS if kw in lowered)
    sathi_score = sum(1 for kw in SATHI_AFFINITY_KEYWORDS if kw in lowered)
    if dost_score == sathi_score:
        return None
    return DOST if dost_score > sathi_score else SATHI


def _build_multi_bot_plan(text: str, mentions: list[BotMention]) -> list[ResponsePlanStep]:
    ordered: list[BotMention] = []
    seen: set[str] = set()
    for mention in sorted(mentions, key=lambda m: m.start):
        if mention.bot not in seen:
            seen.add(mention.bot)
            ordered.append(mention)

    steps: list[ResponsePlanStep] = []
    for i, mention in enumerate(ordered):
        fragment_end = ordered[i + 1].start if i + 1 < len(ordered) else len(text)
        fragment = text[mention.end : fragment_end].strip(" ,.")
        steps.append(ResponsePlanStep(bot=mention.bot, instruction=fragment))
    return steps


class Router:
    def __init__(self) -> None:
        self.last_reply_at: dict[str, float] = {DOST: float("-inf"), SATHI: float("-inf")}

    def route(self, *, text: str, stage_b_addressed_bot: str) -> RouteDecision:
        mentions = find_bot_mentions(text)
        distinct_bots = {m.bot for m in mentions}

        if len(distinct_bots) >= 2:
            return RouteDecision(plan=_build_multi_bot_plan(text, mentions), rule_fired="multi_bot_plan")

        if len(distinct_bots) == 1:
            bot = next(iter(distinct_bots))
            return RouteDecision(plan=[ResponsePlanStep(bot=bot, instruction=text)], rule_fired="explicit_address")

        if stage_b_addressed_bot in (DOST, SATHI):
            return RouteDecision(
                plan=[ResponsePlanStep(bot=stage_b_addressed_bot, instruction=text)],
                rule_fired="stage_b_addressed",
            )

        affinity = classify_persona_affinity(text)
        if affinity is not None:
            return RouteDecision(plan=[ResponsePlanStep(bot=affinity, instruction=text)], rule_fired="persona_affinity")

        chosen = min(self.last_reply_at, key=lambda b: self.last_reply_at[b])
        return RouteDecision(plan=[ResponsePlanStep(bot=chosen, instruction=text)], rule_fired="alternation_fallback")

    def record_reply_completed(self, bot: str, at: float) -> None:
        self.last_reply_at[bot] = at
