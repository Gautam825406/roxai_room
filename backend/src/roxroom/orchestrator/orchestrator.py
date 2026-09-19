"""Orchestrator: the decision funnel from a finalized Utterance to a routing Decision.

Funnel order (all must pass for a bot to speak):
1. Not a duplicate utt_id.
2. Not a bare backchannel ("haan", "hmm", "achha", "ok", ...).
3. Not suppressed (a bot completed a reply <1.5s ago) -- unless this utterance
   explicitly names a bot, which always overrides suppression.
4. should_respond: Stage A short-circuits on explicit mention (respond) or on zero
   signals at all (silence); otherwise Stage B adjudicates the ambiguous middle.
5. Routing picks the bot(s), which is where the actual chosen_bot/plan comes from.

No I/O happens here except the Stage B classifier call (a small, fast model) -- this
class takes a synthetic Utterance stream in tests and needs nothing else running.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from roxroom.context.models import Turn, Utterance
from roxroom.orchestrator.dedupe import UtteranceDedupe
from roxroom.orchestrator.gate import AlwaysNoGate, GateVerdict, StageAResult, StageBGate, evaluate_stage_a, is_backchannel
from roxroom.orchestrator.router import ResponsePlanStep, Router

logger = logging.getLogger("roxroom.orchestrator")

SUPPRESSION_WINDOW_S = 1.5


@dataclass(frozen=True)
class Decision:
    utt_id: str
    respond: bool
    plan: list[ResponsePlanStep]
    rule_fired: str
    stage_a: StageAResult
    stage_b: GateVerdict | None
    latency_ms: float

    @property
    def chosen_bot(self) -> str | None:
        return self.plan[0].bot if len(self.plan) == 1 else None

    def to_log_dict(self) -> dict:
        return {
            "utt_id": self.utt_id,
            "stage_a": self.stage_a.to_log_dict(),
            "stage_b": self.stage_b.to_log_dict() if self.stage_b else None,
            "chosen_bot": self.chosen_bot,
            "plan": [{"bot": s.bot, "instruction": s.instruction} for s in self.plan],
            "rule_fired": self.rule_fired,
            "respond": self.respond,
            "latency_ms": round(self.latency_ms, 2),
        }


class Orchestrator:
    def __init__(
        self,
        *,
        stage_b_gate: StageBGate | None = None,
        router: Router | None = None,
        dedupe: UtteranceDedupe | None = None,
        suppression_window_s: float = SUPPRESSION_WINDOW_S,
        clock=time.monotonic,
    ) -> None:
        self._stage_b_gate = stage_b_gate or AlwaysNoGate()
        self._router = router or Router()
        self._dedupe = dedupe or UtteranceDedupe()
        self._suppression_window_s = suppression_window_s
        self._clock = clock
        self._last_bot_reply_completed_at: float | None = None
        self.decision_log: list[Decision] = []

    async def handle_utterance(self, utterance: Utterance, recent_turns: list[Turn]) -> Decision:
        start = self._clock()

        if self._dedupe.is_duplicate(utterance.utt_id):
            return self._finish(utterance, start, StageAResult(frozenset(), False, False, False), None, [], "duplicate")

        stage_a = evaluate_stage_a(utterance.text)

        if is_backchannel(utterance.text):
            return self._finish(utterance, start, stage_a, None, [], "backchannel")

        if self._is_suppressed(stage_a):
            return self._finish(utterance, start, stage_a, None, [], "suppressed_recent_reply")

        stage_b: GateVerdict | None = None
        addressed_bot = stage_a.explicit_bot or "none"

        if not stage_a.has_explicit_mention:
            if not stage_a.has_any_signal:
                return self._finish(utterance, start, stage_a, None, [], "stage_a_no_signal")
            try:
                stage_b = await self._stage_b_gate.classify(utterance, recent_turns)
            except Exception:
                # LLMStageBGate already guards its own provider call (retry + circuit
                # breaker -> silent fallback, see gate.py) -- this is a second line of
                # defense in case a StageBGate implementation doesn't hold up its end
                # of that contract. Same "silent instead of wrong" principle either way.
                logger.exception("stage B gate raised unexpectedly, defaulting to silent")
                return self._finish(utterance, start, stage_a, None, [], "stage_b_crashed")
            if not stage_b.respond:
                return self._finish(utterance, start, stage_a, stage_b, [], "stage_b_silent")
            addressed_bot = stage_b.addressed_bot

        route = self._router.route(text=utterance.text, stage_b_addressed_bot=addressed_bot)
        return self._finish(utterance, start, stage_a, stage_b, route.plan, route.rule_fired)

    def on_bot_reply_completed(self, bot: str, at: float | None = None) -> None:
        at = at if at is not None else self._clock()
        self._last_bot_reply_completed_at = at
        self._router.record_reply_completed(bot, at)

    def _is_suppressed(self, stage_a: StageAResult) -> bool:
        if self._last_bot_reply_completed_at is None:
            return False
        if stage_a.has_explicit_mention:
            return False  # explicit address always overrides suppression
        elapsed = self._clock() - self._last_bot_reply_completed_at
        return elapsed < self._suppression_window_s

    def _finish(
        self,
        utterance: Utterance,
        start: float,
        stage_a: StageAResult,
        stage_b: GateVerdict | None,
        plan: list[ResponsePlanStep],
        rule_fired: str,
    ) -> Decision:
        latency_ms = (self._clock() - start) * 1000
        decision = Decision(
            utt_id=utterance.utt_id,
            respond=bool(plan),
            plan=plan,
            rule_fired=rule_fired,
            stage_a=stage_a,
            stage_b=stage_b,
            latency_ms=latency_ms,
        )
        self.decision_log.append(decision)
        logger.info(
            "routing decision: %s",
            decision.to_log_dict(),
            extra={"trace_id": decision.utt_id, "stage": "gate_route", **decision.to_log_dict()},
        )
        return decision
