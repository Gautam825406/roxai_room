"""Sequential execution of a routed ResponsePlan across one or more BotAgents, through
the shared speak-lock they already hold their turn on.

Each step's BotAgent.reply_to() reads RoomContext fresh when it builds its prompt, and
the previous step's reply is already in RoomContext by the time the next step runs
(BotAgent.speak() appends it before returning) -- so "each turn seeing the previous
bot's output in context" falls out of the existing plumbing for free; this module only
adds the sequencing and cancellation.

Cancellation here means "stop before starting the next not-yet-started step" -- it does
NOT interrupt a step already mid-reply (that's real barge-in, M6's
BotAgent.cancel()/speak_lock interaction). Good enough for "a human spoke while Sathi's
turn was still queued behind Dost's."

Concurrency note: `execute()` can legitimately be called again before a previous call
has finished (a second utterance arriving while the first plan is still running). A
plain boolean cancel flag would race here -- the new call's own "reset the flag" at
startup could wipe out a cancellation the old call hadn't observed yet. A monotonic
generation counter avoids that: each `execute()` claims a new generation number, and a
loop only keeps going while its own generation is still the current one, so an older
call is invalidated the instant a newer one starts (or `cancel_current()` bumps it)
without any shared mutable flag to race on.
"""
from __future__ import annotations

import logging

from roxroom.bots.bot_agent import BotAgent, SpeakResult
from roxroom.obs.trace import Trace
from roxroom.orchestrator.router import ResponsePlanStep

logger = logging.getLogger("roxroom.orchestrator.plan_executor")


class PlanExecutor:
    def __init__(self, bot_agents: dict[str, BotAgent]) -> None:
        self._bot_agents = bot_agents
        self._generation = 0
        self._running_generations: set[int] = set()

    @property
    def is_running(self) -> bool:
        return bool(self._running_generations)

    def cancel_current(self, reason: str = "human_spoke") -> None:
        if self._running_generations:
            logger.info("plan execution: cancelling remaining steps (%s)", reason)
        self._generation += 1

    async def execute(self, plan: list[ResponsePlanStep], trace: Trace | None = None) -> list[SpeakResult]:
        """`trace`, if given, is only ever passed to the FIRST step -- the <1.5s
        end-of-speech-to-first-audio target (see obs/latency_report.py) is about the
        room's first response to an utterance, not every subsequent bot's turn in a
        multi-bot plan."""
        self._generation += 1
        my_generation = self._generation
        self._running_generations.add(my_generation)
        results: list[SpeakResult] = []
        try:
            for i, step in enumerate(plan):
                if my_generation != self._generation:
                    logger.info(
                        "plan execution: skipping remaining %d step(s) starting at %s (superseded)",
                        len(plan) - i,
                        step.bot,
                    )
                    break
                agent = self._bot_agents.get(step.bot)
                if agent is None:
                    logger.warning("plan execution: no BotAgent registered for %r, skipping", step.bot)
                    continue
                result = await agent.reply_to(step.instruction, trace=trace if i == 0 else None)
                results.append(result)
        finally:
            self._running_generations.discard(my_generation)
        return results
