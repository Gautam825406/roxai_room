import asyncio

import pytest

from roxroom.bots.bot_agent import SpeakResult
from roxroom.obs.trace import Trace
from roxroom.orchestrator.plan_executor import PlanExecutor
from roxroom.orchestrator.router import ResponsePlanStep
from roxroom.orchestrator.speak_lock import FloorOutcome


class FakeAgent:
    """Duck-types BotAgent's async reply_to(instruction, trace=None) -> SpeakResult --
    PlanExecutor doesn't need a real BotAgent (LiveKit connection, persona, etc.) to be
    tested."""

    def __init__(self, bot_id: str, on_reply=None) -> None:
        self.bot_id = bot_id
        self.calls: list[str] = []
        self.traces_received: list = []
        self._on_reply = on_reply

    async def reply_to(self, instruction: str, trace=None) -> SpeakResult:
        self.calls.append(instruction)
        self.traces_received.append(trace)
        if self._on_reply:
            self._on_reply()
        return SpeakResult(text=f"{self.bot_id} says: {instruction}", outcome=FloorOutcome.COMPLETED)


@pytest.mark.asyncio
async def test_single_step_plan_runs_one_agent():
    dost = FakeAgent("dost")
    executor = PlanExecutor({"dost": dost, "sathi": FakeAgent("sathi")})

    results = await executor.execute([ResponsePlanStep(bot="dost", instruction="tumhara naam kya hai")])

    assert dost.calls == ["tumhara naam kya hai"]
    assert len(results) == 1
    assert results[0].text == "dost says: tumhara naam kya hai"


@pytest.mark.asyncio
async def test_multi_step_plan_runs_agents_in_order():
    dost = FakeAgent("dost")
    sathi = FakeAgent("sathi")
    executor = PlanExecutor({"dost": dost, "sathi": sathi})

    results = await executor.execute(
        [
            ResponsePlanStep(bot="dost", instruction="answer karo"),
            ResponsePlanStep(bot="sathi", instruction="example dena"),
        ]
    )

    assert dost.calls == ["answer karo"]
    assert sathi.calls == ["example dena"]
    assert [r.text for r in results] == ["dost says: answer karo", "sathi says: example dena"]


@pytest.mark.asyncio
async def test_missing_bot_in_registry_is_skipped_not_fatal():
    sathi = FakeAgent("sathi")
    executor = PlanExecutor({"sathi": sathi})  # dost not registered

    results = await executor.execute(
        [
            ResponsePlanStep(bot="dost", instruction="answer karo"),
            ResponsePlanStep(bot="sathi", instruction="example dena"),
        ]
    )

    assert sathi.calls == ["example dena"]
    assert len(results) == 1


@pytest.mark.asyncio
async def test_cancel_after_first_step_skips_remaining_steps():
    executor_ref: dict[str, PlanExecutor] = {}

    def cancel_now():
        executor_ref["executor"].cancel_current("human_spoke")

    dost = FakeAgent("dost", on_reply=cancel_now)
    sathi = FakeAgent("sathi")
    executor = PlanExecutor({"dost": dost, "sathi": sathi})
    executor_ref["executor"] = executor

    results = await executor.execute(
        [
            ResponsePlanStep(bot="dost", instruction="answer karo"),
            ResponsePlanStep(bot="sathi", instruction="example dena"),
        ]
    )

    assert dost.calls == ["answer karo"]
    assert sathi.calls == []  # never started
    assert len(results) == 1


@pytest.mark.asyncio
async def test_is_running_reflects_execution_state():
    dost = FakeAgent("dost")
    executor = PlanExecutor({"dost": dost})
    assert executor.is_running is False

    await executor.execute([ResponsePlanStep(bot="dost", instruction="hi")])
    assert executor.is_running is False


@pytest.mark.asyncio
async def test_overlapping_execute_calls_supersede_the_older_plan_not_race_it():
    """Regression test for the boolean-flag race: a fresh execute() call must
    invalidate an older still-running one even though both touch shared state, and
    without the new call's own startup accidentally un-cancelling the old one."""
    dost = FakeAgent("dost")
    sathi = FakeAgent("sathi")
    priya = FakeAgent("priya_bot")  # stand-in for a 3rd step to prove supersession mid-loop
    executor = PlanExecutor({"dost": dost, "sathi": sathi, "priya_bot": priya})

    old_plan_started = asyncio.Event()

    async def slow_reply_to(instruction: str, trace=None) -> SpeakResult:
        old_plan_started.set()
        await asyncio.sleep(0.02)
        return SpeakResult(text="dost slow reply", outcome=FloorOutcome.COMPLETED)

    dost.reply_to = slow_reply_to  # type: ignore[method-assign]

    old_plan_task = asyncio.create_task(
        executor.execute(
            [
                ResponsePlanStep(bot="dost", instruction="first"),
                ResponsePlanStep(bot="sathi", instruction="second (should be skipped)"),
            ]
        )
    )
    await old_plan_started.wait()

    # A second utterance's plan starts while the old one's first step is still in flight.
    new_results = await executor.execute([ResponsePlanStep(bot="priya_bot", instruction="new plan")])
    old_results = await old_plan_task

    assert len(old_results) == 1  # dost's already-in-flight step completed, sathi's did not
    assert sathi.calls == []
    assert [r.text for r in new_results] == ["priya_bot says: new plan"]


@pytest.mark.asyncio
async def test_trace_is_only_passed_to_the_first_step():
    dost = FakeAgent("dost")
    sathi = FakeAgent("sathi")
    executor = PlanExecutor({"dost": dost, "sathi": sathi})
    trace = Trace(trace_id="u1", participant_id="priya")

    await executor.execute(
        [
            ResponsePlanStep(bot="dost", instruction="answer karo"),
            ResponsePlanStep(bot="sathi", instruction="example dena"),
        ],
        trace=trace,
    )

    assert dost.traces_received == [trace]
    assert sathi.traces_received == [None]


@pytest.mark.asyncio
async def test_cancel_before_any_execution_is_a_harmless_noop():
    executor = PlanExecutor({"dost": FakeAgent("dost")})
    executor.cancel_current("nothing running yet")  # should not raise
    dost = FakeAgent("dost")
    executor2 = PlanExecutor({"dost": dost})
    results = await executor2.execute([ResponsePlanStep(bot="dost", instruction="hi")])
    assert len(results) == 1
