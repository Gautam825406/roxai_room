import asyncio

import pytest

from roxroom.orchestrator.speak_lock import FloorOutcome, SpeakLock


@pytest.mark.asyncio
async def test_speak_turn_completes_and_releases():
    lock = SpeakLock()

    async def fn():
        assert lock.holder == "dost"

    outcome = await lock.speak_turn("dost", fn)
    assert outcome == FloorOutcome.COMPLETED
    assert lock.holder is None


@pytest.mark.asyncio
async def test_speak_turn_enforces_mutual_exclusion():
    lock = SpeakLock()
    events: list[str] = []

    async def slow_speak(bot: str):
        events.append(f"{bot}-start")
        await asyncio.sleep(0.05)
        events.append(f"{bot}-end")

    await asyncio.gather(
        lock.speak_turn("dost", lambda: slow_speak("dost")),
        lock.speak_turn("sathi", lambda: slow_speak("sathi")),
    )

    # one bot must fully finish before the other starts -- no interleaving
    assert events in (
        ["dost-start", "dost-end", "sathi-start", "sathi-end"],
        ["sathi-start", "sathi-end", "dost-start", "dost-end"],
    )


@pytest.mark.asyncio
async def test_exception_in_turn_yields_failed_outcome_and_releases_lock():
    lock = SpeakLock()

    async def boom():
        raise RuntimeError("tts blew up")

    outcome = await lock.speak_turn("dost", boom)
    assert outcome == FloorOutcome.FAILED
    assert lock.holder is None
    # lock usable again afterwards
    outcome2 = await lock.speak_turn("sathi", lambda: asyncio.sleep(0))
    assert outcome2 == FloorOutcome.COMPLETED


@pytest.mark.asyncio
async def test_cancellation_yields_interrupted_outcome_and_releases_lock():
    lock = SpeakLock()
    started = asyncio.Event()

    async def hang_forever():
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(lock.speak_turn("dost", hang_forever))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock.holder is None


@pytest.mark.asyncio
async def test_fn_can_explicitly_report_interrupted_without_a_real_cancellation():
    """BotAgent's barge-in path exits its loops via a flag, not asyncio.CancelledError
    -- speak_turn must honor an explicit return value instead of defaulting to
    COMPLETED just because no exception was raised."""
    lock = SpeakLock()
    released = []
    lock.on_floor_released(lambda bot, outcome: released.append((bot, outcome)))

    async def flag_based_barge_in():
        return FloorOutcome.INTERRUPTED  # exits normally, but was actually cut short

    outcome = await lock.speak_turn("dost", flag_based_barge_in)

    assert outcome == FloorOutcome.INTERRUPTED
    assert released == [("dost", FloorOutcome.INTERRUPTED)]
    assert lock.holder is None


@pytest.mark.asyncio
async def test_floor_released_listeners_are_notified_sync_and_async():
    lock = SpeakLock()
    sync_calls = []
    async_calls = []

    def sync_listener(bot, outcome):
        sync_calls.append((bot, outcome))

    async def async_listener(bot, outcome):
        async_calls.append((bot, outcome))

    lock.on_floor_released(sync_listener)
    lock.on_floor_released(async_listener)

    await lock.speak_turn("dost", lambda: asyncio.sleep(0))
    await asyncio.sleep(0)  # let the fire-and-forget async listener task run

    assert sync_calls == [("dost", FloorOutcome.COMPLETED)]
    assert async_calls == [("dost", FloorOutcome.COMPLETED)]
