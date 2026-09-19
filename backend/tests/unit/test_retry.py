import pytest

from roxroom.reliability.retry import RetryError, retry_with_jitter


async def _no_sleep(_seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_succeeds_on_first_try_without_retrying():
    calls = []

    async def fn():
        calls.append(1)
        return "ok"

    result = await retry_with_jitter(fn, sleep=_no_sleep)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("flaky")
        return "recovered"

    result = await retry_with_jitter(fn, max_attempts=3, sleep=_no_sleep, rand=lambda: 0.0)
    assert result == "recovered"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_and_wraps_last_exception():
    async def always_fails():
        raise TimeoutError("provider down")

    with pytest.raises(RetryError) as exc_info:
        await retry_with_jitter(always_fails, max_attempts=3, sleep=_no_sleep, rand=lambda: 0.0)

    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, TimeoutError)


@pytest.mark.asyncio
async def test_only_retries_exceptions_matching_retry_on():
    async def raises_value_error():
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        await retry_with_jitter(raises_value_error, retry_on=(ConnectionError,), sleep=_no_sleep)


@pytest.mark.asyncio
async def test_on_retry_callback_fires_once_per_failed_attempt():
    seen = []

    async def fails_twice_then_ok():
        if len(seen) < 2:
            raise ConnectionError("flaky")
        return "ok"

    result = await retry_with_jitter(
        fails_twice_then_ok,
        max_attempts=3,
        sleep=_no_sleep,
        rand=lambda: 0.0,
        on_retry=lambda attempt, exc: seen.append((attempt, type(exc))),
    )
    assert result == "ok"
    assert seen == [(1, ConnectionError), (2, ConnectionError)]


@pytest.mark.asyncio
async def test_delay_is_bounded_by_max_delay_and_scaled_by_rand():
    delays = []

    async def sleep_spy(seconds: float) -> None:
        delays.append(seconds)

    async def always_fails():
        raise ConnectionError("flaky")

    with pytest.raises(RetryError):
        await retry_with_jitter(
            always_fails,
            max_attempts=4,
            base_delay_s=1.0,
            max_delay_s=2.0,
            sleep=sleep_spy,
            rand=lambda: 1.0,  # no jitter shrinkage -> full computed delay
        )

    # attempt 1 fail -> delay base*2^0=1.0; attempt 2 fail -> base*2^1=2.0; attempt 3
    # fail -> base*2^2=4.0 capped at max_delay_s=2.0; attempt 4 fails, no more sleeps.
    assert delays == [1.0, 2.0, 2.0]
