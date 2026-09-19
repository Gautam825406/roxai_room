import pytest

from roxroom.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


async def _fail():
    raise ConnectionError("down")


async def _ok():
    return "ok"


@pytest.mark.asyncio
async def test_stays_closed_below_failure_threshold():
    breaker = CircuitBreaker("test", failure_threshold=3)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_failure_threshold_reached():
    breaker = CircuitBreaker("test", failure_threshold=3)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_refuses_calls_without_invoking_fn():
    breaker = CircuitBreaker("test", failure_threshold=1)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN

    calls = []

    async def spy():
        calls.append(1)
        return "should not run"

    with pytest.raises(CircuitOpenError):
        await breaker.call(spy)
    assert calls == []


@pytest.mark.asyncio
async def test_half_opens_after_reset_timeout_and_closes_on_success():
    clock = FakeClock()
    breaker = CircuitBreaker("test", failure_threshold=1, reset_timeout_s=30.0, clock=clock)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN

    clock.advance(29.0)
    assert breaker.state == CircuitState.OPEN  # not yet

    clock.advance(2.0)  # now past 30s
    assert breaker.state == CircuitState.HALF_OPEN

    result = await breaker.call(_ok)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_probe_failure_reopens_for_another_full_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker("test", failure_threshold=1, reset_timeout_s=30.0, clock=clock)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    clock.advance(31.0)
    assert breaker.state == CircuitState.HALF_OPEN

    with pytest.raises(ConnectionError):
        await breaker.call(_fail)  # probe fails
    assert breaker.state == CircuitState.OPEN

    clock.advance(29.0)
    assert breaker.state == CircuitState.OPEN  # cooldown restarted, not yet elapsed

    clock.advance(2.0)
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_success_resets_consecutive_failure_count():
    breaker = CircuitBreaker("test", failure_threshold=3)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    await breaker.call(_ok)  # resets the streak

    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    with pytest.raises(ConnectionError):
        await breaker.call(_fail)
    assert breaker.state == CircuitState.CLOSED  # only 2 consecutive since the reset
