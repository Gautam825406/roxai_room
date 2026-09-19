import pytest

from roxroom.reliability.circuit_breaker import CircuitBreaker
from roxroom.reliability.fallback_chain import AllProvidersFailedError, FallbackLink, call_with_fallback


async def _no_sleep(_seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_first_link_success_never_tries_the_rest():
    calls = {"primary": 0, "secondary": 0}

    async def primary():
        calls["primary"] += 1
        return "primary result"

    async def secondary():
        calls["secondary"] += 1
        return "secondary result"

    links = [FallbackLink("primary", primary), FallbackLink("secondary", secondary)]
    result = await call_with_fallback(links)

    assert result == "primary result"
    assert calls == {"primary": 1, "secondary": 0}


@pytest.mark.asyncio
async def test_falls_through_to_second_link_when_first_exhausts_retries():
    async def always_fails():
        raise ConnectionError("primary down")

    async def works():
        return "secondary saved it"

    links = [
        FallbackLink("primary", always_fails, max_attempts=2),
        FallbackLink("secondary", works),
    ]
    result = await call_with_fallback(links)
    assert result == "secondary saved it"


@pytest.mark.asyncio
async def test_raises_all_providers_failed_when_every_link_fails():
    async def always_fails():
        raise ConnectionError("down")

    links = [
        FallbackLink("a", always_fails, max_attempts=1),
        FallbackLink("b", always_fails, max_attempts=1),
    ]
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await call_with_fallback(links)
    assert exc_info.value.attempted == ["a", "b"]


@pytest.mark.asyncio
async def test_open_circuit_on_a_link_skips_straight_to_the_next():
    breaker = CircuitBreaker("flaky", failure_threshold=1)
    calls = {"flaky": 0, "backup": 0}

    async def flaky():
        calls["flaky"] += 1
        raise ConnectionError("down")

    async def backup():
        calls["backup"] += 1
        return "backup result"

    links = [FallbackLink("flaky", flaky, breaker=breaker, max_attempts=1), FallbackLink("backup", backup)]

    # First call: flaky fails once, breaker opens, falls through to backup.
    result1 = await call_with_fallback(links)
    assert result1 == "backup result"
    assert calls == {"flaky": 1, "backup": 1}

    # Second call: breaker for "flaky" is now open -- it should be skipped entirely,
    # not even attempted, going straight to backup.
    result2 = await call_with_fallback(links)
    assert result2 == "backup result"
    assert calls == {"flaky": 1, "backup": 2}  # flaky's call count unchanged
