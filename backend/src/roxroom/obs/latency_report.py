"""Collects completed Trace end-to-end latencies (end-of-speech -> first bot audio)
and reports p50/p95 against the spec's <1.5s target. `summary_table()` is printed on
worker shutdown (see run_bot.py).

Percentile uses linear interpolation between closest ranks (the same convention
numpy's default `percentile` uses) rather than nearest-rank, so p50 of an even-sized
sample isn't biased toward one side.
"""
from __future__ import annotations

TARGET_MS = 1500.0


class LatencyTracker:
    def __init__(self) -> None:
        self._samples: list[float] = []

    def record(self, value_ms: float | None) -> None:
        if value_ms is not None:
            self._samples.append(value_ms)

    @property
    def count(self) -> int:
        return len(self._samples)

    def percentile(self, p: float) -> float | None:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * (p / 100)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        if lower == upper:
            return ordered[lower]
        fraction = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def summary_table(self) -> str:
        if not self._samples:
            return "Latency summary: no completed voice replies recorded this session."

        p50 = self.percentile(50)
        p95 = self.percentile(95)
        assert p50 is not None and p95 is not None
        target_met = "within target" if p95 <= TARGET_MS else "OVER TARGET"

        lines = [
            "Latency summary (end-of-speech -> first bot audio)",
            f"  samples : {self.count}",
            f"  min     : {min(self._samples):.0f}ms",
            f"  p50     : {p50:.0f}ms",
            f"  p95     : {p95:.0f}ms  ({target_met}, target <{TARGET_MS:.0f}ms)",
            f"  max     : {max(self._samples):.0f}ms",
        ]
        return "\n".join(lines)
