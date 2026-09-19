import pytest

from roxroom.obs.latency_report import LatencyTracker


def test_no_samples_reports_nothing_recorded():
    tracker = LatencyTracker()
    assert tracker.percentile(50) is None
    assert "no completed" in tracker.summary_table().lower()


def test_record_ignores_none_values():
    tracker = LatencyTracker()
    tracker.record(None)
    tracker.record(100.0)
    assert tracker.count == 1


def test_percentile_median_of_odd_sample():
    tracker = LatencyTracker()
    for v in [10, 20, 30, 40, 50]:
        tracker.record(v)
    assert tracker.percentile(50) == pytest.approx(30.0)


def test_percentile_median_of_even_sample_interpolates():
    tracker = LatencyTracker()
    for v in [1, 2, 3, 4]:
        tracker.record(v)
    assert tracker.percentile(50) == pytest.approx(2.5)


def test_percentile_95_interpolates_between_closest_ranks():
    tracker = LatencyTracker()
    for v in [10, 20, 30, 40, 50]:
        tracker.record(v)
    assert tracker.percentile(95) == pytest.approx(48.0)


def test_single_sample_returns_that_value_for_any_percentile():
    tracker = LatencyTracker()
    tracker.record(123.0)
    assert tracker.percentile(50) == 123.0
    assert tracker.percentile(95) == 123.0


def test_summary_table_flags_target_met_and_exceeded():
    within = LatencyTracker()
    for v in [500, 600, 700]:
        within.record(v)
    assert "within target" in within.summary_table()

    over = LatencyTracker()
    for v in [2000, 2100, 2200]:
        over.record(v)
    assert "OVER TARGET" in over.summary_table()


def test_summary_table_includes_sample_count_and_min_max():
    tracker = LatencyTracker()
    for v in [100, 500, 900]:
        tracker.record(v)
    table = tracker.summary_table()
    assert "samples : 3" in table
    assert "100ms" in table
    assert "900ms" in table
