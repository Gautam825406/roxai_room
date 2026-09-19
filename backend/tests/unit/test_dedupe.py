from roxroom.orchestrator.dedupe import UtteranceDedupe


def test_first_sighting_is_not_a_duplicate():
    dedupe = UtteranceDedupe()
    assert dedupe.is_duplicate("utt-1") is False


def test_repeated_utt_id_is_a_duplicate():
    dedupe = UtteranceDedupe()
    dedupe.is_duplicate("utt-1")
    assert dedupe.is_duplicate("utt-1") is True


def test_bounded_size_evicts_oldest():
    dedupe = UtteranceDedupe(max_size=3)
    for i in range(3):
        assert dedupe.is_duplicate(f"utt-{i}") is False  # utt-0, utt-1, utt-2 recorded

    dedupe.is_duplicate("utt-3")  # pushes past capacity -> evicts utt-0 (oldest, FIFO)

    assert dedupe.is_duplicate("utt-0") is False  # evicted, looks new again
    assert dedupe.is_duplicate("utt-2") is True  # still remembered
