"""Bounded utt_id dedupe -- an LRU-ish set so a long session doesn't grow this
unbounded. A duplicate utt_id shows up if the transcriber (or a flaky reconnect)
redelivers the same finalized utterance more than once.
"""
from __future__ import annotations

from collections import OrderedDict

DEFAULT_MAX_SIZE = 500


class UtteranceDedupe:
    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size

    def is_duplicate(self, utt_id: str) -> bool:
        """Returns True if already seen; otherwise records it and returns False."""
        if utt_id in self._seen:
            return True
        self._seen[utt_id] = None
        if len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return False
