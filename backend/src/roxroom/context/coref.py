"""Coreference marker detection -- pure text heuristics, no LLM involved.

We don't try to actually resolve "uski"/"ye"/"wahi topic" to a specific referent here.
Per the spec, the more reliable fix is putting the last few salient entities explicitly
in the prompt as a `RECENT ENTITIES:` line (see RoomContext.recent_entities_line) and
letting the LLM do the binding with that hint in hand -- this module just flags *that*
a turn is likely leaning on an antecedent, which is useful for logging/debugging and
for the router's Stage A heuristics later.
"""
from __future__ import annotations

COREFERENCE_MARKERS = {
    "uski", "unki", "uska", "unka", "iska", "iski",
    "isse", "usse", "isko", "usko", "unhe", "unko",
    "ye", "yeh", "wo", "voh", "yehi", "wohi",
}

COREFERENCE_PHRASES = ("wahi topic", "usi topic", "wahi baat")


def contains_coreference_marker(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in COREFERENCE_PHRASES):
        return True
    words = {w.strip(".,?!\"'") for w in lowered.split()}
    return bool(words & COREFERENCE_MARKERS)
