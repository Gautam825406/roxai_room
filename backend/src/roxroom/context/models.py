"""Shared data model for anything that flows through RoomContext.

`Utterance` is the Transcriber's output contract (also reused for room text-chat
messages from M5 onward, so voice and text share one pipeline). `Turn` is what
actually lives in RoomContext's ring buffer -- both voice Utterances and text-chat
messages get normalized into one, so the rest of the system (summary, profiles,
prompt building) doesn't care which modality something arrived on.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["human", "bot"]
Modality = Literal["voice", "text"]


def new_utt_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Utterance:
    participant_id: str
    display_name: str
    text: str
    is_final: bool
    lang: str  # "hi", "en", "hi-en" (code-mixed), ...
    t_start: float  # monotonic seconds
    t_end: float
    utt_id: str = field(default_factory=new_utt_id)
    modality: Modality = "voice"  # "text" for room-chat messages (M5) -- see TranscriberAgent

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")


@dataclass
class Turn:
    speaker_id: str
    display_name: str
    role: Role
    text: str
    ts: float
    modality: Modality = "voice"
    lang: str = "hi-en"
    utt_id: str = field(default_factory=new_utt_id)


@dataclass
class SpeakerProfile:
    participant_id: str
    name: str
    stated_facts: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    language_style: str | None = None
