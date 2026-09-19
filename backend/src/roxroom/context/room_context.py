"""RoomContext: the shared memory the Orchestrator and every BotAgent read from.

Compression policy (documented here since it's the one non-obvious knob):
- Every `COMPRESS_EVERY_N_TURNS` (10) turns added, the oldest `COMPRESS_BATCH_SIZE`
  (5) turns still sitting in the buffer are folded into the rolling summary via the
  injected Summarizer, then dropped from the buffer. That's the "compress the oldest
  half [of the 10-turn window] every 10 turns" rule from the spec.
- `MAX_BUFFERED_TURNS` (20) is a hard safety cap independent of that cadence: if
  compression ever falls behind (e.g. the summarizer is failing), we forcibly drop the
  oldest turns past 20 rather than growing the buffer unbounded. This trades detail for
  memory safety -- a graceful degradation, not a crash.
- Prompt building only ever uses the summary + the last `PROMPT_RECENT_TURNS` (12)
  turns, regardless of how many are currently buffered (up to 20). See README for the
  worked-through token budget.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from roxroom.context.coref import contains_coreference_marker
from roxroom.context.extractor import Extractor, NullExtractor
from roxroom.context.models import Modality, Role, SpeakerProfile, Turn
from roxroom.context.summarizer import NaiveSummarizer, Summarizer

logger = logging.getLogger("roxroom.context.room_context")

COMPRESS_EVERY_N_TURNS = 10
COMPRESS_BATCH_SIZE = 5
MAX_BUFFERED_TURNS = 20
PROMPT_RECENT_TURNS = 12
RECENT_ENTITIES_MAX = 3


@dataclass
class PromptContext:
    rolling_summary: str
    recent_turns: list[Turn]
    recent_entities: list[str]

    def recent_entities_line(self) -> str:
        if not self.recent_entities:
            return ""
        return "RECENT ENTITIES: " + ", ".join(self.recent_entities)


class RoomContext:
    def __init__(
        self,
        *,
        extractor: Extractor | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.turns: deque[Turn] = deque()
        self.rolling_summary: str = ""
        self.speaker_profiles: dict[str, SpeakerProfile] = {}
        self.recent_entities: deque[str] = deque(maxlen=RECENT_ENTITIES_MAX)
        self._extractor = extractor or NullExtractor()
        self._summarizer = summarizer or NaiveSummarizer()
        self._turns_since_compress = 0

    async def add_turn(
        self,
        *,
        speaker_id: str,
        display_name: str,
        role: Role,
        text: str,
        ts: float,
        modality: Modality = "voice",
        lang: str = "hi-en",
    ) -> Turn:
        turn = Turn(
            speaker_id=speaker_id,
            display_name=display_name,
            role=role,
            text=text,
            ts=ts,
            modality=modality,
            lang=lang,
        )
        self.turns.append(turn)
        self._turns_since_compress += 1

        if contains_coreference_marker(text):
            logger.debug("turn %s leans on an antecedent (coref marker present)", turn.utt_id)

        if role == "human":
            await self._update_speaker_profile(turn)

        if self._turns_since_compress >= COMPRESS_EVERY_N_TURNS:
            await self._compress_oldest()

        self._enforce_hard_cap()
        return turn

    def get_speaker_profile(self, participant_id: str) -> SpeakerProfile | None:
        return self.speaker_profiles.get(participant_id)

    def build_prompt_context(self) -> PromptContext:
        recent = list(self.turns)[-PROMPT_RECENT_TURNS:]
        return PromptContext(
            rolling_summary=self.rolling_summary,
            recent_turns=recent,
            recent_entities=list(self.recent_entities),
        )

    async def _update_speaker_profile(self, turn: Turn) -> None:
        recent_context = list(self.turns)[-6:]
        try:
            extraction = await self._extractor.extract(turn, recent_context)
        except Exception:
            # Profile enrichment is best-effort background work, not on the critical
            # reply path -- losing one turn's facts/entities to a flaky extractor call
            # is a fine trade against letting it take down turn processing entirely.
            logger.exception("extractor failed for turn %s, skipping profile update", turn.utt_id)
            return

        profile = self.speaker_profiles.setdefault(
            turn.speaker_id, SpeakerProfile(participant_id=turn.speaker_id, name=turn.display_name)
        )
        for fact in extraction.facts:
            if fact not in profile.stated_facts:
                profile.stated_facts.append(fact)
        for pref in extraction.preferences:
            if pref not in profile.preferences:
                profile.preferences.append(pref)
        if extraction.language_style:
            profile.language_style = extraction.language_style
        for entity in extraction.entities:
            self._push_recent_entity(entity)

    def _push_recent_entity(self, entity: str) -> None:
        entity = entity.strip()
        if not entity:
            return
        if entity in self.recent_entities:
            self.recent_entities.remove(entity)
        self.recent_entities.append(entity)

    async def _compress_oldest(self) -> None:
        batch = [self.turns.popleft() for _ in range(min(COMPRESS_BATCH_SIZE, len(self.turns)))]
        if batch:
            try:
                self.rolling_summary = await self._summarizer.summarize(batch, self.rolling_summary)
            except Exception:
                logger.exception("summarizer failed, falling back to naive concatenation")
                self.rolling_summary = await NaiveSummarizer().summarize(batch, self.rolling_summary)
        self._turns_since_compress = max(0, self._turns_since_compress - COMPRESS_BATCH_SIZE)

    def _enforce_hard_cap(self) -> None:
        while len(self.turns) > MAX_BUFFERED_TURNS:
            dropped = self.turns.popleft()
            logger.warning(
                "hard cap hit: dropping uncompressed turn %s from %s without summarizing",
                dropped.utt_id,
                dropped.speaker_id,
            )
