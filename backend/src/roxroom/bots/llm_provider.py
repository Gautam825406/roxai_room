"""LLMProvider interface. Implemented per-vendor in bots/providers/*.py (M4).

Must stream sentence-sized chunks (not raw tokens) so the caller can start TTS before
generation finishes -- chunk boundaries are the provider's responsibility since only
it sees the raw token stream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def stream_reply(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[str]:
        """Yields text chunks, sentence-chunked where response_format == 'text'.

        For response_format == 'json' (e.g. the Stage B gate call), implementations
        should yield the complete JSON payload as a single chunk once available.
        """
