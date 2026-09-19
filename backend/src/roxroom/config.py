"""Environment/config loading. Single source of truth for connection + identity settings.

Every setting is read once at import-time of `load_config()` so a missing required
value fails fast at process start rather than deep inside a LiveKit callback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class MissingConfigError(RuntimeError):
    pass


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingConfigError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class BotIdentity:
    identity: str
    name: str


@dataclass(frozen=True)
class RoxRoomConfig:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    room_name: str
    dost: BotIdentity
    sathi: BotIdentity


def load_config(dotenv_path: str | None = None) -> RoxRoomConfig:
    load_dotenv(dotenv_path=dotenv_path, override=False)

    return RoxRoomConfig(
        livekit_url=_require("LIVEKIT_URL"),
        livekit_api_key=_require("LIVEKIT_API_KEY"),
        livekit_api_secret=_require("LIVEKIT_API_SECRET"),
        room_name=os.environ.get("ROXROOM_ROOM_NAME", "roxroom-dev").strip(),
        dost=BotIdentity(
            identity=os.environ.get("ROXROOM_DOST_IDENTITY", "roxstar-ai-dost").strip(),
            name=os.environ.get("ROXROOM_DOST_NAME", "Roxstar AI Dost").strip(),
        ),
        sathi=BotIdentity(
            identity=os.environ.get("ROXROOM_SATHI_IDENTITY", "roxstar-ai-sathi").strip(),
            name=os.environ.get("ROXROOM_SATHI_NAME", "Roxstar AI Sathi").strip(),
        ),
    )
