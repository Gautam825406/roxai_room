"""LiveKit access token minting, shared by bot processes and the human-join helper script."""
from __future__ import annotations

from livekit import api

from roxroom.config import RoxRoomConfig


def mint_token(
    config: RoxRoomConfig,
    identity: str,
    name: str,
    room_name: str | None = None,
    hidden: bool = False,
) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=room_name or config.room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        hidden=hidden,
    )
    token = (
        api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(grants)
    )
    return token.to_jwt()
