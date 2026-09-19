"""Mint a LiveKit join token for a human tester.

Usage:
    python scripts/gen_human_token.py "Priya"
    python scripts/gen_human_token.py "Rahul" --room roxroom-dev

Paste the printed URL into a browser to join the dev room via LiveKit's hosted
meet client (https://meet.livekit.io) and confirm you can hear the bot's test tone.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roxroom.config import load_config  # noqa: E402
from roxroom.livekit_token import mint_token  # noqa: E402


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "human"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Display name for the human participant")
    parser.add_argument("--room", default=None, help="Override the default room name")
    args = parser.parse_args()

    config = load_config()
    room_name = args.room or config.room_name
    identity = f"human-{slugify(args.name)}"

    token = mint_token(config, identity=identity, name=args.name, room_name=room_name)

    print(f"identity: {identity}")
    print(f"room:     {room_name}")
    print(f"token:    {token}")
    print()
    print("Join via LiveKit's hosted meet client:")
    print(
        f"  https://meet.livekit.io/custom?liveKitUrl={quote(config.livekit_url)}"
        f"&token={quote(token)}"
    )


if __name__ == "__main__":
    main()
