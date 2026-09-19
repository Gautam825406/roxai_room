"""HTTP token server for the React frontend.

The rest of the project mints LiveKit tokens from short-lived CLI scripts
(`scripts/gen_human_token.py`) or in-process (`run_bot.py`/`main.py`). The
frontend needs the same thing over HTTP: a human types a display name, the
browser asks this server for a join token, then connects to LiveKit directly
with it.

Run: `python -m roxroom.token_server` (see frontend/README.md)
"""
from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from roxroom.config import MissingConfigError, load_config
from roxroom.livekit_token import mint_token

app = FastAPI(title="RoxRoom AI token server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    room: str | None = None


class TokenResponse(BaseModel):
    token: str
    url: str
    identity: str
    room: str
    bots: list[str]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "human"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/token", response_model=TokenResponse)
def issue_token(body: TokenRequest) -> TokenResponse:
    try:
        config = load_config()
    except MissingConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    room_name = body.room or config.room_name
    identity = f"human-{_slugify(body.name)}"
    token = mint_token(config, identity=identity, name=body.name, room_name=room_name)

    return TokenResponse(
        token=token,
        url=config.livekit_url,
        identity=identity,
        room=room_name,
        bots=[config.dost.name, config.sathi.name],
    )


def main() -> None:
    import os

    import uvicorn

    # Render (and most PaaS hosts) assign a dynamic port via $PORT and health-check
    # whatever the service actually binds -- 8787 remains the local-dev default.
    port = int(os.environ.get("PORT", "8787"))
    uvicorn.run("roxroom.token_server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
