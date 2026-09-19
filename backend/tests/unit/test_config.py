from pathlib import Path

import pytest

from roxroom.config import MissingConfigError, load_config
from roxroom.livekit_token import mint_token


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    p = tmp_path / ".env"
    p.write_text(
        "LIVEKIT_URL=wss://example.livekit.cloud\n"
        "LIVEKIT_API_KEY=devkey\n"
        "LIVEKIT_API_SECRET=devsecretdevsecretdevsecret32\n"
    )
    return p


def test_load_config_defaults(env_file: Path):
    cfg = load_config(dotenv_path=str(env_file))
    assert cfg.livekit_url == "wss://example.livekit.cloud"
    assert cfg.room_name == "roxroom-dev"
    assert cfg.dost.name == "Roxstar AI Dost"
    assert cfg.sathi.name == "Roxstar AI Sathi"
    assert cfg.dost.identity != cfg.sathi.identity


def test_load_config_missing_required(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    with pytest.raises(MissingConfigError):
        load_config(dotenv_path=str(empty_env))


def test_mint_token_produces_jwt(env_file: Path):
    cfg = load_config(dotenv_path=str(env_file))
    token = mint_token(cfg, cfg.dost.identity, cfg.dost.name)
    assert token.count(".") == 2  # header.payload.signature
