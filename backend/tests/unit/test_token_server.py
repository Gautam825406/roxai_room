import pytest
from fastapi.testclient import TestClient

from roxroom.config import BotIdentity, MissingConfigError, RoxRoomConfig
from roxroom.token_server import app

FAKE_CONFIG = RoxRoomConfig(
    livekit_url="wss://example.livekit.cloud",
    livekit_api_key="devkey",
    livekit_api_secret="devsecretdevsecretdevsecret32",
    room_name="roxroom-dev",
    dost=BotIdentity(identity="roxstar-ai-dost", name="Roxstar AI Dost"),
    sathi=BotIdentity(identity="roxstar-ai-sathi", name="Roxstar AI Sathi"),
)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr("roxroom.token_server.load_config", lambda: FAKE_CONFIG)
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_issue_token(client: TestClient):
    resp = client.post("/api/token", json={"name": "Priya"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["identity"] == "human-priya"
    assert body["room"] == "roxroom-dev"
    assert body["url"] == FAKE_CONFIG.livekit_url
    assert body["bots"] == ["Roxstar AI Dost", "Roxstar AI Sathi"]
    assert body["token"].count(".") == 2


def test_issue_token_slugifies_and_overrides_room(client: TestClient):
    resp = client.post("/api/token", json={"name": "Amit Kumar!!", "room": "custom-room"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["identity"] == "human-amit-kumar"
    assert body["room"] == "custom-room"


def test_issue_token_rejects_empty_name(client: TestClient):
    resp = client.post("/api/token", json={"name": ""})
    assert resp.status_code == 422


def test_issue_token_surfaces_missing_config(monkeypatch):
    def raise_missing() -> RoxRoomConfig:
        raise MissingConfigError("Missing required environment variable: LIVEKIT_URL")

    monkeypatch.setattr("roxroom.token_server.load_config", raise_missing)
    resp = TestClient(app).post("/api/token", json={"name": "Priya"})
    assert resp.status_code == 500
