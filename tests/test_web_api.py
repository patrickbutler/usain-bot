from datetime import date

import pytest
from fastapi.testclient import TestClient

from usain_bot import agent
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.storage.local import LocalBackend
from usain_bot.web.app import create_app


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


@pytest.fixture
def adapter(fixture_path) -> MockGarminAdapter:
    return MockGarminAdapter(fixture_path)


@pytest.fixture
def client(config, storage, adapter, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        "usain_bot.service.date",
        type("FixedDate", (), {"today": staticmethod(lambda: date(2026, 7, 24)), "fromisoformat": date.fromisoformat}),
    )
    app = create_app(config, storage, adapter)
    return TestClient(app)


class TestStatusEndpoint:
    def test_no_plan_initially(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["has_plan"] is False

    def test_has_plan_after_today(self, client):
        client.get("/api/today")
        resp = client.get("/api/status")
        assert resp.json()["has_plan"] is True


class TestTodayEndpoint:
    def test_returns_recommendation_shape(self, client):
        resp = client.get("/api/today")
        assert resp.status_code == 200
        body = resp.json()
        assert "recommendation" in body
        assert "next_7_days" in body
        assert len(body["next_7_days"]) == 7
        assert "anchors" in body
        assert body["recommendation"]["target_distance_mi"] is not None

    def test_second_call_does_not_bump_plan_version(self, client):
        first = client.get("/api/today").json()
        second = client.get("/api/today").json()
        assert first["plan_version"] == second["plan_version"]

    def test_refresh_recomputes(self, client):
        first = client.get("/api/today").json()
        refreshed = client.post("/api/today/refresh").json()
        assert refreshed["plan_version"] == first["plan_version"] + 1


class TestPlanEndpoint:
    def test_error_before_any_computation(self, client):
        resp = client.get("/api/plan")
        assert resp.json().get("error")

    def test_plan_after_today_computed(self, client):
        client.get("/api/today")
        resp = client.get("/api/plan")
        body = resp.json()
        assert body["plan_version"] >= 1
        assert len(body["weeks"]) > 10

    def test_plan_history(self, client):
        client.get("/api/today")
        client.post("/api/today/refresh")
        resp = client.get("/api/plan/history")
        assert len(resp.json()["versions"]) >= 2


class TestHistoryEndpoint:
    def test_returns_activities_and_weekly_volume(self, client):
        client.post("/api/sync")
        resp = client.get("/api/history?days=90")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["activities"]) > 0
        assert len(body["weekly_volume"]) > 0
        assert all("run_class" in a for a in body["activities"])

    def test_days_param_clamped(self, client):
        resp = client.get("/api/history?days=999999")
        assert resp.status_code == 200


class TestSyncEndpoint:
    def test_sync_returns_summary(self, client):
        resp = client.post("/api/sync")
        assert resp.status_code == 200
        body = resp.json()
        assert body["live"] is True
        assert "message" in body


class TestHealthFlagEndpoint:
    def test_valid_flag_forces_easy(self, client):
        resp = client.post("/api/health-flag", json={"flag": "hip", "note": "twinge"})
        assert resp.status_code == 200
        assert resp.json()["today"]["run_type"] == "easy"

    def test_invalid_flag_rejected(self, client):
        resp = client.post("/api/health-flag", json={"flag": "knee"})
        assert resp.status_code == 400


class TestChatEndpointWithoutApiKey:
    def test_chat_fails_gracefully_without_api_key(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        resp = client.post("/api/chat", json={"message": "what should I run today"})
        assert resp.status_code == 503
        assert "ANTHROPIC_API_KEY" in resp.json()["detail"]

    def test_empty_message_rejected(self, client):
        resp = client.post("/api/chat", json={"message": "   "})
        assert resp.status_code == 400


class TestStaticFrontend:
    def test_index_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "usain-bot" in resp.text

    def test_static_assets_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
