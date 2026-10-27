"""
Integration Tests for Market Data Capture, Health & Replay REST API Endpoints.
"""

import pytest
from starlette.testclient import TestClient
from web.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_market_data_health_endpoint(client):
    res = client.get("/api/market-data/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "capture_engine_active" in data
    assert "queue_depth" in data
    assert "events_persisted_total" in data
    assert "table_record_counts" in data


def test_market_data_sessions_endpoint(client):
    res = client.get("/api/market-data/sessions")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_market_data_snapshots_10s_endpoint(client):
    res = client.get("/api/market-data/history/snapshots-10s?session_date=2026-02-19&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert "snapshots" in data
    assert isinstance(data["snapshots"], list)


def test_market_data_snapshots_10m_endpoint(client):
    res = client.get("/api/market-data/history/snapshots-10m?session_date=2026-02-19")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert "snapshots" in data
    assert isinstance(data["snapshots"], list)


def test_market_data_quality_events_endpoint(client):
    res = client.get("/api/market-data/history/quality-events?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert "events" in data
    assert isinstance(data["events"], list)


def test_replay_lifecycle_endpoints(client):
    # Check initial replay state
    res = client.get("/api/market-data/replay/state")
    assert res.status_code == 200
    state = res.json()
    assert "status" in state

    # Stop / reset replay
    stop_res = client.post("/api/market-data/replay/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "STOPPED"
