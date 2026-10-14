import pytest
from fastapi.testclient import TestClient
from web.api import app
from execution.state_engine import state_engine


def test_chart_reset_endpoint_and_payload_structure():
    client = TestClient(app)

    # 1. Test POST /api/stocks/{symbol}/reset_chart
    res = client.post("/api/stocks/ADVANCE/reset_chart")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["symbol"] == "ADVANCE"
    assert "trade_levels" in data
    tl = data["trade_levels"]
    assert "show_trade_lines" in tl
    assert "is_completed" in tl
    assert "trade_status" in tl

    # 2. Test chart payload for a non-active stock has clean technicals without synthetic trade lines
    res_tech = client.get("/api/stocks/RELIANCE/chart?bars=60")
    if res_tech.status_code == 200:
        tech_data = res_tech.json()
        if tech_data.get("status") == "OK":
            tech_tl = tech_data["trade_levels"]
            # Active positions is not RELIANCE, so show_trade_lines should be False unless active signal exists
            if not tech_data.get("active_position") and not tech_data.get("signal"):
                assert tech_tl["show_trade_lines"] is False
                assert tech_tl["stop_loss"] is None
                assert tech_tl["target_1"] is None
