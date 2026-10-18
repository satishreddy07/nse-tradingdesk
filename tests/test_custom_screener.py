"""
Unit Tests for Custom Screener & Rule Builder Engine (`tests/test_custom_screener.py`)
"""
import pytest
from fastapi.testclient import TestClient

from engines.custom_screener_engine import custom_screener_engine, DEFAULT_EQ_FINAL_CONDITIONS
from web.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_eq_final_default_conditions():
    """Verifies that all 8 conditions from the user's screenshot are configured in EQ-FINAL."""
    presets = custom_screener_engine.get_presets()
    assert "EQ-FINAL" in presets
    eq = presets["EQ-FINAL"]
    assert eq["is_default"] is True
    conditions = eq["conditions"]
    assert len(conditions) == 8

    # Verify key field presence
    fields = [c["field"] for c in conditions]
    assert any("Market Cap" in f for f in fields)
    assert any("Daily Sma( Daily Volume * Daily Close , 50 )" in f for f in fields)
    assert any("Return on capital employed" in f for f in fields)
    assert any("Return on net worth" in f for f in fields)
    assert any("Debt equity ratio" in f for f in fields)
    assert any("Eps after extraordinary items" in f for f in fields)
    assert any("Net profit/reported profit after tax" in f for f in fields)
    assert any("Daily Close" in f for f in fields)


def test_build_scan_clause():
    """Verifies that build_scan_clause generates valid Chartink query string."""
    clause = custom_screener_engine.build_scan_clause(DEFAULT_EQ_FINAL_CONDITIONS)
    assert clause.startswith("( {cash} (")
    assert clause.endswith(") )")
    assert "Market Cap >= 300" in clause
    assert "Daily Close > 75" in clause


def test_preset_save_and_delete():
    """Verifies saving and deleting custom presets in SQLite."""
    test_cond = [
        {"id": "t1", "timeframe": "Daily", "field": "Daily Close", "operator": ">", "value": 100, "raw_clause": "Daily Close > 100"}
    ]
    ok = custom_screener_engine.save_preset("TEST_SCAN", test_cond, "Test description")
    assert ok is True

    presets = custom_screener_engine.get_presets()
    assert "TEST_SCAN" in presets
    assert presets["TEST_SCAN"]["description"] == "Test description"

    # Deleting TEST_SCAN should succeed
    del_ok = custom_screener_engine.delete_preset("TEST_SCAN")
    assert del_ok is True
    assert "TEST_SCAN" not in custom_screener_engine.get_presets()

    # Attempting to delete EQ-FINAL should fail
    del_eq = custom_screener_engine.delete_preset("EQ-FINAL")
    assert del_eq is False


def test_screener_api_endpoints(client):
    """Verifies that REST endpoints return expected schemas."""
    # 1. GET /api/screener/presets
    res = client.get("/api/screener/presets")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "EQ-FINAL" in data["presets"]

    # 2. POST /api/screener/run with small custom condition
    req_body = {
        "preset_name": "EQ-FINAL",
        "custom_conditions": [
            {"id": "c1", "timeframe": "Daily", "field": "Market Cap", "operator": ">=", "value": 100000, "raw_clause": "Market Cap >= 100000"}
        ]
    }
    run_res = client.post("/api/screener/run", json=req_body)
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["status"] == "SUCCESS"
    assert "results" in run_data
    assert isinstance(run_data["results"], list)
    if run_data["results"]:
        top = run_data["results"][0]
        assert "symbol" in top
        assert "close" in top
        assert "per_chg" in top
        assert "volume" in top
        assert "ema20" in top
