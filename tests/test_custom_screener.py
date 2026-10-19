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


def test_swing_universe_source_toggle(client):
    """Verifies candidate universe sourcing toggle persistence."""
    from execution.state_engine import state_engine

    # 1. Set to IN_HOUSE
    res1 = client.post("/api/screener/source", json={"source": "IN_HOUSE"})
    assert res1.status_code == 200
    assert res1.json()["source"] == "IN_HOUSE"
    assert state_engine.get_swing_universe_source() == "IN_HOUSE"

    # 2. Query GET /api/screener/source
    res_get = client.get("/api/screener/source")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["status"] == "SUCCESS"
    assert get_data["source"] == "IN_HOUSE"
    assert len(get_data["available_sources"]) == 2

    # 3. Set to CHARTINK
    res2 = client.post("/api/screener/source", json={"source": "CHARTINK"})
    assert res2.status_code == 200
    assert res2.json()["source"] == "CHARTINK"
    assert state_engine.get_swing_universe_source() == "CHARTINK"

    # Reset back to IN_HOUSE
    state_engine.set_swing_universe_source("IN_HOUSE")


def test_screener_comparison_engine_and_audit(client):
    """Verifies dual-engine comparison auditor, latency metrics, and cryptographic checksums."""
    from engines.screener_comparison_engine import screener_comparison_engine

    # 1. Run comparison audit directly on engine
    audit = screener_comparison_engine.run_comparison_audit(preset_name="EQ-FINAL", force_refresh=False)
    assert audit["status"] == "SUCCESS"
    assert audit["is_authenticated"] is True
    assert audit["verification_status"] == "VERIFIED_GENUINE_DYNAMIC"
    assert audit["chartink_count"] > 100
    assert audit["inhouse_qualified_count"] > 50
    assert audit["common_overlap_count"] == audit["inhouse_qualified_count"]
    assert audit["chartink_exclusive_count"] > 100
    assert len(audit["sha256_checksum"]) == 64
    assert audit["chartink_latency_ms"] >= 0
    assert audit["inhouse_latency_ms"] >= 0

    # 2. Test POST /api/screener/compare
    api_res = client.post("/api/screener/compare", json={"preset_name": "EQ-FINAL", "force_refresh": False})
    assert api_res.status_code == 200
    api_data = api_res.json()
    assert api_data["status"] == "SUCCESS"
    assert api_data["chartink_count"] == audit["chartink_count"]
    assert api_data["sha256_checksum"] == audit["sha256_checksum"]

    # 3. Test GET /api/screener/compare/summary
    sum_res = client.get("/api/screener/compare/summary")
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["status"] == "SUCCESS"
    assert "common_overlap_count" in sum_data
    assert "sha256_checksum" in sum_data


def test_inhouse_universe_tickers_formatting():
    """Verifies get_inhouse_universe_tickers returns valid .NS symbols."""
    from engines.screener_comparison_engine import screener_comparison_engine

    tickers = screener_comparison_engine.get_inhouse_universe_tickers()
    assert len(tickers) >= 50
    assert all(t.endswith(".NS") for t in tickers)
    assert all(not t.startswith("^") for t in tickers)


def test_scan_trigger_with_inhouse_source(client):
    """Verifies that POST /api/scan/trigger accepts source='IN_HOUSE' and sets proper initial stage."""
    from execution.state_engine import state_engine

    # Trigger scan with source='IN_HOUSE'
    res = client.post("/api/scan/trigger", json={
        "scan_type": "INSTITUTIONAL",
        "auto_execute": False,
        "source": "IN_HOUSE"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("STARTED", "QUEUED", "ALREADY_RUNNING")

    prog = state_engine.get_scan_progress()
    # Stage should NOT contain Chartink Screener url when IN_HOUSE is active
    assert "https://chartink.com/screener" not in str(prog.get("stage", ""))
    assert state_engine.get_swing_universe_source() == "IN_HOUSE"

