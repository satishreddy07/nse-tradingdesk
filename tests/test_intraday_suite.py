"""
tests/test_intraday_suite.py
============================
Comprehensive test suite for the Institutional Intraday Multi-Pillar Suite:
  1. VWAP, 15m/30m ORB, and Choppiness Index (CHOP)
  2. Dealer Gamma Exposure (GEX) & Zero-Gamma Flip Level
  3. Real-Time Intraday Sectoral Breadth & Inter-Sectoral Drag
  4. Full Intraday Suite Aggregation & Caching
  5. Flash Bulletins & Priority Wall-Breach Detection
  6. API Endpoint `GET /api/nifty/intraday-advanced`
"""

import pytest
from starlette.testclient import TestClient

from web.api import app
from engines.intraday_analytics_engine import IntradayAnalyticsEngine, intraday_analytics_engine


@pytest.fixture
def client():
    return TestClient(app)


def test_vwap_and_orb_calculations():
    engine = IntradayAnalyticsEngine()
    res = engine.compute_vwap_and_orb()
    assert res is not None
    assert "status" in res
    assert "spot_price" in res
    assert "vwap" in res
    assert "vwap_bands" in res
    assert "orb_15m" in res
    assert "orb_30m" in res
    assert "choppiness_index" in res
    assert "choppiness_regime" in res

    # Check numeric bounds
    assert res["spot_price"] > 10000.0
    assert res["vwap"] > 10000.0
    assert res["vwap_bands"]["upper_1s"] >= res["vwap"]
    assert res["vwap_bands"]["lower_1s"] <= res["vwap"]
    assert 0.0 <= res["choppiness_index"] <= 100.0
    assert res["orb_30m"]["range_pts"] >= 0.0


def test_gamma_exposure_and_zero_flip():
    engine = IntradayAnalyticsEngine()
    # Mock realistic weekly option chain strikes
    dummy_chain = {
        "spot_price": 22700.0,
        "expiry": "16-Oct-2026",
        "strikes": [
            {"strike": 22600, "ce_oi": 40000, "pe_oi": 90000, "ce_iv": 13.0, "pe_iv": 11.0},
            {"strike": 22650, "ce_oi": 30000, "pe_oi": 70000, "ce_iv": 13.2, "pe_iv": 11.2},
            {"strike": 22700, "ce_oi": 60000, "pe_oi": 65000, "ce_iv": 13.5, "pe_iv": 11.5},
            {"strike": 22750, "ce_oi": 80000, "pe_oi": 35000, "ce_iv": 13.8, "pe_iv": 11.8},
            {"strike": 22800, "ce_oi": 110000, "pe_oi": 20000, "ce_iv": 14.0, "pe_iv": 12.0},
        ]
    }
    gex_res = engine.compute_gamma_exposure_and_flip(dummy_chain)
    assert gex_res["status"] == "OK"
    assert "zero_gamma_flip_level" in gex_res
    assert "market_gex_regime" in gex_res
    assert "total_net_gex_cr" in gex_res
    assert "max_gamma_strike" in gex_res

    # Flip level should lie within the strike range
    assert 22500.0 <= gex_res["zero_gamma_flip_level"] <= 22900.0
    assert gex_res["market_gex_regime"] in ("LONG_GAMMA_STABILIZING", "SHORT_GAMMA_VOLATILITY_EXPANSION")
    assert len(gex_res["active_strikes_gex"]) == 5


def test_sectoral_divergence_gauge():
    engine = IntradayAnalyticsEngine()
    sec_res = engine.compute_sectoral_divergence()
    assert sec_res["status"] == "OK"
    assert "breadth_pct_green" in sec_res
    assert 0.0 <= sec_res["breadth_pct_green"] <= 100.0
    assert sec_res["total_sectors"] == 7
    assert "bank_vs_it_alignment" in sec_res
    assert "sector_verdict" in sec_res
    assert len(sec_res["sectors"]) == 7


def test_full_intraday_suite_aggregation():
    suite = intraday_analytics_engine.get_full_intraday_suite(force_refresh=True)
    assert suite["status"] == "OK"
    assert "vwap_and_orb" in suite
    assert "gamma_exposure" in suite
    assert "sectoral_divergence" in suite
    assert suite["spot_price"] > 10000.0


def test_flash_bulletin_and_wall_breach():
    engine = IntradayAnalyticsEngine()
    bulletin = engine.generate_intraday_ai_flash_bulletin(slot_label="TEST_SLOT")
    assert bulletin["status"] == "DISPATCHED"
    assert "bulletin" in bulletin
    assert "Nifty Spot" in bulletin["bulletin"]

    # Test wall breach alerts
    spot = 22750.0
    # Normal case: within walls
    alert_none = engine.check_intraday_wall_breach_and_alert(spot=spot, call_wall=22800, put_wall=22600)
    assert alert_none is None

    # Breach above call wall + 20
    alert_call = engine.check_intraday_wall_breach_and_alert(spot=22825.0, call_wall=22800, put_wall=22600)
    assert alert_call is not None
    assert "CALL WALL BREACH" in alert_call

    # Second check on same day should debounce (avoid duplicate spam)
    alert_repeat = engine.check_intraday_wall_breach_and_alert(spot=22830.0, call_wall=22800, put_wall=22600)
    assert alert_repeat is None


def test_api_intraday_advanced_endpoint(client):
    resp = client.get("/api/nifty/intraday-advanced?force_refresh=false")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "OK"
    assert "vwap_and_orb" in data
    assert "gamma_exposure" in data
    assert "sectoral_divergence" in data
