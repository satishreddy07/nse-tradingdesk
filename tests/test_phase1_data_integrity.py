"""
========================================================================================
Phase 1 Data Integrity Regression Suite (NSE Trading Desk)
========================================================================================
Mandatory verification for Phase 1 remediation items:
1. engines/nifty_analytics_engine.py: get_nifty_prev_close() never falls back to 22811.35.
2. engines/intraday_analytics_engine.py: compute_weekly_theta_decay_curve() and
   compute_delta_neutral_hedging_band() never fall back to 22800.0.
3. engines/options_edge_analytics_engine.py: compute_15m_strike_oi_pulse() and
   compute_strike_gex_profile() never fall back to 22600.0.
4. engines/order_flow_structure_engine.py: compute_cumulative_volume_delta()
   never falls back to 22600.0 or queries depth for strike 22600.

All tests verify that:
- Authentic market data preserves existing successful behavior.
- Missing data / broker exceptions propagate explicit UNAVAILABLE states.
- Synthetic prices (22600.0, 22800.0, 22811.35) are never used.
========================================================================================
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from core.data_quality import now_ist
from engines.nifty_analytics_engine import NiftyAnalyticsEngine, nifty_analytics_engine
from engines.intraday_analytics_engine import IntradayAnalyticsEngine, intraday_analytics_engine
from engines.options_edge_analytics_engine import OptionsEdgeAnalyticsEngine, options_edge_analytics_engine
from engines.order_flow_structure_engine import OrderFlowStructureEngine, order_flow_structure_engine
from web.api import app

client = TestClient(app)


# ======================================================================================
# Item 1: get_nifty_prev_close (engines/nifty_analytics_engine.py)
# ======================================================================================

def test_nifty_prev_close_authentic_data():
    """Verify that when authentic previous close is available, it is returned accurately."""
    today_str = now_ist().strftime("%Y-%m-%d")
    with patch.object(NiftyAnalyticsEngine, "_cached_nifty_prev_close", (today_str, 24850.50)):
        res = NiftyAnalyticsEngine.get_nifty_prev_close()
        assert res == 24850.50
        assert isinstance(res, float)


def test_nifty_prev_close_missing_data_returns_none_never_22811():
    """Verify that when all data sources fail, get_nifty_prev_close returns None, NEVER 22811.35."""
    with patch.object(NiftyAnalyticsEngine, "_cached_nifty_prev_close", None):
        with patch("engines.state_engine.state_engine._get_connection") as mock_conn:
            # Mock DB returning no rows
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor

            with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
                res = NiftyAnalyticsEngine.get_nifty_prev_close()
                assert res is None, f"Expected None on missing data, got: {res}"
                assert res != 22811.35, "CRITICAL: Hardcoded fallback 22811.35 returned!"


def test_nifty_option_chain_caller_handles_none_prev_close():
    """Verify that option chain generator handles None prev_close gracefully without crashing or computing against 22811.35."""
    engine = NiftyAnalyticsEngine()
    mock_chain_res = {
        "status": "OK",
        "spot_price": 24500.0,
        "weekly_exp": "2026-10-29",
        "monthly_exp": "2026-11-26",
        "weekly_rows": [{"strike": 24500, "ce_oi": 1000, "pe_oi": 1000}],
        "monthly_rows": [{"strike": 24500, "ce_oi": 1000, "pe_oi": 1000}],
        "fetched_at": "2026-10-24 15:30:00 IST"
    }
    with patch("engines.kotak_neo_engine.kotak_neo_engine.fetch_live_option_chain", return_value=mock_chain_res):
        with patch.object(engine, "get_nifty_prev_close", return_value=None):
            with patch.object(engine, "_analyze_single_expiry_chain", return_value={"status": "OK", "strikes": []}):
                with patch.object(engine, "_build_options_quant_suggestions", return_value=[]):
                    res = engine._generate_kotak_neo_live_option_chain(24500.0)
                    assert res["status"] == "OK"
                    assert res["spot_price"] == 24500.0
                    assert res["prev_close"] is None
                    assert res["change_pts"] is None
                    assert res["change_pct"] is None


# ======================================================================================
# Item 2: compute_weekly_theta_decay_curve & compute_delta_neutral_hedging_band (engines/intraday_analytics_engine.py)
# ======================================================================================

def test_theta_decay_curve_authentic_spot():
    """Verify that authentic spot computes valid theta decay metrics."""
    engine = IntradayAnalyticsEngine()
    res = engine.compute_weekly_theta_decay_curve(spot_price=24500.0)
    assert res["status"] == "OK"
    assert res["base_atm_straddle_daily_pts"] > 0
    # Formula: 0.8 * 24500 * 0.125 * sqrt(1/365) = ~128.3
    expected_approx = 0.8 * 24500.0 * 0.125 * (1.0 / 365.0) ** 0.5
    assert abs(res["base_atm_straddle_daily_pts"] - expected_approx) < 2.0


def test_theta_decay_curve_missing_spot_returns_unavailable_never_22800():
    """Verify that missing spot returns DATA_UNAVAILABLE and never falls back to 22800.0."""
    engine = IntradayAnalyticsEngine()
    with patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=None):
        with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0}):
            res = engine.compute_weekly_theta_decay_curve(spot_price=None, weekly_chain=None)
            assert res["status"] == "DATA_UNAVAILABLE"
            assert res["base_atm_straddle_daily_pts"] == 0.0
            assert res["current_decay_rate_pts_per_hour"] == 0.0
            assert res["decay_schedule"] == []
            assert "22800" not in str(res)


def test_delta_hedging_band_authentic_spot():
    """Verify that authentic spot computes valid 1-sigma and 2-sigma hedging bands."""
    engine = IntradayAnalyticsEngine()
    res = engine.compute_delta_neutral_hedging_band(spot_price=24500.0, iv=12.5, dte_days=5.0)
    assert res["status"] == "OK"
    assert res["spot_price"] == 24500.0
    assert res["upper_1_sigma"] > 24500.0
    assert res["lower_1_sigma"] < 24500.0
    assert res["upper_2_sigma"] > res["upper_1_sigma"]
    assert res["lower_2_sigma"] < res["lower_1_sigma"]


def test_delta_hedging_band_missing_spot_returns_unavailable_never_22800():
    """Verify that broker exceptions or missing spot return DATA_UNAVAILABLE without calculating bands around 22800."""
    engine = IntradayAnalyticsEngine()
    with patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", side_effect=Exception("Broker timeout")):
        with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0}):
            res = engine.compute_delta_neutral_hedging_band(spot_price=None)
            assert res["status"] == "DATA_UNAVAILABLE"
            assert res["spot_price"] is None
            assert res["upper_1_sigma"] is None
            assert res["lower_1_sigma"] is None
            assert res["delta_status"] == "DATA_UNAVAILABLE"
            assert res["action_required"] is False
            assert "22800" not in str(res)


# ======================================================================================
# Item 3: compute_15m_strike_oi_pulse & compute_strike_gex_profile (engines/options_edge_analytics_engine.py)
# ======================================================================================

def test_options_edge_oi_pulse_authentic_spot():
    """Verify that authentic option chain generates strike buildup classifications around authentic spot."""
    engine = OptionsEdgeAnalyticsEngine()
    mock_chain = {
        "weekly": {
            "spot_price": 24500.0,
            "strikes": [
                {"strike": 24400, "ce_oi": 50000, "pe_oi": 80000, "ce_change_oi": 5000, "pe_change_oi": 10000, "ce_ltp": 150.0, "pe_ltp": 60.0, "ce_change": 10.0, "pe_change": -5.0},
                {"strike": 24500, "ce_oi": 90000, "pe_oi": 90000, "ce_change_oi": 8000, "pe_change_oi": -2000, "ce_ltp": 90.0, "pe_ltp": 90.0, "ce_change": 5.0, "pe_change": -2.0},
                {"strike": 24600, "ce_oi": 120000, "pe_oi": 40000, "ce_change_oi": 15000, "pe_change_oi": -4000, "ce_ltp": 50.0, "pe_ltp": 140.0, "ce_change": -8.0, "pe_change": 12.0},
            ]
        }
    }
    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", return_value=mock_chain):
        pulse = engine.compute_15m_strike_oi_pulse()
        assert pulse["spot_price"] == 24500.0
        assert len(pulse["strikes"]) == 3
        assert pulse["dominant_regime"] in ("LONG_BUILDUP", "SHORT_BUILDUP", "SHORT_COVERING", "LONG_UNWINDING", "NEUTRAL")


def test_options_edge_oi_pulse_missing_spot_returns_unavailable_never_22600():
    """Verify that missing spot or empty chain returns DATA_UNAVAILABLE and never centers around 22600."""
    engine = OptionsEdgeAnalyticsEngine()
    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", return_value={}):
        with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0}):
            pulse = engine.compute_15m_strike_oi_pulse()
            assert pulse["status"] == "DATA_UNAVAILABLE"
            assert pulse["dominant_regime"] == "DATA_UNAVAILABLE"
            assert pulse["strikes"] == []
            assert pulse["spot_price"] == 0.0


def test_strike_gex_profile_authentic_spot():
    """Verify that authentic spot computes valid dealer GEX rows."""
    engine = OptionsEdgeAnalyticsEngine()
    mock_chain = {
        "weekly": {
            "spot_price": 24500.0,
            "strikes": [
                {"strike": 24500, "ce_oi": 80000, "pe_oi": 60000, "ce_iv": 13.5},
            ]
        }
    }
    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", return_value=mock_chain):
        gex = engine.compute_strike_gex_profile()
        assert gex["spot"] == 24500.0
        assert len(gex["strikes"]) == 1
        assert "net_gex_crores" in gex["strikes"][0]
        assert "dealer_bias" in gex["strikes"][0]


def test_strike_gex_profile_missing_spot_returns_unavailable_never_22600():
    """Verify that missing spot returns DATA_UNAVAILABLE with empty strikes and never computes at 22600."""
    engine = OptionsEdgeAnalyticsEngine()
    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", side_effect=Exception("Feed down")):
        with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0}):
            gex = engine.compute_strike_gex_profile()
            assert gex["status"] == "DATA_UNAVAILABLE"
            assert gex["spot"] == 0.0
            assert gex["strikes"] == []
            assert gex["total_net_gex_crores"] == 0.0
            assert gex["gamma_flip_strike"] == 0


# ======================================================================================
# Item 4: compute_cumulative_volume_delta (engines/order_flow_structure_engine.py)
# ======================================================================================

def test_cumulative_volume_delta_authentic_spot():
    """Verify that authentic spot queries depth for the authentic ATM strike, not 22600."""
    engine = OrderFlowStructureEngine()
    mock_depth = {
        "total_buy": 15000,
        "total_sell": 12000,
        "volume": 50000
    }
    with patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=24500.0):
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_option_depth", return_value=mock_depth) as mock_get_depth:
            res = engine.compute_cumulative_volume_delta()
            # Must query strike 24500, NOT 22600!
            assert mock_get_depth.call_count >= 2
            called_strikes = [call.args[0] for call in mock_get_depth.call_args_list]
            assert all(s == 24500 for s in called_strikes), f"Expected strike 24500, but called: {called_strikes}"
            assert 22600 not in called_strikes
            assert "current_cvd" in res
            assert "order_book_imbalance_pct" in res


def test_cumulative_volume_delta_missing_spot_returns_unavailable_never_queries_22600():
    """Verify that missing spot returns DATA_UNAVAILABLE and NEVER queries option depth for strike 22600."""
    engine = OrderFlowStructureEngine()
    with patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=None):
        with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0}):
            with patch("engines.kotak_neo_engine.kotak_neo_engine.get_option_depth") as mock_get_depth:
                res = engine.compute_cumulative_volume_delta()
                # Zero calls to get_option_depth should happen when spot is unavailable
                assert mock_get_depth.call_count == 0, f"Expected 0 calls, but got {mock_get_depth.call_count}!"
                assert res["status"] == "DATA_UNAVAILABLE"
                assert res["instantaneous_delta"] == 0
                assert res["order_book_imbalance_pct"] == 0.0
                assert res["flow_sentiment"] == "NEUTRAL"
                assert "DATA_UNAVAILABLE" in res["absorption_signal"]


# ======================================================================================
# API Endpoint Compatibility Tests
# ======================================================================================

def test_api_options_oi_pulse_contract_compatibility():
    """Verify /api/options/edge/oi-pulse endpoint contract."""
    response = client.get("/api/options/edge/oi-pulse")
    assert response.status_code == 200
    data = response.json()
    assert "strikes" in data
    assert "dominant_regime" in data
    assert "spot_price" in data


def test_api_options_volatility_profile_contract_compatibility():
    """Verify /api/options/edge/volatility-profile endpoint contract."""
    response = client.get("/api/options/edge/volatility-profile")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "strike_gex" in data
    assert "strikes" in data["strike_gex"]
    assert "total_net_gex_crores" in data["strike_gex"]


def test_api_intraday_order_flow_contract_compatibility():
    """Verify /api/intraday/order-flow endpoint contract."""
    response = client.get("/api/intraday/order-flow")
    assert response.status_code == 200
    data = response.json()
    assert "cumulative_volume_delta" in data
    assert "current_cvd" in data["cumulative_volume_delta"]
    assert "flow_sentiment" in data["cumulative_volume_delta"]
    assert "order_book_imbalance_pct" in data["cumulative_volume_delta"]


def test_api_theta_decay_curve_contract_compatibility():
    """Verify /api/nifty/options/theta-decay-curve endpoint contract."""
    response = client.get("/api/nifty/options/theta-decay-curve")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "decay_schedule" in data
    assert "theta_phase" in data


def test_api_delta_hedging_band_contract_compatibility():
    """Verify /api/nifty/options/delta-hedging-band endpoint contract."""
    response = client.get("/api/nifty/options/delta-hedging-band")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "delta_status" in data
    assert "action_required" in data
