# tests/test_intraday_orderflow_and_watchdog.py
"""
Test Suite: Intraday Order Flow, CVD, GEX, ATM IV, and Dynamic Watchdog Sentinel
Verifies zero hardcoding, authentic volume delta calculations, robust null-safety,
and dynamic multi-system watchdog health probes.
"""

import pytest
from unittest.mock import patch, MagicMock
from engines.order_flow_structure_engine import OrderFlowStructureEngine, order_flow_structure_engine
from engines.intraday_analytics_engine import intraday_analytics_engine
from engines.nifty_analytics_engine import nifty_analytics_engine
from engines.kotak_neo_engine import kotak_neo_engine
from engines.omni_watchdog_engine import omni_watchdog


def test_cvd_no_hardcoded_negative_numbers():
    """Verify CVD engine has no -195 or fake hardcoded constants."""
    engine = OrderFlowStructureEngine()
    assert engine._last_cvd_value == 0.0
    assert engine._session_closed_cvd is None

    # When spot is mock-provided
    with patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=24500.0):
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_option_depth", return_value={"total_buy": 100, "total_sell": 100, "volume": 500}):
            res = engine.compute_cumulative_volume_delta()
            assert "current_cvd" in res
            assert res["current_cvd"] != -195, "CVD must not return hardcoded -195"
            assert "timestamp_ist" in res
            assert "data_source" in res
            assert "heavyweights_tracked_count" in res


def test_volume_profile_and_gex_provenance_and_timestamps():
    """Verify Volume Profile and GEX return genuine data provenance and timestamps."""
    vp = order_flow_structure_engine.compute_volume_profile()
    assert "timestamp_ist" in vp
    assert "data_source" in vp
    assert vp.get("data_source") == "AUTHENTIC_INTRADAY_CANDLES"

    fake_chain = {
        "spot_price": 24500.0,
        "expiry": "2026-10-29",
        "strikes": [
            {"strike": 24400, "ce_oi": 1000, "pe_oi": 800, "ce_ltp": 150.0, "pe_ltp": 60.0},
            {"strike": 24500, "ce_oi": 5000, "pe_oi": 4000, "ce_ltp": 80.0, "pe_ltp": 85.0},
            {"strike": 24600, "ce_oi": 3000, "pe_oi": 1200, "ce_ltp": 35.0, "pe_ltp": 140.0},
        ]
    }
    gex = intraday_analytics_engine.compute_gamma_exposure_and_flip(fake_chain)
    assert gex["status"] == "OK"
    assert "timestamp_ist" in gex
    assert "data_source" in gex
    assert gex["data_source"] == "BLACK_SCHOLES_DEALER_GAMMA_CALCULATED"
    assert "zero_gamma_flip_level" in gex
    assert gex["zero_gamma_flip_level"] > 0


def test_black_scholes_iv_calculation_accuracy():
    """Verify Black-Scholes implied volatility produces realistic values and no zeros for active options."""
    # Nifty at 24500, 24500 strike ATM Call trading at ~120 with 3 days to expiry
    S, K = 24500.0, 24500.0
    dte_years = 3.0 / 365.25
    c_ltp = 120.0
    iv = kotak_neo_engine._calc_implied_vol(c_ltp, S, K, dte_years, r=0.065, is_call=True)
    assert 5.0 <= iv <= 30.0, f"Expected realistic IV between 5% and 30%, got {iv}%"

    # Out-of-the-money put: 24300 Put trading at ~45
    p_ltp = 45.0
    p_iv = kotak_neo_engine._calc_implied_vol(p_ltp, S, 24300.0, dte_years, r=0.065, is_call=False)
    assert 5.0 <= p_iv <= 35.0, f"Expected realistic IV between 5% and 35%, got {p_iv}%"


def test_dynamic_watchdog_intraday_sentinel():
    """Verify Watchdog dynamically probes CVD, ATM IV, and GEX."""
    probe = omni_watchdog.probe_subsystem_intraday()
    assert probe["status"] in ("HEALTHY", "DEGRADED", "ALERT")
    assert "running_cvd" in probe["details"]
    assert "atm_iv_and_skew" in probe["details"]
    assert "zero_gamma_flip" in probe["details"]
    assert "tactical_walls" in probe["details"]


def test_intraday_template_null_safety():
    """Verify web/templates/index.html includes null-safe handlers for choppiness and GEX."""
    with open("web/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Null-safe choppiness
    assert "Number.isFinite(cIdx)" in html
    # Safe GEX badge text
    assert "Neutral Gamma" in html
    # Bounded Put/Call Wall Card and KPI cards
    assert "min-h-[84px]" in html or "min-h-[76px]" in html
    assert "overflow-hidden" in html
    assert "updateIntradayActiveOiCard" in html
    assert "Net Active OI (±10" in html


def test_intraday_active_contracts_and_pm10_strikes():
    """Verify ±10 strikes expansion and shares-to-contracts conversion via dynamic lot size."""
    from core.lot_size import resolve_nifty_lot_size
    lot_size = resolve_nifty_lot_size(expiry_date="2026-10-29", strict=False)
    assert lot_size == 65, "2026 contract lot size must be 65 under circular 70616"

    # Create dummy chain with 25 strikes around 24500
    rows = []
    for k in range(24000, 25250, 50):
        rows.append({
            "strikePrice": k,
            "CE": {
                "strikePrice": k,
                "openInterest": 65000,
                "changeinOpenInterest": -6500,
                "lastPrice": 100.0,
                "totalTradedVolume": 10000,
            },
            "PE": {
                "strikePrice": k,
                "openInterest": 130000,
                "changeinOpenInterest": 13000,
                "lastPrice": 100.0,
                "totalTradedVolume": 10000,
            }
        })

    analysis = nifty_analytics_engine._analyze_single_expiry_chain(rows, 24500.0, "2026-10-29", "2026-10-29 15:30:00")
    assert analysis["status"] == "OK"
    oi_flow = analysis["oi_flow"]
    assert oi_flow["active_strikes_count"] == 21, "Must span 21 strikes (±10 strikes around ATM)"
    assert oi_flow["lot_size"] == 65

    # 21 strikes * 130,000 PE OI = 2,730,000 shares = 42,000 contracts
    # 21 strikes * 13,000 PE Chg = 273,000 shares = 4,200 contracts
    # Total Active PE = 42,000 + 4,200 = 46,200 contracts
    assert oi_flow["active_pe_total_contracts"] == 46200
    # 21 strikes * 65,000 CE OI = 1,365,000 shares = 21,000 contracts
    # 21 strikes * (-6,500) CE Chg = -136,500 shares = -2,100 contracts
    # Total Active CE = 21,000 - 2,100 = 18,900 contracts
    assert oi_flow["active_ce_total_contracts"] == 18900
    assert oi_flow["net_active_oi_diff_contracts"] == 46200 - 18900

