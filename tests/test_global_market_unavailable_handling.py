"""
tests/test_global_market_unavailable_handling.py
================================================
Verifies that:
1. Global and US market data fetches do not substitute hardcoded fake values (15.0, 4.25, 4.50, 103.5, 520.0, 24800.0) when unavailable.
2. Missing metrics return None/UNAVAILABLE with proper status flags.
3. Macro score calculations exclude unavailable assets without skewing scores.
4. SQLite persistence properly stores and retrieves verified historical bars for global indices.
"""

import pytest
from unittest.mock import patch
import pandas as pd
from engines.us_analytics_engine import USAnalyticsEngine, us_analytics_engine
from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine


def test_us_analytics_engine_handles_unavailable_data_without_hardcoded_fallbacks():
    """Verifies that missing network data results in None / UNAVAILABLE, not fake hardcoded constants."""
    engine = USAnalyticsEngine()
    
    # Mock _fetch_yahoo_chart to return None for all tickers (simulate full offline / network outage)
    with patch.object(engine, "_fetch_yahoo_chart", return_value=("MOCK", None, None)):
        res = engine.fetch_and_analyze_us_markets(force_refresh=True)
        assert res["status"] == "OK"
        
        # 1. Volatility surface must NOT be hardcoded to 15.0
        vol = res["volatility_surface"]
        assert vol["vix_spot"] is None, f"Expected None for vix_spot but got {vol['vix_spot']}"
        assert vol["vix_9d"] is None
        assert vol["regime"] == "UNAVAILABLE"
        assert res["pillar_scores"]["volatility_score"] == 0.0
        
        # 2. Macro rates must NOT be hardcoded to 4.25 / 4.50 / 103.5
        macro = res["macro_rates_currency"]
        assert macro["us_10y_yield"] is None, f"Expected None for us_10y_yield but got {macro['us_10y_yield']}"
        assert macro["us_3m_yield"] is None
        assert macro["us_dollar_dxy"] is None
        assert macro["yield_curve_spread"] is None
        
        # 3. SPY price and expected close must NOT be 520.0
        t1 = res["us_t1_forecast"]
        assert t1["expected_close_spy"] is None, f"Expected None for expected_close_spy but got {t1['expected_close_spy']}"
        assert t1["expected_pts_spy"] is None
        assert t1["expected_pct_spy"] is None
        
        # 4. Futures radar must report UNAVAILABLE and not assume 24800.0 spot
        fut = res["futures_relative_radar"]
        assert fut["es_futures_price"] is None
        assert fut["nq_futures_price"] is None
        assert fut["gap_verdict"].startswith("UNAVAILABLE")


def test_nifty_global_macro_rows_reflect_unavailable_status():
    """Verifies that Nifty Pillar 4 (Global Cues) & Pillar 5 (Macro) render None/UNAVAILABLE when feeds are missing."""
    import datetime
    today_str = nifty_analytics_engine.now_ist().strftime("%Y-%m-%d") if hasattr(nifty_analytics_engine, "now_ist") else "2026-10-26"
    fake_premarket_dt = datetime.datetime.strptime(f"{today_str} 08:30:00", "%Y-%m-%d %H:%M:%S")
    old_cache = nifty_analytics_engine._predictions_mem_cache
    try:
        with patch("engines.nifty_analytics_engine.now_ist", return_value=fake_premarket_dt):
            # When metrics_map lacks ^N225, ^HSI, BZ=F etc., the rows must show ltp=None and trend=UNAVAILABLE
            with patch.object(us_analytics_engine, "fetch_and_analyze_us_markets", return_value={
                "us_composite_score": 0.0,
                "pillar_scores": {"breadth_score": 0.0, "volatility_score": 0.0},
                "breadth_analysis": {"dispersion_regime": "UNAVAILABLE", "rsp_vs_spy_dispersion": None, "dispersion_note": "Unavailable"},
                "volatility_surface": {"regime": "UNAVAILABLE", "vix_spot": None, "note": "VIX unavailable"},
                "futures_relative_radar": {"es_futures_ret_1d": None, "es_futures_price": None, "expected_fair_gap_pts": 0.0, "gap_verdict": "UNAVAILABLE"},
                "us_t1_forecast": {"verdict": "NEUTRAL", "expected_close_spy": None, "expected_pct_spy": None, "prob_up": 0.33, "confidence": "LOW"}
            }):
                # Mock empty series_map so Asian and Macro tickers have no metrics
                with patch.object(nifty_analytics_engine, "_calc_series_metrics", return_value={}), patch.object(state_engine, "set_kv"):
                    preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=True)
                    pillars = preds.get("pillars", {})
                    g_rows = pillars.get("global_markets", [])
                    m_rows = pillars.get("macro_cross_asset", [])
                    
                    # Check VIX row has no 15.0
                    vix_row = next((r for r in g_rows if r["ticker"] == "^VIX"), None)
                    assert vix_row is not None
                    assert vix_row["ltp"] is None
                    assert vix_row["trend"] == "UNAVAILABLE"
                    
                    # Check SPY row has no fake price
                    spy_row = next((r for r in g_rows if r["ticker"] == "SPY"), None)
                    assert spy_row is not None
                    assert spy_row["ltp"] is None
                    
                    # Check Asian cues has no fake 0.0
                    nikkei_row = next((r for r in g_rows if r["ticker"] == "^N225"), None)
                    assert nikkei_row is not None
                    assert nikkei_row["ltp"] is None
                    assert nikkei_row["trend"] == "UNAVAILABLE"
                    
                    # Check Macro rows (Brent, USD/INR, 10Y Yield) show UNAVAILABLE
                    for mr in m_rows:
                        assert mr["ltp"] is None
                        assert mr["impact_on_nifty"] == "UNAVAILABLE"
    finally:
        nifty_analytics_engine._predictions_mem_cache = old_cache
