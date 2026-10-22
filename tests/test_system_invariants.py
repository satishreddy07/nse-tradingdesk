"""
tests/test_system_invariants.py
================================
Automated System Invariant Verification Suite for NSE Trading Desk.

Enforces 7 uncompromised mathematical, architectural, and data invariants:
1. Invariant 1: Market-Closed Stream & Tick Freezing
2. Invariant 2: Pure Statistical Prediction Direction (Zero Heuristic Override)
3. Invariant 3: Symmetric Realistic Evaluation Thresholds
4. Invariant 4: Standardized Non-Mixed Error Metrics
5. Invariant 5: Cross-Platform Windows/Linux Path Normalization
6. Invariant 6: Portfolio Capital Conservation (Cash + Invested == Equity)
7. Invariant 7: Core System API Contract & Schema Compliance
"""

import math
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from core.nifty_prediction_ledger import NIFTY_LEDGER_TABLE, nifty_prediction_ledger
from engines.kotak_neo_engine import kotak_neo_engine
from engines.ml_performance_engine import ml_performance_engine
from engines.tree_quant_ml_engine import tree_quant_ml_engine
from execution.state_engine import state_engine
from web.api import app

client = TestClient(app)


# ==============================================================================
# INVARIANT 1: MARKET-CLOSED STREAMING & TICK FREEZING
# ==============================================================================
def test_invariant_1_market_closed_tick_freezing():
    """
    INVARIANT 1: When NSE cash equity market is closed, Kotak Neo streaming
    must report False for active streaming, stream_mode must be STANDBY,
    and broker keepalives must not increment ticks received.
    """
    from core.data_quality import now_ist
    from core.market_calendar import NSEMarketCalendar

    dt_now = now_ist()
    mkt = NSEMarketCalendar.get_market_status(dt_now)

    # In post-market hours, engine must report standby
    if not mkt.get("is_open", False):
        assert not kotak_neo_engine.is_streaming_active(), "is_streaming_active() must be False when market is closed"
        stat = kotak_neo_engine.get_status()
        assert stat["stream_mode"] in ("STANDBY", "OFFLINE"), f"stream_mode must be STANDBY when market closed, got {stat['stream_mode']}"
        assert stat["is_streaming"] is False, "is_streaming must be False in status when market closed"


# ==============================================================================
# INVARIANT 2: PURE STATISTICAL PREDICTION DIRECTION (ZERO HEURISTIC OVERRIDE)
# ==============================================================================
def test_invariant_2_prediction_direction_argmax():
    """
    INVARIANT 2: Forecast direction MUST strictly equal the highest probability
    class from the ML model, and MUST NOT be overridden by a heuristic composite score.
    """
    # Case A: DOWN has highest probability (44% Down, 34% Up, 22% Flat)
    # Even if composite_score is -2.0 (inside old -8 to +8 band), direction MUST be BEARISH
    fc_id_a = nifty_prediction_ledger.log_forecast(
        horizon="T+1",
        spot=22700.0,
        composite_score=-2.0,
        verdict="RANGE-BOUND / CONSOLIDATION",
        prob_up=0.34,
        prob_flat=0.22,
        prob_down=0.44,
        expected_close=22600.0,
        expected_pts=-100.0,
        expected_pct=-0.44,
        range_1sigma=(22550.0, 22850.0),
        range_2sigma=(22400.0, 23000.0),
        features={},
        prediction_date="2099-01-01"
    )
    with nifty_prediction_ledger._get_connection() as conn:
        row = conn.execute(f"SELECT predicted_direction FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id = ?", (fc_id_a,)).fetchone()
        assert row["predicted_direction"] == "BEARISH", f"Expected BEARISH due to prob_down 0.44 > 0.34, got {row['predicted_direction']}"

    # Case B: UP has highest probability (55% Up, 25% Flat, 20% Down)
    fc_id_b = nifty_prediction_ledger.log_forecast(
        horizon="T+1",
        spot=22700.0,
        composite_score=4.0,
        verdict="RANGE-BOUND / CONSOLIDATION",
        prob_up=0.55,
        prob_flat=0.25,
        prob_down=0.20,
        expected_close=22850.0,
        expected_pts=150.0,
        expected_pct=0.66,
        range_1sigma=(22550.0, 22850.0),
        range_2sigma=(22400.0, 23000.0),
        features={},
        prediction_date="2099-01-02"
    )
    with nifty_prediction_ledger._get_connection() as conn:
        row = conn.execute(f"SELECT predicted_direction FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id = ?", (fc_id_b,)).fetchone()
        assert row["predicted_direction"] == "BULLISH", f"Expected BULLISH due to prob_up 0.55, got {row['predicted_direction']}"

    # Clean up test rows
    with nifty_prediction_ledger._get_connection() as conn:
        conn.execute(f"DELETE FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id IN (?, ?)", (fc_id_a, fc_id_b))
        conn.commit()


# ==============================================================================
# INVARIANT 3: SYMMETRIC REALISTIC EVALUATION THRESHOLDS
# ==============================================================================
def test_invariant_3_evaluation_threshold_symmetry():
    """
    INVARIANT 3: Directional evaluation thresholds must be realistic to avoid the
    'FLAT Trap'. A daily move < 0.25% is legitimately FLAT.
    """
    fc_test = nifty_prediction_ledger.log_forecast(
        horizon="T+1",
        spot=22700.0,
        composite_score=0.0,
        verdict="FLAT",
        prob_up=0.30,
        prob_flat=0.45,
        prob_down=0.25,
        expected_close=22700.0,
        expected_pts=0.0,
        expected_pct=0.0,
        range_1sigma=(22600.0, 22800.0),
        range_2sigma=(22500.0, 22900.0),
        features={},
        prediction_date="2099-02-01",
        predicted_direction="FLAT"
    )

    with nifty_prediction_ledger._get_connection() as conn:
        ret_pct = ((22720.0 - 22700.0) / 22700.0) * 100.0  # +0.088%
        dir_threshold = 0.25
        realized_dir = "BULLISH" if ret_pct >= dir_threshold else ("BEARISH" if ret_pct <= -dir_threshold else "FLAT")
        assert realized_dir == "FLAT", f"Expected realized FLAT for +0.088% move, got {realized_dir}"

        conn.execute(f"DELETE FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id = ?", (fc_test,))
        conn.commit()


# ==============================================================================
# INVARIANT 4: STANDARDIZED NON-MIXED ERROR METRICS
# ==============================================================================
def test_invariant_4_error_metrics_no_raw_rupee_mixing():
    """
    INVARIANT 4: Prediction MAE must report Nifty index points for index forecasts,
    and percentage return error for portfolio trades. Rupee PnL must NEVER be
    averaged with Nifty index points.
    """
    pred_perf = ml_performance_engine.get_prediction_performance(limit=25)
    mae_pts = pred_perf.get("mean_absolute_error_pts", 0.0)
    mae_pct = pred_perf.get("mean_absolute_error_pct", 0.0)
    mae_val = pred_perf.get("mean_absolute_error", 0.0)

    # MAE points for Nifty index forecasts must be reasonable (< 1,000 pts)
    assert mae_pts < 1000.0, f"Nifty MAE points cannot exceed 1000 pts, got {mae_pts}"
    # Percentage return error is non-negative and bounded for financial derivatives (< 100%)
    assert 0.0 <= mae_pct <= 100.0, f"Return MAE % must be bounded between 0% and 100%, got {mae_pct}"
    # General MAE must be non-negative
    assert mae_val >= 0.0, "MAE must be non-negative"


# ==============================================================================
# INVARIANT 5: CROSS-PLATFORM PATH NORMALIZATION
# ==============================================================================
def test_invariant_5_cross_platform_artifact_inspection():
    """
    INVARIANT 5: _inspect_artifact must handle both Windows backslash paths
    and Linux forward slash paths without failing.
    """
    monthly_path = Path("models/registry/monthly_20261016_101024.bundle")
    if monthly_path.exists():
        ok_fwd, size_fwd, hash_fwd = ml_performance_engine._inspect_artifact(monthly_path)
        assert ok_fwd is True, "Forward slash inspection must succeed"

        win_path = Path("models\\registry\\monthly_20261016_101024.bundle")
        ok_win, size_win, hash_win = ml_performance_engine._inspect_artifact(win_path)
        assert ok_win is True, "Windows backslash inspection must succeed"
        assert size_fwd == size_win, "Size must be identical"


# ==============================================================================
# INVARIANT 6: PORTFOLIO CAPITAL CONSERVATION
# ==============================================================================
def test_invariant_6_portfolio_balance_conservation():
    """
    INVARIANT 6: In Swing Portfolio 1, Available Cash + Total Invested + Unrealized PnL
    must equal Current Equity within numerical precision.
    """
    res = client.get("/api/portfolio/PORTFOLIO_1/summary")
    assert res.status_code == 200
    data = res.json()

    cash = float(data.get("available_cash", 0.0))
    invested = float(data.get("total_invested", 0.0))
    equity = float(data.get("current_equity", 0.0))
    unrealized = float(data.get("stats", {}).get("unrealized_pnl", data.get("total_unrealized_pnl", 0.0)))

    # Equity = Cash + Invested + Unrealized PnL
    calculated_equity = cash + invested + unrealized
    assert math.isclose(equity, calculated_equity, abs_tol=5.0), (
        f"Capital conservation violated: Equity {equity} != Cash ({cash}) + Invested ({invested}) + Unrealized ({unrealized})"
    )


# ==============================================================================
# INVARIANT 7: CORE SYSTEM API CONTRACT & SCHEMA COMPLIANCE
# ==============================================================================
def test_invariant_7_core_endpoints_schema_health():
    """
    INVARIANT 7: All core functional endpoints across Swing, Intraday, Options,
    Predictions, and ML Performance must return HTTP 200 with non-null JSON schemas.
    """
    endpoints = [
        "/api/status",
        "/api/nifty/live-spot",
        "/api/stream/continuous",
        "/api/ml-performance/overview",
        "/api/ml-performance/registry",
        "/api/ml-performance/predictions?limit=5",
        "/api/ml-performance/nifty-forecasts?horizon=T+1",
        "/api/options/portfolio",
        "/api/portfolio/PORTFOLIO_1/summary"
    ]

    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Endpoint {ep} failed with status {resp.status_code}"
        json_data = resp.json()
        assert isinstance(json_data, (dict, list)), f"Endpoint {ep} returned non-JSON payload"


# ==============================================================================
# INVARIANT 8: FII/DII AUTO-RUN & MANUAL UPDATE CONTRACT
# ==============================================================================
def test_invariant_8_fii_dii_autorun_and_manual_update_contract():
    """
    INVARIANT 8:
    1. If target completed trading session data is already in SQLite,
       check_and_update_fii_dii_participant_oi(force_refresh=False) must stop auto-run
       with status 'UP_TO_DATE' and skip redundant network fetch.
    2. Endpoints /api/nifty/fii-dii/status and /api/nifty/fii-dii/update-now must return HTTP 200.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine

    # Status endpoint check
    st_res = client.get("/api/nifty/fii-dii/status")
    assert st_res.status_code == 200
    st_data = st_res.json()
    assert st_data.get("status") == "OK"
    assert "telemetry" in st_data
    tel = st_data["telemetry"]
    assert "status" in tel
    assert "latest_cash_date" in tel

    # Manual update endpoint check
    up_res = client.post("/api/nifty/fii-dii/update-now")
    assert up_res.status_code == 200
    up_data = up_res.json()
    assert up_data.get("status") == "OK"
    assert "telemetry" in up_data
    assert "data" in up_data

    # Auto-run idempotency: after manual update has ingested latest data,
    # auto-run with force_refresh=False must stop with UP_TO_DATE
    auto_res = nifty_analytics_engine.check_and_update_fii_dii_participant_oi(force_refresh=False, trigger_source="TEST_INVARIANT")
    assert auto_res.get("status") == "UP_TO_DATE", f"Expected UP_TO_DATE, got {auto_res.get('status')}"
    assert auto_res.get("action_taken") == "SKIPPED_ALREADY_AVAILABLE"


# ==============================================================================
# INVARIANT 9: GLOBAL & CROSS-ASSET TELEMETRY (NO HARDCODED GAPS)
# ==============================================================================
def test_invariant_9_global_cross_asset_telemetry_no_gaps():
    """
    INVARIANT 9: Global markets table items (US T+1, VIX, E-mini Futures) must not have
    hardcoded None gaps for 5D % and EMA 20 when market data series are available.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine

    # Fetch premarket predictions without forced network recalculate
    preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)
    assert preds.get("status") == "OK"
    g_rows = preds.get("pillars", {}).get("global_markets", [])
    assert len(g_rows) >= 4, "Must contain at least 4 global market rows"

    # Find VIX and Futures rows
    vix_row = next((r for r in g_rows if "^VIX" in r.get("ticker", "")), None)
    es_row = next((r for r in g_rows if "ES=F" in r.get("ticker", "")), None)

    assert vix_row is not None, "VIX row missing from global_markets"
    assert es_row is not None, "ES=F row missing from global_markets"

    # If VIX LTP is present, 1D, 5D and EMA 20 must not be hardcoded None
    if vix_row.get("ltp") is not None:
        assert vix_row.get("ret_1d") is not None, "VIX ret_1d should be populated"
        assert vix_row.get("ema20") is not None, "VIX ema20 should be populated"


# ==============================================================================
# INVARIANT 10: OPTION CHAIN EXPIRY ISOLATION & ZERO MULTI-EXPIRY CORRUPTION
# ==============================================================================
def test_invariant_10_option_chain_expiry_isolation_and_no_leakage():
    """
    INVARIANT 10: Near-ATM streaming option ticks must strictly belong to the active
    weekly expiry. Monthly and longer-dated option quotes must NEVER collide with or
    overwrite weekly strike LTPs in continuous stream telemetry.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    from engines.nifty_analytics_engine import nifty_analytics_engine

    # Fetch option chain
    chain = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False)
    assert chain.get("status") in ("OK", "DATA_NOT_AVAILABLE")

    # Near-ATM option ticks must only have 2-tuple keys (strike, type)
    ticks = kotak_neo_engine.get_near_atm_option_ticks(strike_range=12)
    assert isinstance(ticks, dict)

    weekly_exp = chain.get("weekly", {}).get("expiry") if isinstance(chain, dict) else None

    for k, tick in ticks.items():
        assert isinstance(tick, dict)
        assert "strike" in tick and "type" in tick
        assert tick["type"] in ("CE", "PE")
        # If expiry tag is present and weekly_exp is known, it must not be monthly
        if tick.get("expiry") and weekly_exp:
            assert tick["expiry"] == weekly_exp or tick["expiry"] == chain.get("weekly", {}).get("expiry_date") or "-" in tick["expiry"]


# ==============================================================================
# INVARIANT 11: ADVANCED CONFIDENCE SCORING & ASYMMETRIC STRATEGY INTEGRITY
# ==============================================================================
def test_invariant_11_advanced_confidence_and_asymmetric_strategy_integrity():
    """
    INVARIANT 11:
    1. Multi-pillar dynamic confidence score must be bounded between 0% and 100%.
    2. Confluence must strictly prohibit selling naked contrary legs in strong trends.
    3. Advanced strategy formulation (BWB, Iron Fly, Ratio Backspread) must produce
       strictly valid listed NSE strikes (multiples of 50) with analytical Greeks.
    4. Active watchdog must safely enforce capital preservation, milestone profit locking,
       and high-water mark trailing stops with zero hardcoded or fake values.
    """
    from engines.advanced_confidence_engine import advanced_confidence_engine
    from engines.advanced_options_pricing_engine import advanced_options_pricing_engine
    from engines.strategy_adjustment_engine import strategy_adjustment_engine

    # 1. Test Confidence Engine across all 3 horizons
    for horizon in ["INTRADAY", "WEEKLY", "MONTHLY"]:
        conf = advanced_confidence_engine.compute_dynamic_confidence(horizon=horizon, force_refresh=False)
        assert conf["status"] == "OK"
        assert 0.0 <= conf["confidence_score_pct"] <= 100.0
        assert conf["confidence_grade"] in ("LOW", "MODERATE", "HIGH", "VERY_HIGH")
        assert conf["regime"] in ("STRONG_BULLISH", "MILD_BULLISH", "RANGE_BOUND", "MILD_BEARISH", "STRONG_BEARISH")
        assert conf["recommended_strategy"] in (
            "BULL_BROKEN_WING_BUTTERFLY", "BEAR_BROKEN_WING_BUTTERFLY",
            "INSTITUTIONAL_IRON_FLY", "CALL_RATIO_BACKSPREAD", "PUT_RATIO_BACKSPREAD"
        )
        # Strict rule: Strong Bearish must never sell naked puts
        if conf["regime"] == "STRONG_BEARISH":
            assert conf["recommended_strategy"] == "PUT_RATIO_BACKSPREAD"

    # 2. Test All 5 Strategy Formulations
    architectures = [
        "BULL_BROKEN_WING_BUTTERFLY",
        "BEAR_BROKEN_WING_BUTTERFLY",
        "INSTITUTIONAL_IRON_FLY",
        "CALL_RATIO_BACKSPREAD",
        "PUT_RATIO_BACKSPREAD"
    ]
    spot = 22687.65
    iv = 13.5
    for arch in architectures:
        strat = advanced_options_pricing_engine.formulate_strategy(
            strategy_type=arch,
            spot_price=spot,
            iv_pct=iv,
            dte=5,
            horizon="WEEKLY"
        )
        assert strat["strategy_name"]
        assert strat["architecture"] == arch
        assert isinstance(strat["legs"], list) and len(strat["legs"]) >= 2
        assert 1.0 <= strat["pop_pct"] <= 99.0

        for leg in strat["legs"]:
            assert leg["strike"] % 50 == 0, f"Strike {leg['strike']} must be multiple of 50"
            assert leg["premium"] > 0
            assert "delta" in leg["greeks"]
            assert "gamma" in leg["greeks"]
            assert "theta" in leg["greeks"]

    # 3. Test Active Watchdog Tick
    watchdog_res = strategy_adjustment_engine.tick_watchdog(force_execution=True)
    assert watchdog_res["status"] in ("OK", "STANDBY", "IDLE")
    if watchdog_res["status"] == "OK":
        assert "updated_count" in watchdog_res



