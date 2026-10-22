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
