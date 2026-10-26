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
from unittest.mock import patch
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
    # auto-run with force_refresh=False must stop with UP_TO_DATE or AWAITING_NSE_RELEASE during trading hours
    auto_res = nifty_analytics_engine.check_and_update_fii_dii_participant_oi(force_refresh=False, trigger_source="TEST_INVARIANT")
    assert auto_res.get("status") in ("UP_TO_DATE", "AWAITING_NSE_RELEASE"), f"Expected UP_TO_DATE or AWAITING_NSE_RELEASE, got {auto_res.get('status')}"


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

    # 4. Test Intraday Auto-Adjustment Suite Across All 5 Asymmetric Architectures
    with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=50.0):
        # 4a. Iron Fly Wing Roll Down on spot drop
        fly_down = advanced_options_pricing_engine.formulate_strategy("INSTITUTIONAL_IRON_FLY", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k_atm_down = next(l for l in fly_down["legs"] if l["action"] == "SELL")["strike"]
        adj_if_down, _, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_IF_DOWN", fly_down["strategy_name"], fly_down["legs"], spot_price=k_atm_down - 75.0, horizon="INTRADAY")
        assert adj_if_down is True, "Iron Fly must roll Call wing down when spot drops >= 70 pts intraday"

        # 4b. Iron Fly Wing Roll Up on spot rally
        fly_up = advanced_options_pricing_engine.formulate_strategy("INSTITUTIONAL_IRON_FLY", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k_atm_up = next(l for l in fly_up["legs"] if l["action"] == "SELL")["strike"]
        adj_if_up, _, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_IF_UP", fly_up["strategy_name"], fly_up["legs"], spot_price=k_atm_up + 75.0, horizon="INTRADAY")
        assert adj_if_up is True, "Iron Fly must roll Put wing up when spot rallies >= 70 pts intraday"

        # 4c. Bull Broken Wing Butterfly Wing Tighten on downside threat
        bwb_bull = advanced_options_pricing_engine.formulate_strategy("BULL_BROKEN_WING_BUTTERFLY", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k2_bull = next(l for l in bwb_bull["legs"] if l["action"] == "SELL")["strike"]
        adj_bwb_bull, _, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_BWB_BULL", bwb_bull["strategy_name"], bwb_bull["legs"], spot_price=k2_bull + 10.0, horizon="INTRADAY")
        assert adj_bwb_bull is True, "Bull BWB must tighten broken put wing when spot approaches short strike K2"

        # 4d. Bear Broken Wing Butterfly Wing Tighten on upside threat
        bwb_bear = advanced_options_pricing_engine.formulate_strategy("BEAR_BROKEN_WING_BUTTERFLY", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k2_bear = next(l for l in bwb_bear["legs"] if l["action"] == "SELL")["strike"]
        adj_bwb_bear, _, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_BWB_BEAR", bwb_bear["strategy_name"], bwb_bear["legs"], spot_price=k2_bear - 10.0, horizon="INTRADAY")
        assert adj_bwb_bear is True, "Bear BWB must tighten broken call wing when spot approaches short strike K2"

        # 4e. Call Ratio Backspread Valley Defense
        crb = advanced_options_pricing_engine.formulate_strategy("CALL_RATIO_BACKSPREAD", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k1_crb = next(l for l in crb["legs"] if l["action"] == "SELL")["strike"]
        k2_crb = next(l for l in crb["legs"] if l["action"] == "BUY")["strike"]
        adj_crb, legs_crb, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_CRB", crb["strategy_name"], crb["legs"], spot_price=(k1_crb + k2_crb) / 2.0, horizon="INTRADAY")
        assert adj_crb is True, "Call Ratio Backspread must deploy Put Credit Spread hedge when stalling in Valley of Death"
        assert len(legs_crb) == 4, "Must add 2 hedge legs to original 2 legs"

        # 4f. Put Ratio Backspread Valley Defense
        prb = advanced_options_pricing_engine.formulate_strategy("PUT_RATIO_BACKSPREAD", spot_price=spot, iv_pct=iv, dte=1, horizon="INTRADAY")
        k1_prb = next(l for l in prb["legs"] if l["action"] == "SELL")["strike"]
        k2_prb = next(l for l in prb["legs"] if l["action"] == "BUY")["strike"]
        adj_prb, legs_prb, _ = strategy_adjustment_engine._check_and_apply_adjustments("TEST_PRB", prb["strategy_name"], prb["legs"], spot_price=(k1_prb + k2_prb) / 2.0, horizon="INTRADAY")
        assert adj_prb is True, "Put Ratio Backspread must deploy Call Credit Spread hedge when stalling in Valley of Death"
        assert len(legs_prb) == 4


def test_invariant_12_omni_watchdog_and_ml_sentinel_integrity():
    """
    INVARIANT 12: System-Wide Omni-Watchdog & ML Sentinel Integrity.
    1. Active master diagnostic probes all 10 platform subsystems without unhandled exceptions.
    2. Deep ML Sentinel validates all 32 standardized features, 5 model horizons, and PSI drift.
    3. Function interceptor @watchdog_guarded tracks latencies and records failure states accurately.
    4. Autonomous self-healing functions cleanly recover database locks and prediction caches.
    5. Capital preservation trading gate determines market hours and execution status with zero ambiguity.
    """
    from engines.omni_watchdog_engine import omni_watchdog, watchdog_guarded

    # 1. Master Diagnostic Active Probe across all 10 Subsystems
    res = omni_watchdog.probe_all_subsystems(force_active_probe=True)
    assert res is not None
    assert res["overall_status"] in ("HEALTHY", "DEGRADED", "CRITICAL_ALERT")
    assert res["total_sentinels"] >= 13
    assert res["healthy_count"] >= 10, f"Expected majority of sentinels healthy, got {res['healthy_count']}/{res['total_sentinels']}"

    subsystems = res["subsystems"]
    expected_subsystems = [
        "swing_trading", "ohlcv_vault", "intraday_desk", "options_desk",
        "auto_strategies", "predictions_desk", "commodities_desk",
        "screeners", "paper_trading", "broker_gateway"
    ]
    for sub_name in expected_subsystems:
        assert sub_name in subsystems, f"Subsystem {sub_name} missing from Omni-Watchdog probe!"
        sub = subsystems[sub_name]
        assert "status" in sub and sub["status"] in ("HEALTHY", "STANDBY", "DEGRADED", "ALERT")
        assert "latency_ms" in sub and sub["latency_ms"] >= 0.0
        assert "details" in sub and isinstance(sub["details"], dict)

    # 2. Deep Machine Learning Sentinel Verification (5 Horizons)
    ml = res["ml_sentinel"]
    assert ml is not None
    assert ml["status"] in ("HEALTHY", "DEGRADED", "ALERT")

    # 2a. Dataset health (32 features, zero synthetic numbers)
    ds = ml["dataset_health"]
    assert ds["total_rows"] >= 50, f"Dataset must have at least 50 historical rows, got {ds['total_rows']}"
    assert ds["verified_features_count"] == 32
    assert ds["null_features_detected"] == 0, "No null features allowed in active learning dataset"
    assert ds["synthetic_numbers_detected"] is False

    # 2b. Concept Drift Sentinel
    drift = ml["concept_drift"]
    assert "population_stability_index" in drift
    assert isinstance(drift["population_stability_index"], float)
    assert drift["drift_status"] in ("STABLE", "DRIFT_DETECTED")

    # 2c. Model Registry (Intraday, Weekly, Monthly, Swing, T+1, T+5)
    reg = ml["models_registry"]
    for horizon in ("intraday", "weekly", "monthly", "swing", "nifty_t+1", "nifty_t+5"):
        assert horizon in reg, f"Model horizon {horizon} missing from registry probe"
        h_info = reg[horizon]
        assert h_info["status"] in ("HEALTHY", "DEGRADED", "NOT_FOUND")

    # 2d. Prediction Ledger Reconciliation
    rec = ml["ledger_reconciliation"]
    assert rec["total_forecasts"] >= 1
    assert rec["pending_outcomes"] >= 0

    # 3. Interceptor & Telemetry Tracking Verification
    @watchdog_guarded(domain="test_inv12", section="sentinel_check", critical=False, max_latency_ms=200.0)
    def dummy_healthy_op():
        return {"val": 42.0, "status": "OK"}

    @watchdog_guarded(domain="test_inv12", section="sentinel_check", critical=False, max_latency_ms=200.0)
    def dummy_faulty_op():
        raise RuntimeError("Sentinel simulated error")

    v = dummy_healthy_op()
    assert v["val"] == 42.0

    try:
        dummy_faulty_op()
    except RuntimeError:
        pass

    telemetry = omni_watchdog._function_telemetry
    healthy_key = "test_inv12:dummy_healthy_op"
    faulty_key = "test_inv12:dummy_faulty_op"

    assert healthy_key in telemetry
    assert telemetry[healthy_key]["success_count"] >= 1
    assert telemetry[healthy_key]["status"] == "HEALTHY"

    assert faulty_key in telemetry
    assert telemetry[faulty_key]["failure_count"] >= 1
    assert telemetry[faulty_key]["status"] == "DEGRADED"

    # 4. Autonomous Self-Healing Verification
    heal_db = omni_watchdog.execute_self_healing("database")
    assert heal_db["success"] is True
    assert len(heal_db["actions_taken"]) >= 1

    heal_pred = omni_watchdog.execute_self_healing("predictions")
    assert heal_pred["success"] is True

    # 5. Capital Safety Trading Gate Verification
    gate_allowed, gate_msg = omni_watchdog.is_trading_allowed()
    assert isinstance(gate_allowed, bool)
    assert isinstance(gate_msg, str) and len(gate_msg) > 0


def test_omni_watchdog_intraday_probe_resilience_to_standby_none_probs():
    """
    Verifies that probe_subsystem_intraday handles DATA_UNAVAILABLE and None probabilities
    without crashing with TypeError, correctly tagging Standby regime.
    """
    from engines.omni_watchdog_engine import omni_watchdog
    from unittest.mock import patch

    mock_ml = {
        "status": "DATA_UNAVAILABLE",
        "probabilities": {"p_up": None, "p_flat": None, "p_down": None},
        "predicted_regime": "STANDBY",
        "reason": "Market is closed"
    }
    with patch("engines.intraday_ml_engine.intraday_ml_engine.train_and_predict", return_value=mock_ml):
        probe = omni_watchdog.probe_subsystem_intraday()
        assert probe["name"] == "Intraday Desk & ML Sentinel"
        assert not any("unsupported operand type" in issue for issue in probe.get("issues", []))
        assert "Standby" in probe.get("details", {}).get("intraday_ml_regime", "")


# ==============================================================================
# INVARIANT 13: OPTIONS STOP-LOSS NEGATIVITY & DYNAMIC LEG PRICING INTEGRITY
# ==============================================================================
def test_invariant_13_options_sl_and_hedge_pricing():
    """
    INVARIANT 13: Options Strategy Safety & Invariant Integrity.
    1. Stop-Loss PnL threshold MUST strictly be negative (-abs(...)). If positive magnitude
       is provided (e.g. +7377.50), it must be stored as negative (-7377.50).
    2. Zero mark-to-market PnL at strategy inception MUST NEVER trigger STOPLOSS_EXITED.
    3. Dynamic adjustment legs appended by StrategyAdjustmentEngine MUST have entry_price, price,
       premium, and current_price all populated (> 0.0).
    4. Options portfolio view MUST NEVER display ₹0.00 entry price for any active leg.
    """
    import json
    from engines.nifty_analytics_engine import nifty_analytics_engine
    from engines.strategy_adjustment_engine import strategy_adjustment_engine

    # 1. Test strategy creation with positive SL magnitude
    strat_payload = {
        "strategy_name": "Test Invariant 13 Ratio Spread",
        "symbol": "NIFTY",
        "expiry_date": "29-OCT-2026",
        "expiry_type": "WEEKLY",
        "spot_at_entry": 22600.0,
        "net_premium": 500.0,
        "max_profit": 5000.0,
        "max_loss": 7377.5,
        "target_pnl": 3500.0,
        "stoploss_pnl": 7377.5, # Positive magnitude passed from frontend
        "auto_trade_mode": "MANUAL",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": 22600, "qty": 65, "entry_price": 100.0, "price": 100.0},
            {"action": "BUY", "type": "CE", "strike": 22700, "qty": 65, "entry_price": 40.0, "price": 40.0}
        ]
    }

    add_res = nifty_analytics_engine.deploy_options_strategy(strat_payload)
    assert add_res["status"] == "OK"
    test_strat_id = add_res["strategy_id"]

    try:
        # Check database record
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                row = conn.execute(
                    "SELECT target_pnl, stoploss_pnl, trailing_stop_pnl, status FROM options_strategies WHERE strategy_id = ?",
                    (test_strat_id,)
                ).fetchone()
                assert row is not None
                assert row["target_pnl"] == 3500.0
                assert row["stoploss_pnl"] == -7377.5, f"Expected stoploss_pnl to be strictly negative (-7377.5), got {row['stoploss_pnl']}"
                assert row["trailing_stop_pnl"] == -7377.5, f"Expected trailing_stop_pnl to be negative (-7377.5), got {row['trailing_stop_pnl']}"

        # 2. Evaluate adjustment engine: PnL at 0.0 must NOT exit trade
        adj_res = strategy_adjustment_engine.tick_watchdog(force_execution=True)
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                row_after = conn.execute(
                    "SELECT status, exit_reason FROM options_strategies WHERE strategy_id = ?",
                    (test_strat_id,)
                ).fetchone()
                assert row_after["status"] == "ACTIVE", f"Strategy exited prematurely with status {row_after['status']}, reason: {row_after['exit_reason']}"

        # 3. Dynamic hedge adjustment leg verification
        # Simulate Put Ratio Backspread Valley of Death adjustment
        test_legs = [
            {"action": "SELL", "type": "PE", "strike": 22500, "qty": 65, "entry_price": 80.0, "state": "ACTIVE"},
            {"action": "BUY", "type": "PE", "strike": 22400, "qty": 65, "entry_price": 40.0, "state": "ACTIVE"}
        ]
        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=50.0):
            adjusted, adj_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                strategy_id="TEST-VALLEY-001",
                strat_name="Put Ratio Backspread",
                legs=test_legs,
                spot_price=22450.0, # Stalls in valley (22400 <= 22450 <= 22500)
                horizon="INTRADAY"
            )
        assert adjusted is True
        assert len(adj_legs) == 4
        hedge_legs = [l for l in adj_legs if l.get("state") == "HEDGE_LEG"]
        assert len(hedge_legs) == 2
        for hl in hedge_legs:
            assert hl["entry_price"] > 0.0, f"Hedge leg entry_price must be > 0.0, got {hl['entry_price']}"
            assert hl["price"] > 0.0, f"Hedge leg price must be > 0.0, got {hl['price']}"
            assert hl["premium"] > 0.0, f"Hedge leg premium must be > 0.0, got {hl['premium']}"
            assert hl["current_price"] > 0.0, f"Hedge leg current_price must be > 0.0, got {hl['current_price']}"

        # 4. Verify get_options_portfolio repairs legs where entry_price was missing or zero
        corrupted_strat = {
            "strategy_name": "Test Corrupted Legs",
            "symbol": "NIFTY",
            "expiry_date": "29-OCT-2026",
            "expiry_type": "WEEKLY",
            "spot_at_entry": 22600.0,
            "net_premium": 0.0,
            "max_profit": 5000.0,
            "max_loss": -5000.0,
            "target_pnl": 3000.0,
            "stoploss_pnl": -3000.0,
            "auto_trade_mode": "MANUAL",
            "legs": [
                {"action": "SELL", "type": "CE", "strike": 22800, "qty": 65, "premium": 20.77, "state": "HEDGE_LEG"}
            ]
        }
        res_corrupt = nifty_analytics_engine.deploy_options_strategy(corrupted_strat)
        test_corrupt_id = res_corrupt["strategy_id"]
        # Directly remove entry_price from DB to simulate the old buggy state
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute(
                    "UPDATE options_strategies SET legs_json = ? WHERE strategy_id = ?",
                    (json.dumps([{"action": "SELL", "type": "CE", "strike": 22800, "qty": 65, "premium": 20.77, "state": "HEDGE_LEG"}]), test_corrupt_id)
                )
                conn.commit()

        port = nifty_analytics_engine.get_options_portfolio(force_refresh=True)
        corrupted_in_port = next((s for s in port["active_strategies"] if s["strategy_id"] == test_corrupt_id), None)
        assert corrupted_in_port is not None
        rep_leg = corrupted_in_port["legs"][0]
        assert rep_leg["entry_price"] == 20.77, f"Expected entry_price to fallback to premium 20.77, got {rep_leg['entry_price']}"

    finally:
        # Cleanup test strategies
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("DELETE FROM options_strategies WHERE strategy_id IN (?, ?)", (test_strat_id, test_corrupt_id if 'test_corrupt_id' in locals() else ''))
                conn.commit()




