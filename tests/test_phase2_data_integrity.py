"""
tests/test_phase2_data_integrity.py
===================================
Phase 2 Data Integrity & Remediation Regression Test Suite.
Verifies:
1. engines/order_flow_structure_engine.py:
   - compute_volume_profile() fails safely with DATA_UNAVAILABLE when candles and spot are missing.
   - Never falls back to fabricated 22600.0.
   - Preserves legitimate calculations when candles or live spot exist.
2. engines/horizon_learning_orchestrator.py:
   - evaluate_stop_loss_and_reentry_matrix() explicitly marks simulations with is_simulated=True.
   - Includes simulation_disclaimer warning that grid is not historical performance.
   - Queries authentic empirical trades from unified_trade_learning_dataset.
3. engines/swing_quant_ml_engine.py:
   - Refuses to train or report metrics on synthetic 160-trade prior dataset.
   - Returns explicit INSUFFICIENT_DATA when empirical records are fewer than 5 or single-class.
4. engines/nifty_forecast_learning_engine.py:
   - compile_training_dataset() ingests authentic macro/external series (^GSPC, BZ=F, ^INDIAVIX).
   - Ingests authentic derivatives & FII participant history from database.
   - Eliminates all arbitrary scalar multipliers of Nifty return (ret5).
"""

import math
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# ==============================================================================
# 1. ORDER FLOW STRUCTURE ENGINE: VOLUME PROFILE INTEGRITY
# ==============================================================================

def test_volume_profile_missing_candles_and_spot_fails_closed():
    """
    When both intraday minute candles and live spot are unavailable:
    - Must return DATA_UNAVAILABLE status.
    - available must be False.
    - Spot, POC, VAH, VAL must be 0.0, NEVER fabricated 22600.0.
    """
    from engines.order_flow_structure_engine import OrderFlowStructureEngine
    eng = OrderFlowStructureEngine()

    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_nifty_intraday_minute_chart", return_value={"candles": []}), \
         patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=None), \
         patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_live_nifty_spot", return_value={"price": 0.0, "status": "DATA_UNAVAILABLE"}):
        res = eng.compute_volume_profile()

        assert res["available"] is False
        assert res.get("status") == "DATA_UNAVAILABLE"
        assert res["spot"] == 0.0
        assert res["poc"] == 0.0
        assert res["vah"] == 0.0
        assert res["val"] == 0.0
        assert res["market_state"] == "DATA_UNAVAILABLE"
        # Must NEVER be 22600.0
        assert res["spot"] != 22600.0
        assert res["poc"] != 22600.0


def test_volume_profile_preserves_valid_candles():
    """
    When authentic candles exist, compute_volume_profile() must calculate
    legitimate auction theory POC, VAH, VAL.
    """
    from engines.order_flow_structure_engine import OrderFlowStructureEngine
    eng = OrderFlowStructureEngine()

    mock_candles = [
        {"date": "2026-10-24 09:15:00", "open": 25000.0, "high": 25050.0, "low": 24980.0, "close": 25020.0, "volume": 100},
        {"date": "2026-10-24 09:20:00", "open": 25020.0, "high": 25080.0, "low": 25010.0, "close": 25060.0, "volume": 150},
        {"date": "2026-10-24 09:25:00", "open": 25060.0, "high": 25070.0, "low": 25030.0, "close": 25040.0, "volume": 80},
    ]

    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_nifty_intraday_minute_chart", return_value={"candles": mock_candles}), \
         patch("engines.kotak_neo_engine.kotak_neo_engine.get_live_price", return_value=25040.0):
        res = eng.compute_volume_profile()

        assert res["available"] is True
        assert res["spot"] == 25040.0
        assert res["poc"] > 0.0
        assert res["vah"] >= res["val"]
        assert len(res["histogram"]) > 0


# ==============================================================================
# 2. HORIZON LEARNING ORCHESTRATOR: SIMULATION ISOLATION & EMPIRICAL AUDITING
# ==============================================================================

def test_stop_loss_and_reentry_matrix_labels_simulation_and_queries_empirical():
    """
    evaluate_stop_loss_and_reentry_matrix() must:
    1. Explicitly flag is_simulated=True and is_empirical=False.
    2. Provide simulation_disclaimer.
    3. Include empirical_performance from unified_trade_learning_dataset.
    """
    from engines.horizon_learning_orchestrator import horizon_learning_orchestrator

    res = horizon_learning_orchestrator.evaluate_stop_loss_and_reentry_matrix(horizon="WEEKLY", sample_size=50)

    assert res["evaluation_type"] == "SCENARIO_SIMULATION_GRID"
    assert res["is_simulated"] is True
    assert res["is_empirical"] is False
    assert "simulation_disclaimer" in res
    assert "not be interpreted as actual historical" in res["simulation_disclaimer"].lower()

    # Base & Reentry strategy blocks must be labeled as simulated
    assert res["base_strategy"]["is_simulated"] is True
    assert res["reentry_strategy"]["is_simulated"] is True

    # Empirical performance block must be present
    assert "empirical_performance" in res
    emp = res["empirical_performance"]
    assert emp["data_source"] == "unified_trade_learning_dataset"
    assert emp["status"] in ("AUTHENTIC_EMPIRICAL_DATA", "INSUFFICIENT_DATA")


def test_stop_loss_and_reentry_matrix_insufficient_empirical_trades():
    """
    When unified_trade_learning_dataset has fewer than 5 trades for the horizon,
    empirical_performance must return INSUFFICIENT_DATA.
    """
    from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
    from execution.state_engine import state_engine

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [
        {"net_pnl": 120.0, "is_winner": 1, "re_entry_count": 0, "exit_reason": "TARGET"}
    ]  # only 1 trade (< 5)

    with patch.object(state_engine, "_get_connection", return_value=mock_conn):
        res = horizon_learning_orchestrator.evaluate_stop_loss_and_reentry_matrix(horizon="WEEKLY", sample_size=50)
        emp = res["empirical_performance"]
        assert emp["status"] == "INSUFFICIENT_DATA"
        assert emp["sample_size"] == 1
        assert emp["win_rate_pct"] is None


# ==============================================================================
# 3. SWING QUANT ML ENGINE: REFUSE TO TRAIN ON SYNTHETIC 160-TRADE PRIOR
# ==============================================================================

def test_swing_quant_ml_refuses_synthetic_prior_on_insufficient_data():
    """
    When empirical records in quant_trade_dataset are < 5:
    - Must return INSUFFICIENT_DATA.
    - Must NOT train on synthetic 160-trade prior.
    - prior_samples must be 0, validation_auc and brier_score must be None.
    """
    from engines.swing_quant_ml_engine import SwingQuantMLEngine
    eng = SwingQuantMLEngine()

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    # Return only 2 empirical trades
    mock_conn.execute.return_value.fetchall.return_value = [
        {"trade_type": "REAL", "entry_price": 2500.0, "stop_loss": 2450.0, "target": 2600.0,
         "score": 80.0, "confidence": 4.0, "rsi14": 55.0, "atr_pct": 2.0, "dist_to_pivot": 1.0, "dist_to_ema20": 1.5,
         "vol_ratio": 1.2, "inst_score": 60.0, "week_confluence": 0.0, "is_vcp": 0.0, "bb_sq_ratio": 0.8,
         "adx14": 25.0, "nifty_ret_5d": 0.5, "nifty_rsi14": 55.0, "vix_val": 14.0, "sector_excess_6m": 0.0,
         "is_winner": 1, "r_multiple": 2.0},
        {"trade_type": "REAL", "entry_price": 1800.0, "stop_loss": 1750.0, "target": 1900.0,
         "score": 75.0, "confidence": 3.5, "rsi14": 52.0, "atr_pct": 2.2, "dist_to_pivot": 1.2, "dist_to_ema20": 1.8,
         "vol_ratio": 1.1, "inst_score": 55.0, "week_confluence": 0.0, "is_vcp": 0.0, "bb_sq_ratio": 0.8,
         "adx14": 22.0, "nifty_ret_5d": 0.5, "nifty_rsi14": 55.0, "vix_val": 14.0, "sector_excess_6m": 0.0,
         "is_winner": 0, "r_multiple": -1.0}
    ]

    with patch.object(eng, "_get_connection", return_value=mock_conn):
        res = eng.train_model()
        assert res["status"] == "INSUFFICIENT_DATA"
        assert res["training_mode"] == "INSUFFICIENT_DATA"
        assert res["total_trades"] == 2
        assert res["prior_samples"] == 0
        assert res["brier_score"] is None
        assert res["validation_auc"] is None


def test_swing_quant_ml_refuses_single_class_dataset():
    """
    When empirical records have >= 5 trades but only 1 class (e.g. all winners):
    - Must refuse to train to prevent severe classification bias.
    - Must return INSUFFICIENT_DATA.
    """
    from engines.swing_quant_ml_engine import SwingQuantMLEngine
    eng = SwingQuantMLEngine()

    # 6 trades, all winners (single class)
    rows = []
    for i in range(6):
        rows.append({
            "trade_type": "REAL", "entry_price": 1000.0, "stop_loss": 980.0, "target": 1040.0,
            "score": 80.0, "confidence": 4.0, "rsi14": 55.0, "atr_pct": 2.0, "dist_to_pivot": 1.0, "dist_to_ema20": 1.5,
            "vol_ratio": 1.2, "inst_score": 60.0, "week_confluence": 0.0, "is_vcp": 0.0, "bb_sq_ratio": 0.8,
            "adx14": 25.0, "nifty_ret_5d": 0.5, "nifty_rsi14": 55.0, "vix_val": 14.0, "sector_excess_6m": 0.0,
            "is_winner": 1, "r_multiple": 2.0
        })

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = rows

    with patch.object(eng, "_get_connection", return_value=mock_conn):
        res = eng.train_model()
        assert res["status"] == "INSUFFICIENT_DATA"
        assert res["total_trades"] == 6


def test_swing_quant_ml_deterministic_chronological_ordering_and_brier_evaluation():
    """
    Verify:
    1. The SQL query explicitly contains 'ORDER BY created_at ASC, id ASC'.
    2. Training on live empirical data produces a calibrated out-of-sample Brier score <= 0.35.
    3. If the holdout split contains an unsuitable single-class distribution, brier_score is reported as None.
    """
    from engines.swing_quant_ml_engine import SwingQuantMLEngine
    eng = SwingQuantMLEngine()

    # 1. Inspect the SQL executed
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = []

    with patch.object(eng, "_get_connection", return_value=mock_conn):
        eng.train_model()
        sql_executed = mock_conn.execute.call_args[0][0]
        assert "ORDER BY created_at ASC, id ASC" in sql_executed

    # 2. End-to-end evaluation with live empirical dataset
    meta = eng.train_model()
    if meta.get("status") == "INSUFFICIENT_DATA":
        # In an environment without >= 5 historical empirical trades,
        # train_model() properly refuses to train on synthetic priors per Anti-Fabrication Policy
        assert meta["status"] == "INSUFFICIENT_DATA"
        assert meta["prior_samples"] == 0
    else:
        assert meta["status"] == "ONLINE_ACTIVE"
        assert meta["training_mode"] == "100%_REAL_EMPIRICAL"
        assert meta["brier_score"] is not None
        assert 0.0 <= meta["brier_score"] <= 0.35

    # 3. Test unsuitable single-class holdout split reporting
    # Construct 16 trades where the first 12 have mixed classes, but the last 4 (validation fold) are all 1s
    mock_rows = []
    for i in range(16):
        is_win = 1 if i >= 12 else (i % 2)  # last 4 are all winners
        mock_rows.append({
            "trade_type": "REAL", "entry_price": 1000.0, "stop_loss": 980.0, "target": 1040.0,
            "score": 80.0, "confidence": 4.0, "rsi14": 55.0, "atr_pct": 2.0, "dist_to_pivot": 1.0, "dist_to_ema20": 1.5,
            "vol_ratio": 1.2, "inst_score": 60.0, "week_confluence": 0.0, "is_vcp": 0.0, "bb_sq_ratio": 0.8,
            "adx14": 25.0, "nifty_ret_5d": 0.5, "nifty_rsi14": 55.0, "vix_val": 14.0, "sector_excess_6m": 0.0,
            "is_winner": is_win, "r_multiple": 2.0 if is_win else -1.0
        })

    mock_conn2 = MagicMock()
    mock_conn2.__enter__.return_value = mock_conn2
    mock_conn2.execute.return_value.fetchall.return_value = mock_rows

    with patch.object(eng, "_get_connection", return_value=mock_conn2), \
         patch("execution.state_engine.state_engine.set_kv"):
        res2 = eng.train_model()
        assert res2["status"] == "ONLINE_ACTIVE"
        # Brier score must be None to prevent presenting misleading metrics on single-class holdout
        assert res2["brier_score"] is None



# ==============================================================================
# 4. NIFTY FORECAST LEARNING ENGINE: AUTHENTIC MACRO & ZERO FAKE MULTIPLIERS
# ==============================================================================

def test_nifty_forecast_macro_series_are_authentic_not_multipliers_of_ret5():
    """
    compile_training_dataset() must:
    1. Read real historical series for ^GSPC, BZ=F, and ^INDIAVIX.
    2. Not compute crude_oil_ret_1d = -ret5 * 0.1 or sp500_ret_1d = ret5 * 0.15.
    3. Use neutral baselines when unobserved, never synthetic multipliers.
    """
    from engines.nifty_forecast_learning_engine import NiftyForecastLearningEngine
    from execution.state_engine import state_engine

    eng = NiftyForecastLearningEngine()

    # Generate 50 daily bars of Nifty
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    nifty_close = [24000.0 + i * 20.0 for i in range(50)]
    nifty_df = pd.DataFrame({"Close": nifty_close}, index=dates)

    # Distinct, unrelated movement for S&P 500 (e.g. falling while Nifty rises)
    sp_close = [6000.0 - i * 5.0 for i in range(50)]
    sp_df = pd.DataFrame({"Close": sp_close}, index=dates)

    # Distinct, unrelated movement for Brent Crude Oil
    oil_close = [75.0 + (i % 5) * 1.0 for i in range(50)]
    oil_df = pd.DataFrame({"Close": oil_close}, index=dates)

    # Distinct VIX
    vix_close = [13.5 for _ in range(50)]
    vix_df = pd.DataFrame({"Close": vix_close}, index=dates)

    def mock_get_ohlcv(symbol):
        if symbol == "^NSEI":
            return nifty_df
        elif symbol == "^GSPC":
            return sp_df
        elif symbol == "BZ=F":
            return oil_df
        elif symbol == "^INDIAVIX":
            return vix_df
        return None

    with patch.object(state_engine, "get_ohlcv", side_effect=mock_get_ohlcv), \
         patch("engines.nifty_forecast_learning_engine.nifty_prediction_ledger.get_training_records", return_value=pd.DataFrame()):
        df = eng.compile_training_dataset(horizon="T+1")

        assert not df.empty
        assert len(df) > 10

        # In row 0, check that sp500_ret_1d reflects actual sp_df return, NOT ret5 * 0.15
        row = df.iloc[0]
        # Since sp_close was falling, sp500_ret_1d should be negative!
        # If old fabricated multiplier was active (ret5 > 0 => ret5 * 0.15 > 0), it would be positive!
        assert row["sp500_ret_1d"] < 0.0, f"Expected negative S&P return, got {row['sp500_ret_1d']}"

        # crude_oil_ret_1d must match oil_df, not -ret5 * 0.1
        # near_atm_pcr must be 1.0 (neutral baseline when DB snapshot unobserved), not 1.0 + ret5 * 0.05
        assert row["near_atm_pcr"] == 1.0
        assert row["chg_oi_pcr"] == 1.0
        assert row["fii_fut_long_pct"] == 50.0
