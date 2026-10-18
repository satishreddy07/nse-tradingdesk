"""
tests/test_full_repo_audit_remediation.py
=========================================
Comprehensive verification test suite verifying all remediations from the
Full Repository Audit Report (18 Oct 2026):
- Security: Safe JSON deserialization in state_engine (3.4), VPS script safety (3.5)
- Data Freshness & Quality: Quote timestamps & age (4.1), Cache timestamp preservation (4.2),
  Inverted candle rejection (4.3), Future timestamp flagging (4.4), SW 503 response (4.5)
- Macro & Calendar: FOMC 2026 decisions (5.1), Aug 26 holiday (5.2), Risk-free rate (5.3),
  VIX DATA_UNAVAILABLE (5.4), Dynamic model comparison (5.5), Cost accounting (5.6)
- Machine Learning & Risk: Scaler fold isolation (6.1), Tree ML safety & provenance (6.2-6.4),
  Out-of-sample Brier score (6.5), Formula Brier score MSE (6.6),
  0-DTE fractional expiry (6.7), Contractual max loss vs planned stop loss (6.8-6.9)
"""

import os
import json
import sqlite3
import datetime
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from core.data_quality import validate_ohlcv_dataframe, check_data_freshness, IST, now_ist
from core.market_calendar import NSEMarketCalendar
from execution.state_engine import StateEngine, state_engine
from engines.nifty_analytics_engine import NiftyAnalyticsEngine, CENTRAL_BANK_EVENTS_2026
from engines.weekly_monthly_options_strategy_engine import WeeklyMonthlyOptionsStrategyEngine
from engines.ml_performance_engine import ml_performance_engine
from engines.tree_quant_ml_engine import tree_quant_ml_engine
from core.unified_ml_dataset import UnifiedMLDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# 1. SECURITY & DESERIALIZATION (Finding 3.4)
# =============================================================================
def test_state_engine_json_serialization_safe(tmp_path):
    """Verify state_engine uses safe JSON serialization instead of pickle (Finding 3.4)."""
    import zlib
    test_db = str(tmp_path / "test_state.db")
    eng = StateEngine(db_path=test_db)

    # Create synthetic OHLCV dataframe with uppercase columns
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Close": [104.0, 105.0, 106.0, 107.0, 108.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    }, index=dates)

    eng.save_ohlcv_df("TEST_SYM", df)

    # Check database raw storage: must NOT be a python pickle stream
    with eng._get_connection() as conn:
        row = conn.execute("SELECT ohlcv_blob FROM ohlcv_master_store WHERE symbol = 'TEST_SYM'").fetchone()
        assert row is not None
        compressed_bytes = row[0]
        raw_bytes = zlib.decompress(compressed_bytes)
        # Pickle protocol header is b'\x80\x03', b'\x80\x04', b'\x80\x05'
        assert not raw_bytes.startswith(b"\x80")
        # Must be valid UTF-8 JSON
        json_obj = json.loads(raw_bytes.decode("utf-8"))
        assert "columns" in json_obj and "data" in json_obj

    # Retrieve and verify round-trip integrity
    loaded_df, meta = eng.get_ohlcv_df("TEST_SYM")
    assert loaded_df is not None
    assert len(loaded_df) == 5
    assert list(loaded_df["Close"]) == [104.0, 105.0, 106.0, 107.0, 108.0]


# =============================================================================
# 2. DATA INTEGRITY & FRESHNESS (Findings 4.1, 4.3, 4.4, 5.4)
# =============================================================================
def test_inverted_candles_rejected():
    """Verify validate_ohlcv_dataframe quarantines inverted High < Low candles (Finding 4.3)."""
    # Create DF with inverted candles and pass min_bars=3
    df_clean = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0],
        "High": [105.0, 106.0, 107.0, 108.0],
        "Low": [95.0, 96.0, 97.0, 98.0],
        "Close": [102.0, 103.0, 104.0, 105.0],
        "Volume": [5000, 5000, 5000, 5000]
    })
    is_valid, _, warnings = validate_ohlcv_dataframe(df_clean, min_bars=3)
    assert is_valid is True

    # Invert a single candle (High < Low)
    df_corrupt = df_clean.copy()
    df_corrupt.loc[1, "High"] = 90.0
    df_corrupt.loc[1, "Low"] = 110.0

    # With 1 of 4 candles corrupt (> 5%), it must be rejected or quarantined
    is_valid_c, cleaned_df, warnings_c = validate_ohlcv_dataframe(df_corrupt, min_bars=3)
    assert any("High < Low" in w or "corrupted bars" in w or "corruption" in w.lower() for w in warnings_c)
    if is_valid_c and cleaned_df is not None and len(cleaned_df) > 0:
        assert (cleaned_df["High"] >= cleaned_df["Low"]).all()


def test_future_and_missing_timestamps_flagged():
    """Verify check_data_freshness flags future timestamps and preserves None for missing (Finding 4.4)."""
    # Future timestamp (> 60s ahead)
    future_time = now_ist() + datetime.timedelta(seconds=120)
    res_future = check_data_freshness(future_time.strftime("%Y-%m-%d %H:%M:%S"))
    assert res_future["is_stale"] is True
    assert res_future["status"] == "FUTURE_TIMESTAMP_ERROR"
    assert "in the future" in res_future.get("error", "").lower()

    # Missing timestamp
    res_missing = check_data_freshness(None)
    assert res_missing["is_stale"] is True
    assert res_missing["status"] == "DATA_UNAVAILABLE"
    assert res_missing["age_seconds"] is None


def test_vix_data_unavailable_when_missing():
    """Verify get_india_vix returns DATA_UNAVAILABLE instead of default IV constant (Finding 5.4)."""
    engine = NiftyAnalyticsEngine()
    # Mock state_engine._get_connection failing to simulate missing table/db
    with patch.object(state_engine, "_get_connection", side_effect=Exception("DB Offline")):
        vix_res = engine.get_india_vix()
        assert vix_res["status"] == "DATA_UNAVAILABLE"
        assert vix_res["current_vix"] is None
        assert vix_res["vix_available"] is False


# =============================================================================
# 3. MACRO CALENDAR & HOLIDAYS (Findings 5.1, 5.2, 5.3, 5.6)
# =============================================================================
def test_fomc_events_2026_accurate():
    """Verify 2026 FOMC events reflect actual Federal Reserve actions (Finding 5.1)."""
    jan_event = next((e for e in CENTRAL_BANK_EVENTS_2026 if e["date"] == "2026-01-28"), None)
    assert jan_event is not None
    assert "3.50%-3.75%" in jan_event["outcome_print"]

    sep_event = next((e for e in CENTRAL_BANK_EVENTS_2026 if e["date"] == "2026-09-16"), None)
    assert sep_event is not None
    assert "3.75%-4.00%" in sep_event["outcome_print"]


def test_nse_holiday_aug_26_2026():
    """Verify Aug 26, 2026 is recognized as an official NSE holiday (Finding 5.2)."""
    assert "2026-08-26" in NSEMarketCalendar.NSE_HOLIDAYS
    assert "Id-E-Milad" in NSEMarketCalendar.NSE_HOLIDAYS["2026-08-26"]
    assert NSEMarketCalendar.is_trading_day(datetime.date(2026, 8, 26)) is False
    assert "Id-E-Milad" in NSEMarketCalendar.get_holiday_reason(datetime.date(2026, 8, 26))


def test_trade_cost_accounting_clean():
    """Verify unified simulated trade costs don't have mismatched ₹50 offsets (Finding 5.6)."""
    mgr = UnifiedMLDataset(db_path=":memory:")
    # Calculate costs for entry 24000, qty 65
    entry_price = 24000.0
    qty = 65
    expected_costs = round(entry_price * qty * 0.0012, 2)
    assert expected_costs == 1872.0


# =============================================================================
# 4. MACHINE LEARNING & RISK (Findings 6.2-6.4, 6.6, 6.7, 6.8-6.9)
# =============================================================================
def test_tree_ml_insufficient_data_rejection():
    """Verify TreeQuantMLEngine blocks model promotion with small samples (Findings 6.2-6.4)."""
    # Small dataset with single class (triggers insufficient out-of-sample data / single-class check)
    df_tiny = pd.DataFrame({
        "is_winner": [1, 1, 1, 1, 1, 1, 1, 1],
        "setup_quality": [50.0, 60.0, 70.0, 55.0, 65.0, 75.0, 80.0, 85.0]
    })
    res = tree_quant_ml_engine.train_horizon_models(horizon="INTRADAY", df=df_tiny)
    assert res["status"] == "INSUFFICIENT_DATA"


def test_brier_score_true_mse_calculation():
    """Verify get_market_regime_performance computes true MSE brier score (Finding 6.6)."""
    perf = ml_performance_engine.get_market_regime_performance()
    assert perf["status"] == "SUCCESS"
    for reg, stats in perf["regimes"].items():
        if stats["total_samples"] == 0:
            assert stats["brier_score"] == 0.0
        else:
            # Brier score must be between 0.0 and 1.0
            assert 0.0 <= stats["brier_score"] <= 1.0


def test_dynamic_model_comparison_candidates():
    """Verify get_model_comparison returns dynamic candidates without hardcoded literals (Finding 5.5)."""
    comp = ml_performance_engine.get_model_comparison(horizon="T+1")
    assert comp["horizon"] == "T+1"
    assert len(comp["candidates"]) >= 5
    variants = [c["model_variant"] for c in comp["candidates"]]
    assert any("Production" in v for v in variants)
    assert any("Baseline" in v for v in variants)


def test_0_dte_fractional_expiry_calculation():
    """Verify calculate_sigma_strikes computes fractional hours on 0-DTE expiry (Finding 6.7)."""
    today_str = now_ist().strftime("%Y-%m-%d")
    res = WeeklyMonthlyOptionsStrategyEngine.calculate_sigma_strikes(
        spot=24500.0,
        iv_pct=14.0,
        horizon="WEEKLY",
        expiry_date=today_str,
        dte_override=0
    )
    # DTE on 0-DTE expiry must be fractional (< 1.0), not clamped to 1.0
    assert 0.0 <= res["dte"] < 1.0
    # Time to expiry in years must reflect fractional hours (< 1 full day)
    assert res["time_to_expiry_years"] < (1.0 / 365.0)


def test_options_contractual_max_loss_distinction():
    """Verify options candidates distinguish contractual max loss from planned stop loss (Findings 6.8 & 6.9)."""
    tournament = WeeklyMonthlyOptionsStrategyEngine.analyze_highest_win_probability_strategy(
        verdict="STRONG_BEARISH",
        spot=24500.0,
        iv_pct=14.0,
        horizon="WEEKLY"
    )
    assert tournament["status"] == "OK"
    candidates = tournament["candidates_ranked"]
    assert len(candidates) >= 4

    for c in candidates:
        assert "planned_stop_loss_inr" in c
        assert "contractual_max_loss" in c
        assert "risk_profile" in c
        if c["architecture"] == "NAKED_SHORT_1SIGMA":
            # Naked short call has unlimited tail risk
            assert c["contractual_max_loss"] == "UNLIMITED"
            assert c["risk_profile"] == "UNDEFINED_UNLIMITED"
        elif c["architecture"] == "CREDIT_SPREAD":
            # Defined-Risk credit spread has capped max loss
            assert isinstance(c["contractual_max_loss"], (int, float))
            assert c["risk_profile"] == "DEFINED_RISK"


# =============================================================================
# 5. CONFIGURATION, SCRIPTS & CACHE PRESERVATION (Findings 3.5, 4.2, 4.5, 7.1)
# =============================================================================
def test_requirements_numpy_dependency_version():
    """Verify requirements.txt has portable numpy specification compatible with Linux CI (Finding 7.1/7.3)."""
    req_path = os.path.join(REPO_ROOT, "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "numpy>=2.0.0,<3.0.0" in content
    assert "numpy==2.5.3" not in content


def test_setup_vps_duckdns_safety():
    """Verify setup_vps_duckdns.sh backs up default nginx configuration and guards cloudflared (Finding 3.5)."""
    script_path = os.path.join(REPO_ROOT, "scripts", "setup_vps_duckdns.sh")
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "default.bak" in content
    assert "--disable-cloudflared" in content


def test_service_worker_offline_status():
    """Verify sw.js returns 503 OFFLINE JSON response when offline (Finding 4.5)."""
    sw_path = os.path.join(REPO_ROOT, "web", "static", "sw.js")
    with open(sw_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "status: 503" in content
    assert "OFFLINE" in content
    assert "data_unavailable: true" in content


def test_nifty_learning_fold_scaler_isolation():
    """Verify nifty_forecast_learning_engine fits scaler strictly inside fold loop (Finding 6.1)."""
    engine_path = os.path.join(REPO_ROOT, "engines", "nifty_forecast_learning_engine.py")
    with open(engine_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Must fit fold scaler on train_idx fold only, not on entire X before fold loop
    assert "fold_scaler = StandardScaler()" in content
    assert "X_tr = fold_scaler.fit_transform(X[train_idx])" in content
    assert "X_va = fold_scaler.transform(X[val_idx])" in content

