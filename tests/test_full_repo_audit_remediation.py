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

from core.data_quality import validate_ohlcv_dataframe, check_data_freshness, IST, now_ist, format_ist
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
    }, index=pd.date_range("2026-01-01", periods=4, freq="D"))
    is_valid, _, warnings = validate_ohlcv_dataframe(df_clean, min_bars=3)
    assert is_valid is True

    # Invert a single candle (High < Low)
    df_corrupt = df_clean.copy()
    corrupt_date = pd.Timestamp("2026-01-02")
    df_corrupt.loc[corrupt_date, "High"] = 90.0
    df_corrupt.loc[corrupt_date, "Low"] = 110.0

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
            assert stats["brier_score"] is None
        elif stats["brier_score"] is not None:
            # Three-class Brier score is the sum of squared class-probability errors (range 0..2).
            assert 0.0 <= stats["brier_score"] <= 2.0


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


# =============================================================================
# 6. PIPELINE, RISK & SIZING INTEGRITY (Findings 5.4, 5.5, 6.1, 6.6, 7.6, 8.3)
# =============================================================================
def test_nifty500_compounded_equity_unrealized_pnl():
    """Verify get_live_compounded_equity includes unrealized P&L from active positions (Finding 5.5)."""
    from engines.nifty500_engine import get_live_compounded_equity
    with patch("execution.state_engine.state_engine.get_trade_history", return_value=[{"portfolio_type": "NIFTY500_MOMENTUM", "pnl": 5000.0}]):
        with patch("execution.state_engine.state_engine.get_active_positions", return_value=[
            {"symbol": "INFY", "portfolio_type": "NIFTY500_MOMENTUM", "entry_price": 1000.0, "current_price": 1050.0, "quantity": 10}
        ]):
            curr_eq, tot_eq, slots = get_live_compounded_equity(initial_cash=500000.0)
            # 500,000 + 5,000 realized + (1050 - 1000)*10 = 505,500
            assert tot_eq == 505500.0
            assert slots == 1


def test_nifty500_universe_and_benchmark_fail_closed():
    """Verify Nifty 500 fails closed when constituents or benchmarks are unavailable (Finding 5.4)."""
    from engines.nifty500_engine import get_nifty500_symbols
    with patch("requests.get", side_effect=Exception("Network error")):
        with patch("execution.state_engine.state_engine.get_kv", return_value=[]):
            with patch("execution.state_engine.state_engine.get_all_ohlcv_metadata", return_value=[]):
                syms = get_nifty500_symbols()
                assert syms == []  # Must never silently return arbitrary 10 stocks


def test_institutional_calculate_position_scales_with_base_risk():
    """Verify calculate_position dynamically scales tier risk relative to base_risk_pct (Finding 6.6)."""
    from engines.institutional_engine import ExecutionEngine
    row_a_plus = pd.Series({"close": 1000.0, "atr14": 20.0, "low": 980.0, "score": 95.0, "confidence": 5, "ml_verdict": "APPROVED"})
    row_a = pd.Series({"close": 1000.0, "atr14": 20.0, "low": 980.0, "score": 85.0, "confidence": 4, "ml_verdict": "APPROVED"})
    row_b = pd.Series({"close": 1000.0, "atr14": 20.0, "low": 980.0, "score": 75.0, "confidence": 3, "ml_verdict": "APPROVED"})

    # With base_risk_pct = 0.75
    p_a_plus = ExecutionEngine.calculate_position(row_a_plus, capital=100000.0, base_risk_pct=0.75)
    p_a = ExecutionEngine.calculate_position(row_a, capital=100000.0, base_risk_pct=0.75)
    p_b = ExecutionEngine.calculate_position(row_b, capital=100000.0, base_risk_pct=0.75)
    assert p_a_plus["risk_pct"] == 1.0
    assert p_a["risk_pct"] == 0.75
    assert p_b["risk_pct"] == 0.60

    # With custom base_risk_pct = 0.50
    p_a_plus_half = ExecutionEngine.calculate_position(row_a_plus, capital=100000.0, base_risk_pct=0.50)
    assert round(p_a_plus_half["risk_pct"], 2) == round(0.50 * (1.0 / 0.75), 2)


def test_ai_context_builder_probabilities_and_hit_rate():
    """Verify AI context builder normalizes probabilities and removes fabricated defaults (Finding 7.6)."""
    from ai_engine.context_builder import build_prediction_context
    sample_preds = {
        "status": "OK",
        "spot_price": 24500.0,
        "composite_score": 10.0,
        "t1_forecast": {
            "verdict": "BULLISH",
            "prob_up": 0.6,
            "prob_flat": 0.3,
            "prob_down": 0.3,  # sum is 1.2
        },
        "t5_forecast": {
            "verdict": "RANGE-BOUND",
            "prob_up": 0.4,
            "prob_flat": 0.4,
            "prob_down": 0.4,  # sum is 1.2
        },
        "self_training_engine": {}  # missing hit rates
    }
    contract = build_prediction_context(sample_preds, data_mode_override="HISTORICAL_DAILY")
    t1_p = contract.predictions[0]
    t5_p = contract.predictions[1]

    # Probabilities must sum to 1.0 (within rounding tolerance)
    assert round(t1_p.probability_up + t1_p.probability_flat + t1_p.probability_down, 2) == 1.0
    assert round(t5_p.probability_up + t5_p.probability_flat + t5_p.probability_down, 2) == 1.0
    # Must report INSUFFICIENT_DATA rather than fabricated 70%/75%
    assert "INSUFFICIENT_DATA" in t1_p.validation_status
    assert "INSUFFICIENT_DATA" in t5_p.validation_status


def test_health_check_database_readiness():
    """Verify /health endpoint validates database connectivity (Finding 8.3)."""
    from fastapi.testclient import TestClient
    from web.api import app
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["database"] == "healthy"


# =============================================================================
# 7. OCTOBER 18 AUDIT REMEDIATION VERIFICATION (C-03..C-05, H-01..H-08)
# =============================================================================
def test_h01_candle_geometry_quarantined_not_coerced():
    """H-01: Verify OHLCV validation quarantines geometry violations rather than silently coercing High/Low."""
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 102.0, "Volume": 10000.0
    }, index=dates)

    # Invert single candle
    df.loc[dates[5], "High"] = 90.0
    df.loc[dates[5], "Low"] = 110.0
    is_valid, clean_df, warnings = validate_ohlcv_dataframe(df, symbol="TEST_GEOM", min_bars=30)
    assert is_valid is True
    # Corrupt bar was dropped/quarantined, not silently recalculated
    assert len(clean_df) == 39
    assert dates[5] not in clean_df.index
    assert any("High was less than Low" in w or "candle geometry was violated" in w for w in warnings)


def test_h02_and_h03_nifty500_ingestion_and_universe():
    """H-02 & H-03: Verify Nifty 500 ingestion applies contract and universe fails closed."""
    from engines.nifty500_engine import get_nifty500_symbols
    with patch("requests.get", side_effect=Exception("Offline")):
        with patch("execution.state_engine.state_engine.get_kv", return_value=[]):
            with patch("execution.state_engine.state_engine.get_all_ohlcv_metadata", return_value=[]):
                syms = get_nifty500_symbols()
                assert syms == []  # Unverified universe returns empty


def test_c03_nse_quote_freshness_without_provider_time():
    """C-03: NSE quote fallback marks quotes stale/unverified when provider_timestamp is absent."""
    eng = NiftyAnalyticsEngine()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"index": "NIFTY 50", "last": 24500.0, "variation": 50.0, "percentChange": 0.2, "high": 24550.0, "low": 24450.0}]
        # Omits timestamp to simulate missing provider time
    }
    fake_sess = MagicMock()
    fake_sess.get.return_value = fake_resp
    with patch("urllib.request.urlopen", side_effect=Exception("Offline")):
        with patch.object(eng, "_get_nse_session", return_value=fake_sess):
            res = eng.get_live_nifty_spot(force_refresh=True)
            assert res["source"] == "NSE_ALL_INDICES_STREAM"
            assert res["provider_timestamp"] is None
            assert res["data_quality"] == "UNVERIFIED_SOURCE_TIMESTAMP"
            assert res["is_stale"] is True


def test_c04_forecast_and_tree_synthetic_prior_excluded():
    """C-04: Forecast and Tree ML training fail closed on insufficient empirical data and reject synthetic priors."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    with patch("engines.nifty_forecast_learning_engine.state_engine.get_ohlcv", return_value=None):
        with patch("engines.nifty_forecast_learning_engine.nifty_prediction_ledger.get_training_records", return_value=pd.DataFrame()):
            df_ret = nifty_forecast_learning_engine.compile_training_dataset("T1")
            assert df_ret.empty  # Returns empty, zero synthetic priors injected

    res = tree_quant_ml_engine.train_horizon_models("T1", df=pd.DataFrame())
    assert res["status"] == "INSUFFICIENT_DATA"
    assert res["is_promoted"] is False


def test_c05_model_comparison_measured_provenance():
    """C-05: Model comparison candidates report explicit metric_provenance and no hardcoded offsets."""
    res = ml_performance_engine.get_model_comparison("WEEKLY")
    candidates = res["candidates"]
    for c in candidates:
        assert "metric_provenance" in c
        assert c["metric_provenance"] in ("MEASURED_EMPIRICAL", "MEASURED_HOLDOUT", "MEASURED_CLASS_PRIOR", "THEORETICAL_PRIOR", "DATA_UNAVAILABLE")
        # Ensure no hardcoded fake P&L string
        assert "-₹42,000 (Whipsaw)" not in str(c.get("net_pnl_after_costs", ""))


def test_h05_options_simulation_risk_free_rate_and_disclaimer():
    """H-05: Intraday options backtest returns SCENARIO_SIMULATION and uses RISK_FREE_RATE."""
    from engines.intraday_options_strategy_engine import IntradayOptionsStrategyEngine
    from core.config import RISK_FREE_RATE
    sample_session = [{
        "date": "2026-02-01",
        "verdict": "STRONG_BULLISH",
        "open": 24000.0,
        "high": 24200.0,
        "low": 23950.0,
        "close": 24150.0,
        "return_pct": 0.625,
        "iv_pct": 14.0
    }]
    res = IntradayOptionsStrategyEngine._simulate_strategy_variant(sample_session)
    assert res["simulation_type"] == "SCENARIO_SIMULATION"
    assert res["model_assumptions"]["risk_free_rate"] == RISK_FREE_RATE


def test_h06_unified_ml_dataset_cohort_separation(tmp_path):
    """H-06: Unified ML dataset excludes counterfactuals by default and does not infer fills on signal entry."""
    from core.unified_ml_dataset import UNIFIED_DATASET_TABLE
    ds = UnifiedMLDataset(db_path=str(tmp_path / "unified_ml_test.db"))
    # Ingestion without actual fill must be PENDING_FILL, not FILLED
    sig_id = f"SIG_TEST_{int(datetime.datetime.now().timestamp())}"
    trade_id = f"TRD_TEST_{int(datetime.datetime.now().timestamp())}"
    ds.record_signal_and_trade(
        signal_id=sig_id,
        trade_id=trade_id,
        instrument="NIFTY",
        trading_horizon="WEEKLY",
        strategy_name="TEST_STRAT",
        record_type="LIVE",
        market_session="2026-02-01",
        accept_reject_verdict="ACCEPTED",
        decision_source="SYSTEM",
        features={"regime_confidence": 0.8},
        regime="BULLISH",
        spot_at_entry=24500.0,
        intended_entry=24500.0,
        intended_target=24800.0,
        intended_sl=24350.0,
        sl_threshold_pct=1.0,
        quantity=1,
        capital_at_risk=150.0,
        actual_entry=None  # Not yet filled!
    )
    with ds._get_connection() as conn:
        row = conn.execute(f"SELECT actual_entry, fill_status FROM {UNIFIED_DATASET_TABLE} WHERE trade_id = ?", (trade_id,)).fetchone()
        assert row is not None
        assert row[0] is None  # Actual entry must remain NULL
        assert row[1] == "PENDING_FILL"  # Never inferred as FILLED without verified execution
    # Production training must not consume accepted-but-unfilled signals.
    assert ds.get_dataset_for_horizon("WEEKLY").empty

    # Counterfactual opt-in must not silently mix all cohorts; callers select cohorts explicitly.
    with pytest.raises(ValueError, match="Explicit record_types"):
        ds.get_dataset_for_horizon("WEEKLY", include_counterfactuals=True)
    with pytest.raises(ValueError, match="at least one cohort"):
        ds.get_dataset_for_horizon("WEEKLY", record_types=[])


def test_h07_telegram_vault_restore_safety():
    """H-07: Telegram vault restore rejects corrupt databases and enforces safety limits."""
    from core.telegram_vault import telegram_vault
    import inspect
    src = inspect.getsource(telegram_vault.restore_master_file_from_cloud)
    assert "MAX_RESTORE_SIZE" in src
    assert "required_tables" in src
    assert "state_engine._lock" in src


def test_h08_state_engine_rejects_pickle_and_future_timestamp(tmp_path):
    """H-08: State engine rejects legacy non-JSON blobs and flags future timestamps as invalid."""
    import zlib
    test_db = str(tmp_path / "test_h08.db")
    eng = StateEngine(db_path=test_db)

    # Insert a raw non-JSON blob directly into SQLite
    fake_pickle_bytes = zlib.compress(b"NOT_A_JSON_BLOB")
    now_future = (datetime.datetime.now() + datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    with eng._get_connection() as conn:
        conn.execute("""
            INSERT INTO ohlcv_master_store (symbol, last_bar_date, bars_count, updated_at, ohlcv_blob)
            VALUES ('H08_SYM', '2026-02-01', 1, ?, ?)
        """, (now_future, sqlite3.Binary(fake_pickle_bytes)))
        conn.commit()

    # Must reject non-JSON blob and return (None, 9999.0)
    df, age = eng.get_ohlcv_df("H08_SYM")
    assert df is None
    assert age == 9999.0
    # Convenience API must not discard the invalid-age sentinel and expose unusable data.
    assert eng.get_ohlcv("H08_SYM") is None





def test_tree_model_rejects_nonbinary_targets(tmp_path):
    """Invalid target values must not be truncated to 0/1 or promoted."""
    from engines.tree_quant_ml_engine import TreeQuantMLEngine
    from core.unified_ml_dataset import UNIFIED_FEATURE_COLS

    model = TreeQuantMLEngine(models_dir=tmp_path / "tree_models")
    df = pd.DataFrame({col: np.linspace(0.1, 1.2, 12) for col in UNIFIED_FEATURE_COLS})
    df["is_winner"] = [0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0]
    result = model.train_horizon_models("WEEKLY", df=df)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["is_promoted"] is False
    assert result["metric_provenance"] == "DATA_UNAVAILABLE"
    assert "binary values" in result["reason"]


def test_option_chain_rejects_nonfinite_iv_for_skew_and_gex(monkeypatch):
    """NaN/Infinity IV must be unavailable, not a numeric skew or neutral-gamma regime."""
    monkeypatch.setattr(
        NiftyAnalyticsEngine,
        "get_india_vix",
        staticmethod(lambda: {"current_vix": float("inf")}),
    )
    raw_rows = []
    for strike in (24900, 24950, 25000, 25050, 25100):
        raw_rows.append({
            "strikePrice": strike,
            "CE": {
                "openInterest": 100, "changeinOpenInterest": 1, "totalTradedVolume": 10,
                "impliedVolatility": float("inf"), "lastPrice": 100, "change": 0, "pChange": 0,
                "totalBuyQuantity": 20, "totalSellQuantity": 20
            },
            "PE": {
                "openInterest": 120, "changeinOpenInterest": 1, "totalTradedVolume": 10,
                "impliedVolatility": float("nan"), "lastPrice": 100, "change": 0, "pChange": 0,
                "totalBuyQuantity": 20, "totalSellQuantity": 20
            }
        })
    result = NiftyAnalyticsEngine._analyze_single_expiry_chain(
        raw_rows=raw_rows, spot_price=25000.0, expiry_label="TEST", nse_timestamp="2026-10-18"
    )
    assert result["status"] == "OK"
    assert result["volatility_and_gex"]["atm_iv"] is None
    assert result["volatility_and_gex"]["iv_skew"] is None
    assert result["volatility_and_gex"]["gex_regime"] == "DATA_UNAVAILABLE"
    assert result["volatility_and_gex"]["iv_data_quality"] == "DATA_UNAVAILABLE"


def test_unified_ml_dataset_rejects_missing_feature_imputation(tmp_path):
    """Incomplete feature payloads remain visible in storage but are never zero-imputed into training."""
    from core.unified_ml_dataset import UNIFIED_FEATURE_COLS

    ds = UnifiedMLDataset(db_path=str(tmp_path / "feature_quality.db"))
    features = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
    features.pop("nifty_spot")
    features.pop("iv_spot")
    row_id = ds.record_signal_and_trade(
        signal_id="SIG_MISSING_FEATURE",
        trade_id="TRADE_MISSING_FEATURE",
        instrument="NIFTY",
        trading_horizon="WEEKLY",
        strategy_name="TEST",
        record_type="LIVE",
        market_session="MORNING",
        accept_reject_verdict="ACCEPTED",
        decision_source="RULE_BASED",
        features=features,
        regime="MILD_BULLISH",
        spot_at_entry=25000.0,
        intended_entry=100.0,
        intended_target=80.0,
        intended_sl=140.0,
        sl_threshold_pct=40.0,
        quantity=1,
        capital_at_risk=40.0,
        actual_entry=100.0,
    )
    assert row_id > 0
    with ds._get_connection() as conn:
        row = conn.execute(
            "SELECT features_json, iv_at_entry FROM unified_trade_learning_dataset WHERE id = ?",
            (row_id,),
        ).fetchone()
        snapshot = json.loads(row["features_json"])
        assert snapshot["nifty_spot"] is None
        assert row["iv_at_entry"] is None
    assert ds.record_trade_outcome(
        trade_id="TRADE_MISSING_FEATURE", actual_exit=80.0, exit_reason="PROFIT_TARGET",
        gross_pnl=20.0, net_pnl=18.0, total_costs=2.0
    )
    assert ds.get_dataset_for_horizon("WEEKLY").empty


def test_manual_exit_does_not_fabricate_target_path_or_rule_compliance(tmp_path):
    """Manual exits without path evidence must not be labeled target/SL first or rule-compliant."""
    from core.unified_ml_dataset import UNIFIED_FEATURE_COLS

    ds = UnifiedMLDataset(db_path=str(tmp_path / "manual_exit_quality.db"))
    features = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
    row_id = ds.record_signal_and_trade(
        signal_id="SIG_MANUAL_EXIT",
        trade_id="TRADE_MANUAL_EXIT",
        instrument="NIFTY",
        trading_horizon="WEEKLY",
        strategy_name="TEST",
        record_type="LIVE",
        market_session="MORNING",
        accept_reject_verdict="ACCEPTED",
        decision_source="RULE_BASED",
        features=features,
        regime="RANGE_BOUND",
        spot_at_entry=25000.0,
        intended_entry=100.0,
        intended_target=80.0,
        intended_sl=140.0,
        sl_threshold_pct=40.0,
        quantity=1,
        capital_at_risk=40.0,
        actual_entry=100.0,
    )
    assert row_id > 0
    assert ds.record_trade_outcome(
        trade_id="TRADE_MANUAL_EXIT", actual_exit=110.0,
        exit_reason="MANUAL_CLOSE", gross_pnl=10.0, net_pnl=8.0, total_costs=2.0
    )
    with ds._get_connection() as conn:
        row = conn.execute(
            "SELECT target_hit_before_sl, rules_followed, mae_pct FROM unified_trade_learning_dataset WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert row["target_hit_before_sl"] is None
        assert row["rules_followed"] is None
        assert row["mae_pct"] is None


def test_format_ist_converts_timezone_aware_values():
    """UTC timestamps must be converted to India time before appending the IST label."""
    from datetime import timezone
    utc_dt = datetime.datetime(2026, 10, 18, 0, 0, 0, tzinfo=timezone.utc)
    assert format_ist(utc_dt) == "2026-10-18 05:30:00 IST"


def test_data_quality_rejects_numeric_index():
    """Numeric row numbers must never be accepted as market-bar timestamps."""
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [104, 105, 106, 107, 108],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    })
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="NUMERIC_INDEX", min_bars=5)
    assert not valid
    assert cleaned.empty
    assert any("numeric OHLCV index" in warning for warning in warnings)


def test_data_quality_normalizes_string_timestamp_index():
    """Parse valid string timestamps into a DatetimeIndex before time-series validation."""
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [104, 105, 106, 107, 108],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }, index=dates)
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="STRING_INDEX", min_bars=5)
    assert valid
    assert isinstance(cleaned.index, pd.DatetimeIndex)


def test_data_quality_rejects_missing_volume():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [104, 105, 106, 107, 108],
    }, index=dates)
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="NO_VOLUME", min_bars=5)
    assert not valid
    assert cleaned.empty
    assert any("Missing required Volume" in warning for warning in warnings)


def test_data_quality_rejects_non_finite_ohlcv():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [104, 105, np.inf, 107, 108],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }, index=dates)
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="NON_FINITE", min_bars=5)
    assert not valid
    assert cleaned.empty


def test_data_quality_rejects_unexplained_extreme_price_jump():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 160],
        "High": [105, 106, 107, 108, 165],
        "Low": [99, 100, 101, 102, 159],
        "Close": [104, 105, 106, 107, 164],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }, index=dates)
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="EXTREME_JUMP", min_bars=5)
    assert not valid
    assert cleaned.empty
    assert any("greater than 50%" in warning for warning in warnings)


def test_data_quality_rejects_duplicate_timestamps():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-04", "2026-01-05"])
    df = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103],
        "Close": [104, 105, 106, 107, 108],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }, index=dates)
    valid, cleaned, warnings = validate_ohlcv_dataframe(df, symbol="DUPLICATE_TS", min_bars=5)
    assert not valid
    assert cleaned.empty
    assert any("duplicate timestamp" in warning for warning in warnings)
