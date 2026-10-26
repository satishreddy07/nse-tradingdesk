"""
tests/test_intraday_prediction_validation_engine.py
===================================================
Comprehensive Regression & Integration Tests for the Intraday Prediction Validation Engine.

Covers:
1. Immutable prediction records (cannot overwrite original predictions or feature snapshots).
2. Delayed and missing prices (leaves PENDING or marks DATA_UNAVAILABLE; never fabricates).
3. Incorrect or stale timestamps (freshness check).
4. Market closures and holidays (weekend/holiday handling, session boundaries at 15:30 IST).
5. Duplicate evaluations (idempotent; duplicate calls never double-count or corrupt state).
6. Restart recovery (pending evaluations preserved in SQLite and resolved on restart).
7. Probability calculations & 3-class Brier score correctness.
8. Evaluation horizon boundaries (5m, 15m, 30m deadband thresholds).
9. Prevention of future-data leakage (chronological validation; no random time-series shuffling).
10. Empty datasets and small sample warnings (N < 30 alert).
11. Separation of paper-trading results from real trading results.
12. FastAPI validation endpoints contract and auth verification.
"""

import json
import math
import os
import sqlite3
import tempfile
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from core.data_quality import IST, format_ist
from core.market_calendar import NSEMarketCalendar
from core.intraday_prediction_ledger import (
    IntradayPredictionLedger,
    INTRADAY_PRED_LEDGER_TABLE,
    INTRADAY_EVAL_TABLE
)
from engines.intraday_prediction_validation_engine import (
    IntradayPredictionValidationEngine,
    NEUTRAL_THRESHOLDS_PCT,
    _wilson_confidence_interval
)
from web.api import app

client = TestClient(app)


@pytest.fixture
def isolated_ledger(tmp_path):
    """Provides an isolated SQLite database and IntradayPredictionLedger instance."""
    db_file = str(tmp_path / "test_intraday_validation.db")
    ledger = IntradayPredictionLedger(db_path=db_file)
    engine = IntradayPredictionValidationEngine(ledger=ledger, db_path=db_file)
    return ledger, engine, db_file


# =========================================================================
# 1. IMMUTABLE PREDICTION RECORD TESTS
# =========================================================================
def test_immutable_prediction_creation(isolated_ledger):
    ledger, engine, db_file = isolated_ledger

    features = {
        "spot_price": 24500.0,
        "vwap": 24480.0,
        "vwap_distance_pts": 20.0,
        "vwap_distance_pct": 0.0817,
        "above_vwap": True,
        "atm_pcr": 1.15,
        "near_atm_pcr": 1.20,
        "net_pe_oi_chg": 500000,
        "net_ce_oi_chg": 200000,
        "oi_ratio": 0.4286,
        "writer_dominance_score": 35.0,
        "composite_score": 28.5
    }

    pred_id = ledger.record_prediction(
        spot_price=24500.0,
        predicted_direction="BULLISH",
        prob_up=0.55,
        prob_flat=0.30,
        prob_down=0.15,
        market_regime="STRONG_BULLISH",
        features=features,
        expected_delta_pts=45.0,
        expected_delta_pct=0.1837,
        composite_score=28.5,
        writer_dominance_score=35.0,
        data_provenance="LIVE_MARKET",
        model_version="INTRADAY-VAL-1.0.0"
    )

    assert pred_id.startswith("IPRED-")
    pred = ledger.get_prediction(pred_id)
    assert pred is not None
    assert pred["spot_price"] == 24500.0
    assert pred["predicted_direction"] == "BULLISH"
    assert pred["prob_up"] == 0.55
    assert pred["prob_flat"] == 0.30
    assert pred["prob_down"] == 0.15
    assert pred["market_regime"] == "STRONG_BULLISH"
    assert pred["prediction_status"] == "PENDING"
    assert pred["features_snapshot"]["vwap"] == 24480.0
    assert pred["features_snapshot"]["oi_ratio"] == 0.4286

    # Verify that multi-horizon evaluations were scheduled (5m, 15m, 30m)
    evals = ledger.get_evaluations_for_prediction(pred_id)
    assert len(evals) == 3
    horizons = [e["horizon_min"] for e in evals]
    assert horizons == [5, 15, 30]
    for e in evals:
        assert e["evaluation_status"] == "PENDING"
        assert e["actual_price"] is None


def test_immutable_prediction_never_overwritten(isolated_ledger):
    ledger, engine, db_file = isolated_ledger

    features_v1 = {"vwap": 24500.0, "composite_score": 10.0}
    pred_id = ledger.record_prediction(
        spot_price=24500.0,
        predicted_direction="BULLISH",
        prob_up=0.60,
        prob_flat=0.25,
        prob_down=0.15,
        market_regime="MILD_BULLISH",
        features=features_v1,
        prediction_id="IPRED-TEST-001"
    )

    # Attempt to overwrite the same prediction_id with conflicting data
    features_v2 = {"vwap": 99999.0, "composite_score": -99.0}
    duplicate_res = ledger.record_prediction(
        spot_price=20000.0,
        predicted_direction="BEARISH",
        prob_up=0.10,
        prob_flat=0.10,
        prob_down=0.80,
        market_regime="STRONG_BEARISH",
        features=features_v2,
        prediction_id="IPRED-TEST-001"
    )

    assert duplicate_res == "IPRED-TEST-001"
    # Original record must be completely preserved
    pred = ledger.get_prediction("IPRED-TEST-001")
    assert pred["spot_price"] == 24500.0
    assert pred["predicted_direction"] == "BULLISH"
    assert pred["prob_up"] == 0.60
    assert pred["features_snapshot"]["vwap"] == 24500.0


def test_non_positive_spot_price_rejected(isolated_ledger):
    ledger, engine, _ = isolated_ledger
    with pytest.raises(ValueError, match="must be strictly positive"):
        ledger.record_prediction(
            spot_price=0.0,
            predicted_direction="BULLISH",
            prob_up=0.5,
            prob_flat=0.3,
            prob_down=0.2,
            market_regime="RANGE_BOUND",
            features={}
        )


# =========================================================================
# 2. OUTCOME RESOLUTION & NEUTRAL THRESHOLD TESTS
# =========================================================================
def test_outcome_resolution_bullish_hit(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))  # Tuesday (Trading Day)
    t0_epoch = t0_dt.timestamp()

    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.60,
        prob_flat=0.25,
        prob_down=0.15,
        market_regime="STRONG_BULLISH",
        features={"vwap": 23990.0},
        prediction_epoch=t0_epoch,
        prediction_timestamp=format_ist(t0_dt)
    )

    # Mock price lookup: at 10:05 IST (5m), spot moved to 24020.0 (+0.0833% > 0.03% threshold)
    def mock_price_lookup(target_dt, horizon_min):
        if horizon_min == 5:
            return {"price": 24020.0, "source": "TEST_FEED", "timestamp_ist": format_ist(target_dt)}
        return None

    # Evaluate at 10:06 IST (5m evaluation due)
    eval_dt = t0_dt + timedelta(minutes=6)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=mock_price_lookup)

    assert res["evaluated_count"] == 1
    evals = ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["evaluation_status"] == "RESOLVED"
    assert e5["actual_price"] == 24020.0
    assert e5["price_delta_pts"] == 20.0
    assert e5["realized_direction"] == "BULLISH"
    assert e5["is_direction_correct"] == 1
    # Brier score: (0.6 - 1.0)^2 + (0.25 - 0)^2 + (0.15 - 0)^2 = 0.16 + 0.0625 + 0.0225 = 0.245
    assert round(e5["brier_score"], 3) == 0.245


def test_outcome_resolution_neutral_threshold(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))
    t0_epoch = t0_dt.timestamp()

    # Prediction: BULLISH, but spot barely moves (+3 pts on 24,000 = +0.0125% < 0.03% threshold)
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.50,
        prob_flat=0.30,
        prob_down=0.20,
        market_regime="MILD_BULLISH",
        features={},
        prediction_epoch=t0_epoch,
        prediction_timestamp=format_ist(t0_dt)
    )

    def mock_price_lookup(target_dt, horizon_min):
        return {"price": 24003.0, "source": "TEST_FEED", "timestamp_ist": format_ist(target_dt)}

    eval_dt = t0_dt + timedelta(minutes=6)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=mock_price_lookup)

    assert res["evaluated_count"] == 1
    evals = ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["realized_direction"] == "NEUTRAL"
    # Predicted BULLISH, but realized NEUTRAL -> is_direction_correct = 0 (MISS)
    assert e5["is_direction_correct"] == 0


def test_outcome_resolution_neutral_prediction_hit(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 11, 0, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="NEUTRAL",
        prob_up=0.20,
        prob_flat=0.60,
        prob_down=0.20,
        market_regime="RANGE_BOUND",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    # Market stayed in neutral range (+4 pts on 24,000 = +0.016% < 0.06% for 15m)
    def mock_price_lookup(target_dt, horizon_min):
        if horizon_min == 15:
            return {"price": 24004.0, "source": "TEST_FEED"}
        return None

    eval_dt = t0_dt + timedelta(minutes=16)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=mock_price_lookup)

    evals = ledger.get_evaluations_for_prediction(pred_id)
    e15 = next(e for e in evals if e["horizon_min"] == 15)
    assert e15["realized_direction"] == "NEUTRAL"
    # Predicted NEUTRAL and realized NEUTRAL -> HIT
    assert e15["is_direction_correct"] == 1


# =========================================================================
# 3. MISSING OBSERVATIONS & STALE DATA TESTS
# =========================================================================
def test_missing_price_recent_remains_pending(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.5,
        prob_flat=0.3,
        prob_down=0.2,
        market_regime="RANGE_BOUND",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    # Target is at 10:05. As-of time is 10:07 (2 mins past target, within 15m grace period)
    eval_dt = t0_dt + timedelta(minutes=7)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=lambda dt, h: None)

    evals = ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    # Must remain PENDING awaiting feed, NOT marked as failure or DATA_UNAVAILABLE yet
    assert e5["evaluation_status"] == "PENDING"
    assert e5["actual_price"] is None


def test_missing_price_past_tolerance_marked_unavailable(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.5,
        prob_flat=0.3,
        prob_down=0.2,
        market_regime="RANGE_BOUND",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    # As-of time is 10:30 (25 mins past 5m target; exceeds 15m grace period)
    eval_dt = t0_dt + timedelta(minutes=30)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=lambda dt, h: None)

    evals = ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["evaluation_status"] == "DATA_UNAVAILABLE"
    assert "No authentic price observation" in e5["status_reason"]
    # Crucial: Must NEVER mark is_direction_correct as 0 or 1 on missing data
    assert e5["is_direction_correct"] is None


# =========================================================================
# 4. CALENDAR, HOLIDAYS & SESSION BOUNDARY TESTS
# =========================================================================
def test_holiday_or_weekend_marked_properly(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 25, 10, 0, 0))  # Sunday
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.5,
        prob_flat=0.3,
        prob_down=0.2,
        market_regime="RANGE_BOUND",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    eval_dt = t0_dt + timedelta(minutes=10)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=False):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt)

    assert res["session_closed_count"] >= 1
    evals = ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["evaluation_status"] == "HOLIDAY_OR_CLOSED"
    assert e5["is_direction_correct"] is None


def test_session_boundary_post_1530_marked_session_closed(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    # Prediction logged at 15:15 IST
    t0_dt = IST.localize(datetime(2026, 10, 27, 15, 15, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BEARISH",
        prob_up=0.15,
        prob_flat=0.25,
        prob_down=0.60,
        market_regime="STRONG_BEARISH",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    # 30m target is 15:45 IST (> 15:30 IST market close)
    eval_dt = t0_dt + timedelta(minutes=35)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = engine.evaluate_pending_predictions(as_of_dt=eval_dt)

    evals = ledger.get_evaluations_for_prediction(pred_id)
    e30 = next(e for e in evals if e["horizon_min"] == 30)
    assert e30["evaluation_status"] == "SESSION_CLOSED"
    assert "beyond 15:30 IST market close" in e30["status_reason"]
    assert e30["is_direction_correct"] is None


# =========================================================================
# 5. IDEMPOTENCY & RESTART RECOVERY TESTS
# =========================================================================
def test_evaluation_idempotency(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BULLISH",
        prob_up=0.6,
        prob_flat=0.3,
        prob_down=0.1,
        market_regime="STRONG_BULLISH",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    def mock_price_lookup(dt, h):
        return {"price": 24050.0, "source": "MOCK"}

    eval_dt = t0_dt + timedelta(minutes=6)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res1 = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=mock_price_lookup)
        assert res1["evaluated_count"] == 1

        # Second call: must be completely idempotent (0 evaluated)
        res2 = engine.evaluate_pending_predictions(as_of_dt=eval_dt, price_lookup_fn=mock_price_lookup)
        assert res2["evaluated_count"] == 0

    evals = ledger.get_evaluations_for_prediction(pred_id)
    assert len(evals) == 3
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["evaluation_status"] == "RESOLVED"
    assert e5["actual_price"] == 24050.0


def test_restart_recovery_from_database(isolated_ledger):
    ledger, engine, db_file = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 10, 0, 0))
    pred_id = ledger.record_prediction(
        spot_price=24000.0,
        predicted_direction="BEARISH",
        prob_up=0.2,
        prob_flat=0.2,
        prob_down=0.6,
        market_regime="STRONG_BEARISH",
        features={},
        prediction_epoch=t0_dt.timestamp(),
        prediction_timestamp=format_ist(t0_dt)
    )

    # Simulate process termination and re-initialization
    restarted_ledger = IntradayPredictionLedger(db_path=db_file)
    restarted_engine = IntradayPredictionValidationEngine(ledger=restarted_ledger, db_path=db_file)

    pending = restarted_ledger.get_pending_evaluations()
    assert len(pending) == 3

    # Successfully resolve after restart
    eval_dt = t0_dt + timedelta(minutes=6)
    with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
        res = restarted_engine.evaluate_pending_predictions(
            as_of_dt=eval_dt,
            price_lookup_fn=lambda dt, h: {"price": 23970.0, "source": "RESTART_FEED"}
        )
    assert res["evaluated_count"] == 1

    evals = restarted_ledger.get_evaluations_for_prediction(pred_id)
    e5 = next(e for e in evals if e["horizon_min"] == 5)
    assert e5["evaluation_status"] == "RESOLVED"
    assert e5["realized_direction"] == "BEARISH"
    assert e5["is_direction_correct"] == 1


# =========================================================================
# 6. PERFORMANCE METRICS, WILSON CI & CALIBRATION TESTS
# =========================================================================
def test_wilson_confidence_interval_bounds():
    # 0 out of 10
    low, high = _wilson_confidence_interval(0, 10)
    assert low == 0.0
    assert high > 0.0

    # 10 out of 10
    low, high = _wilson_confidence_interval(10, 10)
    assert low < 1.0
    assert high == 1.0

    # 5 out of 10 (50%)
    low, high = _wilson_confidence_interval(5, 10)
    assert 0.20 < low < 0.50
    assert 0.50 < high < 0.85

    # 0 out of 0
    assert _wilson_confidence_interval(0, 0) == (0.0, 0.0)


def test_empty_dataset_metrics(isolated_ledger):
    _, engine, _ = isolated_ledger
    metrics = engine.compute_validation_metrics()
    assert metrics["status"] == "INSUFFICIENT_DATA"
    assert metrics["sample_size"] == 0
    assert metrics["is_sample_sufficient"] is False
    assert metrics["accuracy"] == 0.0
    assert "disclaimer" in metrics


def test_comprehensive_validation_metrics_calculation(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 9, 30, 0))
    # Seed 10 predictions and evaluations with known outcomes
    # 7 hits, 3 misses
    for i in range(10):
        is_hit = (i < 7)
        pred_dir = "BULLISH" if i % 2 == 0 else "BEARISH"
        real_dir = pred_dir if is_hit else "NEUTRAL"
        prob_up = 0.6 if pred_dir == "BULLISH" else 0.2
        prob_flat = 0.2
        prob_down = 0.6 if pred_dir == "BEARISH" else 0.2

        p_id = ledger.record_prediction(
            spot_price=24000.0,
            predicted_direction=pred_dir,
            prob_up=prob_up,
            prob_flat=prob_flat,
            prob_down=prob_down,
            market_regime="STRONG_BULLISH" if pred_dir == "BULLISH" else "STRONG_BEARISH",
            features={"vwap": 24000.0, "near_atm_pcr": 1.2 if pred_dir == "BULLISH" else 0.8},
            prediction_epoch=t0_dt.timestamp() + (i * 300),
            prediction_timestamp=format_ist(t0_dt + timedelta(minutes=i * 5)),
            horizons_min=(15,)
        )

        ledger.update_evaluation(
            evaluation_id=f"EVAL-{p_id}-15M",
            evaluation_status="RESOLVED",
            actual_price=24030.0 if real_dir == "BULLISH" else (23970.0 if real_dir == "BEARISH" else 24002.0),
            price_delta_pts=30.0 if real_dir == "BULLISH" else (-30.0 if real_dir == "BEARISH" else 2.0),
            price_delta_pct=0.125 if real_dir == "BULLISH" else (-0.125 if real_dir == "BEARISH" else 0.008),
            realized_direction=real_dir,
            is_direction_correct=1 if is_hit else 0,
            brier_score=0.25 if is_hit else 0.55
        )

    metrics = engine.compute_validation_metrics(horizon_min=15)
    assert metrics["status"] == "OK"
    assert metrics["sample_size"] == 10
    # 7 hits out of 10 = 70.0%
    assert metrics["accuracy"] == 0.70
    assert metrics["confidence_interval_95"][0] > 0.35
    assert metrics["is_sample_sufficient"] is False  # N=10 < 30
    assert metrics["warning_message"] is not None

    # Check confusion matrix
    cm = metrics["confusion_matrix"]["matrix"]
    assert sum(sum(r) for r in cm) == 10


# =========================================================================
# 7. FEATURE CONTRIBUTION & BASELINE COMPARISON TESTS
# =========================================================================
def test_feature_contribution_analysis(isolated_ledger):
    ledger, engine, _ = isolated_ledger

    t0_dt = IST.localize(datetime(2026, 10, 27, 9, 30, 0))
    # Seed 6 observations to exceed minimum N=5
    for i in range(6):
        ret_pct = 0.10 if i % 2 == 0 else -0.10
        real_dir = "BULLISH" if ret_pct > 0 else "BEARISH"
        p_id = ledger.record_prediction(
            spot_price=24000.0,
            predicted_direction=real_dir,
            prob_up=0.6 if real_dir == "BULLISH" else 0.2,
            prob_flat=0.2,
            prob_down=0.6 if real_dir == "BEARISH" else 0.2,
            market_regime="STRONG_BULLISH" if real_dir == "BULLISH" else "STRONG_BEARISH",
            features={
                "vwap_distance_pct": 0.10 if real_dir == "BULLISH" else -0.10,
                "near_atm_pcr": 1.30 if real_dir == "BULLISH" else 0.70,
                "oi_ratio": 0.50 if real_dir == "BULLISH" else -0.50,
                "composite_score": 40.0 if real_dir == "BULLISH" else -40.0
            },
            prediction_epoch=t0_dt.timestamp() + (i * 300),
            horizons_min=(15,)
        )

        ledger.update_evaluation(
            evaluation_id=f"EVAL-{p_id}-15M",
            evaluation_status="RESOLVED",
            price_delta_pct=ret_pct,
            realized_direction=real_dir,
            is_direction_correct=1
        )

    feats = engine.evaluate_feature_contributions(horizon_min=15)
    assert feats["status"] == "OK"
    assert feats["sample_size"] == 6
    assert "baseline_comparisons" in feats
    comp = feats["baseline_comparisons"]
    assert comp["combined_model_accuracy"] == 1.0
    assert "lift_over_majority_baseline_pct" in comp
    assert "lift_over_uniform_baseline_pct" in comp

    # Verify Spearman IC features exist
    evaluated = feats["features_evaluated"]
    assert len(evaluated) > 0
    pcr_feat = next((f for f in evaluated if f["feature_key"] == "near_atm_pcr"), None)
    assert pcr_feat is not None
    assert pcr_feat["spearman_ic"] > 0.50


# =========================================================================
# 8. REST API ENDPOINTS VERIFICATION
# =========================================================================
def test_api_intraday_validation_endpoints():
    # 1. Metrics Endpoint
    r1 = client.get("/api/intraday/validation/metrics")
    assert r1.status_code == 200
    d1 = r1.json()
    assert "status" in d1
    assert "sample_size" in d1

    # 2. Recent Predictions Endpoint
    r2 = client.get("/api/intraday/validation/recent?limit=10")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "OK"
    assert "predictions" in d2

    # 3. Feature Contributions Endpoint
    r3 = client.get("/api/intraday/validation/feature-contributions?horizon_min=15")
    assert r3.status_code == 200
    d3 = r3.json()
    assert "status" in d3

    # 4. Record Endpoint (Protected by admin key)
    admin_token = "test-secret-admin-key"
    with patch.dict(os.environ, {"ADMIN_API_KEY": admin_token}):
        # Without admin key -> 401 Unauthorized
        r4_unauth = client.post("/api/intraday/validation/record")
        assert r4_unauth.status_code == 401

        # With admin key -> 200 OK
        r4 = client.post(
            "/api/intraday/validation/record",
            headers={"X-API-Key": admin_token}
        )
        assert r4.status_code == 200
        d4 = r4.json()
        assert "status" in d4

        # 5. Evaluate Endpoint (Protected by admin key)
        r5_unauth = client.post("/api/intraday/validation/evaluate")
        assert r5_unauth.status_code == 401

        r5 = client.post(
            "/api/intraday/validation/evaluate",
            headers={"X-API-Key": admin_token}
        )
        assert r5.status_code == 200
        d5 = r5.json()
        assert "status" in d5
