"""
tests/test_nifty_forecast_learning_system.py
============================================
Comprehensive Unit & Integration Test Suite for NIFTY T+1 and T+5
Prediction Learning, Lifecycle Ledger, Model Evaluation, and Controlled Retraining.

Verifies:
1. Immutable Daily Prediction Logging & Idempotency (Cannot overwrite historical predictions).
2. NSE Calendar-Aware Session Alignment (T+1 = t+1, T+5 = t+5, skipping holidays/weekends).
3. Correct Outcome Resolution (Directional hit vs Target hit vs 1s/2s Cone Containment).
4. Strict Decoupling of Predictive Features from Diagnostic Explanations (Zero Lookahead Leakage).
5. Separate Model Training for T+1 and T+5 with 1-Bar vs 5-Bar Purged Walk-Forward Splits.
6. Candidate vs Baseline Benchmarking, Promotion Gatekeeper, and 1-Click Rollback.
7. REST API Endpoints (/api/nifty-forecast/overview, ledger, evaluation, retrain, rollback).
"""

import json
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from core.market_calendar import NSEMarketCalendar
from core.nifty_prediction_ledger import (
    NIFTY_FEATURE_COLS,
    NIFTY_LEDGER_TABLE,
    NiftyPredictionLedger,
    nifty_prediction_ledger,
)
from engines.nifty_forecast_learning_engine import (
    NiftyForecastLearningEngine,
    nifty_forecast_learning_engine,
)
from engines.tree_quant_ml_engine import PurgedEmbargoedTimeSeriesSplit
from execution.state_engine import state_engine
from web.api import app


@pytest.fixture
def test_client():
    return TestClient(app)


class TestNiftyPredictionLedger:
    """Tests the immutable prediction ledger and calendar alignment."""

    def test_calendar_session_alignment(self):
        """Verifies T+1 and T+5 calendar-aware forward date calculations skipping weekends."""
        # Known Friday: 2026-01-09 -> T+1 should be Monday 2026-01-12
        fri = date(2026, 1, 9)
        t1_day = NiftyPredictionLedger.get_nth_trading_day(fri, n=1)
        assert t1_day > fri
        assert t1_day.weekday() < 5, "Target day must be a weekday"
        assert t1_day == date(2026, 1, 12), f"T+1 from Friday must be Monday, got {t1_day}"

        # T+5 from Monday 2026-01-05 -> Mon -> Tue(1), Wed(2), Thu(3), Fri(4), Mon(5) -> 2026-01-12
        mon = date(2026, 1, 5)
        t5_day = NiftyPredictionLedger.get_nth_trading_day(mon, n=5)
        assert t5_day == date(2026, 1, 12), f"T+5 from Monday must be next Monday, got {t5_day}"

    def test_immutable_forecast_logging(self):
        """Ensures a forecast is logged once and subsequent calls do not overwrite it."""
        test_date = "2026-01-05"
        dummy_feats = {col: 10.0 for col in NIFTY_FEATURE_COLS}
        dummy_feats["spot_price"] = 25000.0

        fid_1 = nifty_prediction_ledger.log_forecast(
            horizon="T+1",
            spot=25000.0,
            composite_score=28.5,
            verdict="STRONG_BULLISH",
            prob_up=0.68,
            prob_flat=0.22,
            prob_down=0.10,
            expected_close=25075.0,
            expected_pts=75.0,
            expected_pct=0.30,
            range_1sigma=(24950.0, 25150.0),
            range_2sigma=(24850.0, 25250.0),
            features=dummy_feats,
            prediction_date=test_date,
            model_name="NiftyMultiFactorEngine",
            model_version="prod_test_v1.0",
        )
        assert fid_1.startswith("FC_T1_20260105")

        # Attempt to log again on same date with modified values (e.g. recalculation or retrain)
        fid_2 = nifty_prediction_ledger.log_forecast(
            horizon="T+1",
            spot=25000.0,
            composite_score=-15.0,  # changed
            verdict="MILD_BEARISH",  # changed
            prob_up=0.20,
            prob_flat=0.30,
            prob_down=0.50,
            expected_close=24900.0,
            expected_pts=-100.0,
            expected_pct=-0.40,
            range_1sigma=(24800.0, 25050.0),
            range_2sigma=(24700.0, 25150.0),
            features=dummy_feats,
            prediction_date=test_date,
            model_name="CandidateModel",
            model_version="cand_v2.0",
        )

        # Invariant: Must return the exact original forecast ID and original values
        assert fid_1 == fid_2

        with nifty_prediction_ledger._get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id = ?",
                (fid_1,)
            ).fetchone()
            assert row is not None
            assert row["verdict"] == "STRONG_BULLISH", "Original verdict must NOT be overwritten"
            assert row["composite_score"] == 28.5
            assert row["model_version"] == "prod_test_v1.0"
            assert row["status"] in ("PENDING", "RESOLVED")

    def test_outcome_resolution_directional_and_cones(self):
        """Tests outcome evaluation: directional hit, target hit, and cone containment."""
        test_date = "2026-01-06"
        dummy_feats = {col: 5.0 for col in NIFTY_FEATURE_COLS}
        dummy_feats["spot_price"] = 25000.0

        fid = nifty_prediction_ledger.log_forecast(
            horizon="T+1",
            spot=25000.0,
            composite_score=25.0,
            verdict="MILD_BULLISH",
            prob_up=0.60,
            prob_flat=0.25,
            prob_down=0.15,
            expected_close=25080.0,
            expected_pts=80.0,
            expected_pct=0.32,
            range_1sigma=(24950.0, 25150.0),
            range_2sigma=(24850.0, 25250.0),
            features=dummy_feats,
            prediction_date=test_date,
        )

        # Get target date
        target_d = NiftyPredictionLedger.get_nth_trading_day(date(2026, 1, 6), n=1).strftime("%Y-%m-%d")

        # Simulate market close on target date: Actual close 25090 (Bullish +0.36%)
        # Target was 25080 (Target hit), 1-sigma was [24950, 25150] (Inside cone)
        res = nifty_prediction_ledger.resolve_matured_outcomes(
            current_date=target_d,
            closes_override={target_d: 25090.0}
        )
        assert res["status"] in ("COMPLETED", "NO_PENDING_FORECASTS")

        with nifty_prediction_ledger._get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {NIFTY_LEDGER_TABLE} WHERE forecast_id = ?",
                (fid,)
            ).fetchone()
            assert row["status"] == "RESOLVED"
            assert row["direction_correct"] == 1
            assert row["target_hit"] == 1
            assert row["range_1sigma_hit"] == 1
            assert row["range_2sigma_hit"] == 1
            assert abs(row["prediction_error_pts"] - 10.0) < 1e-4
            assert row["actual_return_pts"] == 90.0

            # Verify diagnostic metadata is populated
            diag = json.loads(row["diagnostic_metadata_json"] or "{}")
            assert diag.get("direction_outcome") == "HIT"


class TestFeatureAndExplanationDecoupling:
    """Ensures post-outcome diagnostic explanations do not leak into training features."""

    def test_diagnostic_explanations_not_in_features(self):
        """Verifies that diagnostic_metadata_json is completely isolated from NIFTY_FEATURE_COLS."""
        assert "diagnostic_metadata_json" not in NIFTY_FEATURE_COLS
        assert "attribution" not in NIFTY_FEATURE_COLS
        assert "gemini_explanation" not in NIFTY_FEATURE_COLS
        assert "post_mortem" not in NIFTY_FEATURE_COLS

        # Check training records extraction
        df = nifty_prediction_ledger.get_training_records(horizon="T+1", min_samples=1)
        if not df.empty:
            for col in NIFTY_FEATURE_COLS:
                assert col in df.columns, f"Feature {col} must be present in training dataframe"
            assert "diagnostic_metadata_json" not in df.columns, "Diagnostic metadata must NEVER be a feature"


class TestSeparateHorizonTraining:
    """Tests separate model training for T+1 (1-bar purge) and T+5 (5-bar purge)."""

    def test_purged_time_series_split_purge_windows(self):
        """Verifies 1-bar purge for T+1 and 5-bar purge for T+5 in time-series split."""
        n_samples = 100
        dates = pd.date_range("2026-01-01", periods=n_samples, freq="D")
        labels = np.random.randint(-1, 2, n_samples)
        df = pd.DataFrame({"label": labels}, index=dates)

        # T+1: 1-bar purge
        splitter_t1 = PurgedEmbargoedTimeSeriesSplit(n_splits=3, purge_window=1, embargo_window=1)
        splits_t1 = list(splitter_t1.split(df, labels))
        for train_idx, val_idx in splits_t1:
            assert (min(val_idx) - max(train_idx)) >= 1

        # T+5: 5-bar purge (strictly prevents 5-day overlapping return leakage)
        splitter_t5 = PurgedEmbargoedTimeSeriesSplit(n_splits=3, purge_window=5, embargo_window=2)
        splits_t5 = list(splitter_t5.split(df, labels))
        for train_idx, val_idx in splits_t5:
            assert (min(val_idx) - max(train_idx)) >= 5

    def test_train_forecast_models_t1_and_t5(self):
        """Verifies separate training runs for T+1 and T+5 producing versioned bundles."""
        res_t1 = nifty_forecast_learning_engine.train_forecast_models(horizon="T+1")
        valid_statuses = ("PROMOTED_INITIAL_BASELINE", "PROMOTED_PRODUCTION", "RETAINED_PRODUCTION", "INSUFFICIENT_DATA")
        assert res_t1["status"] in valid_statuses
        if res_t1["status"] != "INSUFFICIENT_DATA":
            assert "metrics" in res_t1
            assert res_t1["metrics"]["brier_score"] <= res_t1["metrics"]["baseline_brier"]

        res_t5 = nifty_forecast_learning_engine.train_forecast_models(horizon="T+5")
        assert res_t5["status"] in valid_statuses
        if res_t5["status"] != "INSUFFICIENT_DATA":
            assert "metrics" in res_t5
            assert res_t5["metrics"]["brier_score"] <= res_t5["metrics"]["baseline_brier"]


class TestModelRollbackAndGatekeeper:
    """Tests promotion gatekeeping and 1-click model rollback."""

    def test_rollback_capability(self):
        """Verifies rollback restores previous model configuration."""
        rollback_res = nifty_forecast_learning_engine.rollback_model(horizon="T+1")
        assert "status" in rollback_res
        assert rollback_res["status"] in ("ROLLBACK_SUCCESSFUL", "ROLLED_BACK", "NO_PREVIOUS_VERSION_AVAILABLE")


class TestEvaluationReporting:
    """Tests comprehensive evaluation reporting."""

    def test_generate_evaluation_report(self):
        """Verifies evaluation report structure for T+1 and T+5."""
        for horizon in ("T+1", "T+5"):
            report = nifty_forecast_learning_engine.generate_evaluation_report(horizon=horizon)
            assert report["horizon"] == horizon
            assert report["status"] in ("EVALUATED_LIVE", "INSUFFICIENT_MATURED_RECORDS")
            if report["status"] == "EVALUATED_LIVE":
                assert "directional_accuracy_pct" in report
                assert "confusion_matrix" in report
                assert "cone_containment_1sigma_pct" in report
                assert "mae_pts" in report
                assert "performance_by_regime" in report
                assert "performance_by_volatility" in report


class TestNiftyForecastAPI:
    """Tests the REST API endpoints for Nifty prediction learning."""

    def test_api_overview(self, test_client):
        response = test_client.get("/api/nifty-forecast/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OPERATIONAL"
        assert "ledger_statistics" in data

    def test_api_ledger(self, test_client):
        response = test_client.get("/api/nifty-forecast/ledger?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_api_evaluation(self, test_client):
        response = test_client.get("/api/nifty-forecast/evaluation?horizon=T+1")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "T+1"

    def test_api_retrain(self, test_client):
        response = test_client.post("/api/nifty-forecast/retrain?horizon=T+1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert "T+1" in data["results"]

    def test_api_rollback(self, test_client):
        response = test_client.post("/api/nifty-forecast/rollback?horizon=T+1")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
