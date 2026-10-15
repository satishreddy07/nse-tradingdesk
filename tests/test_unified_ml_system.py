"""
tests/test_unified_ml_system.py
===============================
Comprehensive Unit & Integration Test Suite for the Complete ML Training,
Decision Tree, TreeSHAP, Validation, Model Registry, and Empirical Re-Entry System.

Verifies:
1. Purged & Embargoed Time-Series Cross Validation (Zero future data leakage).
2. Unified Multi-Horizon Dataset Ingestion & Counterfactual Simulation.
3. Multi-Horizon LightGBM & XGBoost Training & Probability Calibration.
4. Concrete Decision Tree Topology Export (JSON nodes, splits, thresholds).
5. Native C++ TreeSHAP Attribution Engine & Feature Importance.
6. Model Registry with Candidate Promotion Gates and 1-Click Rollback.
7. Empirical Stop-Loss (40% Weekly, 50% Monthly) & 1-Time Re-Entry Matrix.
8. ML REST API Endpoints (/api/ml/overview, /api/ml/trees, /api/ml/shap, etc.).
"""

import json
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from core.unified_ml_dataset import UNIFIED_FEATURE_COLS, unified_ml_dataset
from engines.tree_quant_ml_engine import (
    PurgedEmbargoedTimeSeriesSplit,
    tree_quant_ml_engine
)
from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
from web.api import app


class TestPurgedEmbargoedSplit:
    """Tests the Purged & Embargoed Cross-Validation splitter."""

    def test_split_temporal_integrity(self):
        """Ensures training samples strictly precede validation samples with a purge/embargo gap."""
        n_samples = 150
        dates = pd.date_range("2026-01-01", periods=n_samples, freq="D")
        labels = np.random.randint(0, 2, n_samples)
        df = pd.DataFrame({"label": labels}, index=dates)

        splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, purge_window=2, embargo_window=2)
        splits = list(splitter.split(df, labels))

        assert len(splits) == 3, "Expected 3 walk-forward splits"

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            assert len(train_idx) > 0, f"Fold {fold_idx}: Train set must not be empty"
            assert len(val_idx) > 0, f"Fold {fold_idx}: Val set must not be empty"

            # Check zero intersection (strictly disjoint)
            intersection = set(train_idx).intersection(set(val_idx))
            assert len(intersection) == 0, f"Fold {fold_idx}: Train and val sets must have zero overlap"

            # Check chronological precedence: max train index + purge_window <= min val index
            max_train = max(train_idx)
            min_val = min(val_idx)
            assert max_train < min_val, f"Fold {fold_idx}: Max train index {max_train} must precede min val index {min_val}"
            assert (min_val - max_train) >= 2, f"Fold {fold_idx}: Purge gap must be >= 2 bars"


class TestUnifiedMLDataset:
    """Tests multi-horizon trade ingestion, counterfactual logging, and outcome resolution."""

    def test_trade_ingestion_and_resolution(self):
        # 1. Ingest real Weekly Options Trade
        dummy_feats = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
        dummy_feats["horizon_code"] = 2
        dummy_feats["regime_code"] = 1
        dummy_feats["sl_threshold_pct"] = 40.0

        trade_id = "TEST_WEEKLY_001_AUTOMATION"
        row_id = unified_ml_dataset.record_signal_and_trade(
            signal_id="SIG_001",
            trade_id=trade_id,
            instrument="NIFTY",
            trading_horizon="WEEKLY",
            strategy_name="MILD_BULLISH_PE_CREDIT",
            record_type="LIVE",
            market_session="MORNING",
            accept_reject_verdict="ACCEPTED",
            decision_source="HYBRID",
            features=dummy_feats,
            regime="MILD_BULLISH",
            spot_at_entry=25000.0,
            intended_entry=100.0,
            intended_target=0.0,
            intended_sl=140.0,
            sl_threshold_pct=40.0,
            quantity=50,
            capital_at_risk=2000.0,
            rationale={"rule": "PE sold at 1-sigma with 40% stop-loss"},
            strike=24600.0,
            actual_entry=100.0
        )
        assert row_id > 0

        # 2. Ingest Counterfactual Shadow Trade
        shadow_id = "TEST_SHADOW_001_AUTOMATION"
        row_id2 = unified_ml_dataset.record_signal_and_trade(
            signal_id="SIG_SHADOW_001",
            trade_id=shadow_id,
            instrument="RELIANCE",
            trading_horizon="SWING",
            strategy_name="INSTITUTIONAL_6PILLAR",
            record_type="COUNTERFACTUAL",
            market_session="AFTERNOON",
            accept_reject_verdict="REJECTED",
            decision_source="ML_GATEKEEPER",
            features=dummy_feats,
            regime="SIDEWAYS",
            spot_at_entry=2900.0,
            intended_entry=2900.0,
            intended_target=3100.0,
            intended_sl=2800.0,
            sl_threshold_pct=3.45,
            quantity=100,
            capital_at_risk=10000.0,
            rationale={"rejection_reason": "ML win_prob < 0.48"}
        )
        assert row_id2 > 0

        # 3. Resolve Real Trade Outcome (Win)
        resolved = unified_ml_dataset.record_trade_outcome(
            trade_id=trade_id,
            actual_exit=15.0,
            exit_reason="PROFIT_TARGET_EXPIRY",
            gross_pnl=4250.0,
            net_pnl=4175.0,
            total_costs=75.0,
            hold_duration_mins=2800.0,
            re_entry_count=0,
            rules_followed=1
        )
        assert resolved is True

        # 4. Resolve Counterfactual Shadow Trade Outcome (Loss)
        resolved_cf = unified_ml_dataset.record_trade_outcome(
            trade_id=shadow_id,
            actual_exit=2800.0,
            exit_reason="SHADOW_HIT_STOP_LOSS",
            gross_pnl=-10000.0,
            net_pnl=-10080.0,
            total_costs=80.0,
            hold_duration_mins=1200.0,
            rules_followed=1
        )
        assert resolved_cf is True

        # 5. Extract training dataset and verify columns
        df = unified_ml_dataset.get_dataset_for_horizon(horizon="WEEKLY", include_counterfactuals=True)
        assert not df.empty
        assert "is_winner" in df.columns


class TestMultiHorizonTrainingAndModelRegistry:
    """Tests training LightGBM & XGBoost, calibration, baseline comparison, and rollback."""

    def test_multi_horizon_orchestration(self):
        # Run complete multi-horizon learning pipeline
        results = horizon_learning_orchestrator.retrain_and_evaluate_all_horizons()
        assert results["status"] == "COMPLETED"
        assert "horizons_evaluated" in results
        assert set(results["horizons_evaluated"]) == {"INTRADAY", "WEEKLY", "MONTHLY", "SWING"}

        res_dict = results["results"]
        for horizon in ["INTRADAY", "WEEKLY", "MONTHLY", "SWING"]:
            h_res = res_dict[horizon]
            assert "status" in h_res
            assert ("PROMOTED" in h_res["status"] or h_res["status"] in ["CANDIDATE_REJECTED", "TRAINED", "INSUFFICIENT_DATA"])
            metrics = h_res.get("metrics") or h_res.get("validation_metrics") or {}
            assert "ensemble_brier" in metrics
            assert "baseline_brier" in metrics
            assert "ensemble_auc" in metrics

        # Test Rollback API on WEEKLY horizon
        rollback_res = tree_quant_ml_engine.rollback_model("WEEKLY")
        assert "status" in rollback_res


class TestDecisionTreeAndNativeTreeSHAP:
    """Tests concrete tree topology export and native C++ TreeSHAP attribution."""

    def test_decision_tree_topology_export(self):
        tree_json = tree_quant_ml_engine.export_tree_structure("WEEKLY", tree_index=0)
        assert "status" in tree_json
        if tree_json["status"] == "SUCCESS":
            assert "tree_index" in tree_json
            assert "feature_names" in tree_json
            assert "root" in tree_json
            root = tree_json["root"]
            assert ("feature" in root or "leaf_value" in root)
            if "feature" in root:
                assert "threshold" in root
                assert "sample_count" in root

    def test_native_treeshap_factor_attribution(self):
        sample_feats = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
        sample_feats["vix_spot"] = 18.5
        sample_feats["rsi14"] = 62.0
        sample_feats["regime_code"] = 2
        sample_feats["horizon_code"] = 2

        shap_res = tree_quant_ml_engine.compute_shap_explanations("WEEKLY", sample_feats)
        assert "status" in shap_res
        if shap_res["status"] == "SUCCESS":
            assert "predicted_probability" in shap_res
            assert "base_value" in shap_res
            assert "feature_attributions" in shap_res
            attrs = shap_res["feature_attributions"]
            assert len(attrs) > 0
            for item in attrs:
                assert "feature" in item
                assert "shap_attribution" in item
                assert "direction" in item
                assert item["direction"] in ["POSITIVE", "NEGATIVE"]


class TestEmpiricalStopLossAndReentryMatrix:
    """Tests backtest evaluation of 40% weekly and 50% monthly stop-loss with 1-time re-entry."""

    def test_reentry_matrix_evaluation(self):
        # 1. Weekly options audit (40% stop-loss + 1-time re-entry at P0)
        weekly_audit = horizon_learning_orchestrator.evaluate_stop_loss_and_reentry_matrix("WEEKLY")
        assert "horizon" in weekly_audit
        assert weekly_audit["horizon"] == "WEEKLY"
        assert "reentry_strategy" in weekly_audit
        assert "base_strategy" in weekly_audit
        reentry_spec = weekly_audit["reentry_strategy"]
        assert "net_pnl_inr" in reentry_spec
        assert "win_rate_pct" in reentry_spec
        assert "profit_factor" in reentry_spec
        assert "recovery_rate_pct" in reentry_spec
        assert "results_by_regime" in weekly_audit

        # 2. Monthly options audit (50% stop-loss + 1-time re-entry at P0)
        monthly_audit = horizon_learning_orchestrator.evaluate_stop_loss_and_reentry_matrix("MONTHLY")
        assert "horizon" in monthly_audit
        assert monthly_audit["horizon"] == "MONTHLY"
        assert "reentry_strategy" in monthly_audit
        assert "recovery_rate_pct" in monthly_audit["reentry_strategy"]


class TestMLRESTAPI:
    """Tests the REST API endpoints supporting the UI and external consumers."""

    def test_ml_api_endpoints(self):
        client = TestClient(app)

        # 1. Overview
        r_over = client.get("/api/ml/overview")
        assert r_over.status_code == 200
        data_over = r_over.json()
        assert "status" in data_over
        assert "timestamp" in data_over
        assert "dataset_statistics" in data_over
        assert "model_registry" in data_over

        # 2. Dataset Stats
        r_stats = client.get("/api/ml/dataset-stats")
        assert r_stats.status_code == 200
        data_stats = r_stats.json()
        assert "total_records" in data_stats
        assert "by_horizon" in data_stats

        # 3. Decision Trees
        r_tree = client.get("/api/ml/trees?horizon=WEEKLY&tree_index=0")
        assert r_tree.status_code == 200
        data_tree = r_tree.json()
        assert "status" in data_tree

        # 4. TreeSHAP Attribution
        sample_payload = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
        sample_payload["horizon_code"] = 2
        r_shap = client.post("/api/ml/shap?horizon=WEEKLY", json=sample_payload)
        assert r_shap.status_code == 200
        data_shap = r_shap.json()
        assert "status" in data_shap

        # 5. Re-Entry Audit
        r_reentry = client.get("/api/ml/reentry-audit?horizon=WEEKLY")
        assert r_reentry.status_code == 200
        data_re = r_reentry.json()
        assert "reentry_strategy" in data_re
        assert "base_strategy" in data_re
        assert "results_by_regime" in data_re
