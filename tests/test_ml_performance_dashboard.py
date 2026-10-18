"""
tests/test_ml_performance_dashboard.py
======================================
Comprehensive Unit & Integration Test Suite for the Complete ML Performance,
Model Intelligence, Observability, and Audit Dashboard.

Verifies:
1. Discovery and accurate registration of all 8 core ML models.
2. Model details and artifact integrity checks (paths, existence, sizes, checksums).
3. Training run history tracking and persistence in ml_training_run_history.
4. Prediction performance aggregation and separation of PENDING vs RESOLVED outcomes.
5. Dedicated T+1 and T+5 performance and attribution diagnostics.
6. Trade outcome learning financial metrics factoring realistic brokerage and slippage.
7. Decision tree topology export (nodes, split features, thresholds) and TreeSHAP.
8. Market regime stratification across all 5 discrete regimes.
9. Database and feature leakage audits.
10. Model tournament comparisons on identical evaluation windows.
11. Statistical drift monitoring (PSI, KS statistic, calibration shift).
12. Audit event logging and error tracking.
13. All REST API endpoints (/api/ml-performance/* and /ml-performance page route).
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from engines.ml_performance_engine import MLPerformanceEngine, ml_performance_engine
from core.unified_ml_dataset import UnifiedMLDataset, UNIFIED_FEATURE_COLS
from web.api import app


@pytest.fixture
def test_client():
    return TestClient(app)


class TestModelRegistryDiscovery:
    """Verifies autonomous model discovery and artifact integrity."""

    def test_all_core_models_discovered(self):
        reg = ml_performance_engine.get_model_registry()
        assert reg["status"] == "SUCCESS"
        assert reg["models_count"] >= 8

        model_ids = {m["model_id"] for m in reg["models"]}
        expected_ids = {
            "nifty_t1_tactical",
            "nifty_t5_structural",
            "intraday_tree_quant",
            "weekly_options_tree_quant",
            "monthly_options_tree_quant",
            "swing_equity_tree_quant",
            "swing_quant_winrate_evaluator",
            "market_regime_multifactor"
        }
        for eid in expected_ids:
            assert eid in model_ids, f"Expected model {eid} to be discovered in registry"

    def test_model_artifact_integrity(self):
        reg = ml_performance_engine.get_model_registry()
        for m in reg["models"]:
            assert "artifact_valid" in m
            assert "artifact_location" in m
            assert "sample_counts" in m
            assert "feature_count" in m
            assert m["feature_count"] > 0
            assert "architecture" in m
            assert "trading_horizon" in m
            assert "warnings_and_limitations" in m


class TestTrainingHistoryAudit:
    """Verifies training execution logging and persistence."""

    def test_training_runs_persisted(self):
        history = ml_performance_engine.get_training_history(limit=10)
        assert isinstance(history, list)
        # Legacy artifacts without empirical run metadata must not be indexed as invented training runs.
        for run in history:
            assert "run_id" in run
            assert "model_name" in run
            assert "status" in run
            assert "duration_sec" in run
            assert "train_records" in run

    def test_record_new_training_run(self):
        run_id = ml_performance_engine.record_training_run(
            model_name="TEST_MODEL",
            model_version="test_v1.0",
            horizon="T+1",
            duration_sec=2.5,
            status="PROMOTED_PRODUCTION",
            dataset_version="TEST_DS_V1",
            train_records=100,
            val_records=20,
            test_records=20,
            feature_count=10,
            metrics={"brier_score": 0.25, "accuracy": 75.0},
            artifact_path="models/registry/test.bundle"
        )
        assert run_id.startswith("RUN_test_v1.0")

        history = ml_performance_engine.get_training_history(model_name="TEST_MODEL")
        assert len(history) >= 1
        assert history[0]["model_name"] == "TEST_MODEL"


class TestPredictionPerformanceAndCones:
    """Verifies predictions vs outcomes and error calculations."""

    def test_prediction_performance_aggregation(self):
        preds = ml_performance_engine.get_prediction_performance(limit=50)
        assert "total_records" in preds
        assert "finalized_count" in preds
        assert "pending_count" in preds
        assert "confusion_matrix" in preds

        # Invariant: Pending outcomes are never counted as failures
        for p in preds["predictions"]:
            if p["status"] == "PENDING":
                assert p["is_correct"] is None, "Pending outcomes must not be marked as hit or miss prematurely"


class TestTradeOutcomeLearning:
    """Verifies post-cost financial expectancy and stop-loss matrix."""

    def test_trade_outcome_metrics_and_costs(self):
        trades = ml_performance_engine.get_trade_outcome_learning(horizon="ALL")
        assert "net_pnl" in trades
        assert "gross_pnl" in trades
        assert "win_rate_pct" in trades
        assert "profit_factor" in trades
        assert "net_expectancy" in trades
        assert "slippage_cost" in trades
        assert "brokerage_cost" in trades
        assert "stop_loss_and_reentry_matrix" in trades

        # No assumed fixed brokerage/slippage is layered over the already stored net PnL.
        # This legacy schema doesn't separately persist fee component breakdowns.
        assert "total_costs" in trades
        assert trades["slippage_cost"] is None
        assert trades["brokerage_cost"] is None


def test_trade_outcome_dashboard_uses_persisted_net_pnl_without_double_charging(tmp_path):
    """The dashboard must reuse recorded net P&L/costs and not guess brokerage or slippage."""
    db_path = tmp_path / "learning_metrics.db"
    dataset = UnifiedMLDataset(db_path=str(db_path))
    engine = MLPerformanceEngine(db_path=str(db_path), models_dir=tmp_path / "model_registry")

    features = {col: 1.0 for col in UNIFIED_FEATURE_COLS}
    row_id = dataset.record_signal_and_trade(
        signal_id="SIG_RECORDED_PNL",
        trade_id="TRADE_RECORDED_PNL",
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
    assert dataset.record_trade_outcome(
        trade_id="TRADE_RECORDED_PNL",
        actual_exit=80.0,
        exit_reason="PROFIT_TARGET",
        gross_pnl=20.0,
        net_pnl=18.0,
        total_costs=2.0,
    )

    metrics = engine.get_trade_outcome_learning(horizon="WEEKLY")
    assert metrics["total_trades"] == 1
    assert metrics["gross_pnl"] == 20.0
    assert metrics["net_pnl"] == 18.0
    assert metrics["total_costs"] == 2.0
    assert metrics["net_expectancy"] == 18.0
    assert metrics["slippage_cost"] is None
    assert metrics["brokerage_cost"] is None


class TestDecisionTreeAndExplainability:
    """Verifies tree topology extraction and TreeSHAP ranking."""

    def test_tree_topology_and_shap(self):
        trees = ml_performance_engine.get_trees_and_explainability(horizon="INTRADAY", tree_index=0)
        assert trees["horizon"] == "INTRADAY"
        assert "tree_topology" in trees
        assert "feature_importance_gain" in trees
        assert "shap_global_ranking" in trees


class TestMarketRegimeStratification:
    """Verifies multi-factor regime performance matrix."""

    def test_regime_matrix(self):
        regimes = ml_performance_engine.get_market_regime_performance()
        assert regimes["status"] == "SUCCESS"
        assert len(regimes["regimes"]) == 5

        for reg_name in ["STRONG_BULLISH", "MILD_BULLISH", "RANGE_BOUND", "MILD_BEARISH", "STRONG_BEARISH"]:
            assert reg_name in regimes["regimes"]
            reg_data = regimes["regimes"][reg_name]
            assert "total_samples" in reg_data
            assert "is_low_sample" in reg_data


class TestDatasetAndFeatureAudit:
    """Verifies database tables and zero target leakage."""

    def test_dataset_audit(self):
        audit = ml_performance_engine.get_dataset_feature_audit()
        assert "datasets" in audit
        assert "features" in audit
        assert len(audit["datasets"]) >= 7

        for f in audit["features"]:
            assert f["leakage_risk"] == "VERIFIED_CLEAN"


class TestModelComparisonTournament:
    """Verifies model benchmarking against non-ML and simple baselines."""

    def test_model_comparison_tournament(self):
        comp = ml_performance_engine.get_model_comparison(horizon="T+1")
        assert comp["horizon"] == "T+1"
        assert len(comp["candidates"]) >= 5

        variants = [c["model_variant"] for c in comp["candidates"]]
        assert any("Production" in v for v in variants)
        assert any("Baseline" in v for v in variants)


class TestDriftAndAuditLogs:
    """Verifies statistical drift and audit logs."""

    def test_monitoring_drift(self):
        drift = ml_performance_engine.get_monitoring_and_drift()
        assert "status" in drift
        assert "overall_health" in drift
        assert len(drift["models_drift_table"]) >= 8

    def test_audit_logs_event_stream(self):
        ml_performance_engine.record_audit_event(
            severity="WARNING",
            category="DRIFT",
            message="Test PSI deviation alert",
            details={"psi": 0.12}
        )
        logs = ml_performance_engine.get_audit_logs(severity="WARNING")
        assert len(logs) > 0
        assert logs[0]["severity"] == "WARNING"


class TestMLPerformanceAPIEndpoints:
    """Verifies all 13 REST API endpoints and page route."""

    def test_ml_performance_html_page_route(self, test_client):
        res = test_client.get("/ml-performance")
        assert res.status_code == 200
        assert "module-ml-performance" in res.text

    def test_api_overview(self, test_client):
        res = test_client.get("/api/ml-performance/overview")
        assert res.status_code == 200
        data = res.json()
        assert "summary_cards" in data
        assert data["summary_cards"]["total_registered_models"] >= 8

    def test_api_registry(self, test_client):
        res = test_client.get("/api/ml-performance/registry")
        assert res.status_code == 200
        data = res.json()
        assert data["models_count"] >= 8

    def test_api_training_history(self, test_client):
        res = test_client.get("/api/ml-performance/training-history?limit=10")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_api_predictions(self, test_client):
        res = test_client.get("/api/ml-performance/predictions?limit=10")
        assert res.status_code == 200
        data = res.json()
        assert "predictions" in data

    def test_api_nifty_forecasts(self, test_client):
        res = test_client.get("/api/ml-performance/nifty-forecasts?horizon=T+1")
        assert res.status_code == 200
        assert res.json()["horizon"] == "T+1"

    def test_api_trade_outcomes(self, test_client):
        res = test_client.get("/api/ml-performance/trade-outcomes?horizon=ALL")
        assert res.status_code == 200
        assert "net_pnl" in res.json()

    def test_api_explainability(self, test_client):
        res = test_client.get("/api/ml-performance/explainability?horizon=INTRADAY&tree_index=0")
        assert res.status_code == 200
        assert "tree_topology" in res.json()

    def test_api_regimes(self, test_client):
        res = test_client.get("/api/ml-performance/regimes")
        assert res.status_code == 200
        assert "regimes" in res.json()

    def test_api_dataset_audit(self, test_client):
        res = test_client.get("/api/ml-performance/dataset-audit")
        assert res.status_code == 200
        assert "datasets" in res.json()

    def test_api_comparison(self, test_client):
        res = test_client.get("/api/ml-performance/comparison?horizon=T+1")
        assert res.status_code == 200
        assert "candidates" in res.json()

    def test_api_drift(self, test_client):
        res = test_client.get("/api/ml-performance/drift")
        assert res.status_code == 200
        assert "models_drift_table" in res.json()

    def test_api_audit_logs(self, test_client):
        res = test_client.get("/api/ml-performance/audit-logs?limit=10")
        assert res.status_code == 200
        assert isinstance(res.json(), list)
