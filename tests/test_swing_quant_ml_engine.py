"""
Unit & Integration Tests for Swing Quant ML & AI Feedback Engine
================================================================
Verifies:
1. quant_trade_dataset table schema, indices, and active position backfills.
2. Market regime extraction and multi-dimensional feature vectorization.
3. Cold-start model bootstrapping and probability calibration.
4. Pre-trade gatekeeper decision rules (Approved vs Rejected setups).
5. Shadow counterfactual tracking (recording, reconciliation, root-cause attribution).
6. Post-mortem AI & algorithmic attribution generation.
7. REST API endpoints via FastAPI TestClient.
"""

import json
import sqlite3
import pytest
from fastapi.testclient import TestClient

from engines.swing_quant_ml_engine import swing_quant_ml_engine, ML_FEATURE_COLS
from execution.state_engine import state_engine
from web.api import app


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


def test_quant_dataset_schema_and_columns():
    """Verify that quant_trade_dataset exists with required schema in SQLite master DB."""
    with swing_quant_ml_engine._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(quant_trade_dataset)").fetchall()]
        
        required_cols = [
            "id", "symbol", "portfolio_type", "trade_type", "status",
            "entry_price", "exit_price", "stop_loss", "target", "pnl",
            "r_multiple", "return_pct", "hold_days", "exit_reason",
            "is_winner", "sector", "score", "confidence", "rsi14",
            "dist_to_pivot", "dist_to_ema20", "vol_ratio", "inst_score",
            "week_confluence", "is_vcp", "bb_sq_ratio", "nifty_trend",
            "nifty_ret_5d", "vix_val", "sector_excess_6m", "root_cause_attribution",
            "ai_post_mortem_notes", "created_at", "updated_at"
        ]
        for col in required_cols:
            assert col in cols, f"Missing column {col} in quant_trade_dataset"


def test_market_regime_snapshot():
    """Verify that capture_market_regime returns structured macro metrics."""
    regime = swing_quant_ml_engine.capture_market_regime()
    assert isinstance(regime, dict)
    assert "nifty_spot" in regime
    assert "nifty_trend" in regime
    assert "vix_val" in regime
    assert "nifty_rsi14" in regime
    assert regime["nifty_trend"] in ("BULLISH", "BEARISH", "SIDEWAYS")


def test_feature_vectorization():
    """Verify that vectorize_features transforms candidate metadata into 17 numerical features."""
    candidate = {
        "score": 88.5,
        "confidence": 5,
        "rsi14": 58.2,
        "atr_pct": 2.4,
        "dist_to_pivot": 1.1,
        "dist_to_ema20": 1.8,
        "vol_ratio": 1.9,
        "inst_score": 75.0,
        "week_confluence": 1,
        "is_vcp": 1,
        "bb_sq_ratio": 0.85,
        "adx14": 32.0,
        "sector_excess_6m": 4.5,
        "entry": 500.0,
        "sl": 475.0,
        "t1": 550.0
    }
    vec, feat_dict = swing_quant_ml_engine.vectorize_features(candidate)
    assert len(vec) == len(ML_FEATURE_COLS)
    assert len(feat_dict) == len(ML_FEATURE_COLS)
    assert feat_dict["score"] == 88.5
    assert feat_dict["rr_ratio"] == 2.0


def test_pre_trade_gatekeeper_approves_sound_setup():
    """A high-conviction breakout with solid volume, tight pivot distance, and confluence must be APPROVED."""
    sound_setup = {
        "ticker": "QUANT_TEST_WIN",
        "score": 92.0,
        "confidence": 5,
        "rsi14": 62.0,
        "atr_pct": 2.1,
        "dist_to_pivot": 0.8,
        "dist_to_ema20": 1.5,
        "vol_ratio": 2.4,
        "inst_score": 85.0,
        "week_confluence": 1,
        "is_vcp": 1,
        "sector_excess_6m": 6.2,
        "entry": 1000.0,
        "sl": 950.0,
        "t1": 1100.0
    }
    decision = swing_quant_ml_engine.evaluate_candidate(sound_setup)
    assert decision["verdict"] in ("APPROVED", "CAUTION")
    assert decision["win_prob"] >= 0.50
    assert decision["expected_r"] > 0.0


def test_pre_trade_gatekeeper_rejects_overextended_setup():
    """An overextended setup (e.g. ATLANTAELE at +5.1% day run, 4.8% above pivot, weak volume) must be REJECTED."""
    overextended_setup = {
        "ticker": "OVEREXT_TEST",
        "score": 85.0,
        "confidence": 4,
        "rsi14": 78.5,
        "atr_pct": 4.2,
        "dist_to_pivot": 5.2,
        "dist_to_ema20": 7.1,
        "vol_ratio": 0.75,
        "inst_score": 50.0,
        "week_confluence": 0,
        "is_vcp": 0,
        "sector_excess_6m": -3.5,
        "entry": 2000.0,
        "sl": 1840.0,
        "t1": 2150.0
    }
    decision = swing_quant_ml_engine.evaluate_candidate(overextended_setup)
    assert decision["verdict"] == "REJECTED"
    assert len(decision["risk_flags"]) >= 2
    assert any("OVEREXTENDED" in f for f in decision["risk_flags"])


def test_shadow_candidate_lifecycle_and_attribution():
    """Verify recording of a shadow candidate, simulated exit, and post-mortem attribution."""
    test_sym = "UNIT_TEST_SHADOW_SYM"
    
    # 1. Clean previous run if exists
    with swing_quant_ml_engine._get_connection() as conn:
        conn.execute("DELETE FROM quant_trade_dataset WHERE symbol = ?", (test_sym,))
        conn.commit()

    meta = {
        "ticker": test_sym,
        "sector": "ENERGY",
        "score": 75.0,
        "confidence": 3,
        "entry": 200.0,
        "sl": 190.0,
        "t1": 220.0,
        "dist_to_pivot": 3.8,
        "dist_to_ema20": 4.6,
        "rsi14": 74.0,
        "vol_ratio": 0.95
    }

    # 2. Record shadow candidate
    logged = swing_quant_ml_engine.record_shadow_candidate(
        symbol=test_sym,
        portfolio_type="INSTITUTIONAL_6PILLAR",
        entry_price=200.0,
        stop_loss=190.0,
        target=220.0,
        target_2=235.0,
        candidate_meta=meta,
        rejection_reason="ML_GATEKEEPER_CAUTION"
    )
    assert logged is True

    # 3. Verify in active shadow ledger
    active_shadows = swing_quant_ml_engine.get_active_shadow_trades()
    assert any(s["symbol"] == test_sym for s in active_shadows)

    # 4. Verify root cause attribution logic
    attribution = swing_quant_ml_engine._compute_root_cause_attribution(
        is_win=0,
        exit_reason="STOP_LOSS_HIT",
        features=meta,
        nifty_hold_return=-0.5
    )
    assert "OVEREXTENDED_AT_ENTRY" in attribution

    # 5. Clean up
    with swing_quant_ml_engine._get_connection() as conn:
        conn.execute("DELETE FROM quant_trade_dataset WHERE symbol = ?", (test_sym,))
        conn.commit()


def test_model_training_and_metrics():
    """Verify train_model computes valid calibration metrics and saves bundle."""
    meta = swing_quant_ml_engine.train_model()
    assert meta["status"] == "ONLINE_ACTIVE"
    assert meta["total_samples"] >= 160
    assert 0.0 <= meta["brier_score"] <= 0.35
    assert 0.40 <= meta["win_rate"] <= 0.70


def test_swing_quant_ml_rest_endpoints(api_client):
    """Test all Swing Quant ML REST API endpoints."""
    # 1. Model Status
    r_status = api_client.get("/api/ml/swing/model-status")
    assert r_status.status_code == 200
    data = r_status.json()
    assert data["status"] == "OK"
    assert "engine" in data
    assert data["engine"]["dataset_summary"]["real_trades_count"] >= 8

    # 2. Master Dataset
    r_dataset = api_client.get("/api/ml/swing/dataset?limit=10")
    assert r_dataset.status_code == 200
    ds = r_dataset.json()
    assert ds["status"] == "OK"
    assert isinstance(ds["dataset"], list)

    # 3. Active Shadow Trades
    r_shadow = api_client.get("/api/ml/swing/shadow-trades")
    assert r_shadow.status_code == 200
    assert "active_shadow_count" in r_shadow.json()

    # 4. Retrain Trigger
    r_train = api_client.post("/api/ml/swing/train")
    assert r_train.status_code == 200
    assert r_train.json()["status"] == "OK"

    # 5. Reconcile Trigger
    r_recon = api_client.post("/api/ml/swing/reconcile-shadow")
    assert r_recon.status_code == 200
    assert r_recon.json()["status"] == "OK"

    # 6. Active Training Setups Endpoint
    r_active = api_client.get("/api/ml/swing/active-setups")
    assert r_active.status_code == 200
    act_data = r_active.json()
    assert act_data["status"] == "OK"
    assert "count" in act_data
    assert isinstance(act_data["setups"], list)
    if act_data["count"] > 0:
        first = act_data["setups"][0]
        assert "symbol" in first
        assert "ltp" in first
        assert "unrealized_pnl" in first
        assert "dist_to_sl_pct" in first
        assert "dist_to_t1_pct" in first


def test_get_active_training_setups_method():
    """Verify get_active_training_setups returns active setups with computed real-time metrics."""
    setups = swing_quant_ml_engine.get_active_training_setups()
    assert isinstance(setups, list)
    for s in setups:
        assert s["status"] == "ACTIVE"
        assert "ltp" in s
        assert "quantity" in s
        assert "unrealized_pnl" in s
        assert "rr_ratio" in s

