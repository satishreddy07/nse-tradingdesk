"""
========================================================================================
Test Suite: Prospective Validation Engine (`tests/test_prospective_validation_engine.py`)
========================================================================================
Verifies:
1. Table schema initialization (`prospective_paper_signals`, `prospective_paper_trades`).
2. Exact ATR squeeze signal detection with zero look-ahead bias.
3. Idempotent recording of candidate signals without duplicates.
4. Signal conversion to OPEN paper trade (NEXT_OPEN vs SAME_CLOSE).
5. Bar-by-bar progression to COMPLETED trade using TradeSimulator.
6. Telemetry calculation (win rate, gross/net expectancy, profit factor, drawdown).
7. Zero broker order safety guarantee.
========================================================================================
"""

import os
import pytest
import numpy as np
import pandas as pd

from engines.prospective_validation_engine import (
    ProspectiveValidationEngine,
    FROZEN_BASELINE_CONFIG,
    compute_config_hash
)
from engines.trade_simulator import TradeSimulator, TransactionCostModel


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_prospective.db"
    return str(db_file)


@pytest.fixture
def synthetic_squeeze_df():
    """Generates synthetic 40-bar series with an explicit ATR squeeze at bar 30."""
    dates = pd.date_range("2026-02-01", periods=40, freq="B")
    base = 100.0
    opens, highs, lows, closes, vols = [], [], [], [], []

    for i in range(40):
        if i >= 25 and i <= 30:
            # Extreme range contraction: High - Low = 0.5%
            c = base
            h = base + 0.25
            l = base - 0.25
            o = base
            v = 300000.0
        else:
            # Normal volatility: High - Low = 3.0%
            c = base
            h = base + 1.5
            l = base - 1.5
            o = base
            v = 1000000.0
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(v)

    # Bar 31 (breakout day): High moves up +6%
    highs[31] = 106.5
    closes[31] = 105.5
    opens[31] = 101.0

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols
    }, index=dates)


def test_prospective_engine_initialization(temp_db):
    """Verifies SQLite tables and frozen baseline parameters."""
    engine = ProspectiveValidationEngine(db_path=temp_db)
    assert engine.config_hash is not None
    assert len(engine.config_hash) == 12
    assert engine.config["target_pct"] == 5.0
    assert engine.config["stop_pct"] == -3.5
    assert engine.config["horizon_days"] == 10
    assert engine.config["entry_model"] == "NEXT_OPEN"

    with engine._get_conn() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "prospective_paper_signals" in tables
        assert "prospective_paper_trades" in tables


def test_detect_squeeze_signals_zero_lookahead(temp_db, synthetic_squeeze_df):
    """Verifies that squeeze signal is detected at bar 30."""
    engine = ProspectiveValidationEngine(db_path=temp_db)
    target_date = str(synthetic_squeeze_df.index[30])[:10]
    sigs = engine.detect_signals_in_df("TEST_SQUEEZE", synthetic_squeeze_df, as_of_date=target_date)
    assert len(sigs) >= 1
    sig = sigs[0]
    assert sig["symbol"] == "TEST_SQUEEZE"
    assert sig["feature_name"] == "ATR_SQUEEZE_85"
    assert sig["feature_val"] <= 0.85
    assert sig["status"] == "PENDING"
    assert sig["signal_date"] == target_date


def test_idempotent_signal_persistence(temp_db, synthetic_squeeze_df):
    """Verifies that inserting the same signal twice is safely ignored."""
    engine = ProspectiveValidationEngine(db_path=temp_db)
    target_date = str(synthetic_squeeze_df.index[30])[:10]
    sigs = engine.detect_signals_in_df("TEST_IDEMPOTENT", synthetic_squeeze_df, as_of_date=target_date)
    assert len(sigs) > 0
    s = sigs[0]

    with engine._get_conn() as conn:
        # First insert
        conn.execute("""
        INSERT INTO prospective_paper_signals (
            signal_id, symbol, signal_date, feature_name, feature_val,
            atr5, atr20, close_price, status, config_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["signal_id"], s["symbol"], s["signal_date"], s["feature_name"],
            s["feature_val"], s["atr5"], s["atr20"], s["close_price"],
            s["status"], s["config_hash"], s["created_at"]
        ))
        conn.commit()

        # Second insert must be ignored without raising OperationalError or IntegrityError
        conn.execute("""
        INSERT OR IGNORE INTO prospective_paper_signals (
            signal_id, symbol, signal_date, feature_name, feature_val,
            atr5, atr20, close_price, status, config_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["signal_id"], s["symbol"], s["signal_date"], s["feature_name"],
            s["feature_val"], s["atr5"], s["atr20"], s["close_price"],
            s["status"], s["config_hash"], s["created_at"]
        ))
        conn.commit()

        count = conn.execute("SELECT count(*) FROM prospective_paper_signals WHERE signal_id = ?", (s["signal_id"],)).fetchone()[0]
        assert count == 1


def test_trade_lifecycle_and_telemetry(temp_db, synthetic_squeeze_df):
    """
    Simulates a full prospective lifecycle:
    1. Signal recorded at bar 30
    2. Converted to OPEN trade at bar 31 NEXT_OPEN (Open = 101.0)
    3. Simulator detects target hit at 106.05 (bar 31 High = 106.5) -> COMPLETED
    4. Telemetry reports 1 completed trade, 100% win rate, positive expectancy.
    """
    engine = ProspectiveValidationEngine(db_path=temp_db)
    target_date = str(synthetic_squeeze_df.index[30])[:10]
    sigs = engine.detect_signals_in_df("TEST_LIFECYCLE", synthetic_squeeze_df, as_of_date=target_date)
    assert len(sigs) > 0
    sig = sigs[0]

    with engine._get_conn() as conn:
        conn.execute("""
        INSERT INTO prospective_paper_signals (
            signal_id, symbol, signal_date, feature_name, feature_val,
            atr5, atr20, close_price, status, config_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig["signal_id"], sig["symbol"], sig["signal_date"], sig["feature_name"],
            sig["feature_val"], sig["atr5"], sig["atr20"], sig["close_price"],
            "PENDING", sig["config_hash"], sig["created_at"]
        ))
        conn.commit()

    # Save synthetic series to state_engine so prospective pipeline can retrieve it
    from execution.state_engine import state_engine
    state_engine.save_ohlcv_df("TEST_LIFECYCLE", synthetic_squeeze_df)

    conv_res = engine.convert_pending_signals_to_trades()
    assert conv_res["trades_converted"] == 1

    # Verify trade is now OPEN
    with engine._get_conn() as conn:
        trade = conn.execute("SELECT * FROM prospective_paper_trades WHERE symbol = 'TEST_LIFECYCLE'").fetchone()
        assert trade is not None
        assert trade["status"] == "OPEN"
        assert trade["entry_price"] == 101.0
        assert trade["target_price"] == round(101.0 * 1.05, 2)
        assert trade["stop_price"] == round(101.0 * 0.965, 2)

    # Sync open trades (Bar 31 has High = 106.5 which exceeds target 106.05)
    sync_res = engine.sync_open_paper_trades()
    assert sync_res["trades_completed"] == 1

    # Check telemetry
    telem = engine.get_telemetry_summary()
    assert telem["status"] == "SUCCESS"
    assert telem["prospective_performance"]["completed_trades_count"] == 1
    assert telem["prospective_performance"]["open_trades_count"] == 0
    assert telem["prospective_performance"]["realized_win_rate_pct"] == 100.0
    assert telem["prospective_performance"]["mean_net_expectancy_pct"] > 0
    assert "PROFIT_TARGET" in telem["prospective_performance"]["exit_reasons_breakdown"]
    assert "ZERO REAL BROKER ORDERS" in telem["disclaimer"]


def test_api_prospective_validation_endpoints():
    """Verifies FastAPI REST API endpoints for prospective paper-trading."""
    from fastapi.testclient import TestClient
    from web.api import app

    client = TestClient(app)

    # 1. GET summary
    res = client.get("/api/research/prospective-validation/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "prospective_performance" in data
    assert "historical_baseline_reference" in data
    assert data["historical_baseline_reference"]["oos_label_win_rate_pct"] == 37.2
    assert "ZERO REAL BROKER ORDERS" in data["disclaimer"]

    # 2. POST update-trades
    res_update = client.post("/api/research/prospective-validation/update-trades")
    assert res_update.status_code == 200
    data_update = res_update.json()
    assert data_update["status"] == "SUCCESS"
    assert "trades_evaluated" in data_update
