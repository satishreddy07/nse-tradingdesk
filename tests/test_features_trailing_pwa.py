"""
Automated Tests for:
1. Dynamic Trailing Stop-Loss Engine on Options Strategies.
2. Progressive Web App (PWA) manifest.json and sw.js endpoints.
3. Resilient yfinance import fallback.
"""
from datetime import datetime
import json
from unittest.mock import patch, MagicMock
import pytest
import pytz
from fastapi.testclient import TestClient

from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine
from web.api import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_pwa_endpoints(test_client):
    """Verifies that /manifest.json and /sw.js are served properly with correct MIME types and headers."""
    # 1. Manifest
    res_m = test_client.get("/manifest.json")
    assert res_m.status_code == 200
    assert "application/manifest+json" in res_m.headers["content-type"]
    m_data = res_m.json()
    assert m_data["name"] == "NSE Trading Desk — Quant & Options Terminal"
    assert m_data["short_name"] == "NSE Desk"
    assert m_data["display"] == "standalone"
    assert len(m_data["icons"]) >= 2

    # 2. Service Worker
    res_sw = test_client.get("/sw.js")
    assert res_sw.status_code == 200
    assert "javascript" in res_sw.headers["content-type"]
    assert res_sw.headers.get("service-worker-allowed") == "/"
    assert "CACHE_NAME" in res_sw.text


def test_trailing_stop_loss_progression():
    """
    Tests the 3-Tier Dynamic Trailing Stop-Loss system:
    - Initial: SL = -₹2,000, Target = +₹4,000
    - Tier 1: PnL reaches ₹2,100 (>= 50% target) -> Trailed to Breakeven ₹0
    - Tier 2: PnL reaches ₹3,200 (>= 75% target) -> Trailed to Lock 50% profit (+₹2,000)
    - Exit: PnL drops to ₹1,900 -> Exited via TRAILING_STOP_HIT protecting +₹2,000 profit!
    """
    # Clean any dangling test rows to isolate this test without affecting live active strategies
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM options_strategies WHERE strategy_name LIKE 'Test Trailing%'")
            conn.commit()

    strat_payload = {
        "strategy_name": "Test Trailing Bull Spread",
        "symbol": "NIFTY",
        "expiry_date": "2026-10-15",
        "expiry_type": "WEEKLY",
        "spot_at_entry": 24800.0,
        "net_premium": 25.0,
        "max_profit": 4000.0,
        "max_loss": -2000.0,
        "breakevens": [24825.0],
        "risk_reward": "1:2.0",
        "est_margin": 45000.0,
        "pop_pct": 68.0,
        "legs": [
            {"strike": 24800, "type": "CE", "action": "BUY", "qty": 75, "price": 120.0},
            {"strike": 24900, "type": "CE", "action": "SELL", "qty": 75, "price": 66.7}
        ],
        "target_pnl": 4000.0,
        "stoploss_pnl": -2000.0,
        "auto_trade_mode": "INTRADAY",
        "market_condition": "STRONG_BULLISH_MOMENTUM"
    }

    # 1. Deploy Strategy
    deploy_res = nifty_analytics_engine.deploy_options_strategy(strat_payload)
    strat_id = deploy_res["strategy_id"]

    # 2. Check initial DB state
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            row = conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone()
            assert row is not None
            d = dict(row)
            assert d["status"] == "ACTIVE"
            assert d["target_pnl"] == 4000.0
            assert d["stoploss_pnl"] == -2000.0
            assert d["trailing_stop_pnl"] == -2000.0
            assert d["trailing_status"] == "INACTIVE"

    ist_tz = pytz.timezone("Asia/Kolkata")
    dt_midday = datetime(2026, 10, 15, 11, 30, tzinfo=ist_tz)

    # 3. Simulate PnL advancing to Tier 1: +₹2,200 (>= 50% target of ₹4,000)
    mock_port_tier1 = {
        "active_strategies": [
            {"strategy_id": strat_id, "unrealized_pnl": 2200.0}
        ]
    }
    with patch.object(nifty_analytics_engine, "get_options_portfolio", return_value=mock_port_tier1):
        with patch("engines.nifty_analytics_engine.now_ist", return_value=dt_midday):
            reconcile_res = nifty_analytics_engine.reconcile_active_options_strategies_stops_and_targets()
            assert reconcile_res["exited"] == 0

    # Verify Trailing Stop moved to Breakeven ₹0
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            d = dict(conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone())
            assert d["trailing_stop_pnl"] == 0.0
            assert d["highest_pnl_reached"] == 2200.0
            assert "BREAKEVEN" in d["trailing_status"]

    # 4. Simulate PnL advancing to Tier 2: +₹3,200 (>= 75% target of ₹4,000)
    mock_port_tier2 = {
        "active_strategies": [
            {"strategy_id": strat_id, "unrealized_pnl": 3200.0}
        ]
    }
    with patch.object(nifty_analytics_engine, "get_options_portfolio", return_value=mock_port_tier2):
        with patch("engines.nifty_analytics_engine.now_ist", return_value=dt_midday):
            reconcile_res = nifty_analytics_engine.reconcile_active_options_strategies_stops_and_targets()
            assert reconcile_res["exited"] == 0

    # Verify Trailing Stop moved to Lock 50% (+₹2,000)
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            d = dict(conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone())
            assert d["trailing_stop_pnl"] == 2000.0
            assert d["highest_pnl_reached"] == 3200.0
            assert "50PCT" in d["trailing_status"]

    # 5. Simulate Market Pullback: PnL drops to +₹1,900 (fell below trailed SL of +₹2,000)
    mock_port_exit = {
        "active_strategies": [
            {"strategy_id": strat_id, "unrealized_pnl": 1900.0}
        ]
    }
    with patch.object(nifty_analytics_engine, "get_options_portfolio", return_value=mock_port_exit):
        with patch("engines.nifty_analytics_engine.now_ist", return_value=dt_midday):
            reconcile_res = nifty_analytics_engine.reconcile_active_options_strategies_stops_and_targets()
            assert reconcile_res["exited"] == 1
            assert "TRAILING_STOP_HIT" in reconcile_res["actions"][0]["exit_reason"]

    # Verify strategy is now CLOSED with protected profit
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            d = dict(conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone())
            assert d["status"] == "CLOSED"
            assert "TRAILING_STOP_HIT" in d["exit_reason"]
            conn.execute("DELETE FROM options_strategies WHERE strategy_name LIKE 'Test Trailing%'")
            conn.commit()


def test_safe_yfinance_graceful_fallback():
    """Verifies that breadth and sector internals computation does not crash even if yfinance raises ImportError."""
    with patch.dict("sys.modules", {"yfinance": None}):
        breadth_res, sector_res = nifty_analytics_engine._compute_breadth_and_sector_internals()
        assert isinstance(breadth_res, dict)
        assert isinstance(sector_res, dict)
        assert "ad_ratio" in breadth_res
        assert "score" in breadth_res
