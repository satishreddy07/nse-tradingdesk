import json
import pytest
from engines.nifty_analytics_engine import nifty_analytics_engine

@pytest.fixture(autouse=True)
def setup_active_strategy():
    port = nifty_analytics_engine.get_options_portfolio()
    if not port.get("active_strategies"):
        strat_payload = {
            "strategy_name": "Test Active Strategy",
            "symbol": "NIFTY",
            "expiry_date": "29-Oct-2026",
            "expiry_type": "WEEKLY",
            "spot_at_entry": 24500.0,
            "net_premium": 50.0,
            "max_profit": 3750.0,
            "max_loss": -3750.0,
            "target_pnl": 4000.0,
            "stoploss_pnl": -2000.0,
            "legs": [
                {"strike": 24500, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 100.0}
            ]
        }
        nifty_analytics_engine.deploy_options_strategy(strat_payload)
    yield

def test_options_portfolio_clean_state():
    """Verify that only active strategies remain, realized PnL reflects active strategies, and equity curve starts at Base 200,000."""
    nifty_analytics_engine.reset_options_portfolio(keep_active=True)
    port = nifty_analytics_engine.get_options_portfolio()
    assert port["status"] == "OK"
    summary = port["summary"]
    
    assert summary["base_capital"] == 200000.0
    expected_realized = round(sum(float(s.get("realized_pnl", 0.0)) for s in port["active_strategies"]), 2)
    assert summary["total_realized_pnl"] == expected_realized
    assert summary["closed_count"] == 0
    assert summary["active_count"] >= 1
    assert len(port["active_strategies"]) >= 1
    assert len(port["closed_strategies"]) == 0
    
    # Equity curve should have exactly 1 baseline point for Base Capital 2,00,000
    eq_curve = port["equity_curve"]
    assert len(eq_curve) == 1
    assert eq_curve[0]["equity"] == 200000.0
    assert eq_curve[0]["pnl"] == 0.0
    assert "Base" in eq_curve[0]["label"]

def test_reset_options_portfolio_keeps_active():
    """Verify reset_options_portfolio(keep_active=True) retains active strategies and clears closed."""
    res = nifty_analytics_engine.reset_options_portfolio(keep_active=True)
    assert res["status"] == "OK"
    
    port = nifty_analytics_engine.get_options_portfolio()
    assert port["summary"]["active_count"] >= 1
    assert port["summary"]["closed_count"] == 0
    expected_realized = round(sum(float(s.get("realized_pnl", 0.0)) for s in port["active_strategies"]), 2)
    assert port["summary"]["total_realized_pnl"] == expected_realized
    assert len(port["equity_curve"]) == 1
    assert port["equity_curve"][0]["equity"] == 200000.0

def test_get_all_equity_curves():
    """Verify get_all_equity_curves returns valid dictionary with options, swing, and nifty500 curves."""
    curves = nifty_analytics_engine.get_all_equity_curves()
    assert isinstance(curves, dict)
    assert curves["status"] == "OK"
    assert "options_curve" in curves
    assert len(curves["options_curve"]) >= 1
    assert curves["options_curve"][0]["equity"] == 200000.0
