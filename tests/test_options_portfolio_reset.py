import pytest
from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine

def test_options_portfolio_clean_state():
    """Verify that only active strategies remain, realized PnL is 0, and equity curve starts at Base 200,000."""
    nifty_analytics_engine.reset_options_portfolio(keep_active=True)
    port = nifty_analytics_engine.get_options_portfolio()
    assert port["status"] == "OK"
    summary = port["summary"]
    
    assert summary["base_capital"] == 200000.0
    assert summary["total_realized_pnl"] == 0.0
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
    assert port["summary"]["total_realized_pnl"] == 0.0
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

