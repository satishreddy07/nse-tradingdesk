"""
tests/test_options_pricing_and_chain_sync.py
============================================
Comprehensive test suite verifying option pricing integrity, option chain LTP synchronization,
full strikes persistence across windows, key-type normalization, and portfolio MTM reconciliation.
"""

import pytest
import json
import sqlite3
from unittest.mock import patch, MagicMock
from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine


def test_analyze_single_expiry_chain_returns_all_strikes():
    """Verify _analyze_single_expiry_chain returns both display slice and full strikes_data."""
    # Generate 40 strikes around 24500 (from 23500 to 25500)
    raw_rows = []
    for s in range(23500, 25550, 50):
        raw_rows.append({
            "strikePrice": s,
            "CE": {"openInterest": 1000, "changeinOpenInterest": 100, "totalTradedVolume": 500, "impliedVolatility": 13.5, "lastPrice": max(0.5, 24500 - s + 50.0), "change": 1.0, "pChange": 2.0},
            "PE": {"openInterest": 1200, "changeinOpenInterest": 150, "totalTradedVolume": 600, "impliedVolatility": 14.0, "lastPrice": max(0.5, s - 24500 + 50.0), "change": -1.0, "pChange": -2.0}
        })
    
    res = nifty_analytics_engine._analyze_single_expiry_chain(raw_rows, 24500.0, "24-Oct-2026", "2026-10-16 09:15:00")
    assert res["status"] == "OK"
    assert "strikes" in res
    assert "all_strikes" in res
    
    # Display strikes is bounded to ±12 (up to 25 strikes)
    assert len(res["strikes"]) <= 25
    # all_strikes preserves all 41 strikes
    assert len(res["all_strikes"]) == len(raw_rows)
    assert res["all_strikes"][0]["strike"] == 23500
    assert res["all_strikes"][-1]["strike"] == 25500


def test_get_deep_option_chain_normalization():
    """Verify get_deep_option_chain guarantees spot, expiry_date, and atm_iv."""
    mock_chain = {
        "status": "OK",
        "spot_price": 24850.5,
        "weekly": {
            "expiry": "24-Oct-2026",
            "strikes": [{"strike": 24850, "ce_ltp": 120.0, "pe_ltp": 115.0}],
            "all_strikes": [{"strike": 24850, "ce_ltp": 120.0, "pe_ltp": 115.0}],
            "volatility_and_gex": {"atm_iv": 13.8}
        },
        "monthly": {
            "expiry": "28-Nov-2026",
            "strikes": [{"strike": 24850, "ce_ltp": 250.0, "pe_ltp": 240.0}],
            "all_strikes": [{"strike": 24850, "ce_ltp": 250.0, "pe_ltp": 240.0}],
            "volatility_and_gex": {"atm_iv": 14.2}
        }
    }
    with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value=mock_chain):
        deep = nifty_analytics_engine.get_deep_option_chain()
        assert deep["spot"] == 24850.5
        assert deep["weekly"]["expiry_date"] == "24-Oct-2026"
        assert deep["weekly"]["atm_iv"] == 13.8
        assert deep["monthly"]["expiry_date"] == "28-Nov-2026"
        assert deep["monthly"]["atm_iv"] == 14.2


def test_get_options_portfolio_with_far_otm_strikes_and_type_normalization():
    """Verify get_options_portfolio correctly prices far OTM strikes beyond display ±12 and handles int/str/float keys."""
    # Far OTM strikes: 23200 (PE wing) and 25600 (CE wing)
    mock_all_strikes = [
        {"strike": 23200, "ce_ltp": 1650.0, "pe_ltp": 4.5},
        {"strike": 24800, "ce_ltp": 110.0, "pe_ltp": 105.0},
        {"strike": 25600, "ce_ltp": 8.5, "pe_ltp": 780.0},
    ]
    mock_display_strikes = [
        {"strike": 24800, "ce_ltp": 110.0, "pe_ltp": 105.0}
    ]
    mock_chain = {
        "status": "OK",
        "spot_price": 24810.0,
        "weekly": {
            "expiry": "24-Oct-2026",
            "strikes": mock_display_strikes,
            "all_strikes": mock_all_strikes,
            "volatility_and_gex": {"atm_iv": 13.5}
        },
        "monthly": {
            "expiry": "28-Nov-2026",
            "strikes": mock_display_strikes,
            "all_strikes": mock_all_strikes,
            "volatility_and_gex": {"atm_iv": 14.0}
        }
    }

    # Insert a temporary active test strategy with string & float strike keys
    test_strat_id = f"OPT-TEST-SYNC-{pytest.__file__[-6:]}"
    test_legs = [
        {"strike": "25600", "type": "CE", "action": "SELL", "qty": 75, "entry_price": 25.0}, # decayed to 8.5 -> +16.5 pts * 75 = +1237.5
        {"strike": 23200.0, "type": "PE", "action": "SELL", "qty": 75, "entry_price": 10.0}, # decayed to 4.5 -> +5.5 pts * 75 = +412.5
    ]
    
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM options_strategies WHERE strategy_id = ?", (test_strat_id,))
            conn.execute("""
                INSERT INTO options_strategies (
                    strategy_id, strategy_name, symbol, expiry_date, expiry_type,
                    created_at, spot_at_entry, net_premium, max_profit, max_loss,
                    breakevens, risk_reward, est_margin, pop_pct, legs_json, status, realized_pnl,
                    target_pnl, stoploss_pnl, auto_trade_mode, market_condition
                ) VALUES (?, 'Test OTM Sync Strangle', 'NIFTY', '24-Oct-2026', 'WEEKLY',
                    '2026-10-16 09:15:00', 24800.0, 2625.0, 2625.0, 5000.0,
                    '[]', 'Credit', 50000.0, 65.0, ?, 'ACTIVE', 0.0,
                    1312.5, -2625.0, 'MANUAL', 'RANGE_BOUND')
            """, (test_strat_id, json.dumps(test_legs)))
            conn.commit()

    try:
        with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value=mock_chain):
            nifty_analytics_engine._opt_portfolio_cache = None
            port = nifty_analytics_engine.get_options_portfolio(force_refresh=True)
            matching = [s for s in port.get("active_strategies", []) if s.get("strategy_id") == test_strat_id]
            assert len(matching) == 1
            strat = matching[0]
            
            # Check legs MTM
            legs = strat.get("legs", [])
            assert len(legs) == 2
            
            ce_leg = next(l for l in legs if l["type"] == "CE")
            pe_leg = next(l for l in legs if l["type"] == "PE")
            
            # Verify strikes are accurately evaluated from all_strikes, not fallen back to entry_price
            assert ce_leg["current_price"] == 8.5
            assert ce_leg["entry_price"] == 25.0
            assert ce_leg["leg_pnl"] == round((25.0 - 8.5) * 75, 2) # +1237.5
            
            assert pe_leg["current_price"] == 4.5
            assert pe_leg["entry_price"] == 10.0
            assert pe_leg["leg_pnl"] == round((10.0 - 4.5) * 75, 2) # +412.5
            
            total_strat_pnl = ce_leg["leg_pnl"] + pe_leg["leg_pnl"] # +1650.0
            assert strat["unrealized_pnl"] == total_strat_pnl
            assert strat["unrealized_pnl"] > 0 # Legitimate positive PnL on short options decay
    finally:
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("DELETE FROM options_strategies WHERE strategy_id = ?", (test_strat_id,))
                conn.commit()


def test_deploy_options_strategy_populates_missing_entry_prices():
    """Verify deploy_options_strategy populates 0 or missing entry_price from live option chain."""
    mock_chain = {
        "status": "OK",
        "spot_price": 24800.0,
        "weekly": {
            "expiry": "24-Oct-2026",
            "strikes": [{"strike": 24800, "ce_ltp": 94.5, "pe_ltp": 88.0}],
            "all_strikes": [{"strike": 24800, "ce_ltp": 94.5, "pe_ltp": 88.0}],
        }
    }
    
    deploy_req = {
        "strategy_name": "Test Zero Price Ingestion",
        "symbol": "NIFTY",
        "expiry_date": "24-Oct-2026",
        "expiry_type": "WEEKLY",
        "spot_at_entry": 0.0,
        "net_premium": 0.0,
        "legs": [
            {"strike": "24800", "type": "CE", "action": "SELL", "qty": 75, "entry_price": 0.0}
        ]
    }
    
    with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value=mock_chain), \
         patch.object(nifty_analytics_engine, "get_live_nifty_spot", return_value={"price": 24800.0}):
        res = nifty_analytics_engine.deploy_options_strategy(deploy_req)
        assert res["status"] == "OK"
        strat_id = res["strategy_id"]
        
        # Verify in SQLite
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                row = conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone()
                assert row is not None
                legs = json.loads(row["legs_json"])
                assert len(legs) == 1
                assert legs[0]["entry_price"] == 94.5 # populated from chain
                assert legs[0]["strike"] == 24800 # normalized to int
                assert row["spot_at_entry"] == 24800.0 # populated from live spot
                conn.execute("DELETE FROM options_strategies WHERE strategy_id = ?", (strat_id,))
                conn.commit()
