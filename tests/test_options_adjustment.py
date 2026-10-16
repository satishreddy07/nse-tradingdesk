import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import json
from engines.nifty_analytics_engine import nifty_analytics_engine, state_engine, NIFTY_LOT_SIZE

def test_adjust_options_strategy():
    # Deploy a test strategy first
    deploy_data = {
        "strategy_name": "Test Strangle for Adjustment",
        "symbol": "NIFTY",
        "expiry_date": "29-Oct-2026",
        "expiry_type": "MONTHLY",
        "spot_at_entry": 25000.0,
        "net_premium": 250.0 * NIFTY_LOT_SIZE,
        "max_profit": 16250.0,
        "max_loss": 50000.0,
        "breakevens": [24750.0, 25250.0],
        "risk_reward": "1:3",
        "est_margin": 85000.0,
        "pop_pct": 68.0,
        "target_pnl": 8000.0,
        "stoploss_pnl": -10000.0,
        "legs": [
            {"strike": 24800, "type": "PE", "action": "SELL", "qty": NIFTY_LOT_SIZE, "entry_price": 120.0, "iv": 13.0, "expiry_type": "MONTHLY", "expiry_date": "29-Oct-2026"},
            {"strike": 25200, "type": "CE", "action": "SELL", "qty": NIFTY_LOT_SIZE, "entry_price": 130.0, "iv": 13.0, "expiry_type": "MONTHLY", "expiry_date": "29-Oct-2026"}
        ],
        "notes": "Initial deployment test"
    }

    res = nifty_analytics_engine.deploy_options_strategy(deploy_data)
    assert res["status"] == "OK"
    strat_id = res["strategy_id"]

    # Now perform adjustment: Roll the 24800 PE up to 24900 PE and add a hedge 24600 PE BUY
    adjust_data = {
        "strategy_id": strat_id,
        "legs": [
            {"strike": 24900, "type": "PE", "action": "SELL", "qty": NIFTY_LOT_SIZE, "entry_price": 145.0, "iv": 13.2, "expiry_type": "MONTHLY", "expiry_date": "29-Oct-2026"},
            {"strike": 25200, "type": "CE", "action": "SELL", "qty": NIFTY_LOT_SIZE, "entry_price": 130.0, "iv": 13.0, "expiry_type": "MONTHLY", "expiry_date": "29-Oct-2026"},
            {"strike": 24600, "type": "PE", "action": "BUY", "qty": NIFTY_LOT_SIZE, "entry_price": 50.0, "iv": 13.5, "expiry_type": "MONTHLY", "expiry_date": "29-Oct-2026"}
        ],
        "net_premium": 225.0 * NIFTY_LOT_SIZE,
        "max_profit": 14625.0,
        "max_loss": 30000.0,
        "breakevens": [24725.0, 25225.0],
        "est_margin": 65000.0,
        "target_pnl": 7500.0,
        "stoploss_pnl": -8000.0,
        "notes": "Adjusted: rolled PE and added wing hedge"
    }

    adj_res = nifty_analytics_engine.adjust_options_strategy(adjust_data)
    assert adj_res["status"] == "OK"
    assert adj_res["strategy_id"] == strat_id

    # Verify updated values in SQLite
    with state_engine._get_connection() as conn:
        row = conn.execute("SELECT * FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone()
        assert row is not None
        strat = dict(row)
        legs = json.loads(strat["legs_json"])
        assert len(legs) == 3
        assert legs[0]["strike"] == 24900
        assert legs[2]["action"] == "BUY"
        assert strat["est_margin"] == 65000.0
        assert strat["target_pnl"] == 7500.0

    # Clean up test strategy
    exit_res = nifty_analytics_engine.exit_options_strategy(strat_id, exit_reason="TEST_CLEANUP")
    assert exit_res["status"] == "OK"
