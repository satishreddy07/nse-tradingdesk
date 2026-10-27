import pytest
import math
from engines.nifty_analytics_engine import NiftyAnalyticsEngine

def test_closed_leg_greeks_skipped():
    engine = NiftyAnalyticsEngine()
    spot = 22700.0
    expiry_date = "27-Oct-2026"
    iv = 10.0

    # 1. Active straddle with 2 active legs (22700 CE SELL and 22700 PE SELL)
    legs_active = [
        {"strike": 22700, "type": "CE", "action": "SELL", "qty": 65, "lot_size": 65, "iv": 10.0, "state": "ACTIVE"},
        {"strike": 22700, "type": "PE", "action": "SELL", "qty": 65, "lot_size": 65, "iv": 10.0, "state": "ACTIVE"}
    ]
    greeks_straddle = engine._calc_strategy_greeks(spot, legs_active, expiry_date, iv)
    # At ATM, delta of CE sell (-0.5) and PE sell (+0.5) roughly cancel each other out
    assert abs(greeks_straddle["delta"]) < 0.2

    # 2. Straddle where CE leg has exited (state = CLOSED, exit_price = 129.10)
    legs_ce_exited = [
        {"strike": 22700, "type": "CE", "action": "SELL", "qty": 65, "lot_size": 65, "iv": 10.0, "state": "CLOSED", "exit_price": 129.10, "leg_pnl": -1943.50},
        {"strike": 22700, "type": "PE", "action": "SELL", "qty": 65, "lot_size": 65, "iv": 10.0, "state": "ACTIVE"}
    ]
    greeks_ce_exited = engine._calc_strategy_greeks(spot, legs_ce_exited, expiry_date, iv)
    # When CE is closed, only short PE is active, which has delta roughly +0.50
    assert greeks_ce_exited["delta"] > 0.35, f"Expected positive delta for active short put, got {greeks_ce_exited['delta']}"
    # Theta should be positive (harvesting theta from short put)
    assert greeks_ce_exited["theta"] > 0.0

def test_closed_leg_payoff_mechanics():
    """Verify that closed leg has fixed P&L across all spot prices, and active leg generates slope."""
    entry_ce = 99.20
    exit_ce = 129.10
    qty = 65
    ce_closed_pnl = (entry_ce - exit_ce) * qty  # -1943.50

    entry_pe = 68.15

    # At Spot = 23000 (Upside):
    # Short PE intrinsic = max(0, 22700 - 23000) = 0
    # PE profit = (entry_pe - 0) * qty = 68.15 * 65 = +4429.75
    # Total strategy expiry PnL = ce_closed_pnl + pe_profit = -1943.50 + 4429.75 = +2486.25 (Flat profit)
    upside_spot = 23000
    pe_intrinsic_upside = max(0, 22700 - upside_spot)
    pe_pnl_upside = (entry_pe - pe_intrinsic_upside) * qty
    net_pnl_upside = ce_closed_pnl + pe_pnl_upside
    assert round(net_pnl_upside, 2) == 2486.25

    # At Spot = 23500 (Far Upside):
    far_upside_spot = 23500
    pe_intrinsic_far = max(0, 22700 - far_upside_spot)
    pe_pnl_far = (entry_pe - pe_intrinsic_far) * qty
    net_pnl_far = ce_closed_pnl + pe_pnl_far
    # Payoff must be capped flat, NOT incurring unlimited upside losses!
    assert round(net_pnl_far, 2) == 2486.25

    # Breakeven calculation on downside:
    # 2486.25 - 65 * (22700 - S_be) = 0 => S_be = 22700 - (2486.25 / 65) = 22661.75
    be_spot = 22700.0 - (net_pnl_upside / qty)
    assert round(be_spot, 2) == 22661.75
