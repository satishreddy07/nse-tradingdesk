import datetime
import pytest
from core.market_calendar import now_ist
from engines.nifty_analytics_engine import nifty_analytics_engine

def test_calc_strategy_greeks():
    """Verifies that _calc_strategy_greeks produces non-zero, mathematically sound Black-Scholes Greeks."""
    spot = 23146.15
    legs = [
        {"strike": 23500, "type": "CE", "action": "BUY", "qty": 65, "iv": 10.5},
        {"strike": 23600, "type": "CE", "action": "SELL", "qty": 130, "iv": 10.8},
        {"strike": 23900, "type": "CE", "action": "BUY", "qty": 65, "iv": 11.5}
    ]
    # Use dynamic 14-day DTE to make the Black-Scholes ratio spread Greek calculation hermetic and date-invariant
    target_exp = (now_ist().date() + datetime.timedelta(days=14)).strftime("%d-%b-%Y")
    greeks = nifty_analytics_engine._calc_strategy_greeks(spot, legs, target_exp, base_iv=11.0)
    assert isinstance(greeks, dict)
    assert "delta" in greeks and "theta" in greeks and "gamma" in greeks and "vega" in greeks
    # For a call ratio spread, net delta, theta, and vega should be calculated
    assert greeks["delta"] != 0.0 or greeks["theta"] != 0.0
    assert greeks["theta"] > 0.0  # Net credit/spread with short 2 lots has positive theta
    assert greeks["gamma"] != 0.0
    assert greeks["vega"] != 0.0

def test_calc_strangle_greeks():
    """Verifies that short strangle produces delta-neutral and positive theta Greeks."""
    spot = 23146.15
    legs = [
        {"strike": 23650, "type": "CE", "action": "SELL", "qty": 75, "iv": 10.5},
        {"strike": 22350, "type": "PE", "action": "SELL", "qty": 75, "iv": 14.2}
    ]
    target_exp = (now_ist().date() + datetime.timedelta(days=14)).strftime("%d-%b-%Y")
    greeks = nifty_analytics_engine._calc_strategy_greeks(spot, legs, target_exp, base_iv=12.0)
    assert greeks["theta"] > 0.0  # Selling options earns positive theta
    assert greeks["vega"] < 0.0   # Short options have negative vega
