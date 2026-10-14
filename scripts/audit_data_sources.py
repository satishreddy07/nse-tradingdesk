import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.state_engine import state_engine
from engines.intraday_analytics_engine import intraday_analytics_engine
from engines.nifty_analytics_engine import nifty_analytics_engine
from engines.us_analytics_engine import us_analytics_engine

print("=== SECTION 1: SWING TRADING (INSTITUTIONAL 6-PILLAR) ===")
p1_positions = state_engine.get_active_positions('INSTITUTIONAL_6PILLAR')
p1_stats = state_engine.get_portfolio_stats('INSTITUTIONAL_6PILLAR')
print(f"Active Positions ({len(p1_positions)}):")
for p in p1_positions:
    print(f"  {p.get('symbol')}: Qty={p.get('qty')}, Entry=Rs {p.get('entry_price')}, LTP=Rs {p.get('current_price')}, PnL=Rs {p.get('unrealized_pnl')}, T1_Hit={p.get('t1_hit')}, T1_PnL={p.get('t1_pnl')}")

print("\nPortfolio Stats:")
for k, v in p1_stats.items():
    print(f"  {k}: {v}")

print("\n=== SECTION 2: OPTIONS PORTFOLIO & DESK ===")
opt_portfolio = state_engine.get_kv("options_portfolio_state") or {}
active_strats = opt_portfolio.get("active_strategies", [])
print(f"Options Active Strategies ({len(active_strats)}):")
for s in active_strats:
    print(f"  {s.get('strategy_id')}: {s.get('strategy_name')}, Net PnL={s.get('unrealized_pnl')}")

print("\n=== SECTION 3: INTRADAY OPTION CHAIN & SPOT ===")
try:
    dash = nifty_analytics_engine.get_intraday_expiry_dashboard()
    snap = dash.get('latest_snapshot', {})
    print(f"Spot Nifty: {snap.get('spot_price')}, Expiry: {snap.get('expiry_date')}, Near-ATM PCR: {snap.get('near_atm_pcr')}, Max Pain: {snap.get('max_pain')}")
    print(f"ATM Strike: {snap.get('atm_strike')}, Straddle Premium: Rs {snap.get('straddle_premium')}, CE LTP={snap.get('atm_ce_ltp')}, PE LTP={snap.get('atm_pe_ltp')}")
    print(f"Market State: {snap.get('market_state')}")
except Exception as e:
    print(f"Error in Intraday Expiry: {e}")

print("\n=== SECTION 4: NIFTY PREDICTIONS & PREMARKET ===")
try:
    pred = nifty_analytics_engine.compute_nifty_predictions()
    t1_fc = pred.get('t1_forecast', {})
    print(f"Predictions Date: {pred.get('prediction_date')}, Verdict: {t1_fc.get('verdict')}")
    print(f"Spot: {pred.get('spot_price')}, Predicted Range: [{t1_fc.get('lower_range')} - {t1_fc.get('upper_range')}], Confidence: {t1_fc.get('confidence')}%")
    print(f"Pillars Count: {len(pred.get('pillars', {}))}")
except Exception as e:
    print(f"Error in Predictions: {e}")

print("\n=== SECTION 5: US & MACRO INTELLIGENCE ===")
try:
    us_data = us_analytics_engine.fetch_and_analyze_us_markets()
    benchmarks = us_data.get('benchmarks', [])
    for b in benchmarks:
        print(f"  {b.get('name')}: Close={b.get('close')}, 1D Ret={b.get('ret_1d')}%, Trend={b.get('trend')}")
    rates = us_data.get('macro_rates_currency', {})
    print(f"  US10Y: {rates.get('us_10y_yield')}%, DXY: {rates.get('us_dollar_dxy')}, Curve Spread: {rates.get('yield_curve_spread')}")
except Exception as e:
    print(f"Error in US Data: {e}")
