import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from engines.custom_screener_engine import custom_screener_engine
from execution.state_engine import state_engine

print("Running empirical study on Chartink EQ-FINAL stocks...")
scan_res = custom_screener_engine.run_scan(preset_name="EQ-FINAL", force_refresh=False)
stocks = scan_res.get("results", [])
total_stocks = len(stocks)

print(f"Total Screened Stocks: {total_stocks}")

# Filter positive stocks up to Friday close
pos_stocks = [s for s in stocks if float(s.get("per_chg", 0.0) or 0.0) > 0.0]
neg_stocks = [s for s in stocks if float(s.get("per_chg", 0.0) or 0.0) < 0.0]
flat_stocks = [s for s in stocks if float(s.get("per_chg", 0.0) or 0.0) == 0.0]

print(f"Positive on Friday: {len(pos_stocks)} ({len(pos_stocks)/total_stocks*100:.1f}%)")
print(f"Negative on Friday: {len(neg_stocks)} ({len(neg_stocks)/total_stocks*100:.1f}%)")
print(f"Flat on Friday: {len(flat_stocks)} ({len(flat_stocks)/total_stocks*100:.1f}%)")

# Gain brackets
runners_15 = [s for s in pos_stocks if s["per_chg"] >= 15.0]
runners_10_15 = [s for s in pos_stocks if 10.0 <= s["per_chg"] < 15.0]
runners_5_10 = [s for s in pos_stocks if 5.0 <= s["per_chg"] < 10.0]
runners_2_5 = [s for s in pos_stocks if 2.0 <= s["per_chg"] < 5.0]
runners_0_2 = [s for s in pos_stocks if 0.0 < s["per_chg"] < 2.0]

print("\n--- GAIN DISTRIBUTION ON FRIDAY ---")
print(f"Explosive / Circuits (>= +15%): {len(runners_15)} stocks")
print(f"Major Breakouts (+10% to +15%): {len(runners_10_15)} stocks")
print(f"Strong Gainers (+5% to +10%):   {len(runners_5_10)} stocks")
print(f"Moderate (+2% to +5%):          {len(runners_2_5)} stocks")
print(f"Mild (+0% to +2%):              {len(runners_0_2)} stocks")

# 20 EMA Distance distribution among positive stocks
ema_le_5 = [s for s in pos_stocks if s["ema20"] is not None and abs(s["ema20"]) <= 5.0]
ema_5_10 = [s for s in pos_stocks if s["ema20"] is not None and 5.0 < s["ema20"] <= 10.0]
ema_10_20 = [s for s in pos_stocks if s["ema20"] is not None and 10.0 < s["ema20"] <= 20.0]
ema_gt_20 = [s for s in pos_stocks if s["ema20"] is not None and s["ema20"] > 20.0]

print("\n--- 20 EMA DISTANCE DISTRIBUTION AMONG POSITIVE STOCKS ---")
print(f"Within 5% of 20 EMA (Eligible for current BUY_NOW): {len(ema_le_5)} stocks ({len(ema_le_5)/len(pos_stocks)*100:.1f}%)")
print(f"5% to 10% from 20 EMA (Fresh Momentum Breakouts):   {len(ema_5_10)} stocks ({len(ema_5_10)/len(pos_stocks)*100:.1f}%)")
print(f"10% to 20% from 20 EMA (Extended Rallies):          {len(ema_10_20)} stocks ({len(ema_10_20)/len(pos_stocks)*100:.1f}%)")
print(f"Over 20% Extended (Parabolic / Blow-offs):          {len(ema_gt_20)} stocks ({len(ema_gt_20)/len(pos_stocks)*100:.1f}%)")

# Top 15 explosive gainers
print("\n--- TOP 15 EXPLOSIVE RUNNERS ON FRIDAY ---")
for idx, s in enumerate(pos_stocks[:15], 1):
    sym = s["symbol"]
    name = s["name"]
    chg = s["per_chg"]
    close = s["close"]
    vol = s.get("volume_formatted", s["volume"])
    ema20 = s.get("ema20", 0.0)
    print(f"#{idx:2d} | {sym:12s} | Chg: {chg:+6.2f}% | Close: {close:8.2f} | 20EMA Dist: {ema20:+6.2f}% | Vol: {vol} | {name[:25]}")
