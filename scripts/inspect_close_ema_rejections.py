import json
import pandas as pd

with open("scripts/all_latest_signals.json") as f:
    signals = json.load(f)

# Stocks that had Friday dist_to_ema20 <= 5.0%
close_to_ema_stocks = ['SHRIPISTON', 'RPGLIFE', 'AIIL', 'LLOYDSENGG', 'SENCO', 'THANGAMAYL', 'SMSPHARMA', 'RATNAVEER', 'WPIL', 'CELLO', 'GNA', 'WELCORP']

print("=== INSPECTING STOCKS WITHIN 5% OF 20 EMA ON FRIDAY ===")
for s in close_to_ema_stocks:
    if s in signals:
        sig = signals[s]
        print(f"{s:12s} | Score: {sig['score']} | Bucket: {sig['bucket']} | Dist20: {sig['dist_to_ema20']}% | ML: {sig['ml_verdict']} | Rej: {sig['rejection_reason']}")
    else:
        print(f"{s:12s} | NOT IN VPS SIGNALS DB")
