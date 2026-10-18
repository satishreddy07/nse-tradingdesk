import json
import pandas as pd
import numpy as np

with open("scripts/gainers_above_3pct_analysis.json", "r") as f:
    gainers = json.load(f)

df = pd.DataFrame(gainers)

# Sort by Friday gain descending
df.sort_values(by="chg_fri", ascending=False, inplace=True)

print(f"Total Gainers >= 3%: {len(df)}")

# Print top 25 with full metrics
print("\n=== TOP 25 STOCKS CLOSING > 3% ON FRIDAY ===")
cols_to_print = ["symbol", "chg_fri", "vrat_fri", "dist20_fri", "c_thu", "rng_thu", "vrat_thu", "dist20_thu", "archetype", "vps_score", "vps_bucket"]

print(f"{'#':2s} | {'Symbol':12s} | {'Fri Chg%':8s} | {'Fri Vol':7s} | {'Fri EMA20':9s} | {'Thu Rng%':8s} | {'Thu Vol':7s} | {'Thu EMA20':9s} | {'Archetype':22s} | {'Score':5s} | {'VPS Bucket':10s}")
print("-" * 125)

for idx, r in enumerate(df.head(30).to_dict(orient="records"), 1):
    sym = r['symbol']
    fc = f"{r['chg_fri']:+.2f}%"
    fv = f"{r['vrat_fri']:.1f}x"
    fe = f"{r['dist20_fri']:+.1f}%"
    tr = f"{r['rng_thu']:.2f}%"
    tv = f"{r['vrat_thu']:.2f}x"
    te = f"{r['dist20_thu']:+.1f}%"
    arch = r['archetype']
    sc = f"{r['vps_score']:.1f}" if r['vps_score'] else "N/A"
    bk = r['vps_bucket']
    print(f"{idx:2d} | {sym:12s} | {fc:8s} | {fv:7s} | {fe:9s} | {tr:8s} | {tv:7s} | {te:9s} | {arch:22s} | {sc:5s} | {bk:10s}")
