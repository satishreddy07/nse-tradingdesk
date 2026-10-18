import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from engines.accumulation_research_engine import accumulation_research_engine

print("Running test scan with AccumulationResearchEngine...")
res = accumulation_research_engine.run_research_scan(force_refresh=True, min_gain_pct=2.0)

print("\n--- RESEARCH SCAN RESULTS ---")
print(f"Audit ID: {res['audit_id']}")
print(f"Timestamp: {res['run_timestamp']}")
print(f"Total Screened: {res['total_universe']}")
print(f"Positive Movers (>= 2%): {res['positive_movers_count']}")
print(f"Movers >= 3%: {res['movers_3pct_count']}")
print(f"VDU Rate: {res['vdu_rate_pct']}%")
print(f"Tightness Rate: {res['tightness_rate_pct']}%")
print(f"EMA20 Launch Rate: {res['ema20_launch_rate_pct']}%")
print(f"Average Breakout Surge: {res['avg_breakout_surge']}x")
print(f"Execution Duration: {res['execution_duration_sec']}s")
print(f"SHA-256 Hash: {res['sha256_hash']}")

print("\n--- COMMON FINDINGS ---")
for f in res['common_findings']:
    print(f"- {f['title']} [{f['metric']}]: {f['detail']}")

print("\n--- ARCHETYPE DISTRIBUTION ---")
for k, v in res['archetype_distribution'].items():
    print(f"  {k}: {v}")

print("\n--- SAMPLE MOVERS (TOP 5) ---")
for m in res['movers'][:5]:
    print(f"  {m['symbol']:12s} | Chg: {m['chg_pct']:+6.2f}% | Surge: {m['breakout_vol_ratio']:4.1f}x | T-1 Range: {m['pre_range_t1']:4.2f}% | T-1 VDU: {m['pre_vdu_t1']:4.2f}x | T-1 EMA20: {m['dist_ema20_t1']:+5.1f}% | Arch: {m['archetype']}")
