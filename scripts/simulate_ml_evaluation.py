import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import pandas as pd
import numpy as np
from engines.swing_quant_ml_engine import swing_quant_ml_engine

# Load the 70 gainers analysis
with open("scripts/gainers_above_3pct_analysis.json", "r") as f:
    gainers = json.load(f)

print(f"Loaded {len(gainers)} gainers.")
regime = swing_quant_ml_engine.capture_market_regime()
print("Market Regime:", regime)

findings = []

for g in gainers:
    sym = g["symbol"]
    # Construct candidate dict resembling what InstitutionalEngine passes on Friday
    cand = {
        "symbol": sym,
        "ticker": f"{sym}.NS",
        "score": float(g.get("vps_score") or 80.0),
        "confidence": 4,
        "atr_pct": 3.0,
        "rsi14": g["rsi_fri"],
        "dist_to_pivot": max(1.0, g["chg_fri"] - 1.0),
        "dist_to_ema20": g["dist20_fri"],
        "is_vcp": 1 if g["archetype"] in ["Textbook VDU Coil", "Tight Base at EMA"] else 0,
        "week_confluence": 1,
        "vol_ratio": g["vrat_fri"],
        "sector_excess_6m": 2.5,
        "nifty_ret_5d": 0.5,
        "rr_ratio": 2.5,
        "stop_loss_pct": 3.5,
        "holding_days": 7
    }
    
    eval_res = swing_quant_ml_engine.evaluate_candidate(cand, market_regime=regime)
    
    findings.append({
        "symbol": sym,
        "fri_chg": g["chg_fri"],
        "fri_dist20": g["dist20_fri"],
        "thu_rng": g["rng_thu"],
        "thu_vrat": g["vrat_thu"],
        "thu_dist20": g["dist20_thu"],
        "archetype": g["archetype"],
        "win_prob": eval_res["win_prob"],
        "expected_r": eval_res["expected_r"],
        "verdict": eval_res["verdict"],
        "decision": eval_res["decision"],
        "fatal_flags": eval_res["fatal_flags"],
        "risk_flags": eval_res["risk_flags"]
    })

f_df = pd.DataFrame(findings)

print("\n=== SIMULATED ML VERDICTS ON FRIDAY DATA ===")
print("Verdicts:")
print(f_df["verdict"].value_counts())
print("\nDecisions:")
print(f_df["decision"].value_counts())

# Group why they were rejected
fatal_counts = {}
for flags in f_df["fatal_flags"]:
    for f in flags:
        k = f.split(":")[0]
        fatal_counts[k] = fatal_counts.get(k, 0) + 1

print("\nFatal Flag Frequency:")
for k, v in sorted(fatal_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v} stocks ({v/len(f_df)*100:.1f}%)")

low_prob_cnt = (f_df["win_prob"] < 0.48).sum()
low_r_cnt = (f_df["expected_r"] < 0.05).sum()
print(f"Low Win Prob (< 0.48): {low_prob_cnt}")
print(f"Low Expected R (< 0.05): {low_r_cnt}")

# Let's inspect Thursday data simulation (what if we evaluated them on Thursday during accumulation?)
print("\n=== SIMULATING THURSDAY (PRE-BREAKOUT / ACCUMULATION DAY) EVALUATION ===")
thu_findings = []
for g in gainers:
    sym = g["symbol"]
    cand_thu = {
        "symbol": sym,
        "ticker": f"{sym}.NS",
        "score": 78.0,
        "confidence": 4,
        "atr_pct": 2.5,
        "rsi14": g["rsi_thu"],
        "dist_to_pivot": 0.5,
        "dist_to_ema20": g["dist20_thu"],
        "is_vcp": 1 if g["archetype"] in ["Textbook VDU Coil", "Tight Base at EMA"] else 0,
        "week_confluence": 1,
        "vol_ratio": g["vrat_thu"], # NOTE: On Thursday, volume was low (VDU)!
        "sector_excess_6m": 2.0,
        "nifty_ret_5d": 0.5,
        "rr_ratio": 3.0,
        "stop_loss_pct": 2.5,
        "holding_days": 7
    }
    eval_thu = swing_quant_ml_engine.evaluate_candidate(cand_thu, market_regime=regime)
    thu_findings.append({
        "symbol": sym,
        "win_prob": eval_thu["win_prob"],
        "expected_r": eval_thu["expected_r"],
        "verdict": eval_thu["verdict"],
        "decision": eval_thu["decision"],
        "fatal_flags": eval_thu["fatal_flags"],
        "risk_flags": eval_thu["risk_flags"]
    })

thu_df = pd.DataFrame(thu_findings)
print("Thursday Verdicts:")
print(thu_df["verdict"].value_counts())
print("Thursday Decisions:")
print(thu_df["decision"].value_counts())

thu_fatal_counts = {}
for flags in thu_df["fatal_flags"]:
    for f in flags:
        k = f.split(":")[0]
        thu_fatal_counts[k] = thu_fatal_counts.get(k, 0) + 1
print("Thursday Fatal Flags:", thu_fatal_counts)

thu_risk_counts = {}
for flags in thu_df["risk_flags"]:
    for f in flags:
        k = f.split(":")[0]
        thu_risk_counts[k] = thu_risk_counts.get(k, 0) + 1
print("Thursday Risk Flags:")
for k, v in sorted(thu_risk_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v} stocks")
