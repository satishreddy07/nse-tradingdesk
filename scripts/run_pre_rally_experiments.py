"""
Pre-Rally Pattern Discovery & Validation Engine: Research Iteration Experiments.
Evaluates predefined parameter spaces, combinatorial confluences, statistical confidence intervals,
confusion matrices, return distributions, transaction costs, and regime splits.
"""

import sys
import os
import math
import sqlite3
import json
from typing import Dict, Any, List, Tuple, Callable
import numpy as np

def wilson_score_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = wins / n
    denominator = 1.0 + z * z / n
    center_adjusted_probability = p + z * z / (2.0 * n)
    adjusted_standard_deviation = math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return round(max(0.0, lower_bound) * 100.0, 1), round(min(1.0, upper_bound) * 100.0, 1)

def run_experiments(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Load all 2,369 events with parsed features
    c.execute("""
        SELECT 
            event_id, symbol, event_date, entry_price, forward_return,
            mfe_pct, mae_pct, time_to_target, is_winner, is_in_sample,
            features_json
        FROM pre_rally_events
        ORDER BY event_date ASC, symbol ASC
    """)
    rows = c.fetchall()

    events = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features_json"])
        events.append(d)

    n_total = len(events)
    is_events = [e for e in events if e["is_in_sample"] == 1]
    oos_events = [e for e in events if e["is_in_sample"] == 0]

    n_is = len(is_events)
    n_oos = len(oos_events)

    tot_wins_is = sum(1 for e in is_events if e["is_winner"] == 1)
    tot_wins_oos = sum(1 for e in oos_events if e["is_winner"] == 1)
    tot_losses_oos = n_oos - tot_wins_oos

    baseline_is = round((tot_wins_is / n_is) * 100.0, 2)
    baseline_oos = round((tot_wins_oos / n_oos) * 100.0, 2)

    print("=" * 110)
    print(f"EXPERIMENT DATASET: {n_total} Total Events across 35 Liquid Stocks")
    print(f"In-Sample:      {n_is} events | Wins: {tot_wins_is} | Baseline Win Rate: {baseline_is}%")
    print(f"Out-of-Sample:  {n_oos} events | Wins: {tot_wins_oos} | Baseline Win Rate: {baseline_oos}%")
    print(f"Split Date:     {is_events[-1]['event_date']}")
    print("=" * 110)

    # Predefined Search Space of Hypotheses (Individual & Combinations)
    # Each entry has: (id, name, predicate_func)
    hypotheses = [
        # --- A. ATR Squeeze Parameter Sensitivity Sweep ---
        ("ATR_SQ_075", "ATR Squeeze <= 0.75 (Extreme)", lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.75),
        ("ATR_SQ_080", "ATR Squeeze <= 0.80 (Tight)", lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.80),
        ("ATR_SQ_085", "ATR Squeeze <= 0.85 (Baseline Squeeze)", lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85),
        ("ATR_SQ_090", "ATR Squeeze <= 0.90 (Mild)", lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.90),
        ("ATR_SQ_095", "ATR Squeeze <= 0.95 (Loose)", lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.95),

        # --- B. Volume Fade Parameter Sensitivity Sweep ---
        ("VOL_FADE_3D", "Consecutive Volume Fade (3-Day Decline)", lambda f: f.get("vol_declining_3d", 0) == 1),
        ("VOL_FADE_2D", "Consecutive Volume Fade (2-Day Decline: V_t1 < V_t2)", lambda f: f.get("vdu_ratio_t1", 1.0) < f.get("vdu_ratio_t2", 1.0)),
        ("VOL_SMA5_08", "Volume Fade vs 5-Day SMA (Vol <= 0.80x SMA5)", lambda f: f.get("vol_ratio_5d", 1.0) <= 0.80),
        ("VDU_08", "Volume Dry-Up (VDU <= 0.80x SMA20)", lambda f: f.get("vdu_ratio_t1", 1.0) <= 0.80),
        ("VDU_05", "Extreme Volume Dry-Up (VDU <= 0.50x SMA20)", lambda f: f.get("vdu_ratio_t1", 1.0) <= 0.50),

        # --- C. Trend Structure & Moving Average Alignment ---
        ("TREND_ALIGN", "Bullish Trend Alignment (EMA20 >= EMA50)", lambda f: f.get("trend_aligned", 0) == 1),
        ("EMA20_COIL", "20 EMA Coiling Anchor (|Dist| <= 2.5%)", lambda f: abs(f.get("dist_ema20_t1", 99.0)) <= 2.5),
        ("EMA20_SLOPE", "Rising 20 EMA Slope (5d Slope >= 0.0%)", lambda f: f.get("ema20_slope_5d", -99.0) >= 0.0),

        # --- D. Price Range & Compression ---
        ("TIGHT_30", "Daily Volatility Contraction (Range <= 3.0%)", lambda f: f.get("daily_range_t1", 99.0) <= 3.0),
        ("NR4_BAR", "Narrow Range 4 Bar (NR4 Session)", lambda f: f.get("is_nr4", 0) == 1),

        # --- E. 2-Way Confluences (Investigating Leading Candidates) ---
        (
            "PAIR_ATRSQ_VOLFADE3D", 
            "ATR Squeeze (<=0.85) + 3-Day Volume Fade", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("vol_declining_3d", 0) == 1
        ),
        (
            "PAIR_ATRSQ_VOLFADE2D", 
            "ATR Squeeze (<=0.85) + 2-Day Volume Fade", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and (f.get("vdu_ratio_t1", 1.0) < f.get("vdu_ratio_t2", 1.0))
        ),
        (
            "PAIR_ATRSQ_TREND", 
            "ATR Squeeze (<=0.85) + Trend Alignment (EMA20 >= EMA50)", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("trend_aligned", 0) == 1
        ),
        (
            "PAIR_VOLFADE3D_TREND", 
            "3-Day Volume Fade + Trend Alignment", 
            lambda f: f.get("vol_declining_3d", 0) == 1 and f.get("trend_aligned", 0) == 1
        ),
        (
            "PAIR_VOLFADE3D_EMA20", 
            "3-Day Volume Fade + 20 EMA Anchor (|Dist| <= 2.5%)", 
            lambda f: f.get("vol_declining_3d", 0) == 1 and abs(f.get("dist_ema20_t1", 99.0)) <= 2.5
        ),
        (
            "PAIR_VDU_ATRSQ", 
            "Volume Dry-Up (VDU <= 0.8x) + ATR Squeeze (<=0.85)", 
            lambda f: f.get("vdu_ratio_t1", 1.0) <= 0.80 and f.get("atr_contraction_ratio", 1.0) <= 0.85
        ),
        (
            "PAIR_EXTVDU_ATRSQ", 
            "Extreme VDU (<= 0.5x) + ATR Squeeze (<=0.85)", 
            lambda f: f.get("vdu_ratio_t1", 1.0) <= 0.50 and f.get("atr_contraction_ratio", 1.0) <= 0.85
        ),
        (
            "PAIR_VDU_VOLFADE3D", 
            "Volume Dry-Up (VDU <= 0.8x) + 3-Day Volume Fade", 
            lambda f: f.get("vdu_ratio_t1", 1.0) <= 0.80 and f.get("vol_declining_3d", 0) == 1
        ),

        # --- F. 3-Way & 4-Way High-Conviction Confluences ---
        (
            "TRIPLE_ATRSQ_VOLFADE_TREND", 
            "ATR Squeeze (<=0.85) + 3-Day Vol Fade + Trend Align", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("vol_declining_3d", 0) == 1 and f.get("trend_aligned", 0) == 1
        ),
        (
            "TRIPLE_ATRSQ_VOLFADE_EMA20", 
            "ATR Squeeze (<=0.85) + 3-Day Vol Fade + 20 EMA Anchor", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("vol_declining_3d", 0) == 1 and abs(f.get("dist_ema20_t1", 99.0)) <= 2.5
        ),
        (
            "TRIPLE_ATRSQ_VOLFADE_TIGHT", 
            "ATR Squeeze (<=0.85) + 3-Day Vol Fade + Tight Range (<=3%)", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("vol_declining_3d", 0) == 1 and f.get("daily_range_t1", 99.0) <= 3.0
        ),
        (
            "TRIPLE_ATRSQ_TREND_EMA20", 
            "ATR Squeeze (<=0.85) + Trend Align + 20 EMA Anchor", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("trend_aligned", 0) == 1 and abs(f.get("dist_ema20_t1", 99.0)) <= 2.5
        ),
        (
            "QUAD_COMPRESSION_LAUNCH", 
            "Quad Confluence (ATR Squeeze + Vol Fade + Trend Align + 20 EMA Anchor)", 
            lambda f: f.get("atr_contraction_ratio", 1.0) <= 0.85 and f.get("vol_declining_3d", 0) == 1 and f.get("trend_aligned", 0) == 1 and abs(f.get("dist_ema20_t1", 99.0)) <= 2.5
        )
    ]

    print(f"\nEvaluating {len(hypotheses)} Predefined Hypotheses Across Both In-Sample and Out-of-Sample Sets...\n")

    results = []

    for h_id, name, fn in hypotheses:
        # Match events
        m_is = [e for e in is_events if fn(e["features"])]
        m_oos = [e for e in oos_events if fn(e["features"])]
        m_all = m_is + m_oos

        n_is_m = len(m_is)
        n_oos_m = len(m_oos)
        n_all_m = len(m_all)

        w_is = sum(1 for e in m_is if e["is_winner"] == 1)
        w_oos = sum(1 for e in m_oos if e["is_winner"] == 1)

        p_is = round((w_is / n_is_m) * 100.0, 1) if n_is_m > 0 else 0.0
        p_oos = round((w_oos / n_oos_m) * 100.0, 1) if n_oos_m > 0 else 0.0

        lift_is = round(p_is / baseline_is, 2) if n_is_m > 0 else 0.0
        lift_oos = round(p_oos / baseline_oos, 2) if n_oos_m > 0 else 0.0
        delta_pp = round(p_oos - baseline_oos, 1) if n_oos_m > 0 else 0.0

        ci_low, ci_high = wilson_score_interval(w_oos, n_oos_m)
        stability = round(min(2.0, p_oos / (p_is + 1e-9)), 2) if (n_is_m >= 3 and n_oos_m >= 3) else 0.0

        # Confusion Matrix Metrics (OOS)
        tp = w_oos
        fp = n_oos_m - w_oos
        fn = tot_wins_oos - tp
        tn = tot_losses_oos - fp

        recall_oos = round((tp / (tot_wins_oos + 1e-9)) * 100.0, 1)
        fpr_oos = round((fp / (tot_losses_oos + 1e-9)) * 100.0, 1)

        # Forward Return Distribution & Risk Excursions (OOS)
        fwd_returns_oos = [e["forward_return"] for e in m_oos]
        mfes_oos = [e["mfe_pct"] for e in m_oos]
        maes_oos = [e["mae_pct"] for e in m_oos]

        avg_fwd_ret = round(float(np.mean(fwd_returns_oos)), 2) if n_oos_m > 0 else 0.0
        median_fwd_ret = round(float(np.median(fwd_returns_oos)), 2) if n_oos_m > 0 else 0.0
        std_fwd_ret = round(float(np.std(fwd_returns_oos)), 2) if n_oos_m > 0 else 0.0
        avg_mfe = round(float(np.mean(mfes_oos)), 2) if n_oos_m > 0 else 0.0
        avg_mae = round(float(np.mean(maes_oos)), 2) if n_oos_m > 0 else 0.0
        mfe_mae_ratio = round(abs(avg_mfe) / (abs(avg_mae) + 1e-9), 2) if n_oos_m > 0 else 0.0

        # Realistic Transaction Cost Impact (Roundtrip 0.35% Brokerage + STT + Taxes)
        # Net Expectancy = WinRate * (Avg Win Return - Cost) + (1 - WinRate) * (Avg Loss Return - Cost)
        win_returns = [e["forward_return"] for e in m_oos if e["is_winner"] == 1]
        loss_returns = [e["forward_return"] for e in m_oos if e["is_winner"] == 0]
        avg_win_ret = float(np.mean(win_returns)) if win_returns else 0.0
        avg_loss_ret = float(np.mean(loss_returns)) if loss_returns else 0.0
        roundtrip_cost = 0.35
        p_win_dec = p_oos / 100.0
        net_expectancy = round((p_win_dec * (avg_win_ret - roundtrip_cost)) + ((1.0 - p_win_dec) * (avg_loss_ret - roundtrip_cost)), 2) if n_oos_m > 0 else 0.0

        # Honest Classification per Step 6
        if n_oos_m < 8 or n_all_m < 15:
            classification = "Promising but insufficiently validated" if lift_oos >= 1.15 else "Degraded or rejected"
        elif lift_oos >= 1.20 and delta_pp >= +7.0 and stability >= 0.70 and n_oos_m >= 15:
            classification = "Validated out of sample"
        elif lift_oos >= 1.05 and delta_pp >= +1.5:
            classification = "Exploratory candidate"
        elif lift_oos < 0.90 or delta_pp <= -3.0:
            classification = "Degraded or rejected"
        else:
            classification = "No demonstrated edge"

        results.append({
            "id": h_id,
            "name": name,
            "n_all": n_all_m,
            "n_is": n_is_m,
            "n_oos": n_oos_m,
            "w_is": w_is,
            "w_oos": w_oos,
            "p_is": p_is,
            "p_oos": p_oos,
            "lift_is": lift_is,
            "lift_oos": lift_oos,
            "delta_pp": delta_pp,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "stability": stability,
            "recall_oos": recall_oos,
            "fpr_oos": fpr_oos,
            "avg_fwd_ret": avg_fwd_ret,
            "median_fwd_ret": median_fwd_ret,
            "std_fwd_ret": std_fwd_ret,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "mfe_mae_ratio": mfe_mae_ratio,
            "net_expectancy": net_expectancy,
            "classification": classification
        })

    # Sort results by Lift OOS descending
    results.sort(key=lambda x: (x["lift_oos"], x["n_oos"]), reverse=True)

    # Print Summary Table
    print(f"{'Hypothesis Name':<45} | {'N_all':<5} | {'N_oos':<5} | {'OOS Win%':<8} | {'Delta':<7} | {'OOS Lift':<8} | {'95% CI':<14} | {'Net Exp':<7} | {'Classification':<26}")
    print("-" * 145)

    for r in results:
        delta_str = f"{r['delta_pp']:+5.1f}%"
        ci_str = f"[{r['ci_low']:>4.1f}%, {r['ci_high']:>4.1f}%]"
        exp_str = f"{r['net_expectancy']:+5.2f}%"
        print(f"{r['name']:<45} | {r['n_all']:<5} | {r['n_oos']:<5} | {r['p_oos']:>7.1f}% | {delta_str:<7} | {r['lift_oos']:>7.2f}x | {ci_str:<14} | {exp_str:<7} | {r['classification']:<26}")

    # Step 3 Detailed Comparisons & Diagnostic Tests
    print("\n" + "=" * 110)
    print("STEP 3 & 5: DETAILED COMPARATIVE EVIDENCE & RISK EXCURSIONS")
    print("=" * 110)

    # 1. Compare ATR Squeeze parameter sensitivity
    print("\n1. ATR SQUEEZE THRESHOLD SENSITIVITY SWEEP:")
    atr_variants = [r for r in results if r["id"].startswith("ATR_SQ_")]
    for r in atr_variants:
        print(f"   - {r['name']:<35}: N_oos={r['n_oos']:<3} | Win%={r['p_oos']}% | Lift={r['lift_oos']}x | MFE/MAE={r['mfe_mae_ratio']} | Expectancy={r['net_expectancy']:+0.2f}%")

    # 2. Compare Volume Fade variants
    print("\n2. VOLUME FADE DEFINITION VARIANTS:")
    vol_variants = [r for r in results if r["id"].startswith("VOL_") or r["id"].startswith("VDU_")]
    for r in vol_variants:
        print(f"   - {r['name']:<45}: N_oos={r['n_oos']:<3} | Win%={r['p_oos']}% | Lift={r['lift_oos']}x | MFE/MAE={r['mfe_mae_ratio']} | Expectancy={r['net_expectancy']:+0.2f}%")

    # 3. Intersections with Leading Candidates
    print("\n3. COMBINATIONS & INTERSECTIONS (Component vs Combination Lift):")
    comb_keys = [
        "ATR_SQ_085", "VOL_FADE_3D", "PAIR_ATRSQ_VOLFADE3D", 
        "PAIR_ATRSQ_TREND", "PAIR_VOLFADE3D_TREND", "PAIR_VOLFADE3D_EMA20",
        "PAIR_VDU_ATRSQ", "PAIR_EXTVDU_ATRSQ", "PAIR_VDU_VOLFADE3D",
        "TRIPLE_ATRSQ_VOLFADE_TREND", "TRIPLE_ATRSQ_VOLFADE_EMA20", "QUAD_COMPRESSION_LAUNCH"
    ]
    key_results = {r["id"]: r for r in results}
    for k in comb_keys:
        if k in key_results:
            r = key_results[k]
            print(f"   - {r['name']:<55}: N_oos={r['n_oos']:<3} | Win%={r['p_oos']}% | Lift={r['lift_oos']}x | NetExp={r['net_expectancy']:+0.2f}% | Class={r['classification']}")

    conn.close()

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "/var/lib/nse-tradingdesk/swing_trades.db"
    run_experiments(db)
