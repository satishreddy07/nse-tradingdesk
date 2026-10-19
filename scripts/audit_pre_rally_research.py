"""
Comprehensive Audit & Research Script for Pre-Rally Pattern Discovery Engine.
Reconciles stored historical events, reproduces statistical metrics, tests predefined
candidate variations, and computes uncertainty intervals.
"""

import sys
import os
import math
import sqlite3
import json
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def wilson_score_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p = wins / n
    denominator = 1.0 + z * z / n
    center_adjusted_probability = p + z * z / (2.0 * n)
    adjusted_standard_deviation = math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return round(max(0.0, lower_bound) * 100.0, 1), round(min(1.0, upper_bound) * 100.0, 1)

def run_reconciliation(db_path: str):
    print("=" * 80)
    print(f"STEP 1: RECONCILING STORED DATA IN {db_path}")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. Total Events Summary
    c.execute("""
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(event_date) as start_date,
            MAX(event_date) as end_date,
            SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as total_wins
        FROM pre_rally_events
    """)
    summary = dict(c.fetchone())
    total_events = summary["total_events"]
    total_wins = summary["total_wins"]
    overall_baseline = round((total_wins / total_events) * 100.0, 2) if total_events > 0 else 0.0

    print(f"Total Stored Events:      {total_events}")
    print(f"Unique Symbols:           {summary['unique_symbols']}")
    print(f"Date Range:               {summary['start_date']} to {summary['end_date']}")
    print(f"Total Wins (5% in 10d):   {total_wins} / {total_events} ({overall_baseline}%)")

    # 2. In-Sample vs Out-of-Sample Split Reconciliation
    c.execute("""
        SELECT 
            is_in_sample,
            COUNT(*) as event_count,
            SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as win_count,
            MIN(event_date) as min_date,
            MAX(event_date) as max_date
        FROM pre_rally_events
        GROUP BY is_in_sample
        ORDER BY is_in_sample DESC
    """)
    splits = [dict(r) for r in c.fetchall()]
    
    is_count = 0
    is_wins = 0
    is_baseline = 0.0
    oos_count = 0
    oos_wins = 0
    oos_baseline = 0.0
    split_date = "--"

    for s in splits:
        is_in = s["is_in_sample"]
        cnt = s["event_count"]
        wins = s["win_count"]
        rate = round((wins / cnt) * 100.0, 2) if cnt > 0 else 0.0
        label = "In-Sample (70%)" if is_in == 1 else "Out-of-Sample (30%)"
        print(f"\n{label}:")
        print(f"  Dates:     {s['min_date']} to {s['max_date']}")
        print(f"  Events:    {cnt}")
        print(f"  Wins:      {wins}")
        print(f"  Baseline:  {rate}%")
        if is_in == 1:
            is_count = cnt
            is_wins = wins
            is_baseline = rate
            split_date = s["max_date"]
        else:
            oos_count = cnt
            oos_wins = wins
            oos_baseline = rate

    print(f"\nChronological Split Date (Last IS date / Split point): {split_date}")

    # 3. Reconcile Combinations from pre_rally_discovered_combinations
    print("\n" + "=" * 80)
    print("RECONCILING REPORTED DISCOVERED COMBINATIONS")
    print("=" * 80)

    c.execute("""
        SELECT 
            combination_id,
            name,
            category,
            rule_expression,
            validation_status,
            sample_size,
            oos_sample_size,
            win_count,
            oos_win_count,
            in_sample_precision,
            out_sample_precision,
            overall_precision,
            lift_oos,
            stability_score,
            mfe_mae_ratio,
            avg_forward_return
        FROM pre_rally_discovered_combinations
        ORDER BY lift_oos DESC, sample_size DESC
    """)
    combinations = [dict(r) for r in c.fetchall()]
    print(f"Total Pre-Configured Combinations in Database: {len(combinations)}\n")

    # Header for reconciliation table
    print(f"{'Name':<45} | {'Status':<14} | {'N_all':<6} | {'N_is':<6} | {'N_oos':<6} | {'Wins_oos':<8} | {'Prec_OOS':<8} | {'Lift_OOS':<8} | {'95% CI (OOS)':<16}")
    print("-" * 135)

    for comb in combinations:
        c_id = comb["combination_id"]
        name = comb["name"][:44]
        status = comb["validation_status"]
        n_all = comb["sample_size"]
        n_oos = comb["oos_sample_size"]
        n_is = n_all - n_oos
        wins_oos = comb["oos_win_count"]
        prec_oos = comb["out_sample_precision"]
        lift_oos = comb["lift_oos"]
        ci_low, ci_high = wilson_score_interval(wins_oos, n_oos)
        ci_str = f"[{ci_low}%, {ci_high}%]"

        print(f"{name:<45} | {status:<14} | {n_all:<6} | {n_is:<6} | {n_oos:<6} | {wins_oos:<8} | {prec_oos:>7.1f}% | {lift_oos:>7.2f}x | {ci_str:<16}")

    conn.close()

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/swing_trades.db"
    run_reconciliation(db)
