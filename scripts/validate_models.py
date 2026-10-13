"""
scripts/validate_models.py
==========================
Standalone Model Validation & Calibration Diagnostic Utility.
Executes and prints full mathematical validation metrics:
  1. 180-Day Walk-Forward Historical Hit Rates (T+1 & T+5)
  2. Brier Score Calibration & Reliability Index
  3. Online Self-Training Learned Weights vs Base Weights
  4. Forward-Testing Ledger (Prediction vs Actual Settlement & Attribution)
  5. US Next-Day Model & Futures Relative Radar
"""

import sys
import os
import argparse
import json

# Force UTF-8 on Windows terminal if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.nifty_analytics_engine import nifty_analytics_engine
from engines.us_analytics_engine import us_analytics_engine


def run_model_validation(force_refresh: bool = False, verbose_ledger: bool = True):
    print("=" * 80)
    print(" [*] NSE TRADING DESK -- PREDICTION MODEL VALIDATION & CALIBRATION REPORT ")
    print("=" * 80)

    # 1. Fetch Nifty Predictions & Validation Data
    print("\n[1/3] Loading Nifty 50 Multi-Factor & Walk-Forward Validation Engine...")
    preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=force_refresh)
    
    spot = preds.get("spot_price", 0.0)
    score_t1 = preds.get("composite_score", 0.0)
    pillar_scores = preds.get("pillar_scores", {})
    t1_fc = preds.get("t1_forecast", {})
    t5_fc = preds.get("t5_forecast", {})
    
    t1_verdict = t1_fc.get("verdict", "N/A")
    t5_verdict = t5_fc.get("verdict", "N/A")
    
    print(f" -> Current Spot: Rs {spot:,.2f}")
    print(f" -> T+1 Composite Score: {score_t1:+.1f} | Verdict: {t1_verdict} (Target: Rs {t1_fc.get('expected_close', 0.0):,.1f})")
    print(f" -> T+5 Verdict: {t5_verdict} (Target: Rs {t5_fc.get('expected_close', 0.0):,.1f})")

    # 2. Walk-Forward Historical Calibration
    val = preds.get("validation_and_calibration", {})
    lookback = val.get("lookback_days", 0)
    t1_hr = val.get("t1_hit_rate_pct", 0.0)
    t5_hr = val.get("t5_hit_rate_pct", 0.0)
    brier = val.get("brier_score_t1", 0.0)
    d_vol = val.get("realized_daily_vol_pct", 0.0)
    w_vol = val.get("realized_5d_vol_pct", 0.0)
    t1_probs = val.get("t1_probs", {})
    t5_probs = val.get("t5_probs", {})

    print("\n" + "-" * 80)
    print(" [1] 180-DAY WALK-FORWARD HISTORICAL CALIBRATION (Out-of-Sample)")
    print("-" * 80)
    print(f"  * Lookback Window          : {lookback} aligned trading sessions")
    print(f"  * Signals Evaluated (T+1)  : {val.get('t1_signals_evaluated', 0)}")
    print(f"  * Signals Evaluated (T+5)  : {val.get('t5_signals_evaluated', 0)}")
    print(f"  * T+1 Directional Hit Rate : {t1_hr:.1f}%")
    print(f"  * T+5 Directional Hit Rate : {t5_hr:.1f}%")
    print(f"  * Brier Calibration Score  : {brier:.4f} (Benchmark: < 0.25 is well-calibrated)")
    print(f"  * Realized Daily Return Vol: {d_vol:.2f}% (Annualized: ~{d_vol * 15.87:.1f}%)")
    print(f"  * Realized 5-Day Return Vol: {w_vol:.2f}%")
    print(f"  * Calibrated T+1 Probs     : Up: {t1_probs.get('up',0)*100:.1f}% | Flat: {t1_probs.get('flat',0)*100:.1f}% | Down: {t1_probs.get('down',0)*100:.1f}%")
    print(f"  * Calibrated T+5 Probs     : Up: {t5_probs.get('up',0)*100:.1f}% | Flat: {t5_probs.get('flat',0)*100:.1f}% | Down: {t5_probs.get('down',0)*100:.1f}%")

    # 3. Online Self-Training & Adaptive Weights
    st = preds.get("self_training_engine", {})
    base_w = st.get("base_weights", {})
    w_t1 = st.get("learned_weights_t1", {})
    w_t5 = st.get("learned_weights_t5", {})

    print("\n" + "-" * 80)
    print(" [2] ONLINE SELF-TRAINING & ADAPTIVE PILLAR CALIBRATION")
    print("-" * 80)
    print(f"  * Sessions Graded          : {st.get('sessions_evaluated', 0)} real market sessions")
    print(f"  * Recent T+1 Hit Rate      : {st.get('t1_hit_rate_pct', 0.0)}%")
    print(f"  * Recent T+5 Hit Rate      : {st.get('t5_hit_rate_pct', 0.0)}%")
    print(f"  * Average T+1 Error Points : {st.get('avg_t1_error_pts', 0.0)} pts")
    print(f"  * Learning Algorithm       : {st.get('learning_method', 'N/A')}")
    print("\n  Pillar Weights Evolution (Base vs Learned T+1 vs Learned T+5):")
    print(f"  {'Pillar Name':<16} | {'Base Weight':<12} | {'Learned T+1':<12} | {'Learned T+5':<12}")
    print("  " + "-" * 60)
    for k in base_w:
        b = base_w.get(k, 0.0)
        l1 = w_t1.get(k, b)
        l5 = w_t5.get(k, b)
        print(f"  {k.capitalize():<16} | {b:<12.3f} | {l1:<12.3f} | {l5:<12.3f}")

    # 4. Recent Forward-Testing Ledger
    ledger = preds.get("prediction_ledger", [])
    if verbose_ledger and ledger:
        print("\n" + "-" * 80)
        print(f" [3] RECENT FORWARD-TESTING SETTLEMENT LEDGER (Last {min(8, len(ledger))} Sessions)")
        print("-" * 80)
        print(f"  {'Date':<11} | {'Spot':<9} | {'T+1 Call':<25} | {'Actual Close':<12} | {'Error':<10} | {'Status'}")
        print("  " + "-" * 76)
        for row in ledger[:8]:
            dt = row.get("prediction_date", "")
            s0 = row.get("spot_at_prediction", 0.0)
            verd = row.get("t1_verdict", "")
            act = row.get("actual_t1_close")
            act_str = f"Rs {act:,.1f}" if act is not None else "Pending"
            err = row.get("t1_error_pts")
            err_str = f"{err:+.1f} pts" if err is not None else "--"
            status = row.get("t1_validated", "PENDING")
            print(f"  {dt:<11} | {s0:<9.1f} | {verd[:24]:<25} | {act_str:<12} | {err_str:<10} | {status}")

    # 5. US Market Predictor & Relative Radar
    print("\n" + "-" * 80)
    print(" [4] US NEXT-DAY PREDICTION & FUTURES RELATIVE RADAR")
    print("-" * 80)
    try:
        us_intel = us_analytics_engine.fetch_and_analyze_us_markets(force_refresh=force_refresh)
        us_t1 = us_intel.get("us_t1_forecast", {})
        radar = us_intel.get("futures_relative_radar", {})

        print(f"  * US Composite Score      : {us_t1.get('composite_score', 0.0):+.1f} / 100")
        print(f"  * US Next-Day Verdict     : {us_t1.get('verdict', 'N/A')} (Confidence: {us_t1.get('confidence', 'N/A')})")
        print(f"  * US Probabilities        : Up: {us_t1.get('prob_up', 0)*100:.1f}% | Flat: {us_t1.get('prob_flat', 0)*100:.1f}% | Down: {us_t1.get('prob_down', 0)*100:.1f}%")
        print(f"  * SPY Expected Drift      : {us_t1.get('expected_pts_spy', 0.0):+.2f} pts ({us_t1.get('expected_pct_spy', 0.0):+.2f}%)")
        print(f"  * Futures Beta Transm.    : beta = {radar.get('beta_sp_to_nifty', 0.52):.2f}")
        print(f"  * Expected Nifty Fair Gap : {radar.get('expected_fair_gap_pts', 0.0):+.1f} pts ({radar.get('expected_fair_gap_pct', 0.0):+.2f}%)")
        print(f"  * Opening Gap Verdict     : {radar.get('gap_verdict', 'N/A')}")
    except Exception as e:
        print(f"  [Notice] US Intelligence preview warning: {e}")

    print("\n" + "=" * 80)
    print(" [OK] MODEL CALIBRATION & VALIDATION DIAGNOSTIC COMPLETED ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and calibrate trading desk models.")
    parser.add_argument("--refresh", action="store_true", help="Force recomputation and live market data sync")
    parser.add_argument("--compact", action="store_true", help="Omit detailed session-by-session ledger")
    args = parser.parse_args()

    run_model_validation(force_refresh=args.refresh, verbose_ledger=not args.compact)
