import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
from engines.custom_screener_engine import custom_screener_engine

print("Step 1: Fetching Chartink EQ-FINAL scan...")
scan_res = custom_screener_engine.run_scan(preset_name="EQ-FINAL", force_refresh=False)
all_stocks = scan_res.get("results", [])
print(f"Total screened stocks: {len(all_stocks)}")

# Filter gainers >= 3.0%
gainers = [s for s in all_stocks if float(s.get("per_chg", 0.0) or 0.0) >= 3.0]
gainers.sort(key=lambda s: float(s.get("per_chg", 0.0) or 0.0), reverse=True)
print(f"Total gainers >= 3.0% on Friday: {len(gainers)}")

symbols = [s["symbol"] for s in gainers]

# Step 2: Load latest_signals from local file downloaded from VPS
print("\nStep 2: Loading VPS latest_signals...")
with open("scripts/all_latest_signals.json", "r") as f:
    vps_signals = json.load(f)
print(f"Loaded {len(vps_signals)} total signals from VPS.")
matched = sum(1 for s in symbols if s in vps_signals)
print(f"Matched {matched} of {len(symbols)} gainers in VPS latest_signals table.")

# Step 3: Download OHLCV data
print("\nStep 3: Downloading daily OHLCV data from Yahoo Finance...")
tickers = [f"{s}.NS" for s in symbols]
hist = yf.download(tickers, period="2mo", interval="1d", group_by='ticker', progress=False)

# Step 4: Analyze each stock's pre-Friday accumulation and Friday breakout
print("\nStep 4: Running quantitative accumulation analysis...")

records = []
for s in symbols:
    t = f"{s}.NS"
    try:
        if t not in hist.columns.levels[0]:
            continue
        df = hist[t].dropna()
        if len(df) < 25:
            continue
        
        # Calculate indicators
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']
        
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        vol_sma20 = vol.rolling(20).mean()
        
        daily_range = (high - low) / low * 100.0
        pct_change = close.pct_change() * 100.0
        vol_ratio = vol / vol_sma20
        dist_ema20 = (close - ema20) / ema20 * 100.0
        dist_ema50 = (close - ema50) / ema50 * 100.0
        
        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi14 = 100 - (100 / (1 + rs))
        
        # Get Friday (last row) and Thursday (last-1 row)
        idx_fri = -1
        idx_thu = -2
        idx_wed = -3
        
        date_fri = df.index[idx_fri].strftime('%Y-%m-%d')
        date_thu = df.index[idx_thu].strftime('%Y-%m-%d')
        
        c_fri = float(close.iloc[idx_fri])
        chg_fri = float(pct_change.iloc[idx_fri])
        rng_fri = float(daily_range.iloc[idx_fri])
        vol_fri = int(vol.iloc[idx_fri])
        vrat_fri = float(vol_ratio.iloc[idx_fri])
        dist20_fri = float(dist_ema20.iloc[idx_fri])
        dist50_fri = float(dist_ema50.iloc[idx_fri])
        rsi_fri = float(rsi14.iloc[idx_fri])
        
        c_thu = float(close.iloc[idx_thu])
        chg_thu = float(pct_change.iloc[idx_thu])
        rng_thu = float(daily_range.iloc[idx_thu])
        vol_thu = int(vol.iloc[idx_thu])
        vrat_thu = float(vol_ratio.iloc[idx_thu])
        dist20_thu = float(dist_ema20.iloc[idx_thu])
        dist50_thu = float(dist_ema50.iloc[idx_thu])
        rsi_thu = float(rsi14.iloc[idx_thu])
        
        rng_wed = float(daily_range.iloc[idx_wed])
        vrat_wed = float(vol_ratio.iloc[idx_wed])
        
        # Accumulation criteria (Thursday)
        # 1. Volatility contraction: range <= 3.0%
        # 2. Volume Dry-Up (VDU): vol_ratio <= 0.8x
        # 3. EMA 20 Proximity: dist_ema20 between -3.0% and +4.0%
        # 4. Multi-day tightness: avg range Wed & Thu <= 3.5%
        avg_pre_rng = (rng_thu + rng_wed) / 2.0
        avg_pre_vrat = (vrat_thu + vrat_wed) / 2.0
        
        is_tight = rng_thu <= 3.0
        is_vdu = vrat_thu <= 0.8
        is_near_ema20 = abs(dist20_thu) <= 4.0
        is_near_ema50 = abs(dist50_thu) <= 5.0
        
        # Accumulation Quality Score (0 to 100)
        accum_score = 0
        if is_tight: accum_score += 30
        if rng_thu <= 2.0: accum_score += 15
        if is_vdu: accum_score += 25
        if vrat_thu <= 0.5: accum_score += 10
        if is_near_ema20: accum_score += 20
        
        # Determine Pre-Breakout Archetype
        if is_tight and is_vdu and is_near_ema20:
            archetype = "Textbook VDU Coil"
        elif is_near_ema20 and is_vdu:
            archetype = "VDU at EMA Support"
        elif is_tight and is_near_ema20:
            archetype = "Tight Base at EMA"
        elif dist20_thu > 5.0:
            archetype = "Momentum Continuation"
        elif dist20_thu < -4.0 and is_near_ema50:
            archetype = "50 EMA Deep Pullback"
        else:
            archetype = "Loose / Consolidating"
            
        sig = vps_signals.get(s, {})
        vps_bucket = sig.get('bucket', 'NOT_IN_DB')
        vps_score = sig.get('score', 0.0)
        vps_ml = sig.get('ml_verdict', 'NONE')
        vps_rej = sig.get('rejection_reason', 'NONE')
        
        records.append({
            'symbol': s,
            'c_fri': c_fri,
            'chg_fri': chg_fri,
            'rng_fri': rng_fri,
            'vrat_fri': vrat_fri,
            'dist20_fri': dist20_fri,
            'rsi_fri': rsi_fri,
            'c_thu': c_thu,
            'chg_thu': chg_thu,
            'rng_thu': rng_thu,
            'vrat_thu': vrat_thu,
            'dist20_thu': dist20_thu,
            'rsi_thu': rsi_thu,
            'avg_pre_rng': avg_pre_rng,
            'avg_pre_vrat': avg_pre_vrat,
            'accum_score': accum_score,
            'archetype': archetype,
            'vps_bucket': vps_bucket,
            'vps_score': vps_score,
            'vps_ml': vps_ml,
            'vps_rej': vps_rej
        })
    except Exception as e:
        print(f"Error analyzing {s}: {e}")

res_df = pd.DataFrame(records)
print(f"Successfully analyzed {len(res_df)} stocks.")

# Save results to JSON for deeper inspection
res_df.to_json("scripts/gainers_above_3pct_analysis.json", orient="records", indent=2)

print("\n--- SUMMARY OF FINDINGS ---")
print(f"Total Analyzed: {len(res_df)}")
print(f"Average Friday Gain: {res_df['chg_fri'].mean():.2f}%")
print(f"Average Friday Volume Surge: {res_df['vrat_fri'].mean():.2f}x")
print(f"Average Thursday Range: {res_df['rng_thu'].mean():.2f}%")
print(f"Average Thursday Volume Ratio: {res_df['vrat_thu'].mean():.2f}x")
print(f"Average Thursday Dist to 20 EMA: {res_df['dist20_thu'].mean():.2f}%")

print("\n--- PRE-FRIDAY ACCUMULATION BEHAVIOR (THURSDAY) ---")
tight_cnt = (res_df['rng_thu'] <= 3.0).sum()
vdu_cnt = (res_df['vrat_thu'] <= 0.8).sum()
extreme_vdu_cnt = (res_df['vrat_thu'] <= 0.5).sum()
near_ema20_cnt = (res_df['dist20_thu'].abs() <= 4.0).sum()
textbook_cnt = ((res_df['rng_thu'] <= 3.5) & (res_df['vrat_thu'] <= 0.8) & (res_df['dist20_thu'].abs() <= 4.0)).sum()

print(f"Tight Range on Thursday (<= 3.0%): {tight_cnt} ({tight_cnt/len(res_df)*100:.1f}%)")
print(f"Volume Dry-Up on Thursday (<= 0.8x): {vdu_cnt} ({vdu_cnt/len(res_df)*100:.1f}%)")
print(f"Extreme Volume Dry-Up (<= 0.5x):    {extreme_vdu_cnt} ({extreme_vdu_cnt/len(res_df)*100:.1f}%)")
print(f"Resting within 4% of 20 EMA:        {near_ema20_cnt} ({near_ema20_cnt/len(res_df)*100:.1f}%)")
print(f"TEXTBOOK COIL (Tight + VDU + EMA):  {textbook_cnt} ({textbook_cnt/len(res_df)*100:.1f}%)")

print("\n--- ARCHETYPE BREAKDOWN ---")
print(res_df['archetype'].value_counts())

print("\n--- VPS ENGINE CLASSIFICATION ON FRIDAY ---")
print("Bucket counts:")
print(res_df['vps_bucket'].value_counts())
print("\nML Verdict counts:")
print(res_df['vps_ml'].value_counts())

print("\n--- TOP REJECTION REASONS ON FRIDAY ---")
print(res_df[res_df['vps_ml'] == 'REJECTED']['vps_rej'].value_counts().head(10))
