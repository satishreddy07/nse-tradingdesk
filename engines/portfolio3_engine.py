#!/usr/bin/env python3
# ==============================================================================
# AUTOMATED PAPER TRADING ENGINE (HYBRID SQUEEZE + SPLIT EXIT)
# ==============================================================================
import os
import sys
import sqlite3
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

DB_PATH = os.environ.get("P3_DB_PATH", "paper_portfolio.db")
FRICTION_PCT = 0.30  # 0.30% delivery brokerage, STT, and slippage deduction

SCAN_CLAUSE = "( {cash} (  market cap >=  300 and  daily sma(  daily volume *  daily close , 50 ) >=  10000000 and  yearly return on capital employed percentage >=  10 and  yearly return on net worth percentage >=  10 and  yearly debt equity ratio <=  1.5 and  quarterly eps after extraordinary items basic >  0 and  quarterly net profit/reported profit after tax >  0 and  daily close >  75 ) )"

# ------------------------------------------------------------------------------
# 1. DATABASE SETUP
# ------------------------------------------------------------------------------
def init_db(db_path=None):
    target_db = db_path or DB_PATH
    conn = sqlite3.connect(target_db)
    cur = conn.cursor()
    
    # Watchlist for T-1 Coils awaiting breakout confirmation
    cur.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            setup_date TEXT,
            trigger_high REAL,
            base_low REAL,
            days_tracked INTEGER DEFAULT 0,
            status TEXT DEFAULT 'PENDING'
        )
    ''')

    # Active and Historical Paper Trades
    cur.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            entry_date TEXT,
            entry_price REAL,
            stop_loss REAL,
            risk REAL,
            target_p1 REAL,
            p1_filled INTEGER DEFAULT 0,
            p1_exit_price REAL DEFAULT 0,
            p1_exit_date TEXT,
            p2_active INTEGER DEFAULT 1,
            p2_exit_price REAL DEFAULT 0,
            p2_exit_date TEXT,
            status TEXT DEFAULT 'OPEN',
            net_ret_pct REAL DEFAULT 0,
            r_multiple REAL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# ------------------------------------------------------------------------------
# 2. CHARTINK SCRAPER
# ------------------------------------------------------------------------------
def get_chartink_universe():
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://chartink.com/screener/"
    }
    try:
        resp = session.get("https://chartink.com/screener/", headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        csrf_token = soup.find("meta", {"name": "csrf-token"})["content"]

        post_headers = {
            **headers,
            "X-CSRF-TOKEN": csrf_token,
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }
        r = session.post(
            "https://chartink.com/screener/process",
            headers=post_headers,
            data={"scan_clause": SCAN_CLAUSE},
            timeout=20
        )
        data = r.json().get("data", [])
        return [f"{str(sym['nsecode']).strip()}.NS" for sym in data]
    except Exception as e:
        print(f"[!] Chartink fetch error: {e}")
        return []

# ------------------------------------------------------------------------------
# 3. DATA FETCHER HELPER
# ------------------------------------------------------------------------------
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        close = pd.to_numeric(df['Close'], errors='coerce')
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        vol = pd.to_numeric(df['Volume'], errors='coerce')

        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA10'] = close.ewm(span=10, adjust=False).mean()
        df['SMA50'] = close.rolling(50).mean()
        df['Vol_SMA20'] = vol.rolling(20).mean()

        prev_c = close.shift(1)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        df['ATR5'] = tr.rolling(5).mean()
        df['ATR20'] = tr.rolling(20).mean()
        return df
    except Exception:
        return None

# ------------------------------------------------------------------------------
# 4. STEP A: UPDATE ACTIVE OPEN POSITIONS
# ------------------------------------------------------------------------------
def update_open_positions(db_path=None):
    target_db = db_path or DB_PATH
    conn = sqlite3.connect(target_db)
    cur = conn.cursor()
    cur.execute("SELECT * FROM positions WHERE status = 'OPEN'")
    open_pos = cur.fetchall()

    if not open_pos:
        conn.close()
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n[*] Tracking {len(open_pos)} active paper positions...")

    for row in open_pos:
        pos_id, ticker, entry_date, entry_price, stop_loss, risk, target_p1, p1_filled, p1_exit_p, p1_exit_d, p2_act, p2_exit_p, p2_exit_d, status, _, _ = row
        df = get_stock_data(ticker, period="1mo")
        if df is None:
            continue

        today_bar = df.iloc[-1]
        t_high = float(today_bar['High'])
        t_low = float(today_bar['Low'])
        t_close = float(today_bar['Close'])
        t_ema10 = float(today_bar['EMA10'])

        updated = False

        # Phase 1: Target not yet reached
        if not p1_filled:
            if t_low <= stop_loss:
                # Stopped out completely
                p1_exit_p = stop_loss
                p2_exit_p = stop_loss
                p1_filled = 1
                p2_act = 0
                status = 'CLOSED'
                p1_exit_d = today_str
                p2_exit_d = today_str
                updated = True
            elif t_high >= target_p1:
                # Bank Part 1, move stop to breakeven
                p1_filled = 1
                p1_exit_p = target_p1
                p1_exit_d = today_str
                stop_loss = entry_price
                updated = True
        
        # Phase 2: Part 1 locked, managing Part 2 runner
        if p1_filled and p2_act and status == 'OPEN':
            if t_low <= stop_loss:
                p2_exit_p = stop_loss
                p2_act = 0
                status = 'CLOSED'
                p2_exit_d = today_str
                updated = True
            elif t_close < t_ema10:
                p2_exit_p = t_close
                p2_act = 0
                status = 'CLOSED'
                p2_exit_d = today_str
                updated = True

        if updated:
            net_ret = 0.0
            r_mult = 0.0
            if status == 'CLOSED':
                p1_r = (p1_exit_p - entry_price) / risk if risk > 0 else 0.0
                p2_r = (p2_exit_p - entry_price) / risk if risk > 0 else 0.0
                r_mult = round((0.5 * p1_r) + (0.5 * p2_r), 2)

                p1_ret = (p1_exit_p - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
                p2_ret = (p2_exit_p - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
                net_ret = round((0.5 * p1_ret) + (0.5 * p2_ret) - FRICTION_PCT, 2)

            cur.execute('''
                UPDATE positions SET
                    stop_loss = ?, p1_filled = ?, p1_exit_price = ?, p1_exit_date = ?,
                    p2_active = ?, p2_exit_price = ?, p2_exit_date = ?, status = ?,
                    net_ret_pct = ?, r_multiple = ?
                WHERE id = ?
            ''', (stop_loss, p1_filled, p1_exit_p, p1_exit_d, p2_act, p2_exit_p, p2_exit_d, status, net_ret, r_mult, pos_id))
            conn.commit()

    conn.close()

# ------------------------------------------------------------------------------
# 5. STEP B: EVALUATE WATCHLIST & TRIGGER NEW ENTRIES
# ------------------------------------------------------------------------------
def evaluate_watchlist(db_path=None):
    target_db = db_path or DB_PATH
    conn = sqlite3.connect(target_db)
    cur = conn.cursor()
    cur.execute("SELECT ticker, setup_date, trigger_high, base_low, days_tracked FROM watchlist WHERE status = 'PENDING'")
    watchlist_items = cur.fetchall()

    if not watchlist_items:
        conn.close()
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n[*] Auditing {len(watchlist_items)} coiled candidates on watchlist...")

    for item in watchlist_items:
        ticker, setup_date, trigger_high, base_low, days_tracked = item
        df = get_stock_data(ticker, period="1mo")
        if df is None:
            continue

        bar = df.iloc[-1]
        c_high = float(bar['High'])
        c_low = float(bar['Low'])
        c_close = float(bar['Close'])
        c_vol = float(bar['Volume'])
        c_vol_sma = float(bar['Vol_SMA20'])
        c_ema20 = float(bar['EMA20'])

        # 1. Breakout Trigger Confirmation (Trigger High * 1.001 with volume pacing >= 3.12x)
        trigger_level = trigger_high * 1.001
        if (c_high >= trigger_level) and (c_vol >= c_vol_sma * 3.12):
            entry_price = round(max(trigger_level, float(bar['Open'])), 2)
            stop_loss = round(base_low * 0.998, 2)
            risk = round(entry_price - stop_loss, 2)

            if risk > 0:
                target_p1 = round(entry_price + (2.0 * risk), 2)
                cur.execute('''
                    INSERT INTO positions (ticker, entry_date, entry_price, stop_loss, risk, target_p1)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ticker, today_str, entry_price, stop_loss, risk, target_p1))
                cur.execute("UPDATE watchlist SET status = 'TRIGGERED' WHERE ticker = ?", (ticker,))
                conn.commit()
                print(f"   🚀 TRIGGERED BUY: {ticker} @ ₹{entry_price} | SL: ₹{stop_loss} | Target (+2R): ₹{target_p1}")
                continue

        # 2. Structural Breakdown (Close below 20 EMA or volume churn)
        if (c_close < c_ema20) or (c_vol > c_vol_sma and c_close <= trigger_high):
            cur.execute("UPDATE watchlist SET status = 'INVALIDATED' WHERE ticker = ?", (ticker,))
            conn.commit()
            continue

        # 3. Inside Bar / Roll or Expiry
        if c_vol <= c_vol_sma * 0.80:
            days_tracked += 1
            if days_tracked >= 2:
                cur.execute("UPDATE watchlist SET status = 'EXPIRED' WHERE ticker = ?", (ticker,))
            else:
                trigger_high = max(trigger_high, c_high)
                base_low = min(base_low, c_low)
                cur.execute("UPDATE watchlist SET trigger_high = ?, base_low = ?, days_tracked = ? WHERE ticker = ?",
                            (trigger_high, base_low, days_tracked, ticker))
            conn.commit()
        else:
            cur.execute("UPDATE watchlist SET status = 'EXPIRED' WHERE ticker = ?", (ticker,))
            conn.commit()

    conn.close()

# ------------------------------------------------------------------------------
# 6. STEP C: SCAN FOR NEW HYBRID SETUPS
# ------------------------------------------------------------------------------
def scan_and_populate_watchlist(db_path=None, custom_tickers=None):
    target_db = db_path or DB_PATH
    print("\n[*] Querying Chartink for quality universe...")
    tickers = custom_tickers if custom_tickers is not None else get_chartink_universe()
    if not tickers:
        print("[!] No tickers returned from screener.")
        return 0

    conn = sqlite3.connect(target_db)
    cur = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')
    new_candidates = 0

    print(f"[*] Scanning {len(tickers)} stocks for Hybrid Squeeze setup...")

    def check_ticker(ticker):
        df = get_stock_data(ticker, period="3mo")
        if df is None or len(df) < 25:
            return None
        
        c_close = df['Close'].iloc[-1]
        c_high = df['High'].iloc[-1]
        c_low = df['Low'].iloc[-1]
        c_vol = df['Volume'].iloc[-1]
        
        ema20 = df['EMA20'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        vol_sma20 = df['Vol_SMA20'].iloc[-1]
        atr5 = df['ATR5'].iloc[-1]
        atr20 = df['ATR20'].iloc[-1]

        # Hybrid Setup Conditions
        vdu = (c_vol <= vol_sma20 * 0.80)
        atr_squeeze = (atr5 / atr20 <= 0.80) if (atr20 > 0 and pd.notna(atr5) and pd.notna(atr20)) else False
        uptrend = (ema20 > sma50) and (c_close >= ema20 * 0.98)
        proximity = (abs(c_close - ema20) / ema20 <= 0.04)
        tight_range = ((c_high - c_low) / c_low <= 0.035)

        # 3-Day Volume Fade
        vol_fade = (c_vol < df['Volume'].iloc[-2]) and (df['Volume'].iloc[-2] < df['Volume'].iloc[-3])

        if vdu and atr_squeeze and uptrend and proximity and tight_range and vol_fade:
            return {
                'ticker': ticker,
                'high': round(float(c_high), 2),
                'low': round(float(c_low), 2)
            }
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                # Add to watchlist if not already active or pending
                cur.execute("SELECT status FROM positions WHERE ticker = ? AND status = 'OPEN'", (res['ticker'],))
                if not cur.fetchone():
                    cur.execute('''
                        INSERT OR REPLACE INTO watchlist (ticker, setup_date, trigger_high, base_low, days_tracked, status)
                        VALUES (?, ?, ?, ?, 0, 'PENDING')
                    ''', (res['ticker'], today_str, res['high'], res['low']))
                    conn.commit()
                    new_candidates += 1

    conn.close()
    print(f"✅ Scanning complete. Added {new_candidates} new candidates to the watchlist.")
    return new_candidates

# ------------------------------------------------------------------------------
# 7. PORTFOLIO & WATCHLIST DASHBOARD REPORT
# ------------------------------------------------------------------------------
def generate_report(db_path=None):
    target_db = db_path or DB_PATH
    conn = sqlite3.connect(target_db)
    
    # 1. Active Watchlist Table
    df_watch = pd.read_sql_query("SELECT ticker, setup_date, trigger_high, base_low, days_tracked FROM watchlist WHERE status = 'PENDING'", conn)
    print("\n" + "="*70)
    print("CURRENT WATCHLIST (AWAITING BREAKOUT CONFIRMATION)")
    print("="*70)
    if not df_watch.empty:
        print(tabulate(df_watch, headers=["Ticker", "Coil Date", "Trigger High (₹)", "Base Low (₹)", "Days Coiled"], tablefmt="fancy_grid", showindex=False))
    else:
        print("No pending coils on watchlist.")

    # 2. Active Positions Table
    df_pos = pd.read_sql_query("SELECT ticker, entry_date, entry_price, stop_loss, target_p1, p1_filled, p2_active FROM positions WHERE status = 'OPEN'", conn)
    print("\n" + "="*70)
    print("ACTIVE PAPER POSITIONS")
    print("="*70)
    if not df_pos.empty:
        df_pos['Part1_Status'] = df_pos['p1_filled'].apply(lambda x: "Booked (+2R)" if x else "Waiting")
        df_pos['Part2_Status'] = df_pos['p2_active'].apply(lambda x: "Trailing 10 EMA" if x else "Closed")
        view_cols = ['ticker', 'entry_date', 'entry_price', 'stop_loss', 'target_p1', 'Part1_Status', 'Part2_Status']
        print(tabulate(df_pos[view_cols], headers=["Ticker", "Entry Date", "Entry (₹)", "Current Stop (₹)", "Target P1 (₹)", "Part 1", "Part 2"], tablefmt="fancy_grid", showindex=False))
    else:
        print("No active open positions.")

    # 3. Closed Trades Summary
    df_closed = pd.read_sql_query("SELECT ticker, entry_date, p2_exit_date, net_ret_pct, r_multiple FROM positions WHERE status = 'CLOSED'", conn)
    if not df_closed.empty:
        wins = df_closed[df_closed['net_ret_pct'] > 0]
        wr = (len(wins) / len(df_closed)) * 100
        avg_r = df_closed['r_multiple'].mean()
        avg_ret = df_closed['net_ret_pct'].mean()
        print("\n" + "="*70)
        print("CLOSED TRADES PERFORMANCE")
        print("="*70)
        print(f"Total Completed Trades : {len(df_closed)}")
        print(f"Win Rate               : {wr:.1f}%")
        print(f"Average Expectancy     : {avg_r:+.2f}R")
        print(f"Average Net Return     : {avg_ret:+.2f}% (Net of 0.30% charges)")
    conn.close()

# ------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    # Step 1: Manage running trades
    update_open_positions()
    # Step 2: Trigger new breakouts
    evaluate_watchlist()
    # Step 3: Scan for next-day candidates
    scan_and_populate_watchlist()
    # Step 4: Display portfolio dashboard
    generate_report()
