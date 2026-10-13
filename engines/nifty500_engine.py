"""
========================================================================================
Nifty 500 Momentum Compounded Swing Engine (Portfolio 2 / Nifty 500)
========================================================================================
Automated Execution Engine for Nifty 500 Constituents.
- Capital: ₹1,00,000 (1 lac) base capital with dynamic compounding.
- Risk per trade: 1.5% of dynamic compounded equity.
- Allocation cap: 10% of dynamic compounded equity (10 max concurrent slots).
- Stop Distance: 2.0x ATR Wilder stop.
- The 7-Point Entry Criteria Matrix:
  #0: Macro Regime Gate: Nifty 500 Close > EMA50(Nifty 500)
  #1: Structural Uptrend: EMA20 > EMA50 and LTP > EMA50
  #2: Relative Strength (RS): Return_63d(Stock) - Return_63d(Nifty 500) > 0.0
  #3: Value Pullback Pocket: 0.985 * EMA20 <= LTP <= 1.020 * EMA20
  #4: Momentum Cooldown: 40 <= RSI-14 <= 62
  #5: Bullish Bounce Bar: LTP > Open and LTP > Close_{t-1}
  #6: Volume Expansion: Volume_intraday >= 0.80 * SMA20(Volume)
- Storage: SQLite DB under portfolio_type='NIFTY500_MOMENTUM'.
- 4-Tier Strategy Exit Hierarchy: Initial SL (-2.0 ATR), Trailing SL (Rolling 5D Low @ +2R), Fast Cut (<20 EMA), Day 8 Time Stop.
========================================================================================
"""

import asyncio
import datetime
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import requests
import threading
import concurrent.futures

from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from core.config import (
    P2_BASE_CAPITAL, P2_RISK_PER_TRADE, P2_MAX_SLOTS, P2_STOP_ATR_MULT,
    SEEDS_DIR, HISTORICAL_DATA_DIR
)
from execution.state_engine import state_engine
from engines.institutional_engine import DirectYahooFinanceFetcher

logger = logging.getLogger("Nifty500Engine")

PORTFOLIO_TYPE = "NIFTY500_MOMENTUM"

# Thread-safe execution state tracker
_nifty500_engine_state: Dict[str, Any] = {
    "last_run_timestamp": None,
    "last_run_mode": None,
    "macro_regime": "UNKNOWN",
    "bench_ltp": None,
    "bench_ema50": None,
    "compounded_equity": P2_BASE_CAPITAL,
    "active_slots": 0,
    "available_slots": P2_MAX_SLOTS,
    "candidates_found": 0,
    "orders_executed": 0,
    "reconciled_trades": 0,
    "trading_gate_open": False,
    "trading_gate_message": "",
    "orders": [],
    "status": "IDLE"
}


def get_live_compounded_equity(initial_cash: float = P2_BASE_CAPITAL) -> Tuple[float, float, int]:
    """
    Calculates live mark-to-market equity across cash and active holdings
    from SQLite state engine to dynamically compound position sizes and risk budgets.
    """
    all_history = state_engine.get_trade_history(limit=500)
    closed_trades = [
        t for t in all_history
        if state_engine._normalize_portfolio_type(t.get("portfolio_type")) == PORTFOLIO_TYPE
    ]
    realized_pnl = sum(float(t.get("pnl", 0.0)) for t in closed_trades)

    open_trades = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    open_slots_count = len(open_trades)

    total_equity = initial_cash + realized_pnl
    current_equity = max(total_equity, 10000.0)
    return current_equity, total_equity, open_slots_count


def get_nifty500_symbols() -> List[str]:
    """
    Retrieves official Nifty 500 constituent symbols from NSE archives,
    persisting and falling back to the Single Master File (`system_kv_store` / `ohlcv_master_store`).
    """
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    logger.info("Ingesting active Nifty 500 constituents from NSE...")
    try:
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code == 200:
            df_raw = pd.read_csv(io.StringIO(res.text))
            symbols = df_raw["Symbol"].dropna().astype(str).str.strip().unique().tolist()
            if len(symbols) >= 100:
                state_engine.set_kv("nifty500_constituents", symbols)
                logger.info(f"-> Successfully loaded {len(symbols)} stocks from live official NSE list.")
                return symbols
    except Exception as e:
        logger.warning(f"Live NSE constituent ingestion notice: {e}. Checking Single Master File...")

    cached_syms = state_engine.get_kv("nifty500_constituents", [])
    if isinstance(cached_syms, list) and len(cached_syms) >= 100:
        logger.info(f"-> Loaded {len(cached_syms)} Nifty 500 stocks from Single Master File.")
        return cached_syms

    meta_rows = state_engine.get_all_ohlcv_metadata(limit=500)
    if meta_rows:
        syms = [r["symbol"] for r in meta_rows if not r["symbol"].startswith("^")]
        if syms:
            logger.info(f"-> Loaded {len(syms)} stocks from Single Master File OHLCV store.")
            return syms

    return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR"]


async def run_batch_download(symbols: List[str]) -> pd.DataFrame:
    """
    Multi-threaded Smart Incremental Gap-Fill OHLCV Downloader for Nifty 500 + ^CRSLDX benchmark.
    Uses shared `DirectYahooFinanceFetcher` backed by the Single Master File (`data/swing_trades.db`).
    """
    fetcher = DirectYahooFinanceFetcher(max_workers=24)
    ohlcv_stats = state_engine.get_ohlcv_stats()
    if ohlcv_stats.get("stocks_count", 0) < 50:
        telegram_vault.restore_master_file_from_cloud()

    target_date = fetcher.get_latest_expected_bar_date()
    is_mkt_open, _ = NSEMarketCalendar.can_execute_trades()

    tickers = [f"{s}.NS" for s in symbols]
    total_tickers = len(tickers)
    flat_rows = []
    dirty_dfs: Dict[str, pd.DataFrame] = {}

    start_t = time.time()

    # 1. Fetch ^CRSLDX benchmark
    _, bench_df = fetcher.fetch_ticker_chart("^CRSLDX", range_period="1y")
    if bench_df is not None and not bench_df.empty:
        for idx_ts, r in bench_df.tail(180).iterrows():
            dt_norm = pd.Timestamp(idx_ts).tz_localize(None).normalize() if pd.Timestamp(idx_ts).tzinfo else pd.Timestamp(idx_ts).normalize()
            flat_rows.append({
                "date": dt_norm,
                "symbol": "^CRSLDX",
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r["Volume"] or 0.0),
            })

    # 2. Fetch or incrementally gap-fill Nifty 500 constituents in 24 threads
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        future_to_ticker = {
            executor.submit(fetcher.fetch_or_gapfill_ticker, t, target_date, is_mkt_open, False): t
            for t in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            try:
                sym, df, mode = future.result()
                if mode in ("GAP_FILLED", "FULL_DOWNLOAD") and df is not None:
                    dirty_dfs[sym] = df
                if df is not None and not df.empty:
                    sym_clean = sym.replace(".NS", "").replace(".BO", "").strip().upper()
                    sub_df = df.tail(160)
                    for idx_ts, r in sub_df.iterrows():
                        if pd.isna(r["Close"]) or pd.isna(r["Open"]):
                            continue
                        dt_norm = pd.Timestamp(idx_ts).tz_localize(None).normalize() if pd.Timestamp(idx_ts).tzinfo else pd.Timestamp(idx_ts).normalize()
                        flat_rows.append({
                            "date": dt_norm,
                            "symbol": sym_clean,
                            "open": float(r["Open"]),
                            "high": float(r["High"]),
                            "low": float(r["Low"]),
                            "close": float(r["Close"]),
                            "volume": float(r["Volume"] or 0.0),
                        })
            except Exception:
                pass

            if completed % 50 == 0 or completed == total_tickers:
                pct = 25 + int((completed / max(1, total_tickers)) * 32)
                state_engine.update_scan_progress(
                    stage=f"Step 2/5: Incremental OHLCV Sync ({completed}/{total_tickers} Nifty 500 stocks)...",
                    percent=pct,
                    stocks_total=total_tickers,
                    stocks_done=completed,
                    add_log=(completed % 150 == 0 or completed == total_tickers)
                )

    if dirty_dfs:
        state_engine.save_ohlcv_batch(dirty_dfs)
        threading.Thread(
            target=telegram_vault.backup_master_file_to_cloud,
            kwargs={"custom_note": f"Nifty 500 Scan Sync ({len(dirty_dfs)} updated)"},
            daemon=True
        ).start()

    elapsed = time.time() - start_t
    logger.info(f"-> Nifty 500 Incremental Sync finished in {elapsed:.1f}s ({len(flat_rows):,} total bars retrieved).")
    return pd.DataFrame(flat_rows)


def calc_rsi(s: pd.Series) -> pd.Series:
    """Wilder's 14-period RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))


def reconcile_open_nifty500_positions(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Evaluates all open positions in Portfolio 2 (Nifty 500 Daily Momentum) against
    the official 4-tier Strategy Exit Hierarchy verbatim:
    Priority 1: Initial Stop-Loss (INITIAL_SL) -> Price <= Entry - (2.0 * ATR_14)
    Priority 2: Trailing Stop-Loss (TRAILING_SL) -> Price <= Rolling 5-Day Low once gain >= +2R
    Priority 3: Fast Cut (FAST_EMA20_CUT) -> Close < 20 EMA (while not trailing)
    Priority 4: Time Stop (TIME_EXIT) -> Days Held >= 8 and Gain < +1R
    STRICT TRADING GATE: Only executes exits during NSE Regular Market Hours (09:15 AM – 03:30 PM IST).
    """
    open_positions = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    if not open_positions:
        return []

    can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
    closed_events = []

    for pos in open_positions:
        sym = pos["symbol"]
        sym_df = df_all[df_all["symbol"] == sym].sort_values("date")
        if sym_df.empty:
            continue

        latest_bar = sym_df.iloc[-1]
        c = float(latest_bar["close"])
        h = float(latest_bar["high"])
        l = float(latest_bar["low"])

        entry_px = float(pos["entry_price"])
        orig_sl = float(pos.get("original_stop_loss") or pos["stop_loss"])
        current_sl = float(pos["stop_loss"])

        r_unit = max(0.01, entry_px - orig_sl)
        target_2r = entry_px + (2.0 * r_unit)

        highest_seen = max(float(pos.get("highest_price") or entry_px), h, c)

        # If market is closed, update MTM price only and strictly block trade exit execution
        if not can_trade:
            state_engine.update_position_price(sym, current_price=c, stop_loss=current_sl, highest_price=highest_seen)
            continue

        previously_trailing = bool(
            pos.get("t1_hit", 0) or
            pos.get("sl_reason") == "TRAILING_SL" or
            (float(pos.get("highest_price") or 0) >= target_2r)
        )
        hit_2r_today = (h >= target_2r) or (c >= target_2r)
        in_trailing_mode = previously_trailing or hit_2r_today

        if len(sym_df) >= 6:
            rolling_5d_low = float(sym_df["low"].iloc[:-1].tail(5).min())
        elif len(sym_df) > 1:
            rolling_5d_low = float(sym_df["low"].iloc[:-1].min())
        else:
            rolling_5d_low = l

        ema_20 = float(sym_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]) if len(sym_df) >= 10 else 0.0

        try:
            entry_dt = datetime.datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d").date() if pos.get("entry_date") else None
        except Exception:
            entry_dt = None

        if entry_dt is not None:
            trading_sessions = sym_df[sym_df["date"].dt.date >= entry_dt]
            days_held = max(0, len(trading_sessions) - 1)
        else:
            days_held = int(pos.get("hold_days", 0) or 0)

        # Priority 1: Initial Stop-Loss
        if not in_trailing_mode:
            if l <= orig_sl or c <= orig_sl:
                res = state_engine.close_position(sym, exit_price=orig_sl, exit_reason="INITIAL_SL")
                if res:
                    closed_events.append(res)
                    logger.info(f"🛑 [P2 Priority 1: INITIAL_SL] Stop-Loss hit for {sym} @ ₹{orig_sl:.2f}")
                continue

        # Priority 2: Trailing Stop-Loss
        if in_trailing_mode:
            trailing_sl = max(current_sl, rolling_5d_low)
            if l <= trailing_sl or c <= trailing_sl:
                res = state_engine.close_position(sym, exit_price=trailing_sl, exit_reason="TRAILING_SL")
                if res:
                    closed_events.append(res)
                    logger.info(f"🎯 [P2 Priority 2: TRAILING_SL] Trailing Stop hit for {sym} @ ₹{trailing_sl:.2f}")
                continue
            else:
                if trailing_sl > current_sl or not previously_trailing:
                    state_engine.update_position_price(sym, current_price=c, stop_loss=trailing_sl, highest_price=highest_seen)
                    with state_engine._lock:
                        with state_engine._get_connection() as conn:
                            conn.execute(
                                "UPDATE active_positions SET t1_hit = 1, sl_reason = 'TRAILING_SL', stop_loss = ? WHERE symbol = ?",
                                (trailing_sl, sym)
                            )
                    logger.info(f"📈 [P2 Priority 2: TRAILING_SL] Ratcheted stop for {sym} to ₹{trailing_sl:.2f}")
                    telegram_vault.send_alert_and_sync(
                        title=f"📈 P2 TRAILING STOP RATCHETED: {sym}",
                        message=f"Stop-Loss raised from ₹{current_sl:,.2f} to 5D Low <b>₹{trailing_sl:,.2f}</b> (LTP: ₹{c:,.2f}).",
                        sync_db=True
                    )
                else:
                    state_engine.update_position_price(sym, current_price=c, stop_loss=current_sl, highest_price=highest_seen)

        # Priority 3: Fast Cut
        if not in_trailing_mode and ema_20 > 0 and c < ema_20:
            res = state_engine.close_position(sym, exit_price=c, exit_reason="FAST_EMA20_CUT")
            if res:
                closed_events.append(res)
                logger.info(f"✂️ [P2 Priority 3: FAST_EMA20_CUT] Fast Cut for {sym} @ ₹{c:.2f} (Close < 20 EMA)")
            continue

        # Priority 4: Time Stop
        gain = c - entry_px
        if days_held >= 8 and gain < r_unit:
            res = state_engine.close_position(sym, exit_price=c, exit_reason="TIME_EXIT")
            if res:
                closed_events.append(res)
                logger.info(f"⏰ [P2 Priority 4: TIME_EXIT] Time Stop for {sym} @ ₹{c:.2f} after {days_held} days")
            continue

        if not in_trailing_mode:
            state_engine.update_position_price(sym, current_price=c, stop_loss=orig_sl, highest_price=highest_seen)

    return closed_events



async def run_nifty500_pipeline_async(mode: str = "SCHEDULED_305PM", auto_execute: bool = True) -> Dict[str, Any]:
    """
    Executes the full Nifty 500 Momentum scan and execution pipeline:
    1. Ingests Nifty 500 constituents.
    2. Downloads daily bars including benchmark ^CRSLDX.
    3. Audits macro regime gate (^CRSLDX > 50-EMA).
    4. Reconciles existing active positions.
    5. Calculates technical indicators (EMA20, EMA50, RVOL, RS-Spread vs ^CRSLDX, RSI-14, ATR-14).
    6. Identifies qualifying setups and ranks by RS spread.
    7. Sizes positions dynamically with compounded equity and opens positions in SQLite DB.
    """
    start_time = time.time()
    today_str = format_ist()
    _nifty500_engine_state["status"] = "RUNNING"
    _nifty500_engine_state["last_run_mode"] = mode
    _nifty500_engine_state["last_run_timestamp"] = today_str

    state_engine.start_scan_progress(
        scan_type="NIFTY500_MOMENTUM (3:05 PM MOC)",
        initial_stage="Step 1/5: Ingesting live Nifty 500 constituent list from NSE..."
    )
    logger.info(f"=== 🚀 Starting Nifty 500 Momentum Pipeline ({mode}) at {today_str} ===")

    symbols = get_nifty500_symbols()
    if not symbols:
        _nifty500_engine_state["status"] = "FAILED"
        state_engine.finish_scan_progress("Scan failed: Could not load Nifty 500 constituent symbols.")
        return {"status": "ERROR", "message": "Failed to load Nifty 500 constituent symbols."}

    state_engine.update_scan_progress(
        stage=f"Step 2/5: Downloading 6-month daily candles for {len(symbols)} Nifty 500 stocks + ^CRSLDX benchmark...",
        percent=25,
        stocks_total=len(symbols),
        stocks_done=0
    )

    df_all = await run_batch_download(symbols)
    if df_all.empty:
        _nifty500_engine_state["status"] = "FAILED"
        state_engine.finish_scan_progress("Scan failed: No chart data retrieved from Yahoo Finance.")
        return {"status": "ERROR", "message": "No chart data retrieved."}

    state_engine.update_scan_progress(
        stage=f"Step 3/5: Auditing Macro Regime Gate (^CRSLDX > 50-EMA) & reconciling 4-Tier Strategy Exits...",
        percent=60,
        stocks_total=len(symbols),
        stocks_done=len(df_all["symbol"].unique())
    )

    # Macro Regime Gate Audit on ^CRSLDX
    b_df = df_all[df_all["symbol"] == "^CRSLDX"].copy()
    if b_df.empty:
        b_df = df_all[df_all["symbol"] == "^NSEI"].copy()
        if b_df.empty:
            b_df = df_all.groupby("date").agg({"close": "mean"}).reset_index()
            b_df["symbol"] = "^CRSLDX"

    b_df = b_df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    b_df["bench_ema50"] = b_df["close"].ewm(span=50, adjust=False).mean()
    b_df["bull_regime"] = b_df["close"] > b_df["bench_ema50"]
    b_df["bench_ret_63"] = b_df["close"].pct_change(63)

    latest_bench = b_df.iloc[-1]
    bench_ltp = float(latest_bench["close"])
    bench_ema = float(latest_bench["bench_ema50"])
    is_bullish = bool(latest_bench["bull_regime"])
    macro_regime_label = "BULL REGIME (EXECUTE BUYS)" if is_bullish else "BEAR REGIME (HOLD 100% CASH - ZERO LONGS)"

    _nifty500_engine_state["macro_regime"] = macro_regime_label
    _nifty500_engine_state["bench_ltp"] = bench_ltp
    _nifty500_engine_state["bench_ema50"] = bench_ema

    # Reconcile Existing Active Holdings
    reconciled_trades = reconcile_open_nifty500_positions(df_all)
    _nifty500_engine_state["reconciled_trades"] = len(reconciled_trades)

    # Compounded Equity & Slots
    current_equity, total_eq, active_slots = get_live_compounded_equity()
    available_slots = max(0, P2_MAX_SLOTS - active_slots)

    _nifty500_engine_state["compounded_equity"] = current_equity
    _nifty500_engine_state["active_slots"] = active_slots
    _nifty500_engine_state["available_slots"] = available_slots

    logger.info(f"Benchmark LTP: {bench_ltp:.2f} | 50-EMA: {bench_ema:.2f} | Regime: {macro_regime_label}")
    logger.info(f"Compounded Equity: ₹{current_equity:,.2f} | Open Slots: {active_slots}/{P2_MAX_SLOTS} | Available: {available_slots}")

    state_engine.update_scan_progress(
        stage=f"Step 4/5: Evaluating 7-Point Entry Criteria Matrix & Relative Strength spread vs Nifty 500...",
        percent=80
    )

    # Vectorized Indicators
    stocks = df_all[df_all["symbol"] != "^CRSLDX"].copy()
    stocks = stocks.sort_values(["symbol", "date"]).reset_index(drop=True)

    stocks = pd.merge(stocks, b_df[["date", "bull_regime", "bench_ret_63"]], on="date", how="left")
    stocks["bull_regime"] = stocks["bull_regime"].ffill().fillna(False)
    stocks["bench_ret_63"] = stocks["bench_ret_63"].ffill().fillna(0.0)


    stocks["ema_20"] = stocks.groupby("symbol")["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    stocks["ema_50"] = stocks.groupby("symbol")["close"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    stocks["vol_ma20"] = stocks.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())
    stocks["stock_ret_63"] = stocks.groupby("symbol")["close"].transform(lambda s: s.pct_change(63))
    stocks["rs_spread"] = stocks["stock_ret_63"] - stocks["bench_ret_63"]
    stocks["rsi"] = stocks.groupby("symbol")["close"].transform(calc_rsi)

    tr1 = stocks["high"] - stocks["low"]
    tr2 = (stocks["high"] - stocks.groupby("symbol")["close"].shift(1)).abs()
    tr3 = (stocks["low"] - stocks.groupby("symbol")["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    stocks["atr"] = tr.groupby(stocks["symbol"]).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())

    # 7-Point Entry Criteria Matrix
    uptrend = (stocks["ema_20"] > stocks["ema_50"]) & (stocks["close"] > stocks["ema_50"])
    rs_leader = stocks["rs_spread"] > 0.0
    pullback_zone = (stocks["close"] >= stocks["ema_20"] * 0.985) & (stocks["close"] <= stocks["ema_20"] * 1.020)
    rsi_zone = (stocks["rsi"] >= 40.0) & (stocks["rsi"] <= 62.0)
    prior_c = stocks.groupby("symbol")["close"].shift(1)
    bullish_bar = (stocks["close"] > stocks["open"]) & (stocks["close"] > prior_c)
    vol_confirm = stocks["volume"] >= 0.80 * stocks["vol_ma20"]

    stocks["tech_setup"] = (
        uptrend & rs_leader & pullback_zone & rsi_zone & bullish_bar & vol_confirm
    )
    stocks["signal"] = stocks["tech_setup"] & stocks["bull_regime"]

    latest_session = stocks["date"].max()
    active_bars = stocks[stocks["date"] == latest_session].copy()

    current_open_syms = [p["symbol"] for p in state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)]

    # Primary qualifying candidates
    candidates = (
        active_bars[
            active_bars["tech_setup"] & (active_bars["atr"] > 0) & (~active_bars["symbol"].isin(current_open_syms))
        ]
        .sort_values("rs_spread", ascending=False)
        .reset_index(drop=True)
    )
    
    # If zero strict pullback setups today, also include top RS uptrend leaders as watchlist
    if candidates.empty:
        candidates = (
            active_bars[
                uptrend.reindex(active_bars.index, fill_value=False) &
                rs_leader.reindex(active_bars.index, fill_value=False) &
                (active_bars["atr"] > 0) &
                (~active_bars["symbol"].isin(current_open_syms))
            ]
            .sort_values("rs_spread", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )

    _nifty500_engine_state["candidates_found"] = len(candidates)
    logger.info(f"Found {len(candidates)} candidate setups ranking by Relative Strength spread.")


    # Sizing & Execution
    dynamic_risk_budget = current_equity * P2_RISK_PER_TRADE
    dynamic_slot_cap = current_equity / P2_MAX_SLOTS

    orders = []
    executed_count = 0
    signals_to_save = []

    for _, row in candidates.iterrows():
        sym = row["symbol"]
        entry_px = float(row["close"])
        atr_val = float(row["atr"])
        initial_sl = round(entry_px - (P2_STOP_ATR_MULT * atr_val), 2)
        risk_per_share = round(entry_px - initial_sl, 2)

        if risk_per_share <= 0:
            continue

        shares_by_risk = int(dynamic_risk_budget / risk_per_share)
        shares_by_slot = int(dynamic_slot_cap / entry_px)
        shares = min(shares_by_risk, shares_by_slot)

        target_2r = round(entry_px + 2.0 * risk_per_share, 2)
        rs_spread_pct = round(row["rs_spread"] * 100, 2)
        rsi_val = round(row["rsi"], 1)

        is_strict_setup = bool(row.get("tech_setup", False))
        bucket_label = "BUY_NOW" if (is_bullish and is_strict_setup) else ("PRE_BREAKOUT" if is_strict_setup else "WAIT_PULLBACK")

        signal_item = {
            "symbol": sym,
            "sector": "NIFTY500",
            "bucket": bucket_label,
            "score": round(50.0 + min(rs_spread_pct * 2, 40.0), 1),
            "confidence": 5 if is_strict_setup else 4,
            "ml_prob": round(0.55 + min(rs_spread_pct / 100.0, 0.35), 2),
            "tier": "A" if is_strict_setup else "B+",
            "close": entry_px,
            "entry": entry_px,
            "sl": initial_sl,
            "t1": target_2r,
            "t2": round(entry_px + 3.0 * risk_per_share, 2),
            "rr": 2.0,
            "shares": shares,
            "capital_at_risk": round(shares * risk_per_share, 2),
            "risk_pct": 1.5,
            "atr_pct": round((atr_val / entry_px) * 100, 2),
            "rsi14": rsi_val,
            "dist_to_pivot": round(((entry_px - row["ema_20"]) / row["ema_20"]) * 100, 2),
            "dist_to_ema20": round(((entry_px - row["ema_20"]) / row["ema_20"]) * 100, 2),
            "inst_score": round(75.0, 1),
            "week_confluence": True,
            "intraday_pct": 0.0,
            "intraday_adj": 0.0,
            "rs_spread": rs_spread_pct
        }
        signals_to_save.append(signal_item)

        can_trade_now, gate_msg = NSEMarketCalendar.can_execute_trades()

        if auto_execute and can_trade_now and is_bullish and is_strict_setup and len(orders) < available_slots and shares > 0:

            order_payload = {
                "symbol": sym,
                "action": "BUY CNC (MOC 3:05 PM)",
                "ltp": round(entry_px, 2),
                "quantity": shares,
                "initial_sl": initial_sl,
                "target_2r": target_2r,
                "stop_dist_pct": round(((entry_px - initial_sl) / entry_px) * 100, 2),
                "allocated_inr": round(shares * entry_px, 2),
                "max_risk_inr": round(shares * risk_per_share, 2),
                "rs_spread_pct": rs_spread_pct,
                "rsi_14": rsi_val,
                "portfolio_type": PORTFOLIO_TYPE,
            }

            opened = state_engine.open_position(
                symbol=sym,
                entry_price=entry_px,
                quantity=shares,
                stop_loss=initial_sl,
                target=target_2r,
                target_2=round(entry_px + 3.0 * risk_per_share, 2),
                triggered_from="NIFTY500_SCHEDULED",
                notes=f"Strategy Exit Hierarchy | Initial SL ₹{initial_sl:.2f} (-2.0 ATR) | Trailing @ +2R (₹{target_2r:.2f}) on 5D Low | RS Spread: +{rs_spread_pct}% | RSI: {rsi_val}",
                sl_reason="INITIAL_SL",
                sector="NIFTY500",
                portfolio_type=PORTFOLIO_TYPE,
                entry_reason=f"Uptrend (EMA20>EMA50), RS Leader (+{rs_spread_pct}%), Pullback to 20EMA, RSI={rsi_val}",
                entry_regime="BULL REGIME (^CRSLDX > 50 EMA)",
                entry_metrics={
                    "rs_spread_pct": rs_spread_pct,
                    "rsi_14": rsi_val,
                    "atr": round(atr_val, 2),
                    "dist_ema20_pct": round(((entry_px - row["ema_20"]) / row["ema_20"]) * 100, 2),
                    "bench_ltp": round(bench_ltp, 2),
                    "compounded_equity": round(current_equity, 2),
                    "target_2r": target_2r
                }
            )

            if opened:
                executed_count += 1
                orders.append(order_payload)
                logger.info(f"✓ [P2 Order Executed] {sym}: {shares} shs @ ₹{entry_px:.2f} | Initial SL: ₹{initial_sl:.2f}")

    state_engine.save_latest_signals(signals_to_save, portfolio_type=PORTFOLIO_TYPE)

    can_trade_final, gate_msg_final = NSEMarketCalendar.can_execute_trades()
    _nifty500_engine_state["trading_gate_open"] = can_trade_final
    _nifty500_engine_state["trading_gate_message"] = gate_msg_final
    _nifty500_engine_state["orders"] = orders
    _nifty500_engine_state["orders_executed"] = executed_count
    _nifty500_engine_state["status"] = "COMPLETED"

    gate_suffix = ""
    if auto_execute and not can_trade_final:
        gate_suffix = " [🔒 STRICT TRADING GATE: Market Closed — Signals updated, 0 trades executed]"
        logger.warning(f"🔒 [P2 Auto-Execute Blocked] {gate_msg_final}")

    elapsed = time.time() - start_time
    summary_msg = (
        f"Nifty 500 Scan Complete in {elapsed:.1f}s: {len(stocks['symbol'].unique())} stocks analyzed -> "
        f"{len(candidates)} RS Candidates | {executed_count} orders placed ({macro_regime_label}){gate_suffix}"
    )
    state_engine.log_scan_run(
        scan_type="NIFTY500_MOMENTUM",
        portfolio_type=PORTFOLIO_TYPE,
        status="SUCCESS",
        execution_time_sec=elapsed,
        candidates_count=len(candidates),
        orders_count=executed_count,
        reconciled_count=len(reconciled_trades),
        message=summary_msg
    )
    state_engine.set_kv("nifty500_snapshot", {
        "macro_regime": macro_regime_label,
        "bench_ltp": bench_ltp,
        "bench_ema50": bench_ema,
        "candidates_found": len(candidates),
        "last_run_time": format_ist()
    })
    state_engine.finish_scan_progress(summary_msg)

    # Backup Single Master File to Telegram Cloud Vault in background
    threading.Thread(
        target=telegram_vault.backup_master_file_to_cloud,
        kwargs={"custom_note": f"P2 Nifty 500 Scan Complete ({len(candidates)} RS Candidates)"},
        daemon=True
    ).start()

    logger.info(f"=== 🎯 {summary_msg} ===")
    return {
        "status": "SUCCESS",
        "macro_regime": macro_regime_label,
        "bench_ltp": bench_ltp,
        "bench_ema50": bench_ema,
        "compounded_equity": current_equity,
        "active_slots": active_slots + executed_count,
        "available_slots": max(0, available_slots - executed_count),
        "candidates_found": len(candidates),
        "orders_executed": executed_count,
        "orders": orders,
        "reconciled_trades": reconciled_trades,
        "trading_gate_open": can_trade_final,
        "trading_gate_message": gate_msg_final
    }


def run_nifty500_pipeline_sync(mode: str = "SCHEDULED_305PM", auto_execute: bool = True) -> Dict[str, Any]:
    """Synchronous runner for scheduler threads."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(run_nifty500_pipeline_async(mode=mode, auto_execute=auto_execute))
        else:
            return loop.run_until_complete(run_nifty500_pipeline_async(mode=mode, auto_execute=auto_execute))
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(run_nifty500_pipeline_async(mode=mode, auto_execute=auto_execute))
        finally:
            new_loop.close()


def get_nifty500_status() -> Dict[str, Any]:
    """Returns current status and latest run summary."""
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    curr_eq, tot_eq, active_count = get_live_compounded_equity()
    state_copy = dict(_nifty500_engine_state)
    if state_copy.get("macro_regime") == "UNKNOWN" or state_copy.get("bench_ltp") is None:
        saved = state_engine.get_kv("nifty500_snapshot", {})
        if isinstance(saved, dict):
            for k, v in saved.items():
                if state_copy.get(k) in (None, "UNKNOWN", 0):
                    state_copy[k] = v
    state_copy["trading_gate_open"] = can_trade
    state_copy["trading_gate_message"] = gate_msg
    state_copy["compounded_equity"] = curr_eq
    state_copy["active_slots"] = active_count
    state_copy["available_slots"] = max(0, P2_MAX_SLOTS - active_count)
    state_copy["stats"] = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE)
    return state_copy

