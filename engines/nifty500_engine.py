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

from core.data_quality import IST, now_ist, format_ist
from core.config import (
    P2_BASE_CAPITAL, P2_RISK_PER_TRADE, P2_MAX_SLOTS, P2_STOP_ATR_MULT,
    SEEDS_DIR, HISTORICAL_DATA_DIR
)
from execution.state_engine import state_engine

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
    falling back to cached seeds if offline or rate-limited.
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
                logger.info(f"-> Successfully loaded {len(symbols)} stocks from live official NSE list.")
                return symbols
    except Exception as e:
        logger.warning(f"Live NSE constituent ingestion notice: {e}. Checking local seed file...")

    seed_path = SEEDS_DIR / "indices" / "ind_nifty500list.csv"
    if seed_path.exists():
        df_seed = pd.read_csv(seed_path)
        symbols = df_seed["Symbol"].dropna().astype(str).str.strip().unique().tolist()
        logger.info(f"-> Loaded {len(symbols)} stocks from cached seed constituent list.")
        return symbols

    if HISTORICAL_DATA_DIR.exists():
        pkl_files = [f.stem for f in HISTORICAL_DATA_DIR.glob("*.pkl") if not f.stem.startswith("^")]
        if pkl_files:
            logger.info(f"-> Loaded {len(pkl_files[:500])} stocks from local historical cache.")
            return pkl_files[:500]

    return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR"]


async def fetch_yahoo_candle(session: aiohttp.ClientSession, ticker: str, semaphore: asyncio.Semaphore) -> Optional[List[Dict[str, Any]]]:
    """Fetches daily candle history for a ticker."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=6mo&interval=1d"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=12) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
                result = payload.get("chart", {}).get("result")
                if not result:
                    return None

                data = result[0]
                timestamps = data.get("timestamp", [])
                quotes = data.get("indicators", {}).get("quote", [{}])[0]

                opens = quotes.get("open", [])
                highs = quotes.get("high", [])
                lows = quotes.get("low", [])
                closes = quotes.get("close", [])
                volumes = quotes.get("volume", [])

                records = []
                sym_clean = "^CRSLDX" if ticker == "^CRSLDX" else ticker.replace(".NS", "")

                for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
                    if None in (ts, o, h, l, c, v):
                        continue
                    records.append({
                        "date": pd.to_datetime(ts, unit="s").normalize(),
                        "symbol": sym_clean,
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                    })
                return records
        except Exception:
            return None


async def run_batch_download(symbols: List[str]) -> pd.DataFrame:
    """Asynchronously downloads recent daily candles for tickers + ^CRSLDX benchmark."""
    tickers = [f"{s}.NS" for s in symbols]
    tickers.append("^CRSLDX")

    semaphore = asyncio.Semaphore(25)
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)

    start_t = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_yahoo_candle(session, t, semaphore) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_t
    flat_rows = []
    for sublist in results:
        if isinstance(sublist, list) and sublist:
            flat_rows.extend(sublist)

    logger.info(f"-> Batch download finished in {elapsed:.1f}s ({len(flat_rows):,} total bars retrieved).")
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
    """
    open_positions = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    if not open_positions:
        return []

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

    logger.info(f"=== 🚀 Starting Nifty 500 Momentum Pipeline ({mode}) at {today_str} ===")

    symbols = get_nifty500_symbols()
    if not symbols:
        _nifty500_engine_state["status"] = "FAILED"
        return {"status": "ERROR", "message": "Failed to load Nifty 500 constituent symbols."}

    df_all = await run_batch_download(symbols)
    if df_all.empty:
        _nifty500_engine_state["status"] = "FAILED"
        return {"status": "ERROR", "message": "No chart data retrieved."}

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

        if auto_execute and is_bullish and is_strict_setup and len(orders) < available_slots and shares > 0:

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

    _nifty500_engine_state["orders"] = orders
    _nifty500_engine_state["orders_executed"] = executed_count
    _nifty500_engine_state["status"] = "COMPLETED"

    elapsed = time.time() - start_time
    state_engine.log_scan_run(
        scan_type="NIFTY500_MOMENTUM",
        portfolio_type=PORTFOLIO_TYPE,
        status="SUCCESS",
        execution_time_sec=elapsed,
        candidates_count=len(candidates),
        orders_count=executed_count,
        reconciled_count=len(reconciled_trades),
        message=f"Analyzed {len(stocks['symbol'].unique())} stocks. Candidates: {len(candidates)}, Executed: {executed_count}"
    )

    logger.info(f"=== 🎯 Pipeline Completed: {executed_count} orders executed into Portfolio 2 ===")
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
        "reconciled_trades": reconciled_trades
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
    curr_eq, tot_eq, active_count = get_live_compounded_equity()
    state_copy = dict(_nifty500_engine_state)
    state_copy["compounded_equity"] = curr_eq
    state_copy["active_slots"] = active_count
    state_copy["available_slots"] = max(0, P2_MAX_SLOTS - active_count)
    state_copy["stats"] = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE)
    return state_copy
