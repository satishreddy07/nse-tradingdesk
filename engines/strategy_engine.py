"""
========================================================================================
Portfolio 3: Pre-Breakout Coil & Intraday Volume Confirmation Strategy Engine
File: engines/strategy_engine.py
========================================================================================
Automated Strategy Engine for NSE Cash Equities:
1. Watchlist Ingestion (EOD Pre-Breakout Coils):
   - Volume Dry-Up (VDU): T-1 Volume <= 0.5x to 0.8x of 20-day SMA volume.
   - 20 EMA Proximity: Close within +/- 3.8% of 20-day EMA.
   - Price Range Contraction: T-1 daily high-low range <= 3.2%.
   - Trend Filter: Close >= 20 EMA, and 20 EMA > 50 SMA.
2. Live Intraday Trigger & Validation (09:15 - 10:15 IST):
   - Scheduled check every 30 minutes (09:45, 10:15 IST).
   - Entry Condition: Current price crosses above yesterday's High (T-1 High).
   - Volume Confirmation Filter: Time-adjusted volume pacing >= 3.12x of 20-day average.
3. Risk Management & Stop Loss:
   - Initial Stop-Loss (1R): Strictly anchored at T-1 Low * 0.998 (0.2% safety buffer).
   - Fixed Risk Capital Model: Risk 1% of total account capital per trade.
   - Quantity Sizing: floor((Account Capital * 1%) / Risk per share).
4. The Split Exit Plan (Automated Order Management):
   - Once filled, split the position quantity 50 / 50:
     - First Half (50% Qty): Place an automated target order at +2.0R profit from entry.
     - Second Half (50% Qty): Hold as a runner, and dynamically trail the stop-loss
       upward using the 10-day Exponential Moving Average (10 EMA) on daily candles.
========================================================================================
"""

import asyncio
import concurrent.futures
import datetime
import json
import logging
import math
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import requests

from core.config import (
    P3_BASE_CAPITAL,
    P3_RISK_PER_TRADE,
    P3_MAX_SLOTS,
    P3_ALLOCATION_CAP_PCT,
    P3_TARGET_R,
    P3_VOLUME_PACING_THRESHOLD,
    DB_PATH,
    CHARTINK_URL
)
from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from engines.institutional_engine import (
    DirectYahooFinanceFetcher,
    get_intraday_volume_factor,
    _fast_ewm_array,
    get_stock_sector
)
from execution.state_engine import state_engine

logger = logging.getLogger("Portfolio3StrategyEngine")

PORTFOLIO_TYPE = "PORTFOLIO_3_PRE_BREAKOUT"
ENGINE_VERSION = "1.0.0-PROD-PORTFOLIO3"

# In-memory thread-safe state tracking dictionary
_portfolio3_state: Dict[str, Any] = {
    "version": ENGINE_VERSION,
    "portfolio_type": PORTFOLIO_TYPE,
    "status": "IDLE",
    "last_scan_timestamp": None,
    "last_intraday_check_timestamp": None,
    "last_reconcile_timestamp": None,
    "base_capital": P3_BASE_CAPITAL,
    "risk_per_trade_pct": P3_RISK_PER_TRADE,
    "max_slots": P3_MAX_SLOTS,
    "volume_pacing_threshold": P3_VOLUME_PACING_THRESHOLD,
    "target_r_multiple": P3_TARGET_R,
    "watchlist_count": 0,
    "active_positions_count": 0,
    "active_slots_used": 0,
    "available_slots": P3_MAX_SLOTS,
    "open_legs": {},         # symbol -> dual-leg detail dict
    "recent_triggers": [],   # list of recent breakout triggers
    "recent_exits": []       # list of recent split exits
}

_state_lock = threading.RLock()


# ==============================================================================
# TECHNICAL INDICATOR & COIL EVALUATION HELPERS
# ==============================================================================

def evaluate_prebreakout_coil(
    df: pd.DataFrame,
    symbol: str,
    vdu_min: float = 0.0,
    vdu_max: float = 0.80,
    ema20_dist_max_pct: float = 3.8,
    range_contraction_max_pct: float = 3.2
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether a stock meets the 4 core Pre-Breakout Coil rules on completed T-1 daily bar:
    1. Volume Dry-Up (VDU): T-1 Volume <= 0.5x to 0.8x of 20-day SMA volume.
    2. 20 EMA Proximity: Close within +/- 3.8% of 20-day EMA.
    3. Price Range Contraction: T-1 daily high-low range <= 3.2% of Close.
    4. Trend Filter: Close >= 20 EMA, and 20 EMA > 50 SMA.
    """
    if df is None or len(df) < 55:
        return None

    # Determine T-1 index: ensure we evaluate the latest completed daily bar
    now = now_ist()
    last_dt = df.index[-1]
    is_today_partial = (
        isinstance(last_dt, (pd.Timestamp, datetime.datetime))
        and last_dt.date() == now.date()
        and now.time() < datetime.time(15, 30)
    )

    t1_idx = -2 if (is_today_partial and len(df) >= 56) else -1
    t1_date = df.index[t1_idx]
    t1_date_str = t1_date.strftime("%Y-%m-%d") if hasattr(t1_date, "strftime") else str(t1_date)[:10]

    close_arr = df["Close"].to_numpy(dtype=np.float64, copy=False)
    high_arr = df["High"].to_numpy(dtype=np.float64, copy=False)
    low_arr = df["Low"].to_numpy(dtype=np.float64, copy=False)
    vol_arr = df["Volume"].to_numpy(dtype=np.float64, copy=False)

    n_bars = len(close_arr)
    eval_pos = n_bars + t1_idx if t1_idx < 0 else t1_idx

    if eval_pos < 50:
        return None

    c_t1 = float(close_arr[eval_pos])
    h_t1 = float(high_arr[eval_pos])
    l_t1 = float(low_arr[eval_pos])
    v_t1 = float(vol_arr[eval_pos])

    if c_t1 <= 0 or h_t1 <= 0 or l_t1 <= 0 or h_t1 < l_t1:
        return None

    # 1. 20-day SMA of Volume ending at T-1
    vol_slice = vol_arr[max(0, eval_pos - 19) : eval_pos + 1]
    avg_vol_20 = float(np.mean(vol_slice)) if len(vol_slice) >= 15 else float(np.mean(vol_slice or [1.0]))
    if avg_vol_20 <= 0:
        avg_vol_20 = 1.0

    # Volume Dry-Up (VDU)
    vdu_ratio = round(v_t1 / avg_vol_20, 3)
    vdu_passed = (vdu_min <= vdu_ratio <= vdu_max)

    # 2. 20 EMA and 50 SMA ending at T-1
    ema20_series = _fast_ewm_array(close_arr[: eval_pos + 1], span=20)
    ema20_t1 = float(ema20_series[-1]) if len(ema20_series) > 0 else c_t1

    close_50_slice = close_arr[max(0, eval_pos - 49) : eval_pos + 1]
    sma50_t1 = float(np.mean(close_50_slice)) if len(close_50_slice) >= 30 else ema20_t1

    ema20_dist_pct = round(abs(c_t1 - ema20_t1) / ema20_t1 * 100.0, 2)
    ema20_prox_passed = (ema20_dist_pct <= ema20_dist_max_pct)

    # 3. Price Range Contraction
    range_contraction_pct = round(((h_t1 - l_t1) / c_t1) * 100.0, 2)
    range_contraction_passed = (range_contraction_pct <= range_contraction_max_pct)

    # 4. Trend Filter: Close >= 20 EMA and 20 EMA > 50 SMA
    trend_filter_passed = (c_t1 >= ema20_t1) and (ema20_t1 > sma50_t1)

    all_passed = (vdu_passed and ema20_prox_passed and range_contraction_passed and trend_filter_passed)

    if not all_passed:
        return None

    # Calculate 10-day EMA for trailing stop benchmark
    ema10_series = _fast_ewm_array(close_arr[: eval_pos + 1], span=10)
    ema10_t1 = float(ema10_series[-1]) if len(ema10_series) > 0 else c_t1

    # Initial Stop-Loss strictly anchored at T-1 Low * 0.998 (0.2% buffer)
    initial_sl = round(l_t1 * 0.998, 2)
    breakout_trigger_price = round(h_t1, 2)
    risk_per_share = round(breakout_trigger_price - initial_sl, 2)

    return {
        "symbol": symbol.replace(".NS", "").replace(".BO", "").strip().upper(),
        "t_minus_1_date": t1_date_str,
        "t_minus_1_close": round(c_t1, 2),
        "t_minus_1_high": breakout_trigger_price,
        "t_minus_1_low": round(l_t1, 2),
        "initial_sl": initial_sl,
        "risk_per_share": risk_per_share,
        "vdu_ratio": vdu_ratio,
        "avg_vol_20": round(avg_vol_20, 0),
        "ema20": round(ema20_t1, 2),
        "ema50": round(sma50_t1, 2),
        "ema10": round(ema10_t1, 2),
        "ema20_dist_pct": ema20_dist_pct,
        "range_contraction_pct": range_contraction_pct,
        "status": "WATCHING"
    }


def calculate_volume_pacing(
    live_volume: float,
    avg_volume_20: float,
    current_time: Optional[datetime.datetime] = None
) -> float:
    """
    Computes time-adjusted volume pacing relative to the 20-day average volume
    for the current time of day in NSE regular market hours (09:15 to 15:30 IST, 375 mins).
    """
    if avg_volume_20 <= 0:
        return 1.0

    if current_time is None:
        current_time = now_ist()

    # Get intraday volume factor curve (U-shaped equity distribution)
    vol_factor = get_intraday_volume_factor(current_time)
    expected_vol = avg_volume_20 * vol_factor

    if expected_vol <= 0:
        return 1.0

    pacing = live_volume / expected_vol
    return round(float(pacing), 2)


# ==============================================================================
# STRATEGY ENGINE CORE CLASS
# ==============================================================================

class Portfolio3StrategyEngine:
    """
    Automated Execution & Split Exit Engine for Portfolio 3 (Pre-Breakout Coil Strategy).
    Manages:
    1. Watchlist Ingestion from EOD daily OHLCV.
    2. Live 30-minute breakout scans & volume confirmation gating (09:15 - 10:15 IST).
    3. Fixed 1% risk capital position sizing & OMS order execution.
    4. Automated 50/50 Split Exit: +2.0R target lock-in on Leg 1 & 10 EMA trailing runner on Leg 2.
    """

    _instance: Optional["Portfolio3StrategyEngine"] = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(Portfolio3StrategyEngine, cls).__new__(cls)
            return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.db_path = db_path or str(DB_PATH)
        self.portfolio_type = PORTFOLIO_TYPE
        self.base_capital = P3_BASE_CAPITAL
        self.risk_per_trade_pct = P3_RISK_PER_TRADE
        self.max_slots = P3_MAX_SLOTS
        self.allocation_cap_pct = P3_ALLOCATION_CAP_PCT
        self.target_r = P3_TARGET_R
        self.volume_pacing_threshold = P3_VOLUME_PACING_THRESHOLD

        self.fetcher = DirectYahooFinanceFetcher(max_workers=20)
        self._sync_state_from_db()
        self._initialized = True
        logger.info(f"✓ Portfolio 3 Strategy Engine initialized ({PORTFOLIO_TYPE}).")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _sync_state_from_db(self):
        """Re-synchronizes in-memory tracking dictionary from SQLite database."""
        with _state_lock:
            try:
                with self._get_connection() as conn:
                    # 1. Count active watchlist items
                    w_count = conn.execute(
                        "SELECT COUNT(*) FROM p3_prebreakout_watchlist WHERE status = 'WATCHING'"
                    ).fetchone()[0]

                    # 2. Query active positions for Portfolio 3
                    active_positions = state_engine.get_active_positions(portfolio_type=self.portfolio_type)
                    pos_count = len(active_positions)

                    # 3. Query dual-leg details
                    legs_rows = conn.execute(
                        "SELECT * FROM p3_position_legs WHERE overall_status IN ('OPEN', 'PARTIAL_CLOSED')"
                    ).fetchall()

                    open_legs = {}
                    for row in legs_rows:
                        item = dict(row)
                        open_legs[item["symbol"]] = item

                    _portfolio3_state["watchlist_count"] = w_count
                    _portfolio3_state["active_positions_count"] = pos_count
                    _portfolio3_state["active_slots_used"] = pos_count
                    _portfolio3_state["available_slots"] = max(0, self.max_slots - pos_count)
                    _portfolio3_state["open_legs"] = open_legs
            except Exception as e:
                logger.error(f"Error syncing Portfolio 3 state from DB: {e}")

    def get_stock_daily_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches or retrieves cached/gapfilled daily OHLCV DataFrame for a symbol."""
        clean_sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        try:
            df = state_engine.get_ohlcv(clean_sym)
            if df is not None and len(df) >= 55:
                return df
        except Exception:
            pass
        try:
            _, df_gap, _ = self.fetcher.fetch_or_gapfill_ticker(clean_sym)
            if df_gap is not None and not df_gap.empty and len(df_gap) >= 55:
                return df_gap
        except Exception:
            pass
        try:
            _, df_chart = self.fetcher.fetch_ticker_chart(clean_sym, range_period="1y")
            if df_chart is not None and not df_chart.empty and len(df_chart) >= 55:
                return df_chart
        except Exception:
            pass
        return None

    # --------------------------------------------------------------------------
    # 1. WATCHLIST INGESTION (EOD PRE-BREAKOUT COILS)
    # --------------------------------------------------------------------------

    def ingest_prebreakout_watchlist(
        self,
        universe: Optional[List[str]] = None,
        sync_cloud: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scans candidate universe for EOD Pre-Breakout Coils and persists candidates
        into `p3_prebreakout_watchlist` with status 'WATCHING'.
        """
        start_t = time.time()
        logger.info("🔍 [Portfolio 3] Starting EOD Pre-Breakout Coil Watchlist Ingestion...")

        # If universe is not provided, fetch from state_engine master OHLCV or Chartink/Nifty
        symbols_to_scan = universe
        if not symbols_to_scan:
            try:
                meta_list = state_engine.get_all_ohlcv_metadata(limit=800)
                symbols_to_scan = [m["symbol"] for m in meta_list if m.get("bars_count", 0) >= 55]
            except Exception:
                symbols_to_scan = []

        if not symbols_to_scan:
            # Fallback to liquid default universe
            symbols_to_scan = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
                "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
                "BAJFINANCE", "TATAMOTORS", "SUNPHARMA", "ULTRACEMCO", "NTPC", "POWERGRID",
                "INDUSINDBK", "ADANIENT", "JSWSTEEL", "TATASTEEL", "COALINDIA", "HINDALCO",
                "TECHM", "WIPRO", "HCLTECH", "ONGC", "M&M", "BEL", "HAL", "TRENT"
            ]

        logger.info(f"Scanning {len(symbols_to_scan)} symbols for Pre-Breakout Coils...")
        candidates: List[Dict[str, Any]] = []

        # Batch-fetch or retrieve OHLCV
        now_dt = now_ist()
        scan_date_str = now_dt.strftime("%Y-%m-%d")

        for sym in symbols_to_scan:
            clean_sym = sym.replace(".NS", "").replace(".BO", "").strip().upper()
            try:
                df = self.get_stock_daily_df(clean_sym)
                if df is None or len(df) < 55:
                    continue

                coil = evaluate_prebreakout_coil(df, clean_sym)
                if coil:
                    candidates.append(coil)
            except Exception as sym_err:
                logger.debug(f"Error evaluating coil for {clean_sym}: {sym_err}")

        # Persist into SQLite p3_prebreakout_watchlist
        inserted_count = 0
        now_str = format_ist(now_dt)
        with self._get_connection() as conn:
            for c in candidates:
                try:
                    conn.execute("""
                    INSERT INTO p3_prebreakout_watchlist (
                        symbol, scan_date, t_minus_1_date, t_minus_1_close, t_minus_1_high,
                        t_minus_1_low, initial_sl, t_minus_1_range_pct, vdu_ratio,
                        ema20, ema50, ema20_dist_pct, avg_vol_20, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'WATCHING', ?, ?)
                    ON CONFLICT(symbol, scan_date) DO UPDATE SET
                        t_minus_1_close = excluded.t_minus_1_close,
                        t_minus_1_high = excluded.t_minus_1_high,
                        t_minus_1_low = excluded.t_minus_1_low,
                        initial_sl = excluded.initial_sl,
                        t_minus_1_range_pct = excluded.t_minus_1_range_pct,
                        vdu_ratio = excluded.vdu_ratio,
                        ema20 = excluded.ema20,
                        ema50 = excluded.ema50,
                        ema20_dist_pct = excluded.ema20_dist_pct,
                        avg_vol_20 = excluded.avg_vol_20,
                        updated_at = excluded.updated_at
                    """, (
                        c["symbol"], scan_date_str, c["t_minus_1_date"], c["t_minus_1_close"],
                        c["t_minus_1_high"], c["t_minus_1_low"], c["initial_sl"],
                        c["range_contraction_pct"], c["vdu_ratio"], c["ema20"], c["ema50"],
                        c["ema20_dist_pct"], c["avg_vol_20"], now_str, now_str
                    ))
                    inserted_count += 1
                except Exception as db_err:
                    logger.error(f"Failed to persist watchlist candidate {c['symbol']}: {db_err}")
            conn.commit()

        elapsed = round(time.time() - start_t, 2)
        logger.info(f"✓ [Portfolio 3 Watchlist Ingested] {len(candidates)} coil candidates saved in {elapsed}s.")

        # Update in-memory state
        with _state_lock:
            _portfolio3_state["last_scan_timestamp"] = now_str
            _portfolio3_state["watchlist_count"] = inserted_count

        if sync_cloud and candidates:
            tickers_summary = ", ".join([c["symbol"] for c in candidates[:10]])
            telegram_vault.send_alert_and_sync(
                title=f"📋 P3 PRE-BREAKOUT WATCHLIST INGESTED ({len(candidates)} Stocks)",
                message=(
                    f"Identified <b>{len(candidates)}</b> pre-breakout coils meeting VDU (<=0.8x), "
                    f"20 EMA proximity (+-3.8%), range contraction (<=3.2%), and trend filter (Close>=20 EMA>50 SMA).\n"
                    f"Top candidates: <code>{tickers_summary}</code>\n"
                    f"Intraday 30m breakout scanning will activate tomorrow 09:15-10:15 IST."
                ),
                sync_db=True
            )

        return candidates

    def get_active_watchlist(self) -> List[Dict[str, Any]]:
        """Returns the list of currently active watchlist candidates."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM p3_prebreakout_watchlist
            WHERE status = 'WATCHING'
            ORDER BY id DESC
            """).fetchall()
            return [dict(r) for r in rows]

    # --------------------------------------------------------------------------
    # 2. LIVE INTRADAY TRIGGER & VALIDATION (09:15 - 10:15 IST)
    # --------------------------------------------------------------------------

    def is_intraday_window_active(self, dt: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
        """Validates if the 09:15 - 10:15 IST morning breakout window is active."""
        if dt is None:
            dt = now_ist()

        if not NSEMarketCalendar.is_trading_day(dt.date()):
            return False, f"Today ({dt.strftime('%Y-%m-%d')}) is not an NSE trading session."

        t = dt.time()
        start_time = datetime.time(9, 15)
        end_time = datetime.time(10, 15)

        if start_time <= t <= end_time:
            return True, f"Active morning breakout window (Current: {t.strftime('%H:%M:%S')} IST)."
        elif t < start_time:
            return False, f"Pre-market: Market opens at 09:15 IST (Current: {t.strftime('%H:%M:%S')} IST)."
        else:
            return False, f"Breakout window closed at 10:15 IST (Current: {t.strftime('%H:%M:%S')} IST)."

    async def async_fetch_symbols_quotes(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Asynchronously fetches real-time quote metadata (LTP, Day High, Day Low, Volume)
        for watchlist symbols via Yahoo Finance v8 API.
        """
        quotes: Dict[str, Dict[str, Any]] = {}
        if not symbols:
            return quotes

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        connector = aiohttp.TCPConnector(limit=15, ssl=False)
        timeout = aiohttp.ClientTimeout(total=8)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            tasks = []
            for sym in symbols:
                clean_sym = sym.replace(".NS", "").replace(".BO", "").strip().upper()
                yf_sym = f"{clean_sym}.NS"
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}?range=1d&interval=30m"
                tasks.append(self._fetch_single_quote(session, clean_sym, url))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, dict) and "symbol" in res:
                    quotes[res["symbol"]] = res

        return quotes

    async def _fetch_single_quote(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        url: str
    ) -> Optional[Dict[str, Any]]:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    chart = data.get("chart", {}).get("result", [])
                    if chart:
                        meta = chart[0].get("meta", {})
                        ltp = float(meta.get("regularMarketPrice") or 0.0)
                        high = float(meta.get("regularMarketDayHigh") or ltp)
                        low = float(meta.get("regularMarketDayLow") or ltp)
                        volume = float(meta.get("regularMarketVolume") or 0.0)

                        # Extract 30m candle data if available
                        timestamps = chart[0].get("timestamp", [])
                        indicators = chart[0].get("indicators", {}).get("quote", [{}])[0]
                        bars = []
                        if timestamps and indicators.get("close"):
                            for i, ts in enumerate(timestamps):
                                c = indicators.get("close", [])[i]
                                v = indicators.get("volume", [])[i] or 0.0
                                if c is not None:
                                    bars.append({"timestamp": ts, "close": float(c), "volume": float(v)})

                        return {
                            "symbol": symbol,
                            "ltp": ltp,
                            "high": high,
                            "low": low,
                            "volume": volume,
                            "bars_30m": bars
                        }
        except Exception as e:
            logger.debug(f"Async quote fetch error for {symbol}: {e}")
        return None

    def evaluate_breakout_candidate(
        self,
        watch_item: Dict[str, Any],
        quote_data: Dict[str, Any],
        now_dt: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Evaluates live entry condition and volume confirmation filter for a watchlist item:
        - Entry Condition: LTP > T-1 High.
        - Volume Confirmation: Time-adjusted volume pacing >= 3.12x.
        """
        if now_dt is None:
            now_dt = now_ist()

        sym = watch_item["symbol"]
        trigger_price = float(watch_item["t_minus_1_high"])
        initial_sl = float(watch_item["initial_sl"])
        avg_vol_20 = float(watch_item["avg_vol_20"])

        ltp = float(quote_data.get("ltp") or 0.0)
        day_high = float(quote_data.get("high") or ltp)
        live_vol = float(quote_data.get("volume") or 0.0)

        price_crossed = (ltp > trigger_price) or (day_high > trigger_price)
        pacing = calculate_volume_pacing(live_vol, avg_vol_20, now_dt)
        volume_confirmed = (pacing >= self.volume_pacing_threshold)

        should_enter = price_crossed and volume_confirmed

        return {
            "symbol": sym,
            "ltp": ltp,
            "trigger_price": trigger_price,
            "initial_sl": initial_sl,
            "risk_per_share": round(trigger_price - initial_sl, 2),
            "live_volume": live_vol,
            "avg_vol_20": avg_vol_20,
            "volume_pacing": pacing,
            "price_crossed": price_crossed,
            "volume_confirmed": volume_confirmed,
            "should_enter": should_enter,
            "reason": (
                "CONFIRMED_BREAKOUT" if should_enter else
                ("UNCONFIRMED_VOLUME_PACE" if price_crossed and not volume_confirmed else "PRICE_NOT_TRIGGERED")
            )
        }

    # --------------------------------------------------------------------------
    # 3. RISK MANAGEMENT & OMS ORDER DISPATCH
    # --------------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        initial_sl: float,
        account_capital: Optional[float] = None
    ) -> Tuple[int, float, float]:
        """
        Calculates position quantity based on the fixed 1% risk capital model:
        Risk Capital = Total Account Capital * 1%
        Risk per Share = Entry Price - Initial Stop Loss
        Quantity = floor(Risk Capital / Risk per Share)
        Subject to maximum position allocation cap (20% of capital).
        """
        capital = account_capital if account_capital is not None else self.base_capital
        risk_budget = capital * self.risk_per_trade_pct  # 1% of capital
        risk_per_share = entry_price - initial_sl

        if risk_per_share <= 0:
            return 0, 0.0, 0.0

        raw_qty = math.floor(risk_budget / risk_per_share)

        # Ensure minimum quantity of 2 for 50/50 split
        if raw_qty < 2:
            raw_qty = 2

        # Cap by maximum position allocation (e.g. 20% of capital)
        max_investment = capital * self.allocation_cap_pct
        if raw_qty * entry_price > max_investment:
            raw_qty = math.floor(max_investment / entry_price)

        total_qty = max(0, int(raw_qty))
        actual_risk_capital = round(total_qty * risk_per_share, 2)

        return total_qty, round(risk_per_share, 2), actual_risk_capital

    def execute_breakout_order(
        self,
        eval_result: Dict[str, Any],
        bypass_gate: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a confirmed breakout order via OMS Backend:
        1. Sizes the position using 1% fixed risk capital model.
        2. Splits quantity into Leg 1 (50% target @ +2.0R) and Leg 2 (50% runner with 10 EMA trailing).
        3. Persists into `active_positions` via `state_engine.open_position()`.
        4. Persists dual-leg tracking into `p3_position_legs`.
        5. Updates watchlist status to 'TRIGGERED'.
        6. Sends Telegram trade alert.
        """
        sym = eval_result["symbol"]
        entry_price = float(eval_result["ltp"])
        initial_sl = float(eval_result["initial_sl"])

        # Check slot availability
        active_positions = state_engine.get_active_positions(portfolio_type=self.portfolio_type)
        if len(active_positions) >= self.max_slots:
            logger.warning(f"🔒 [P3 Risk Gate Blocked] Max slots reached ({len(active_positions)}/{self.max_slots}). Cannot enter {sym}.")
            return None

        # Check if already held in active portfolio
        existing = [p for p in active_positions if p["symbol"] == sym]
        if existing:
            logger.info(f"ℹ️ [P3 Risk Gate Blocked] {sym} is already active in Portfolio 3.")
            return None

        # Calculate position size
        total_qty, risk_per_share, risk_capital = self.calculate_position_size(entry_price, initial_sl)
        if total_qty < 2:
            logger.warning(f"⚠️ [P3 Sizing Blocked] Sized quantity ({total_qty}) too small for 50/50 split in {sym}.")
            return None

        # Split 50 / 50
        leg1_qty = total_qty // 2
        leg2_qty = total_qty - leg1_qty

        # Target 1 = Entry + 2.0R
        target_2r = round(entry_price + (self.target_r * risk_per_share), 2)

        now_dt = now_ist()
        now_str = format_ist(now_dt)
        entry_date_str = now_dt.strftime("%Y-%m-%d")
        entry_time_str = now_dt.strftime("%H:%M:%S")

        sector = get_stock_sector(sym)

        # 1. Open unified position in active_positions table
        opened = state_engine.open_position(
            symbol=sym,
            entry_price=entry_price,
            quantity=total_qty,
            stop_loss=initial_sl,
            target=target_2r,
            target_2=0.0,
            triggered_from="P3_PRE_BREAKOUT_COIL",
            notes=f"P3 Split Exit: Leg 1 ({leg1_qty} shs) @ +2R ₹{target_2r:.2f}, Leg 2 ({leg2_qty} shs) 10 EMA Runner",
            sl_reason="INITIAL_SL_T1_LOW",
            sector=sector,
            portfolio_type=self.portfolio_type,
            entry_reason="PRE_BREAKOUT_COIL_CONFIRMED_3.12X_PACING",
            entry_regime="MOMENTUM_BREAKOUT",
            bypass_gate=bypass_gate
        )

        if not opened:
            logger.error(f"❌ Failed to open OMS position for {sym} via StateEngine.")
            return None

        # 2. Persist granular dual-leg tracking into p3_position_legs
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO p3_position_legs (
                symbol, portfolio_type, entry_date, entry_time, entry_price, total_quantity,
                initial_sl, risk_per_share, risk_capital, leg1_qty, leg1_target_price,
                leg1_status, leg2_qty, leg2_status, leg2_current_sl, leg2_trailing_ema10,
                overall_status, current_price, highest_price, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 'OPEN', ?, ?, 'OPEN', ?, ?, ?, ?)
            ON CONFLICT(symbol, entry_date) DO UPDATE SET
                entry_price = excluded.entry_price,
                total_quantity = excluded.total_quantity,
                leg1_qty = excluded.leg1_qty,
                leg2_qty = excluded.leg2_qty,
                updated_at = excluded.updated_at
            """, (
                sym, self.portfolio_type, entry_date_str, entry_time_str, entry_price, total_qty,
                initial_sl, risk_per_share, risk_capital, leg1_qty, target_2r,
                leg2_qty, initial_sl, initial_sl,
                entry_price, entry_price, now_str, now_str
            ))

            # Update watchlist status to 'TRIGGERED'
            conn.execute("""
            UPDATE p3_prebreakout_watchlist
            SET status = 'TRIGGERED', updated_at = ?
            WHERE symbol = ? AND status = 'WATCHING'
            """, (now_str, sym))
            conn.commit()

        # Update in-memory state tracking
        with _state_lock:
            _portfolio3_state["open_legs"][sym] = {
                "symbol": sym,
                "entry_price": entry_price,
                "total_quantity": total_qty,
                "leg1_qty": leg1_qty,
                "leg1_target_price": target_2r,
                "leg1_status": "OPEN",
                "leg2_qty": leg2_qty,
                "leg2_status": "OPEN",
                "leg2_current_sl": initial_sl,
                "overall_status": "OPEN",
                "entry_date": entry_date_str
            }
            _portfolio3_state["recent_triggers"].append({
                "symbol": sym,
                "entry_price": entry_price,
                "time": now_str,
                "pacing": eval_result.get("volume_pacing")
            })

        logger.info(
            f"🚀 [P3 ORDER EXECUTED] {sym}: {total_qty} shares @ ₹{entry_price:.2f}. "
            f"SL: ₹{initial_sl:.2f} (-1R), Leg 1: {leg1_qty} shs @ +2R ₹{target_2r:.2f}, "
            f"Leg 2: {leg2_qty} shs (10 EMA Runner). Pacing: {eval_result.get('volume_pacing')}x."
        )

        telegram_vault.send_alert_and_sync(
            title=f"🚀 P3 BREAKOUT EXECUTED: {sym}",
            message=(
                f"Entered <b>{sym}</b> @ <b>₹{entry_price:,.2f}</b> (Qty: {total_qty} shs)\n"
                f"• Volume Pacing: <b>{eval_result.get('volume_pacing')}x</b> (>= 3.12x threshold)\n"
                f"• Initial Stop-Loss: ₹{initial_sl:,.2f} (1R = ₹{risk_per_share:.2f})\n"
                f"• <b>Leg 1 (50% Qty):</b> {leg1_qty} shs Target @ +2.0R (₹{target_2r:,.2f})\n"
                f"• <b>Leg 2 (50% Qty):</b> {leg2_qty} shs Trailing Runner (10 EMA)\n"
                f"• Risk Capital: ₹{risk_capital:,.2f} (1.0% account risk)"
            ),
            sync_db=True
        )

        return {
            "symbol": sym,
            "entry_price": entry_price,
            "total_quantity": total_qty,
            "initial_sl": initial_sl,
            "leg1_target": target_2r,
            "leg1_qty": leg1_qty,
            "leg2_qty": leg2_qty,
            "volume_pacing": eval_result.get("volume_pacing"),
            "timestamp": now_str
        }

    # --------------------------------------------------------------------------
    # 4. AUTOMATED SPLIT EXIT STRATEGY & POSITION RECONCILIATION
    # --------------------------------------------------------------------------

    def reconcile_open_positions(
        self,
        bypass_gate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Automated Split Exit Strategy:
        Monitors active positions in Portfolio 3:
        1. Checks Initial SL: if LTP <= Initial SL before Target 1, closes entire position (-1.0R).
        2. Checks Leg 1 Target (+2.0R): if High >= Target, books 50% profit, records partial exit,
           and ratchets Leg 2 stop to at least Breakeven.
        3. Checks Leg 2 Runner: calculates daily 10 EMA. If 10 EMA > current SL, ratchets SL upward.
           If LTP <= trailing SL or daily Close drops below 10 EMA, exits remaining 50% runner.
        """
        start_t = time.time()
        active_positions = state_engine.get_active_positions(portfolio_type=self.portfolio_type)

        if not active_positions:
            return []

        logger.info(f"🔄 [Portfolio 3 Split Exit] Reconciling {len(active_positions)} active positions...")
        now_dt = now_ist()
        now_str = format_ist(now_dt)
        closed_events: List[Dict[str, Any]] = []

        can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
        if not can_trade and not bypass_gate:
            logger.info(f"Strict market gate closed ({gate_reason}). Updating prices only.")

        for pos in active_positions:
            sym = pos["symbol"]
            entry_px = float(pos["entry_price"])
            qty = int(pos["quantity"])
            current_sl = float(pos["stop_loss"])
            target_1 = float(pos["target"])
            t1_hit = int(pos.get("t1_hit") or 0)

            # Query dual-leg details from p3_position_legs
            leg_info = None
            with self._get_connection() as conn:
                r = conn.execute(
                    "SELECT * FROM p3_position_legs WHERE symbol = ? AND overall_status IN ('OPEN', 'PARTIAL_CLOSED')",
                    (sym,)
                ).fetchone()
                if r:
                    leg_info = dict(r)

            try:
                # Fetch recent daily OHLCV including today's live candle
                df = self.get_stock_daily_df(sym)
                if df is None or df.empty:
                    continue

                c = float(df["Close"].iloc[-1])
                h = float(df["High"].iloc[-1])
                l = float(df["Low"].iloc[-1])

                # Compute 10-day EMA for trailing stop
                close_arr = df["Close"].to_numpy(dtype=np.float64, copy=False)
                ema10_series = _fast_ewm_array(close_arr, span=10)
                ema10_val = round(float(ema10_series[-1]), 2) if len(ema10_series) > 0 else c

                # Update MTM current price
                state_engine.update_position_price(
                    sym,
                    current_price=c,
                    highest_price=max(float(pos.get("highest_price") or entry_px), h),
                    portfolio_type=self.portfolio_type
                )

                if not can_trade and not bypass_gate:
                    continue

                # -------------------------------------------------------------
                # SCENARIO A: INITIAL STOP-LOSS HIT (Before Target 1)
                # -------------------------------------------------------------
                if not t1_hit and c <= current_sl:
                    closed = state_engine.close_position(
                        sym,
                        exit_price=c,
                        exit_reason="INITIAL_SL",
                        portfolio_type=self.portfolio_type,
                        bypass_gate=bypass_gate
                    )
                    if closed:
                        closed_events.append(closed)

                    realized_loss = closed.get("pnl", 0.0) if closed else round((c - entry_px) * qty, 2)

                    # Update p3_position_legs
                    with self._get_connection() as conn:
                        conn.execute("""
                        UPDATE p3_position_legs
                        SET leg1_status = 'STOPPED_OUT', leg1_exit_price = ?, leg1_exit_date = ?,
                            leg2_status = 'STOPPED_OUT', leg2_exit_price = ?, leg2_exit_date = ?,
                            overall_status = 'CLOSED', total_realized_pnl = ?, updated_at = ?
                        WHERE symbol = ? AND overall_status IN ('OPEN', 'PARTIAL_CLOSED')
                        """, (c, now_str, c, now_str, realized_loss, now_str, sym))
                        conn.commit()

                    logger.info(f"🛑 [P3 Full Stop Hit] {sym} closed at Initial SL ₹{c:.2f} (Loss: ₹{realized_loss:.2f}).")
                    telegram_vault.send_alert_and_sync(
                        title=f"🛑 P3 INITIAL STOP HIT: {sym}",
                        message=f"Closed entire position in <b>{sym}</b> @ ₹{c:.2f}. PnL: -₹{abs(closed.get('pnl', 0.0)):.2f}.",
                        sync_db=True
                    )
                    continue

                # -------------------------------------------------------------
                # SCENARIO B: LEG 1 TARGET HIT (+2.0R)
                # -------------------------------------------------------------
                if not t1_hit and h >= target_1:
                    leg1_qty = qty // 2
                    pnl_leg1 = round((target_1 - entry_px) * leg1_qty, 2)
                    rem_qty = qty - leg1_qty

                    # Move stop loss to Breakeven (entry_px) or 10 EMA (whichever is higher)
                    new_sl_leg2 = round(max(entry_px, ema10_val), 2)

                    # Update unified active_positions
                    with state_engine._lock:
                        with state_engine._get_connection() as conn:
                            conn.execute("""
                            UPDATE active_positions
                            SET quantity = ?, stop_loss = ?, t1_hit = 1, t1_pnl = ?,
                                sl_reason = 'BREAKEVEN_OR_10EMA', updated_at = ?
                            WHERE symbol = ? AND portfolio_type = ?
                            """, (rem_qty, new_sl_leg2, pnl_leg1, now_str, sym, self.portfolio_type))

                    # Update p3_position_legs
                    with self._get_connection() as conn:
                        conn.execute("""
                        UPDATE p3_position_legs
                        SET leg1_status = 'TARGET_HIT', leg1_exit_price = ?, leg1_exit_date = ?, leg1_pnl = ?,
                            leg2_current_sl = ?, leg2_trailing_ema10 = ?, overall_status = 'PARTIAL_CLOSED',
                            total_realized_pnl = ?, updated_at = ?
                        WHERE symbol = ? AND overall_status = 'OPEN'
                        """, (target_1, now_str, pnl_leg1, new_sl_leg2, ema10_val, pnl_leg1, now_str, sym))
                        conn.commit()

                    logger.info(
                        f"🎯 [P3 Leg 1 Target Hit] {sym}: Booked 50% ({leg1_qty} shs) @ +2R ₹{target_1:.2f} "
                        f"(PnL: +₹{pnl_leg1:.2f}). Leg 2 stop ratcheted to ₹{new_sl_leg2:.2f}."
                    )
                    telegram_vault.send_alert_and_sync(
                        title=f"🎯 P3 TARGET 1 HIT (+2.0R): {sym}",
                        message=(
                            f"Locked in 50% profit on <b>{sym}</b> @ <b>₹{target_1:.2f}</b> (PnL: +₹{pnl_leg1:,.2f}).\n"
                            f"Remaining {rem_qty} shares trailing stop ratcheted to <b>₹{new_sl_leg2:.2f}</b> (10 EMA / Breakeven)."
                        ),
                        sync_db=True
                    )
                    t1_hit = 1
                    current_sl = new_sl_leg2

                # -------------------------------------------------------------
                # SCENARIO C: LEG 2 RUNNER TRAILING WITH 10 EMA
                # -------------------------------------------------------------
                if t1_hit:
                    # Ratchet trailing stop upward only
                    ratchet_sl = round(max(current_sl, ema10_val), 2)
                    if ratchet_sl > current_sl:
                        state_engine.update_position_price(
                            sym,
                            current_price=c,
                            stop_loss=ratchet_sl,
                            portfolio_type=self.portfolio_type
                        )
                        with self._get_connection() as conn:
                            conn.execute("""
                            UPDATE p3_position_legs
                            SET leg2_current_sl = ?, leg2_trailing_ema10 = ?, updated_at = ?
                            WHERE symbol = ? AND overall_status = 'PARTIAL_CLOSED'
                            """, (ratchet_sl, ema10_val, now_str, sym))
                            conn.commit()

                        logger.info(f"📈 [P3 Trailing Stop Ratcheted] {sym}: Raised to ₹{ratchet_sl:.2f} (10 EMA).")
                        current_sl = ratchet_sl

                    # Check exit for Leg 2 runner
                    if c <= current_sl:
                        closed = state_engine.close_position(
                            sym,
                            exit_price=c,
                            exit_reason="10EMA_TRAILING_STOP",
                            portfolio_type=self.portfolio_type,
                            bypass_gate=bypass_gate
                        )
                        if closed:
                            closed_events.append(closed)

                        total_leg2_pnl = round((c - entry_px) * qty, 2)
                        with self._get_connection() as conn:
                            conn.execute("""
                            UPDATE p3_position_legs
                            SET leg2_status = 'TRAILED_OUT', leg2_exit_price = ?, leg2_exit_date = ?,
                                leg2_pnl = ?, overall_status = 'CLOSED',
                                total_realized_pnl = total_realized_pnl + ?, updated_at = ?
                            WHERE symbol = ? AND overall_status = 'PARTIAL_CLOSED'
                            """, (c, now_str, total_leg2_pnl, total_leg2_pnl, now_str, sym))
                            conn.commit()

                        logger.info(f"🏁 [P3 Runner Exited] {sym} closed @ ₹{c:.2f} on 10 EMA trailing stop (Runner PnL: +₹{total_leg2_pnl:.2f}).")
                        telegram_vault.send_alert_and_sync(
                            title=f"🏁 P3 RUNNER EXITED (10 EMA): {sym}",
                            message=f"Closed runner in <b>{sym}</b> @ ₹{c:.2f}. Runner PnL: +₹{total_leg2_pnl:,.2f}.",
                            sync_db=True
                        )
            except Exception as pos_err:
                logger.error(f"Error reconciling Portfolio 3 position {sym}: {pos_err}")

        self._sync_state_from_db()
        return closed_events

    # --------------------------------------------------------------------------
    # 5. ASYNC SCHEDULED INTRADAY CYCLE RUNNER
    # --------------------------------------------------------------------------

    async def async_run_intraday_breakout_cycle(
        self,
        bypass_window: bool = False,
        bypass_gate: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the full 30-minute intraday breakout validation cycle:
        1. Checks if morning window (09:15 - 10:15 IST) is active.
        2. Retrieves watchlist candidates with status 'WATCHING'.
        3. Asynchronously fetches live quotes & 30m volume data.
        4. Validates breakout price trigger and volume pacing >= 3.12x.
        5. Executes confirmed breakouts via OMS.
        6. Reconciles open positions & automated split exits.
        """
        start_t = time.time()
        now_dt = now_ist()
        now_str = format_ist(now_dt)

        # 1. Window check
        is_window, window_msg = self.is_intraday_window_active(now_dt)
        if not is_window and not bypass_window:
            logger.info(f"⏳ [Portfolio 3 Breakout Cycle Skipped] {window_msg}")
            return {
                "status": "WINDOW_INACTIVE",
                "message": window_msg,
                "timestamp": now_str,
                "executed_orders": 0
            }

        # 2. Get active watchlist
        watchlist = self.get_active_watchlist()
        if not watchlist:
            logger.info("ℹ️ [Portfolio 3] Watchlist is currently empty. Run EOD ingestion first.")
            return {
                "status": "EMPTY_WATCHLIST",
                "message": "Watchlist is empty.",
                "timestamp": now_str,
                "executed_orders": 0
            }

        symbols = [item["symbol"] for item in watchlist]

        # 3. Async fetch live quotes
        quotes = await self.async_fetch_symbols_quotes(symbols)

        # 4. Evaluate candidates & execute confirmed breakouts
        executed_orders = []
        evaluations = []

        for item in watchlist:
            sym = item["symbol"]
            q = quotes.get(sym)
            if not q or q.get("ltp", 0.0) <= 0:
                continue

            eval_res = self.evaluate_breakout_candidate(item, q, now_dt)
            evaluations.append(eval_res)

            if eval_res["should_enter"]:
                order_res = self.execute_breakout_order(eval_res, bypass_gate=bypass_gate)
                if order_res:
                    executed_orders.append(order_res)

        # 5. Reconcile open positions and split exits
        closed_trades = self.reconcile_open_positions(bypass_gate=bypass_gate)

        elapsed = round(time.time() - start_t, 2)
        with _state_lock:
            _portfolio3_state["last_intraday_check_timestamp"] = now_str
            _portfolio3_state["status"] = "IDLE"

        return {
            "status": "SUCCESS",
            "timestamp": now_str,
            "watchlist_evaluated": len(watchlist),
            "quotes_fetched": len(quotes),
            "breakouts_confirmed": len(executed_orders),
            "executed_orders": executed_orders,
            "reconciled_closed_trades": len(closed_trades),
            "evaluations": evaluations,
            "execution_sec": elapsed
        }

    def run_intraday_breakout_cycle(
        self,
        bypass_window: bool = False,
        bypass_gate: bool = False
    ) -> Dict[str, Any]:
        """Synchronous wrapper for scheduled background workers or test runners."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running inside an active event loop, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.async_run_intraday_breakout_cycle(
                            bypass_window=bypass_window,
                            bypass_gate=bypass_gate
                        )
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.async_run_intraday_breakout_cycle(
                        bypass_window=bypass_window,
                        bypass_gate=bypass_gate
                    )
                )
        except Exception:
            return asyncio.run(
                self.async_run_intraday_breakout_cycle(
                    bypass_window=bypass_window,
                    bypass_gate=bypass_gate
                )
            )

    # --------------------------------------------------------------------------
    # 6. TELEMETRY & DASHBOARD REPORTING
    # --------------------------------------------------------------------------

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Provides full operational telemetry for Portfolio 3 dashboard."""
        self._sync_state_from_db()
        stats = state_engine.get_portfolio_stats(portfolio_type=self.portfolio_type)
        can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
        is_window, window_msg = self.is_intraday_window_active()

        with _state_lock:
            state_copy = dict(_portfolio3_state)

        return {
            "portfolio_type": self.portfolio_type,
            "title": "Portfolio 3: Pre-Breakout Coil & Volume Confirmation Engine",
            "base_capital": self.base_capital,
            "current_equity": round(
                self.base_capital + stats.get("total_realized_pnl", 0.0) + stats.get("unrealized_pnl", 0.0), 2
            ),
            "risk_per_trade_pct": self.risk_per_trade_pct * 100.0,
            "risk_per_trade_inr": round(self.base_capital * self.risk_per_trade_pct, 2),
            "max_slots": self.max_slots,
            "active_positions_count": stats.get("active_count", 0),
            "available_slots": stats.get("slots_available", self.max_slots),
            "available_cash": stats.get("available_cash", self.base_capital),
            "total_invested": stats.get("total_invested", 0.0),
            "open_risk": stats.get("open_risk", 0.0),
            "watchlist_count": state_copy.get("watchlist_count", 0),
            "volume_pacing_threshold": self.volume_pacing_threshold,
            "target_r_multiple": self.target_r,
            "trading_gate_open": can_trade,
            "trading_gate_message": gate_msg,
            "intraday_breakout_window_active": is_window,
            "intraday_breakout_window_message": window_msg,
            "last_scan_timestamp": state_copy.get("last_scan_timestamp"),
            "last_intraday_check_timestamp": state_copy.get("last_intraday_check_timestamp"),
            "stats": stats,
            "open_legs": state_copy.get("open_legs", {})
        }


# Global singleton instance
portfolio3_strategy_engine = Portfolio3StrategyEngine()
