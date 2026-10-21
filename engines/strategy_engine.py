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
from bs4 import BeautifulSoup
from engines.portfolio3_engine import get_chartink_universe, SCAN_CLAUSE, FRICTION_PCT

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
    ema20_dist_max_pct: float = 4.0,
    range_contraction_max_pct: float = 3.5,
    atr_squeeze_max: float = 0.80,
    enforce_vol_fade: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether a stock meets the 6 core Pre-Breakout Hybrid Squeeze rules on completed T-1 daily bar:
    1. Primary Uptrend: EMA20 > SMA50 and Close >= EMA20 * 0.98
    2. Mean-Reversion Anchor: |Close - EMA20| / EMA20 <= 0.04 (4.0%)
    3. Daily Range Contraction: (High - Low) / Low <= 0.035 (3.5% maximum range)
    4. Volume Dry-Up (VDU): Volume <= 0.80 * Vol_SMA20
    5. 3-Day Progressive Volume Fade: Volume_{T-3} > Volume_{T-2} > Volume_{T-1}
    6. Dynamic Volatility Squeeze: ATR5 / ATR20 <= 0.80
    """
    if df is None or len(df) < 50:
        return None

    # Determine T-1 index: ensure we evaluate the latest completed daily bar
    now = now_ist()
    last_dt = df.index[-1]
    is_today_partial = (
        isinstance(last_dt, (pd.Timestamp, datetime.datetime))
        and last_dt.date() == now.date()
        and now.time() < datetime.time(15, 30)
    )

    t1_idx = -2 if (is_today_partial and len(df) >= 51) else -1
    t1_date = df.index[t1_idx]
    t1_date_str = t1_date.strftime("%Y-%m-%d") if hasattr(t1_date, "strftime") else str(t1_date)[:10]

    close_arr = df["Close"].to_numpy(dtype=np.float64, copy=False)
    high_arr = df["High"].to_numpy(dtype=np.float64, copy=False)
    low_arr = df["Low"].to_numpy(dtype=np.float64, copy=False)
    vol_arr = df["Volume"].to_numpy(dtype=np.float64, copy=False)

    n_bars = len(close_arr)
    eval_pos = n_bars + t1_idx if t1_idx < 0 else t1_idx

    if eval_pos < 25:
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
    vdu_passed = (vdu_ratio <= vdu_max)

    # 2. 20 EMA and 50 SMA ending at T-1
    ema20_series = _fast_ewm_array(close_arr[: eval_pos + 1], span=20)
    ema20_t1 = float(ema20_series[-1]) if len(ema20_series) > 0 else c_t1

    close_50_slice = close_arr[max(0, eval_pos - 49) : eval_pos + 1]
    sma50_t1 = float(np.mean(close_50_slice)) if len(close_50_slice) >= 30 else ema20_t1

    # 1. Primary Uptrend: EMA20 > SMA50 and Close >= EMA20 * 0.98
    uptrend_passed = (ema20_t1 > sma50_t1) and (c_t1 >= ema20_t1 * 0.98)

    # 2. Mean-Reversion Anchor: |Close - EMA20| / EMA20 <= 0.04
    ema20_dist_pct = round(abs(c_t1 - ema20_t1) / ema20_t1 * 100.0, 2)
    ema20_prox_passed = (abs(c_t1 - ema20_t1) / ema20_t1 <= (ema20_dist_max_pct / 100.0))

    # 3. Price Range Contraction: (High - Low) / Low <= 0.035
    range_contraction_ratio = (h_t1 - l_t1) / l_t1
    range_contraction_pct = round(range_contraction_ratio * 100.0, 2)
    range_contraction_passed = (range_contraction_ratio <= (range_contraction_max_pct / 100.0))

    # 5. 3-Day Progressive Volume Fade: Volume_{T-3} > Volume_{T-2} > Volume_{T-1}
    vol_fade_passed = True
    if enforce_vol_fade and eval_pos >= 2:
        v_prev1 = float(vol_arr[eval_pos - 1])
        v_prev2 = float(vol_arr[eval_pos - 2])
        vol_fade_passed = (v_t1 < v_prev1) and (v_prev1 < v_prev2)
    elif enforce_vol_fade:
        vol_fade_passed = False

    # 6. Dynamic Volatility Squeeze: ATR5 / ATR20 <= 0.80
    tr_list = []
    start_tr = max(1, eval_pos - 25)
    for i in range(start_tr, eval_pos + 1):
        tr = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1])
        )
        tr_list.append(tr)

    atr5 = float(np.mean(tr_list[-5:])) if len(tr_list) >= 5 else 1.0
    atr20 = float(np.mean(tr_list[-20:])) if len(tr_list) >= 20 else (atr5 if atr5 > 0 else 1.0)
    atr_squeeze_ratio = round(atr5 / atr20, 3) if atr20 > 0 else 1.0
    atr_squeeze_passed = (atr_squeeze_ratio <= atr_squeeze_max)

    all_passed = (
        uptrend_passed
        and ema20_prox_passed
        and range_contraction_passed
        and vdu_passed
        and vol_fade_passed
        and atr_squeeze_passed
    )

    if not all_passed:
        return None

    # Calculate 10-day EMA for trailing stop benchmark
    ema10_series = _fast_ewm_array(close_arr[: eval_pos + 1], span=10)
    ema10_t1 = float(ema10_series[-1]) if len(ema10_series) > 0 else c_t1

    # Initial Stop-Loss strictly anchored at Base Low * 0.998 (0.2% buffer)
    trigger_high = round(h_t1, 2)
    base_low = round(l_t1, 2)
    initial_sl = round(base_low * 0.998, 2)
    entry_est = round(trigger_high * 1.001, 2)
    risk_per_share = round(entry_est - initial_sl, 2)
    target_p1 = round(entry_est + (2.0 * risk_per_share), 2)

    return {
        "symbol": symbol.replace(".NS", "").replace(".BO", "").strip().upper(),
        "t_minus_1_date": t1_date_str,
        "t_minus_1_close": round(c_t1, 2),
        "t_minus_1_high": trigger_high,
        "t_minus_1_low": base_low,
        "trigger_high": trigger_high,
        "base_low": base_low,
        "initial_sl": initial_sl,
        "risk_per_share": risk_per_share,
        "target_p1": target_p1,
        "vdu_ratio": vdu_ratio,
        "avg_vol_20": round(avg_vol_20, 0),
        "ema20": round(ema20_t1, 2),
        "ema50": round(sma50_t1, 2),
        "ema10": round(ema10_t1, 2),
        "ema20_dist_pct": ema20_dist_pct,
        "range_contraction_pct": range_contraction_pct,
        "atr5": round(atr5, 2),
        "atr20": round(atr20, 2),
        "atr_squeeze_ratio": atr_squeeze_ratio,
        "vol_fade_passed": 1 if vol_fade_passed else 0,
        "days_tracked": 0,
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
                        "SELECT COUNT(DISTINCT symbol) FROM p3_prebreakout_watchlist WHERE status = 'WATCHING'"
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

                    # Restore timestamps from KV or DB if missing in memory
                    if not _portfolio3_state.get("last_scan_timestamp"):
                        saved_scan = state_engine.get_kv("p3_last_scan_timestamp")
                        if not saved_scan:
                            max_u = conn.execute("SELECT MAX(updated_at) FROM p3_prebreakout_watchlist").fetchone()
                            if max_u and max_u[0]:
                                saved_scan = max_u[0]
                        if saved_scan:
                            _portfolio3_state["last_scan_timestamp"] = saved_scan

                    if not _portfolio3_state.get("last_intraday_check_timestamp"):
                        saved_check = state_engine.get_kv("p3_last_intraday_check_timestamp")
                        if saved_check:
                            _portfolio3_state["last_intraday_check_timestamp"] = saved_check
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

        # If universe is not provided, fetch from Chartink screener
        symbols_to_scan = universe
        if not symbols_to_scan:
            try:
                symbols_to_scan = get_chartink_universe()
                if symbols_to_scan:
                    logger.info(f"✓ Retrieved {len(symbols_to_scan)} fundamental stocks from Chartink screener.")
            except Exception as ce:
                logger.warning(f"Chartink fetch failed ({ce}), falling back to master metadata...")
                symbols_to_scan = []

        if not symbols_to_scan:
            try:
                meta_list = state_engine.get_all_ohlcv_metadata(limit=800)
                symbols_to_scan = [m["symbol"] for m in meta_list if m.get("bars_count", 0) >= 50]
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
                if df is None or len(df) < 50:
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
                    conn.execute(
                        "UPDATE p3_prebreakout_watchlist SET status = 'SUPERSEDED', updated_at = ? WHERE symbol = ? AND status = 'WATCHING' AND scan_date != ?",
                        (now_str, c["symbol"], scan_date_str)
                    )
                    conn.execute("""
                    INSERT INTO p3_prebreakout_watchlist (
                        symbol, scan_date, t_minus_1_date, t_minus_1_close, t_minus_1_high,
                        t_minus_1_low, trigger_high, base_low, initial_sl, t_minus_1_range_pct, vdu_ratio,
                        ema20, ema50, ema20_dist_pct, avg_vol_20, atr5, atr20, vol_fade_passed,
                        days_tracked, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'WATCHING', ?, ?)
                    ON CONFLICT(symbol, scan_date) DO UPDATE SET
                        t_minus_1_close = excluded.t_minus_1_close,
                        t_minus_1_high = excluded.t_minus_1_high,
                        t_minus_1_low = excluded.t_minus_1_low,
                        trigger_high = excluded.trigger_high,
                        base_low = excluded.base_low,
                        initial_sl = excluded.initial_sl,
                        t_minus_1_range_pct = excluded.t_minus_1_range_pct,
                        vdu_ratio = excluded.vdu_ratio,
                        ema20 = excluded.ema20,
                        ema50 = excluded.ema50,
                        ema20_dist_pct = excluded.ema20_dist_pct,
                        avg_vol_20 = excluded.avg_vol_20,
                        atr5 = excluded.atr5,
                        atr20 = excluded.atr20,
                        vol_fade_passed = excluded.vol_fade_passed,
                        updated_at = excluded.updated_at
                    """, (
                        c["symbol"], scan_date_str, c["t_minus_1_date"], c["t_minus_1_close"],
                        c["t_minus_1_high"], c["t_minus_1_low"], c["trigger_high"], c["base_low"],
                        c["initial_sl"], c["range_contraction_pct"], c["vdu_ratio"], c["ema20"],
                        c["ema50"], c["ema20_dist_pct"], c["avg_vol_20"], c.get("atr5", 0.0),
                        c.get("atr20", 0.0), c.get("vol_fade_passed", 1), now_str, now_str
                    ))
                    inserted_count += 1
                except Exception as db_err:
                    logger.error(f"Failed to persist watchlist candidate {c['symbol']}: {db_err}")
            conn.commit()

        elapsed = round(time.time() - start_t, 2)
        logger.info(f"✓ [Portfolio 3 Watchlist Ingested] {len(candidates)} coil candidates saved in {elapsed}s.")

        # Update in-memory state & persistent KV
        with _state_lock:
            _portfolio3_state["last_scan_timestamp"] = now_str
            _portfolio3_state["watchlist_count"] = inserted_count
        state_engine.set_kv("p3_last_scan_timestamp", now_str)
        try:
            state_engine.record_system_activity(
                category="PORTFOLIO_3",
                action="Pre-Breakout Coil Watchlist Ingestion",
                status="SUCCESS",
                summary=f"P3 Coil Watchlist Ingestion complete: {len(candidates)} candidates saved at {now_str}.",
                details={"count": len(candidates), "top_candidates": [c["symbol"] for c in candidates[:10]]},
                execution_sec=elapsed
            )
        except Exception:
            pass

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

    def get_active_watchlist(self, include_all: bool = False) -> List[Dict[str, Any]]:
        """Returns the list of currently active watchlist candidates enriched with live LTP and spreads."""
        from execution.state_engine import state_engine
        try:
            from engines.kotak_neo_engine import kotak_neo_engine
        except ImportError:
            kotak_neo_engine = None
        with self._get_connection() as conn:
            if include_all:
                rows = conn.execute("""
                SELECT * FROM p3_prebreakout_watchlist
                WHERE scan_date >= date('now', '-2 days')
                AND id IN (
                    SELECT MAX(id) FROM p3_prebreakout_watchlist
                    GROUP BY symbol
                )
                ORDER BY id DESC
                """).fetchall()
            else:
                rows = conn.execute("""
                SELECT * FROM p3_prebreakout_watchlist
                WHERE status IN ('WATCHING', 'TRIGGERED')
                AND id IN (
                    SELECT MAX(id) FROM p3_prebreakout_watchlist
                    WHERE status IN ('WATCHING', 'TRIGGERED')
                    GROUP BY symbol
                )
                ORDER BY id DESC
                """).fetchall()
            items = [dict(r) for r in rows]

        with self._get_connection() as c2:
            for item in items:
                sym = item.get('symbol', '')
                k_ltp = kotak_neo_engine.get_live_price(sym) if kotak_neo_engine else None
                pos_row = c2.execute("SELECT current_price FROM active_positions WHERE symbol = ?", (sym,)).fetchone()
                sig_row = c2.execute("SELECT close, intraday_pct FROM latest_signals WHERE symbol = ?", (sym,)).fetchone()
                ltp = k_ltp or (pos_row[0] if pos_row and pos_row[0] else None) or (sig_row[0] if sig_row and sig_row[0] else None) or item.get('t_minus_1_close')
                t1_close = float(item.get('t_minus_1_close') or 0.0)
                if ltp and t1_close > 0:
                    day_chg = round((float(ltp) - t1_close) / t1_close * 100, 2)
                else:
                    day_chg = sig_row[1] if sig_row and sig_row[1] is not None else 0.0
                item['current_price'] = ltp
                item['ltp'] = ltp
                item['day_change_pct'] = day_chg
                trigger_high = float(item.get('trigger_high') or item.get('t_minus_1_high') or 0.0)
                if trigger_high > 0 and ltp:
                    item['distance_to_trigger_pct'] = round((float(ltp) - trigger_high) / trigger_high * 100, 2)
                else:
                    item['distance_to_trigger_pct'] = 0.0
        return items

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

        # Kotak Neo Direct Path: Zero Yahoo scraping when Kotak Neo is active
        try:
            from engines.kotak_neo_engine import kotak_neo_engine
            if kotak_neo_engine._is_authenticated:
                kotak_neo_engine.fetch_rest_quotes(symbols)
                for sym in symbols:
                    clean_sym = sym.replace(".NS", "").replace(".BO", "").strip().upper()
                    tick = kotak_neo_engine.get_live_tick(clean_sym)
                    if tick and tick.get("ltp"):
                        quotes[clean_sym] = {
                            "symbol": clean_sym,
                            "ltp": float(tick["ltp"]),
                            "high": float(tick.get("high") or tick["ltp"]),
                            "low": float(tick.get("low") or tick["ltp"]),
                            "volume": int(tick.get("volume") or 0),
                            "change_pct": float(tick.get("change_pct") or 0.0),
                            "timestamp": tick.get("updated_at")
                        }
                if len(quotes) == len(symbols):
                    return quotes
        except Exception:
            pass

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
        Evaluates live entry condition, volume confirmation, invalidation, and rolling inside bar:
        - Trigger Condition: Price crosses Trigger High * 1.001 with volume pacing >= 3.12x Vol_SMA20.
        - Entry Execution: Entry Price = max(Trigger High * 1.001, Open, LTP).
        - Initial Stop-Loss (1R): Stop = Base Low * 0.998.
        - Risk Denominator (1R): Risk = Entry Price - Stop.
        - Invalidation: Discard immediately if price closes below 20 EMA, volume expands without breaking high, or 2 sessions pass.
        - Rolling Inside Bar: If volume <= 0.80x and price holds 20 EMA on Day 1, roll Trigger High and keep tracking.
        """
        if now_dt is None:
            now_dt = now_ist()

        sym = watch_item["symbol"]
        trigger_high = float(watch_item.get("trigger_high") or watch_item.get("t_minus_1_high") or 0.0)
        base_low = float(watch_item.get("base_low") or watch_item.get("t_minus_1_low") or 0.0)
        avg_vol_20 = float(watch_item.get("avg_vol_20") or 1.0)
        days_tracked = int(watch_item.get("days_tracked") or 0)
        ema20 = float(watch_item.get("ema20") or 0.0)

        ltp = float(quote_data.get("ltp") or 0.0)
        day_high = float(quote_data.get("high") or ltp)
        day_low = float(quote_data.get("low") or ltp)
        open_px = float(quote_data.get("open") or quote_data.get("open_price") or ltp)
        live_vol = float(quote_data.get("volume") or 0.0)

        trigger_level = round(trigger_high * 1.001, 2)
        price_crossed = (ltp >= trigger_level) or (day_high >= trigger_level) or (ltp > trigger_high)
        pacing = calculate_volume_pacing(live_vol, avg_vol_20, now_dt)
        volume_confirmed = (pacing >= self.volume_pacing_threshold)

        should_enter = price_crossed and volume_confirmed

        entry_price = round(max(trigger_level, open_px, ltp), 2)
        initial_sl = round(base_low * 0.998, 2)
        risk_per_share = round(max(0.01, entry_price - initial_sl), 2)
        target_p1 = round(entry_price + (2.0 * risk_per_share), 2)

        # Invalidation & Inside Bar Rolling
        is_invalidated = False
        invalidation_reason = ""
        is_inside_bar_roll = False
        new_trigger_high = trigger_high
        new_base_low = base_low
        next_days_tracked = days_tracked

        if not should_enter:
            if ema20 > 0 and ltp < ema20:
                is_invalidated = True
                invalidation_reason = "CLOSED_BELOW_20EMA"
            elif avg_vol_20 > 0 and live_vol > avg_vol_20 and day_high <= trigger_high:
                is_invalidated = True
                invalidation_reason = "VOLUME_EXPANSION_WITHOUT_BREAKOUT"
            elif days_tracked >= 2:
                is_invalidated = True
                invalidation_reason = "MAX_SESSIONS_EXPIRED"
            elif not price_crossed and avg_vol_20 > 0 and live_vol <= avg_vol_20 * 0.80 and (ema20 == 0 or ltp >= ema20):
                is_inside_bar_roll = True
                next_days_tracked = days_tracked + 1
                new_trigger_high = round(max(trigger_high, day_high), 2)
                new_base_low = round(min(base_low, day_low), 2)

        return {
            "symbol": sym,
            "ltp": ltp,
            "open": open_px,
            "day_high": day_high,
            "day_low": day_low,
            "trigger_high": trigger_high,
            "trigger_level": trigger_level,
            "trigger_price": trigger_high,
            "base_low": base_low,
            "entry_price": entry_price,
            "initial_sl": initial_sl,
            "risk_per_share": risk_per_share,
            "target_p1": target_p1,
            "live_volume": live_vol,
            "avg_vol_20": avg_vol_20,
            "volume_pacing": pacing,
            "price_crossed": price_crossed,
            "volume_confirmed": volume_confirmed,
            "should_enter": should_enter,
            "is_invalidated": is_invalidated,
            "invalidation_reason": invalidation_reason,
            "is_inside_bar_roll": is_inside_bar_roll,
            "new_trigger_high": new_trigger_high,
            "new_base_low": new_base_low,
            "next_days_tracked": next_days_tracked,
            "days_tracked": days_tracked,
            "reason": (
                "CONFIRMED_BREAKOUT" if should_enter else
                ("UNCONFIRMED_VOLUME_PACE" if price_crossed and not volume_confirmed else
                 ("INVALIDATED_" + invalidation_reason if is_invalidated else
                  ("INSIDE_BAR_ROLL" if is_inside_bar_roll else "PRICE_NOT_TRIGGERED")))
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
        entry_price = float(eval_result.get("entry_price") or eval_result["ltp"])
        initial_sl = float(eval_result["initial_sl"])
        risk_per_share = float(eval_result.get("risk_per_share") or (entry_price - initial_sl))
        target_2r = float(eval_result.get("target_p1") or round(entry_price + (self.target_r * risk_per_share), 2))

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
            sl_reason="INITIAL_SL_BASE_LOW",
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
                overall_status, current_price, highest_price, risk, target_p1, p1_filled,
                p2_active, friction_pct, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 'OPEN', ?, ?, 'OPEN', ?, ?, ?, ?, 0, 1, ?, ?, ?)
            ON CONFLICT(symbol, entry_date) DO UPDATE SET
                entry_price = excluded.entry_price,
                total_quantity = excluded.total_quantity,
                leg1_qty = excluded.leg1_qty,
                leg2_qty = excluded.leg2_qty,
                risk = excluded.risk,
                target_p1 = excluded.target_p1,
                updated_at = excluded.updated_at
            """, (
                sym, self.portfolio_type, entry_date_str, entry_time_str, entry_price, total_qty,
                initial_sl, risk_per_share, risk_capital, leg1_qty, target_2r,
                leg2_qty, initial_sl, initial_sl,
                entry_price, entry_price, risk_per_share, target_2r, FRICTION_PCT, now_str, now_str
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
        Path-Dependent Split-Exit Engine:
        Splits positions 50% / 50% upon entry:
        1. Pre-Target Stop-Out: If daily Low <= Stop-Loss before High >= Target_+2R,
           liquidate both Part 1 and Part 2 at Stop-Loss (full -1.0R loss).
        2. Target Reached: If daily High >= Entry + (2.0 * Risk):
           - Part 1 (50% qty) books profit at +2.0R.
           - Stop-Loss for Part 2 is raised to Breakeven (Entry Price).
        3. Runner Trailing: Part 2 trails dynamically day-by-day and exits when daily
           Close < EMA10 or when the breakeven stop is breached.
        4. Friction Deduction: Automatically subtract 0.30% round-trip charges
           (brokerage, STT, exchange fees, slippage) from the blended net percentage return.
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
                # SCENARIO A: PRE-TARGET STOP-OUT (Before Target 1)
                # -------------------------------------------------------------
                # If daily Low <= Stop-Loss before High >= Target_+2R,
                # liquidate both Part 1 and Part 2 at Stop-Loss (full -1.0R loss).
                if not t1_hit and (l <= current_sl or c <= current_sl):
                    stop_exit_price = current_sl
                    closed = state_engine.close_position(
                        sym,
                        exit_price=stop_exit_price,
                        exit_reason="INITIAL_SL",
                        portfolio_type=self.portfolio_type,
                        bypass_gate=bypass_gate
                    )
                    if closed:
                        closed_events.append(closed)

                    leg1_loss = round((stop_exit_price - entry_px) * (qty // 2), 2)
                    leg2_loss = round((stop_exit_price - entry_px) * (qty - (qty // 2)), 2)
                    total_loss = closed.get("pnl", 0.0) if closed else round((stop_exit_price - entry_px) * qty, 2)

                    p1_ret = (stop_exit_price - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                    p2_ret = (stop_exit_price - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                    net_ret = round((0.5 * p1_ret) + (0.5 * p2_ret) - FRICTION_PCT, 2)
                    r_mult = -1.0

                    # Update p3_position_legs
                    with self._get_connection() as conn:
                        conn.execute("""
                        UPDATE p3_position_legs
                        SET leg1_status = 'STOPPED_OUT', leg1_exit_price = ?, leg1_exit_date = ?, leg1_pnl = ?,
                            leg2_status = 'STOPPED_OUT', leg2_exit_price = ?, leg2_exit_date = ?, leg2_pnl = ?,
                            p1_filled = 1, p2_active = 0, overall_status = 'CLOSED',
                            total_realized_pnl = ?, net_ret_pct = ?, r_multiple = ?, updated_at = ?
                        WHERE symbol = ? AND overall_status IN ('OPEN', 'PARTIAL_CLOSED')
                        """, (
                            stop_exit_price, now_str, leg1_loss,
                            stop_exit_price, now_str, leg2_loss,
                            total_loss, net_ret, r_mult, now_str, sym
                        ))
                        conn.commit()

                    logger.info(f"🛑 [P3 Pre-Target Stop Hit] {sym} closed at Initial SL ₹{stop_exit_price:.2f} (Loss: ₹{total_loss:.2f}, -1.0R).")
                    telegram_vault.send_alert_and_sync(
                        title=f"🛑 P3 INITIAL STOP HIT: {sym}",
                        message=f"Closed entire position in <b>{sym}</b> @ ₹{stop_exit_price:.2f}. PnL: -₹{abs(total_loss):.2f} (-1.0R, net return: {net_ret}%).",
                        sync_db=True
                    )
                    continue

                # -------------------------------------------------------------
                # SCENARIO B: TARGET REACHED (+2.0R)
                # -------------------------------------------------------------
                # If daily High >= Entry + (2.0 * Risk):
                # - Part 1 (50% qty) books profit at +2.0R.
                # - Stop-Loss for Part 2 is raised to Breakeven (Entry Price).
                if not t1_hit and h >= target_1:
                    leg1_qty = qty // 2
                    rem_qty = qty - leg1_qty
                    pnl_leg1 = round((target_1 - entry_px) * leg1_qty, 2)

                    # Move stop loss to Breakeven (entry_px)
                    new_sl_leg2 = round(entry_px, 2)

                    # Update unified active_positions
                    with state_engine._lock:
                        with state_engine._get_connection() as conn:
                            conn.execute("""
                            UPDATE active_positions
                            SET quantity = ?, stop_loss = ?, t1_hit = 1, t1_pnl = ?,
                                sl_reason = 'BREAKEVEN_STOP', updated_at = ?
                            WHERE symbol = ? AND portfolio_type = ?
                            """, (rem_qty, new_sl_leg2, pnl_leg1, now_str, sym, self.portfolio_type))

                    # Update p3_position_legs
                    with self._get_connection() as conn:
                        conn.execute("""
                        UPDATE p3_position_legs
                        SET leg1_status = 'TARGET_HIT', leg1_exit_price = ?, leg1_exit_date = ?, leg1_pnl = ?,
                            p1_filled = 1, leg2_current_sl = ?, leg2_trailing_ema10 = ?, overall_status = 'PARTIAL_CLOSED',
                            total_realized_pnl = ?, updated_at = ?
                        WHERE symbol = ? AND overall_status = 'OPEN'
                        """, (target_1, now_str, pnl_leg1, new_sl_leg2, ema10_val, pnl_leg1, now_str, sym))
                        conn.commit()

                    logger.info(
                        f"🎯 [P3 Part 1 Target Hit] {sym}: Booked 50% ({leg1_qty} shs) @ +2R ₹{target_1:.2f} "
                        f"(PnL: +₹{pnl_leg1:.2f}). Part 2 stop ratcheted to Breakeven ₹{new_sl_leg2:.2f}."
                    )
                    telegram_vault.send_alert_and_sync(
                        title=f"🎯 P3 TARGET 1 HIT (+2.0R): {sym}",
                        message=(
                            f"Locked in 50% profit on <b>{sym}</b> @ <b>₹{target_1:.2f}</b> (PnL: +₹{pnl_leg1:,.2f}).\n"
                            f"Remaining {rem_qty} shares stop ratcheted to Breakeven (<b>₹{new_sl_leg2:.2f}</b>)."
                        ),
                        sync_db=True
                    )
                    t1_hit = 1
                    current_sl = new_sl_leg2
                    qty = rem_qty
                    continue

                # -------------------------------------------------------------
                # SCENARIO C: RUNNER TRAILING WITH 10 EMA / BREAKEVEN
                # -------------------------------------------------------------
                # Runner Trailing: Part 2 trails dynamically day-by-day and exits when
                # daily Close < EMA10 or when the breakeven stop is breached.
                elif t1_hit:
                    # Trailing stop can follow 10 EMA if 10 EMA > Breakeven
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

                    breakeven_breached = (l <= current_sl)
                    ema10_violated = (c < ema10_val)

                    if breakeven_breached or ema10_violated:
                        p2_exit_px = current_sl if breakeven_breached else c
                        exit_reason = "BREAKEVEN_STOP" if breakeven_breached else "10EMA_TRAILING_STOP"
                        closed = state_engine.close_position(
                            sym,
                            exit_price=p2_exit_px,
                            exit_reason=exit_reason,
                            portfolio_type=self.portfolio_type,
                            bypass_gate=bypass_gate
                        )
                        if closed:
                            closed_events.append(closed)

                        total_leg2_pnl = round((p2_exit_px - entry_px) * qty, 2)
                        p1_exit_p = float(leg_info.get("leg1_exit_price") or target_1) if leg_info else target_1
                        risk_val = float(leg_info.get("risk_per_share") or leg_info.get("risk") or (target_1 - entry_px) / 2.0) if leg_info else (target_1 - entry_px) / 2.0
                        if risk_val <= 0:
                            risk_val = 1.0

                        p1_r = (p1_exit_p - entry_px) / risk_val
                        p2_r = (p2_exit_px - entry_px) / risk_val
                        r_mult = round((0.5 * p1_r) + (0.5 * p2_r), 2)

                        p1_ret = (p1_exit_p - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                        p2_ret = (p2_exit_px - entry_px) / entry_px * 100.0 if entry_px > 0 else 0.0
                        net_ret = round((0.5 * p1_ret) + (0.5 * p2_ret) - FRICTION_PCT, 2)

                        with self._get_connection() as conn:
                            conn.execute("""
                            UPDATE p3_position_legs
                            SET leg2_status = 'TRAILED_OUT', leg2_exit_price = ?, leg2_exit_date = ?,
                                leg2_pnl = ?, p2_active = 0, overall_status = 'CLOSED',
                                total_realized_pnl = total_realized_pnl + ?,
                                net_ret_pct = ?, r_multiple = ?, updated_at = ?
                            WHERE symbol = ? AND overall_status = 'PARTIAL_CLOSED'
                            """, (p2_exit_px, now_str, total_leg2_pnl, total_leg2_pnl, net_ret, r_mult, now_str, sym))
                            conn.commit()

                        logger.info(f"🏁 [P3 Runner Exited] {sym} closed @ ₹{p2_exit_px:.2f} ({exit_reason}, Net Ret: {net_ret}%, {r_mult}R).")
                        telegram_vault.send_alert_and_sync(
                            title=f"🏁 P3 RUNNER EXITED: {sym}",
                            message=f"Closed runner in <b>{sym}</b> @ ₹{p2_exit_px:.2f}. Runner PnL: +₹{total_leg2_pnl:,.2f} ({r_mult}R, Net Return: {net_ret}%).",
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
            elif eval_res.get("is_invalidated"):
                with self._get_connection() as conn:
                    conn.execute(
                        "UPDATE p3_prebreakout_watchlist SET status = 'INVALIDATED', updated_at = ? WHERE symbol = ? AND status = 'WATCHING'",
                        (now_str, sym)
                    )
                    conn.commit()
                logger.info(f"🚫 [P3 Candidate Invalidated] {sym}: {eval_res.get('invalidation_reason')}")
            elif eval_res.get("is_inside_bar_roll"):
                if eval_res.get("next_days_tracked", 0) >= 2:
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE p3_prebreakout_watchlist SET status = 'EXPIRED', updated_at = ? WHERE symbol = ? AND status = 'WATCHING'",
                            (now_str, sym)
                        )
                        conn.commit()
                    logger.info(f"⌛ [P3 Candidate Expired] {sym}: Tracked 2 sessions without triggering.")
                else:
                    with self._get_connection() as conn:
                        conn.execute("""
                            UPDATE p3_prebreakout_watchlist
                            SET trigger_high = ?, base_low = ?, days_tracked = ?, updated_at = ?
                            WHERE symbol = ? AND status = 'WATCHING'
                        """, (
                            eval_res["new_trigger_high"],
                            eval_res["new_base_low"],
                            eval_res["next_days_tracked"],
                            now_str,
                            sym
                        ))
                        conn.commit()
                    logger.info(f"🔄 [P3 Inside Bar Roll] {sym}: High rolled to {eval_res['new_trigger_high']}, Low to {eval_res['new_base_low']}, Day={eval_res['next_days_tracked']}.")

        # 5. Reconcile open positions and split exits
        closed_trades = self.reconcile_open_positions(bypass_gate=bypass_gate)

        elapsed = round(time.time() - start_t, 2)
        with _state_lock:
            _portfolio3_state["last_intraday_check_timestamp"] = now_str
            _portfolio3_state["status"] = "IDLE"
        state_engine.set_kv("p3_last_intraday_check_timestamp", now_str)

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
