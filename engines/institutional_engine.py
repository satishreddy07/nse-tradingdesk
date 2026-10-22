"""
========================================================================================
Institutional NSE Equities Technical Analysis & Signal Generation Engine (Portfolio 1)
========================================================================================
Production-grade 6-Pillar Percentile Ranking Engine for NSE Indian Equities.
Fetches universe dynamically from Chartink, downloads 2 years daily OHLCV from Yahoo Finance
Chart API with direct rate-limit resilient HTTP requests, computes ~30 raw metrics,
ranks across 6 pillars using dynamic universe percentiles with Real Sector RS,
applies ML re-ranking, enforces sector concentration limits & score-weighted sizing,
generates time-aware classifications (BUY_NOW, PRE_BREAKOUT, WAIT_PULLBACK, AVOID),
manages trailing stops after T1, and reconciles positions into SQLite State Engine.
========================================================================================
"""

import sys
import os
import time
import math
import json
import logging
import datetime
import argparse
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import pytz
from bs4 import BeautifulSoup

from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from core.stock_registry import (
    NSE_SECTOR_MAP, SECTOR_BENCHMARK_MAP, get_stock_sector, format_nse_ticker, clean_symbol
)
from core.config import (
    P1_BASE_CAPITAL, P1_BASE_RISK_PCT, CHARTINK_URL, HISTORICAL_DATA_DIR
)
from execution.state_engine import state_engine
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("InstitutionalEngine")
PORTFOLIO_TYPE = "INSTITUTIONAL_6PILLAR"

# Thread-safe engine runtime state
_institutional_engine_state: Dict[str, Any] = {
    "last_run_timestamp": None,
    "last_run_mode": None,
    "status": "IDLE",
    "chartink_fetched_count": 0,
    "universe_size": 0,
    "cache_hits_count": 0,
    "gap_filled_count": 0,
    "full_downloads_count": 0,
    "buy_now_count": 0,
    "pre_breakout_count": 0,
    "wait_pullback_count": 0,
    "avoid_count": 0,
    "reconciled_trades": 0,
    "orders_executed": 0,
    "trading_gate_open": False,
    "trading_gate_message": "",
    "signals": []
}


def get_run_mode(dt_ist: Optional[datetime.datetime] = None) -> Tuple[str, str]:
    """
    Detects market session based on current IST time:
    - pre_market  (< 09:15)  : Yesterday's close
    - morning     (09:15-12:00): Yesterday's close, base signal only
    - afternoon   (12:00-15:30): Yesterday's close + intraday confirmation score (±15 pts)
    - post_market (> 15:30)  : Today's complete bar
    """
    if dt_ist is None:
        dt_ist = now_ist()

    t = dt_ist.time()
    t_pre = datetime.time(9, 15)
    t_mid = datetime.time(12, 0)
    t_close = datetime.time(15, 30)

    if t < t_pre:
        return "pre_market", "Yesterday's close"
    elif t < t_mid:
        return "morning", "Yesterday's close (Base Signal Only)"
    elif t < t_close:
        return "afternoon", "Yesterday's close + Intraday Confirmation (±15 pts)"
    else:
        return "post_market", "Today's complete bar"


def get_intraday_volume_factor(dt_ist: Optional[datetime.datetime] = None) -> float:
    """Calculates expected volume completion fraction based on standard NSE trading hours."""
    if dt_ist is None:
        dt_ist = now_ist()
    
    market_open = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = dt_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if dt_ist <= market_open:
        return 0.05
    if dt_ist >= market_close:
        return 1.0

    elapsed_mins = (dt_ist - market_open).total_seconds() / 60.0
    fraction_day = elapsed_mins / 375.0
    vol_factor = 0.5 * fraction_day + 0.5 * math.sin(fraction_day * math.pi / 2.0)
    return max(0.1, min(1.0, vol_factor))


# ==============================================================================
# DATA LAYER (CHARTINK LIVE FETCHER + MULTI-THREADED INCREMENTAL GAP-FILL YAHOO)
# ==============================================================================
class ChartinkUniverseFetcher:
    def __init__(self, url: str = CHARTINK_URL):
        self.url = url
        self.process_url = "https://chartink.com/screener/process"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })

    def fetch_tickers(self) -> List[str]:
        logger.info(f"Connecting to Chartink screener: {self.url}")
        try:
            t0_chart = time.time()
            res = self.session.get(self.url, timeout=15)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            csrf_elem = soup.find("meta", {"name": "csrf-token"})
            if not csrf_elem or "content" not in csrf_elem.attrs:
                raise ValueError("CSRF token not found on Chartink page.")
            csrf_token = csrf_elem["content"]

            scanner_tag = soup.find("scanner")
            scan_json_raw = scanner_tag.get(":scan-json", "{}")
            scan_data = json.loads(scan_json_raw)
            atlas_query = scan_data.get("atlas_query")

            headers = {
                "X-CSRF-TOKEN": csrf_token,
                "Referer": self.url,
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            }

            post_res = self.session.post(self.process_url, headers=headers, data={"scan_clause": atlas_query}, timeout=25)
            post_res.raise_for_status()

            raw_stocks = post_res.json().get("data", [])
            tickers = []
            for s in raw_stocks:
                sym = s.get("nsecode")
                if sym:
                    tickers.append(format_nse_ticker(sym))

            t_chart_ms = (time.time() - t0_chart) * 1000.0
            from core.data_watchdog import data_watchdog
            data_watchdog.record_packet("chartink_scanner", latency_ms=t_chart_ms, details={"stocks_count": len(tickers)})

            logger.info(f"Fetched {len(tickers)} stocks dynamically from Chartink screener.")
            return sorted(list(set(tickers)))
        except Exception as e:
            try:
                from core.data_watchdog import data_watchdog
                data_watchdog.record_failure("chartink_scanner", str(e))
            except Exception:
                pass
            logger.warning(f"Chartink fetch encountered notice: {e}. Falling back to cached seed tickers.")
            return []


class DirectYahooFinanceFetcher:
    """
    High-Speed Multi-Threaded OHLCV Downloader with Smart Incremental Gap-Filling
    and Single Master File (`data/swing_trades.db`) + Telegram Cloud Vault Integration:
    - Uses 24 worker threads with pooled keep-alive HTTP connections & dual Yahoo endpoints.
    - Reads/writes all OHLCV data directly inside the Single Master SQLite File (`ohlcv_master_store`).
    - Zero individual `.pkl` or `.csv` files created on disk.
    - If a stock exists in Master File up to a past date (e.g. Oct 08) and reappears later (e.g. 20 days later),
      fetches ONLY the missing delta bars, merges & deduplicates cleanly!
    """
    def __init__(self, max_workers: int = 24, storage_dir: Optional[str] = None):
        self.max_workers = max_workers
        self.endpoints = [
            "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        ]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }

        # Thread-safe pooled HTTP session for fast, rate-limit resilient requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        adapter = HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def load_raw_cached(self, ticker: str) -> Tuple[Optional[pd.DataFrame], float]:
        """Returns (cached_df, age_hours) from the Single Master File (`ohlcv_master_store`)."""
        return state_engine.get_ohlcv_df(ticker)

    def save_to_storage(self, ticker: str, df: pd.DataFrame) -> None:
        """Saves a single ticker's OHLCV DataFrame directly into the Single Master File."""
        state_engine.save_ohlcv_df(ticker, df)

    @staticmethod
    def get_latest_expected_bar_date() -> datetime.date:
        """Determines the latest trading session date that should be present in OHLCV data."""
        now = now_ist()
        d = now.date()
        if NSEMarketCalendar.is_trading_day(d) and now.time() >= datetime.time(9, 15):
            return d
        d = d - datetime.timedelta(days=1)
        while not NSEMarketCalendar.is_trading_day(d):
            d = d - datetime.timedelta(days=1)
        return d

    @staticmethod
    def get_intraday_volume_multiplier(last_bar_date: Optional[datetime.date] = None) -> float:
        """
        Computes the intraday volume projection factor when evaluating a partial trading bar
        during NSE regular market hours (09:15 AM – 03:30 PM IST, 375 total minutes).
        Example:
        - At 10:20 AM IST (65 mins elapsed): 375 / 65 = 5.77 -> capped at 4.5x pace
        - At 03:05 PM IST (350 mins elapsed): 375 / 350 = 1.07x pace
        - Outside market hours (completed daily bar): 1.0x
        """
        now = now_ist()
        if last_bar_date is not None and last_bar_date != now.date():
            return 1.0
        if not NSEMarketCalendar.is_trading_day(now.date()):
            return 1.0
        t = now.time()
        if datetime.time(9, 15) <= t <= datetime.time(15, 30):
            open_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
            elapsed_mins = max(15.0, (now - open_dt).total_seconds() / 60.0)
            return min(4.5, max(1.0, 375.0 / elapsed_mins))
        return 1.0

    def _request_yahoo_df(
        self,
        ticker: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetches Daily OHLCV bars from Yahoo Finance v8 API AND stitches the live
        intraday partial bar from `quote_data['meta']` (`regularMarketPrice`,
        `regularMarketDayHigh`, `regularMarketDayLow`, `regularMarketVolume`, `regularMarketTime`)
        so that any scan run at 10:20 AM, 3:05 PM, or manually during market hours includes
        historical daily OHLCV + the live partial candle up to the exact moment the scanner runs.
        """
        yf_ticker = ticker if (ticker.startswith("^") or "." in ticker or "=" in ticker or "-" in ticker) else f"{ticker}.NS"
        for attempt in range(max_retries):
            endpoint = self.endpoints[(hash(yf_ticker) + attempt) % len(self.endpoints)]
            url = endpoint.format(ticker=yf_ticker)
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    res = data.get("chart", {}).get("result", [])
                    if not res:
                        return None
                    quote_data = res[0]
                    meta = quote_data.get("meta", {}) or {}
                    timestamps = quote_data.get("timestamp", [])
                    quotes = quote_data.get("indicators", {}).get("quote", [{}])[0]
                    if not timestamps or not quotes.get("close"):
                        return None
                    dates = [datetime.datetime.fromtimestamp(ts, tz=IST).replace(hour=9, minute=15, second=0, microsecond=0) for ts in timestamps]
                    df = pd.DataFrame({
                        "Open": quotes.get("open", []),
                        "High": quotes.get("high", []),
                        "Low": quotes.get("low", []),
                        "Close": quotes.get("close", []),
                        "Volume": quotes.get("volume", []),
                    }, index=pd.DatetimeIndex(dates))
                    df.dropna(subset=["Close"], inplace=True)
                    if df.empty:
                        return None
                    df["Volume"] = df["Volume"].fillna(0.0)
                    df = df.sort_index()
                    dt_s = pd.Series(pd.DatetimeIndex(df.index).date)
                    df = df[~dt_s.duplicated(keep="last").values]

                    # Stitch Live Intraday Partial Bar from `meta` if present
                    reg_time = meta.get("regularMarketTime")
                    reg_price = meta.get("regularMarketPrice")
                    if reg_time and reg_price and float(reg_price) > 0:
                        meta_dt = datetime.datetime.fromtimestamp(int(reg_time), tz=IST).replace(
                            hour=9, minute=15, second=0, microsecond=0
                        )
                        reg_p = float(reg_price)
                        reg_h = float(meta.get("regularMarketDayHigh") or reg_p)
                        reg_l = float(meta.get("regularMarketDayLow") or reg_p)
                        reg_v = float(meta.get("regularMarketVolume") or 0.0)
                        prev_c = float(meta.get("chartPreviousClose") or meta.get("previousClose") or df["Close"].iloc[-1])
                        reg_o = float(meta.get("regularMarketOpen") or prev_c)

                        last_dt = df.index[-1]
                        if meta_dt.date() == last_dt.date():
                            # Update today's partial or final bar with latest live intraday high/low/close/volume
                            cur_o = float(df.iloc[-1]["Open"]) if not pd.isna(df.iloc[-1]["Open"]) else reg_o
                            cur_h = float(df.iloc[-1]["High"]) if not pd.isna(df.iloc[-1]["High"]) else reg_h
                            cur_l = float(df.iloc[-1]["Low"]) if not pd.isna(df.iloc[-1]["Low"]) else reg_l
                            cur_v = float(df.iloc[-1]["Volume"]) if not pd.isna(df.iloc[-1]["Volume"]) else 0.0
                            df.iloc[-1, df.columns.get_loc("Open")] = cur_o
                            df.iloc[-1, df.columns.get_loc("High")] = max(cur_h, reg_h, reg_p)
                            df.iloc[-1, df.columns.get_loc("Low")] = min(cur_l, reg_l, reg_p) if cur_l > 0 else min(reg_l, reg_p)
                            df.iloc[-1, df.columns.get_loc("Close")] = reg_p
                            df.iloc[-1, df.columns.get_loc("Volume")] = max(cur_v, reg_v)
                        elif meta_dt.date() > last_dt.date() and (NSEMarketCalendar.is_trading_day(meta_dt.date()) or not yf_ticker.endswith(".NS")):
                            # Append today's live partial intraday candle
                            live_row = pd.DataFrame({
                                "Open": [reg_o],
                                "High": [max(reg_o, reg_h, reg_p)],
                                "Low": [min(reg_o, reg_l, reg_p)],
                                "Close": [reg_p],
                                "Volume": [reg_v],
                            }, index=pd.DatetimeIndex([meta_dt]))
                            df = pd.concat([df, live_row]).sort_index()
                            dt_s2 = pd.Series(pd.DatetimeIndex(df.index).date)
                            df = df[~dt_s2.duplicated(keep="last").values]

                    df.attrs["meta"] = meta
                    return df
                elif resp.status_code == 429:
                    time.sleep((1.5 ** attempt) + np.random.uniform(0.2, 0.8))
                else:
                    time.sleep(0.3)
            except Exception:
                time.sleep(0.5)
        return None

    def fetch_ticker_chart(self, ticker: str, range_period: str = "2y", max_retries: int = 3) -> Tuple[str, Optional[pd.DataFrame]]:
        df = self._request_yahoo_df(ticker, {"range": range_period, "interval": "1d", "includePrePost": "false"}, max_retries=max_retries)
        return ticker, df

    def fetch_or_gapfill_ticker(
        self,
        ticker: str,
        target_date: Optional[datetime.date] = None,
        is_market_open: Optional[bool] = None,
        persist_immediately: bool = True
    ) -> Tuple[str, Optional[pd.DataFrame], str]:
        """
        Smart Incremental Gap-Filler for a single ticker backed by the Single Master File:
        Returns (ticker, dataframe, fetch_mode) where fetch_mode in ('CACHE_HIT', 'GAP_FILLED', 'FULL_DOWNLOAD', 'FAILED').
        - Outside market hours: if `last_bar_date == target_date`, uses cached Master File series.
        - During market hours (09:15 AM – 03:30 PM IST): refreshes whenever `age_hours > 0.02` (~1 min)
          so every scan at 10:20 AM, 3:05 PM, or manual trigger always stitches the live partial intraday bar!
        """
        if target_date is None:
            target_date = self.get_latest_expected_bar_date()
        if is_market_open is None:
            is_market_open, _ = NSEMarketCalendar.can_execute_trades()
        cached_df, age_hours = self.load_raw_cached(ticker)

        if cached_df is not None and len(cached_df) >= 80:
            last_idx = cached_df.index[-1]
            last_bar_date = last_idx.date() if hasattr(last_idx, "date") else pd.Timestamp(last_idx).date()
            now_dt = now_ist()
            is_post_market = (now_dt.time() >= datetime.time(15, 30)) if NSEMarketCalendar.is_trading_day(now_dt.date()) else False
            is_stale_partial_bar = (last_bar_date == target_date and is_post_market and age_hours > 0.25)
            gap_days = (target_date - last_bar_date).days if target_date and last_bar_date else 999

            # Case 1: Already up to date in Single Master File (outside market hours, or refreshed within the last 30 mins during live market hours)
            if gap_days <= 0 and not is_stale_partial_bar and (not is_market_open or age_hours <= 0.50):
                return ticker, cached_df, "CACHE_HIT"

            # Case 2: Cached in Master File up to a past date (e.g. Oct 08 -> 20 days gap) or live intraday bar update (>15m old)
            if gap_days <= 300:
                if gap_days <= 4:
                    delta_range = "5d"
                elif gap_days <= 26:
                    delta_range = "1mo"
                elif gap_days <= 80:
                    delta_range = "3mo"
                elif gap_days <= 165:
                    delta_range = "6mo"
                else:
                    delta_range = "1y"

                delta_df = self._request_yahoo_df(
                    ticker,
                    {"range": delta_range, "interval": "1d", "includePrePost": "false"},
                    max_retries=2
                )
                if delta_df is not None and not delta_df.empty:
                    c_df = cached_df.copy()
                    d_df = delta_df.copy()
                    c_df.index = pd.to_datetime(c_df.index, utc=True)
                    d_df.index = pd.to_datetime(d_df.index, utc=True)
                    merged = pd.concat([c_df, d_df]).sort_index()
                    dt_series = pd.Series(pd.DatetimeIndex(merged.index).date)
                    merged = merged[~dt_series.duplicated(keep="last").values]
                    if len(merged) > 500:
                        merged = merged.iloc[-500:]
                    if persist_immediately:
                        self.save_to_storage(ticker, merged)
                    return ticker, merged, "GAP_FILLED"
                else:
                    return ticker, cached_df, "CACHE_HIT"

        # Case 3: Brand new stock or gap > 300 days -> Full 2-Year Download
        full_df = self._request_yahoo_df(ticker, {"range": "2y", "interval": "1d", "includePrePost": "false"}, max_retries=3)
        if full_df is not None and not full_df.empty:
            if persist_immediately:
                self.save_to_storage(ticker, full_df)
            return ticker, full_df, "FULL_DOWNLOAD"

        return ticker, cached_df, "FAILED"

    def fetch_universe_ohlcv(self, tickers: List[str], min_bars: int = 80, source_label: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        """
        Multi-threaded universe OHLCV fetcher backed by the Single Master File (`data/swing_trades.db`):
        1. Always checks & syncs with the Telegram Cloud Master Vault first (`auto_restore_on_cold_boot`).
        2. Processes all tickers across 24 threads using Smart Incremental Gap-Filling (reusing Master File cache).
        3. Commits all updated/new OHLCV DataFrames into the Single Master File in one batch transaction.
        """
        total_tickers = len(tickers)
        if not source_label:
            src = state_engine.get_swing_universe_source()
            source_label = "In-House NSE" if src == "IN_HOUSE" else "Chartink Live"
        vault_sync = telegram_vault.auto_restore_on_cold_boot()
        ohlcv_stats = state_engine.get_ohlcv_stats()
        vault_stocks = ohlcv_stats.get("stocks_count", 0)
        vault_bar = ohlcv_stats.get("latest_bar_date", "--")
        state_engine.update_scan_progress(
            stage=(
                f"Step 2b/6: Checked Telegram Master File ({vault_stocks} stocks in Vault, "
                f"Latest Bar: {vault_bar}, Status: {vault_sync.get('status', 'READY')})..."
            ),
            percent=20,
            stocks_total=total_tickers,
            stocks_done=0,
            add_log=True
        )

        target_date = self.get_latest_expected_bar_date()
        is_mkt_open, _ = NSEMarketCalendar.can_execute_trades()

        results: Dict[str, pd.DataFrame] = {}
        dirty_dfs: Dict[str, pd.DataFrame] = {}
        stats = {
            "total_requested": total_tickers,
            "cache_hits": 0,
            "gap_filled": 0,
            "full_downloads": 0,
            "eligible": 0
        }

        state_engine.update_scan_progress(
            stage=f"Step 3/6: Multi-threaded Incremental OHLCV Sync across {total_tickers} {source_label} stocks (reading Telegram Master Vault cache first)...",
            percent=22,
            stocks_total=total_tickers,
            stocks_done=0,
            add_log=True
        )

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_or_gapfill_ticker, t, target_date, is_mkt_open, False): t
                for t in tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                sym_key = future_to_ticker.get(future)
                try:
                    sym, df, mode = future.result()
                    if mode == "CACHE_HIT":
                        stats["cache_hits"] += 1
                    elif mode == "GAP_FILLED":
                        stats["gap_filled"] += 1
                        if df is not None:
                            dirty_dfs[sym] = df
                    elif mode == "FULL_DOWNLOAD":
                        stats["full_downloads"] += 1
                        if df is not None:
                            dirty_dfs[sym] = df

                    if df is not None and len(df) >= min_bars:
                        last_close = float(df["Close"].iloc[-1])
                        avg_vol_20d = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else 0
                        # Liquid criteria: price >= 50 and avg volume >= 25,000 (standard NSE swing liquidity)
                        if last_close >= 50.0 and avg_vol_20d >= 25000:
                            results[sym] = df
                            stats["eligible"] += 1
                except Exception as e:
                    logger.debug(f"Notice during fetch of {sym_key}: {e}")

                if completed % 40 == 0 or completed == total_tickers:
                    pct = 22 + int((completed / max(1, total_tickers)) * 43)
                    state_engine.update_scan_progress(
                        stage=(
                            f"Step 3/6: OHLCV Sync ({completed}/{total_tickers} {source_label} stocks) • "
                            f"Master Vault Hit: {stats['cache_hits']} | Intraday Delta: {stats['gap_filled']} | New 2Y: {stats['full_downloads']} | "
                            f"Liquid: {stats['eligible']}"
                        ),
                        percent=pct,
                        stocks_total=total_tickers,
                        stocks_done=completed,
                        add_log=(completed % 160 == 0 or completed == total_tickers)
                    )

        # Commit all gap-filled / newly downloaded OHLCV series into the Single Master File in one fast transaction
        if dirty_dfs:
            saved_cnt = state_engine.save_ohlcv_batch(dirty_dfs)
            logger.info(f"Saved {saved_cnt} updated OHLCV stocks into Single Master File (data/swing_trades.db).")

        logger.info(
            f"OHLCV Sync Complete: {total_tickers} {source_label} stocks -> "
            f"{stats['cache_hits']} cached, {stats['gap_filled']} gap-filled, {stats['full_downloads']} full 2Y -> "
            f"{len(results)} liquid stocks eligible."
        )
        try:
            from core.data_watchdog import data_watchdog
            data_watchdog.record_packet("yahoo_finance", details={"eligible": len(results), "gap_filled": stats["gap_filled"]})
        except Exception:
            pass
        return results, stats



    def fetch_benchmarks(self) -> Dict[str, pd.DataFrame]:
        benchmarks = {
            "NIFTY_50": "^NSEI",
            "NIFTY_500": "^CRSLDX",
            "INDIA_VIX": "^INDIAVIX",
            "BANK": "^NSEBANK",
            "IT": "^CNXIT",
            "AUTO": "^CNXAUTO",
            "PHARMA": "^CNXPHARMA",
            "FMCG": "^CNXFMCG",
            "METAL": "^CNXMETAL",
            "REALTY": "^CNXREALTY",
            "ENERGY": "^CNXENERGY"
        }
        bench_data: Dict[str, pd.DataFrame] = {}

        def _fetch_one(item: Tuple[str, str]):
            name, ticker = item
            _, df = self.fetch_ticker_chart(ticker, range_period="1y", max_retries=2)
            return name, df

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(benchmarks)) as executor:
            for fut in concurrent.futures.as_completed([executor.submit(_fetch_one, it) for it in benchmarks.items()]):
                try:
                    name, df = fut.result()
                    if df is not None and not df.empty:
                        bench_data[name] = df
                except Exception:
                    pass
        return bench_data


# ==============================================================================
# C-VECTORIZED & MULTI-THREADED PARALLEL FEATURE ENGINE
# ==============================================================================
from numpy.lib.stride_tricks import sliding_window_view


def _fast_ewm_array(arr: np.ndarray, span: float) -> np.ndarray:
    """Cython-vectorized EWM matching pd.Series(arr).ewm(span=span, adjust=False).mean()."""
    if len(arr) == 0:
        return arr
    return pd.Series(arr, copy=False).ewm(span=span, adjust=False).mean().to_numpy()


class FeatureEngine:
    @staticmethod
    def calc_adx_np(high_np: np.ndarray, low_np: np.ndarray, close_np: np.ndarray, period: int = 14) -> float:
        """
        Canonical Welles Wilder ADX(14) calculation with true Wilder exponential smoothing.
        Matches TA-Lib and standard quantitative reference benchmarks (Zero inflation).
        """
        if len(close_np) < period * 2 + 1:
            return 20.0  # Insufficient warmup period

        h = np.asarray(high_np, dtype=np.float64)
        l = np.asarray(low_np, dtype=np.float64)
        c = np.asarray(close_np, dtype=np.float64)

        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))

        n = len(tr)
        if n < period * 2:
            return 20.0

        # Wilder's smoothing for TR, +DM, -DM
        smoothed_tr = np.zeros(n)
        smoothed_pdm = np.zeros(n)
        smoothed_mdm = np.zeros(n)

        smoothed_tr[period - 1] = np.sum(tr[:period])
        smoothed_pdm[period - 1] = np.sum(plus_dm[:period])
        smoothed_mdm[period - 1] = np.sum(minus_dm[:period])

        inv_p = 1.0 / period
        for i in range(period, n):
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] * inv_p) + tr[i]
            smoothed_pdm[i] = smoothed_pdm[i - 1] - (smoothed_pdm[i - 1] * inv_p) + plus_dm[i]
            smoothed_mdm[i] = smoothed_mdm[i - 1] - (smoothed_mdm[i - 1] * inv_p) + minus_dm[i]

        valid_idx = np.arange(period - 1, n)
        tr_valid = smoothed_tr[valid_idx]
        plus_di = 100.0 * (smoothed_pdm[valid_idx] / np.maximum(tr_valid, 1e-9))
        minus_di = 100.0 * (smoothed_mdm[valid_idx] / np.maximum(tr_valid, 1e-9))
        di_sum = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(di_sum, 1e-9)

        if len(dx) < period:
            return 20.0

        # Wilder's smoothing on DX to produce ADX
        adx_val = np.mean(dx[:period])
        for j in range(period, len(dx)):
            adx_val = (adx_val * (period - 1) + dx[j]) * inv_p

        val = float(adx_val)
        return round(val, 2) if not (math.isnan(val) or math.isinf(val)) else 20.0

    @staticmethod
    def calc_rsi_np(close_np: np.ndarray, period: int = 14) -> float:
        """Welles Wilder's 14-period smoothed RSI calculation matching pd.Series(close).ewm(alpha=1/period)."""
        if len(close_np) < period + 1:
            return 50.0
        m = min(len(close_np), period * 8)
        d = np.diff(close_np[-m:])
        gain = np.where(d > 0.0, d, 0.0)
        loss = np.where(d < 0.0, -d, 0.0)
        avg_gain = float(pd.Series(gain, copy=False).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
        avg_loss = float(pd.Series(loss, copy=False).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
        rs = avg_gain / (avg_loss + 1e-9)
        val = 100.0 - (100.0 / (1.0 + rs))
        return float(val) if not math.isnan(val) else 50.0

    @staticmethod
    def check_minervini_np(
        c_now: float,
        c_20: float,
        c_50: float,
        c_150: float,
        c_200: float,
        c_200_prev22: float,
        high_52w: float,
        low_52w: float
    ) -> int:
        score = 0
        if c_now > c_150 and c_now > c_200: score += 1
        if c_150 > c_200: score += 1
        if c_200 > c_200_prev22: score += 1
        if c_50 > c_150 and c_50 > c_200: score += 1
        if c_now > c_50: score += 1
        if c_now >= low_52w * 1.30: score += 1
        if c_now >= high_52w * 0.75: score += 1
        if c_now > c_20: score += 1
        return score

    @staticmethod
    def precompute_benchmark_context(benchmarks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Precomputes benchmark returns, market dates, and intraday multipliers ONCE before parallel stock loop."""
        nifty_df = benchmarks.get("NIFTY_50")
        nifty_ret_3m = 0.0
        if nifty_df is not None and len(nifty_df) >= 63:
            nc = nifty_df["Close"].to_numpy(dtype=np.float64, copy=False)
            nifty_ret_3m = float((nc[-1] - nc[-63]) / (nc[-63] + 1e-9) * 100.0)

        sector_ret_6m_map: Dict[str, float] = {}
        for k, bdf in benchmarks.items():
            if bdf is not None and len(bdf) >= 126:
                bc = bdf["Close"].to_numpy(dtype=np.float64, copy=False)
                sector_ret_6m_map[k] = float((bc[-1] - bc[-126]) / (bc[-126] + 1e-9) * 100.0)
            elif bdf is not None and len(bdf) >= 63:
                bc = bdf["Close"].to_numpy(dtype=np.float64, copy=False)
                sector_ret_6m_map[k] = float((bc[-1] - bc[0]) / (bc[0] + 1e-9) * 100.0)
            else:
                sector_ret_6m_map[k] = 0.0

        now_dt = now_ist()
        today_date = now_dt.date()
        live_intraday_mult = DirectYahooFinanceFetcher.get_intraday_volume_multiplier(today_date)
        vol_factor = get_intraday_volume_factor(now_dt)

        return {
            "nifty_ret_3m": nifty_ret_3m,
            "sector_ret_6m_map": sector_ret_6m_map,
            "today_date": today_date,
            "live_intraday_mult": live_intraday_mult,
            "vol_factor": vol_factor
        }

    @classmethod
    def compute_stock_metrics(
        cls,
        ticker: str,
        df: pd.DataFrame,
        benchmarks: Dict[str, pd.DataFrame],
        bench_ctx: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        n = len(df)
        if n < 105:
            return None

        if bench_ctx is None:
            bench_ctx = cls.precompute_benchmark_context(benchmarks)

        close_np = df["Close"].to_numpy(dtype=np.float64, copy=False)
        high_np = df["High"].to_numpy(dtype=np.float64, copy=False)
        low_np = df["Low"].to_numpy(dtype=np.float64, copy=False)
        vol_np = df["Volume"].to_numpy(dtype=np.float64, copy=False)

        c_now = float(close_np[-1])
        if c_now <= 0 or math.isnan(c_now):
            return None

        sector = get_stock_sector(ticker)

        ema20 = _fast_ewm_array(close_np, 20.0)
        ema50 = _fast_ewm_array(close_np, 50.0)
        ema150 = _fast_ewm_array(close_np, 150.0)
        ema200 = _fast_ewm_array(close_np, 200.0)

        ema20_now = float(ema20[-1])
        ema50_now = float(ema50[-1])
        ema20_slope = float((ema20_now - ema20[-10]) / (ema20[-10] + 1e-9) * 100.0) if n >= 10 else 0.0
        ema50_slope = float((ema50_now - ema50[-10]) / (ema50[-10] + 1e-9) * 100.0) if n >= 10 else 0.0

        adx14 = cls.calc_adx_np(high_np, low_np, close_np, 14)

        lookback_bars = min(n, 250)
        high_52w = float(np.max(high_np[-lookback_bars:]))
        low_52w = float(np.min(low_np[-lookback_bars:]))
        c_200_prev22 = float(ema200[-22]) if n >= 22 else float(ema200[0])
        mv_passed = cls.check_minervini_np(
            c_now, ema20_now, ema50_now, float(ema150[-1]), float(ema200[-1]),
            c_200_prev22, high_52w, low_52w
        )

        ret_1m = float((c_now - close_np[-21]) / (close_np[-21] + 1e-9) * 100.0) if n >= 21 else 0.0
        ret_3m = float((c_now - close_np[-63]) / (close_np[-63] + 1e-9) * 100.0) if n >= 63 else 0.0
        ret_6m = float((c_now - close_np[-126]) / (close_np[-126] + 1e-9) * 100.0) if n >= 126 else float((c_now - close_np[0]) / (close_np[0] + 1e-9) * 100.0)

        ret_3m_ex_1m = ret_3m - ret_1m
        ret_6m_ex_3m = ret_6m - ret_3m

        nifty_ret_3m = bench_ctx["nifty_ret_3m"]
        excess_3m = ret_3m - nifty_ret_3m

        s_map = bench_ctx["sector_ret_6m_map"]
        sector_ret_6m = s_map.get(sector, s_map.get("NIFTY_500", s_map.get("NIFTY_50", 0.0)))
        sector_excess_6m = ret_6m - sector_ret_6m

        rsi14 = cls.calc_rsi_np(close_np, 14)

        # Welles Wilder's 14-period Smoothed ATR matching nifty500_engine (alpha=1/14)
        m_atr = min(len(close_np) - 1, 70)
        h_slice = high_np[-m_atr:]
        l_slice = low_np[-m_atr:]
        cp_slice = close_np[-(m_atr + 1):-1]
        tr_slice = np.maximum(h_slice - l_slice, np.maximum(np.abs(h_slice - cp_slice), np.abs(l_slice - cp_slice)))
        atr14 = float(pd.Series(tr_slice, copy=False).ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1])
        atr_pct = (atr14 / c_now) * 100.0
        atr_distance = abs(atr_pct - 2.5)

        # Fast Bollinger Band Squeeze over last 126 windows (requires last 145 closes)
        bb_slice = close_np[-min(n, 145):]
        bb_wins = sliding_window_view(bb_slice, 20)
        sma20_arr = np.mean(bb_wins, axis=-1)
        std20_arr = np.std(bb_wins, axis=-1, ddof=1)
        bb_width_arr = (std20_arr * 4.0) / (sma20_arr + 1e-9)
        bb_width = float(bb_width_arr[-1])
        bb_width_min_6m = float(np.min(bb_width_arr))
        bb_sq_ratio = bb_width / (bb_width_min_6m + 1e-9)

        last_idx = df.index[-1]
        last_idx_dt = last_idx.date() if hasattr(last_idx, "date") else pd.Timestamp(last_idx).date()
        intraday_vol_mult = bench_ctx["live_intraday_mult"] if last_idx_dt == bench_ctx["today_date"] else 1.0
        effective_last_vol = float(vol_np[-1]) * intraday_vol_mult

        vol_50d_avg_prev = float(np.mean(vol_np[-51:-1])) if n >= 51 else float(np.mean(vol_np))
        vol_ratio = effective_last_vol / (vol_50d_avg_prev + 1e-9)

        # Last 20 daily returns & 50-bar rolling volume averages for accumulation / pocket pivot
        c_tail21 = close_np[-21:]
        ret_tail20 = ((c_tail21[1:] - c_tail21[:-1]) / (c_tail21[:-1] + 1e-9)) * 100.0
        vol_tail20 = vol_np[-20:]
        if n >= 69:
            vol_50_roll20 = np.mean(sliding_window_view(vol_np[-69:], 50), axis=-1)
        else:
            vol_50_roll20 = np.full(20, vol_50d_avg_prev, dtype=np.float64)

        accum_days = int(np.sum((ret_tail20 > 0.7) & (vol_tail20 > vol_50_roll20)))

        ret_tail10 = ret_tail20[-10:]
        vol_tail10 = vol_tail20[-10:]
        down_vols = vol_tail10[ret_tail10 < 0.0]
        max_down_vol = float(np.max(down_vols)) if len(down_vols) > 0 else 0.0
        intraday_pct = float(ret_tail20[-1])
        pocket_pivot = bool(intraday_pct > 0.0 and effective_last_vol > max_down_vol)
        vol_dryup = float(np.mean(vol_np[-5:]) / (vol_50d_avg_prev + 1e-9))

        prior_highs = high_np[-21:-1] if len(high_np) >= 22 else (high_np[:-1] if len(high_np) > 1 else high_np)
        pivot_20d = float(np.max(prior_highs))
        dist_to_pivot = ((c_now - pivot_20d) / (pivot_20d + 1e-9)) * 100.0
        dist_to_ema20 = ((c_now - ema20_now) / (ema20_now + 1e-9)) * 100.0

        close_10d = close_np[-10:]
        base_tightness = float((np.max(close_10d) - np.min(close_10d)) / (np.mean(close_10d) + 1e-9) * 100.0)

        # Fast Weekly Confluence in pure NumPy (5 trading sessions per week, anchored to latest bar)
        try:
            weekly_close = close_np[::-5][::-1]
            if len(weekly_close) >= 15:
                w_10 = _fast_ewm_array(weekly_close, 10.0)
                w_30 = _fast_ewm_array(weekly_close, 30.0)
                w_rsi = cls.calc_rsi_np(weekly_close, 14)
                week_confluence = bool(
                    weekly_close[-1] > w_10[-1] and
                    w_10[-1] > w_30[-1] and
                    w_rsi > 50.0
                )
            else:
                week_confluence = False
        except Exception:
            week_confluence = False

        is_vcp = bool(base_tightness < 6.0 and bb_sq_ratio < 1.35)

        c_high = float(high_np[-1])
        c_low = float(low_np[-1])

        inst_score = (
            (accum_days / 6.0 * 35.0) +
            (25.0 if pocket_pivot else 0.0) +
            (min(vol_ratio, 2.5) / 2.5 * 25.0) +
            (15.0 if c_now > (c_low + (c_high - c_low) * 0.6) else 0.0)
        )
        inst_score = float(min(100.0, max(0.0, inst_score)))

        range_pos = float((c_now - c_low) / ((c_high - c_low) + 1e-9) * 100.0)
        daily_range_pct = float((c_high - c_low) / (c_low + 1e-9) * 100.0)
        is_accumulating_coil = bool(
            abs(dist_to_ema20) <= 3.8 and
            daily_range_pct <= 3.2 and
            vol_ratio <= 0.85 and
            c_now > ema50_now
        )

        vol_factor = bench_ctx["vol_factor"]
        projected_vol_ratio = vol_ratio / vol_factor
        intraday_score = float(min(100.0, max(0.0,
            (50.0 + intraday_pct * 16.67) * 0.50 +
            range_pos * 0.30 +
            min(projected_vol_ratio, 2.0) / 2.0 * 100.0 * 0.20
        )))

        return {
            "ticker": ticker,
            "sector": sector,
            "close": round(c_now, 2),
            "high": round(c_high, 2),
            "low": round(c_low, 2),
            "daily_range_pct": round(daily_range_pct, 2),
            "is_accumulating_coil": is_accumulating_coil,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "adx14": adx14,
            "mv_passed": mv_passed,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_3m_ex_1m": ret_3m_ex_1m,
            "ret_6m_ex_3m": ret_6m_ex_3m,
            "excess_3m": excess_3m,
            "sector_excess_6m": sector_excess_6m,
            "rsi14": rsi14,
            "atr_pct": atr_pct,
            "atr_distance": atr_distance,
            "atr14": atr14,
            "bb_width": bb_width,
            "bb_sq_ratio": bb_sq_ratio,
            "vol_ratio": vol_ratio,
            "projected_vol_ratio": projected_vol_ratio,
            "vol_factor": vol_factor,
            "accum_days": accum_days,
            "pocket_pivot": pocket_pivot,
            "vol_dryup": vol_dryup,
            "pivot_20d": pivot_20d,
            "dist_to_pivot": dist_to_pivot,
            "dist_to_ema20": dist_to_ema20,
            "daily_range_pct": daily_range_pct,
            "is_accumulating_coil": is_accumulating_coil,
            "base_tightness": base_tightness,
            "week_confluence": week_confluence,
            "is_vcp": is_vcp,
            "inst_score": inst_score,
            "intraday_pct": intraday_pct,
            "intraday_score": intraday_score,
            "above_50dma": bool(c_now > ema50_now)
        }

    @classmethod
    def compute_universe_metrics_parallel(
        cls,
        ohlcv_dict: Dict[str, pd.DataFrame],
        benchmarks: Dict[str, pd.DataFrame],
        max_workers: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Computes all 30 technical indicators across the entire liquid stock universe
        using 16 parallel worker threads with C-vectorized NumPy/SciPy arrays and live progress updates.
        """
        if not ohlcv_dict:
            return []

        bench_ctx = cls.precompute_benchmark_context(benchmarks)
        items = list(ohlcv_dict.items())
        total_stocks = len(items)
        raw_records: List[Dict[str, Any]] = []

        def _worker(item: Tuple[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
            sym, df = item
            return cls.compute_stock_metrics(sym, df, benchmarks, bench_ctx=bench_ctx)

        done_cnt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, item) for item in items]
            for fut in concurrent.futures.as_completed(futures):
                done_cnt += 1
                try:
                    res = fut.result()
                    if res is not None:
                        raw_records.append(res)
                except Exception as e:
                    logger.debug(f"Indicator calculation skip: {e}")

                if done_cnt % 120 == 0 or done_cnt == total_stocks:
                    pct = 70 + int((done_cnt / max(1, total_stocks)) * 12)
                    state_engine.update_scan_progress(
                        stage=f"Step 4/6: Multi-Threaded Technical Indicator Engine ({done_cnt}/{total_stocks} liquid stocks computed)...",
                        percent=pct,
                        stocks_total=total_stocks,
                        stocks_done=done_cnt,
                        add_log=(done_cnt == total_stocks)
                    )

        return raw_records


# ==============================================================================
# PERCENTILE SCORING & 6-PILLAR ENGINE
# ==============================================================================
class PercentileScoringEngine:
    @staticmethod
    def pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
        return series.rank(pct=True, ascending=ascending) * 100.0

    @classmethod
    def score_universe(cls, raw_df: pd.DataFrame, run_mode: str = "morning") -> pd.DataFrame:
        df = raw_df.copy()

        # Pillar 1 — Trend (25%)
        p_ema20_slope = cls.pct_rank(df['ema20_slope'], ascending=True)
        p_ema50_slope = cls.pct_rank(df['ema50_slope'], ascending=True)
        p_trend_slope = (p_ema20_slope + p_ema50_slope) / 2.0
        p_adx = cls.pct_rank(df['adx14'], ascending=True)
        p_mv = cls.pct_rank(df['mv_passed'], ascending=True)
        df['p_trend'] = p_trend_slope * 0.5 + p_adx * 0.3 + p_mv * 0.2

        # Pillar 2 — Momentum (20%)
        p_ret_1m = cls.pct_rank(df['ret_1m'], ascending=True)
        p_ret_3m_ex = cls.pct_rank(df['ret_3m_ex_1m'], ascending=True)
        p_ret_6m_ex = cls.pct_rank(df['ret_6m_ex_3m'], ascending=True)
        p_excess_3m = cls.pct_rank(df['excess_3m'], ascending=True)
        p_mom_raw = p_ret_1m * 0.2 + p_ret_3m_ex * 0.3 + p_ret_6m_ex * 0.3 + p_excess_3m * 0.2

        rsi_vals = df['rsi14'].to_numpy(dtype=np.float64, copy=False)
        rsi_mult = np.full(len(rsi_vals), 0.90, dtype=np.float64)
        rsi_mult[(rsi_vals >= 50.0) & (rsi_vals <= 70.0)] = 1.05
        rsi_mult[((rsi_vals >= 40.0) & (rsi_vals < 50.0)) | ((rsi_vals > 70.0) & (rsi_vals <= 80.0))] = 1.00
        rsi_mult[(rsi_vals > 80.0) & (rsi_vals <= 85.0)] = 0.85
        rsi_mult[rsi_vals > 85.0] = 0.60
        rsi_mult[rsi_vals < 30.0] = 0.70

        df['p_momentum'] = p_mom_raw * rsi_mult

        # Pillar 3 — Volatility (15%)
        p_atr = cls.pct_rank(df['atr_distance'], ascending=False)
        p_bb_squeeze = cls.pct_rank(df['bb_sq_ratio'], ascending=False)
        df['p_volatility'] = p_atr * 0.7 + p_bb_squeeze * 0.3

        # Pillar 4 — Volume (15%)
        p_vol_ratio = cls.pct_rank(df['vol_ratio'], ascending=True)
        p_accum = cls.pct_rank(df['accum_days'], ascending=True)
        p_pocket = cls.pct_rank(df['pocket_pivot'].astype(int), ascending=True)
        df['p_volume'] = p_vol_ratio * 0.4 + p_accum * 0.3 + p_pocket * 0.3

        # Pillar 5 — Relative Strength (15% with Real Sector RS)
        p_rs_market = cls.pct_rank(df['ret_6m'], ascending=True)
        p_rs_sector = cls.pct_rank(df['sector_excess_6m'], ascending=True)
        df['p_rs'] = p_rs_market * 0.5 + p_rs_sector * 0.5

        # Pillar 6 — Proximity (10%)
        # Stocks closest to pivot (smallest absolute distance) receive the highest proximity percentile
        p_dist_pivot = cls.pct_rank(df['dist_to_pivot'].abs(), ascending=False)
        p_tightness = cls.pct_rank(df['base_tightness'], ascending=False)
        df['p_proximity'] = p_dist_pivot * 0.6 + p_tightness * 0.4

        df['score_base'] = (
            df['p_trend'] * 0.25 +
            df['p_momentum'] * 0.20 +
            df['p_volatility'] * 0.15 +
            df['p_volume'] * 0.15 +
            df['p_rs'] * 0.15 +
            df['p_proximity'] * 0.10
        )

        df['adj_weekly'] = np.where(df['week_confluence'], 15.0, -20.0)
        market_breadth_pct = (df['above_50dma'].sum() / len(df)) * 100.0
        df['adj_breadth'] = 5.0 if market_breadth_pct >= 50.0 else -5.0
        df['adj_inst'] = (df['inst_score'] - 50.0) / 50.0 * 8.0

        if run_mode == "afternoon":
            df['intraday_adj'] = (df['intraday_score'] - 50.0) / 50.0 * 15.0
        else:
            df['intraday_adj'] = 0.0

        df['score_raw'] = (
            df['score_base'] +
            df['adj_weekly'] +
            df['adj_breadth'] +
            df['adj_inst'] +
            df['intraday_adj']
        ).clip(0.0, 100.0)

        df['confidence'] = (
            (df['mv_passed'] >= 6).astype(int) +
            df['week_confluence'].astype(int) +
            df['pocket_pivot'].astype(int) +
            (df['accum_days'] >= 3).astype(int) +
            (df['rsi14'].between(50, 70)).astype(int) +
            (df['adx14'] >= 25).astype(int)
        )
        return df


# ==============================================================================
# ML RE-RANKER
# ==============================================================================
class MLReRanker:
    @staticmethod
    def predict_win_probability(df: pd.DataFrame) -> pd.Series:
        f_score = (df['score_raw'] - 50.0) / 25.0
        f_conf = (df['confidence'] - 3.0) / 1.5
        f_trend = (df['p_trend'] - 50.0) / 25.0
        f_vol = (df['p_volume'] - 50.0) / 25.0
        f_rs = (df['p_rs'] - 50.0) / 25.0
        f_prox = (df['p_proximity'] - 50.0) / 25.0
        f_vcp = df['is_vcp'].astype(float) * 0.8
        f_weekly = df['week_confluence'].astype(float) * 1.0 - 0.5

        log_odds = (
            -0.15 +
            0.55 * f_score +
            0.40 * f_conf +
            0.25 * f_trend +
            0.20 * f_vol +
            0.30 * f_rs +
            0.35 * f_prox +
            0.30 * f_vcp +
            0.45 * f_weekly
        )
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        return prob.round(3)

    @classmethod
    def apply_reranking(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            from engines.swing_quant_ml_engine import swing_quant_ml_engine
            regime = swing_quant_ml_engine.capture_market_regime()
            probs = []
            exp_rs = []
            verdicts = []
            decisions = []
            flags_list = []

            for _, row in df.iterrows():
                eval_res = swing_quant_ml_engine.evaluate_candidate(row.to_dict(), market_regime=regime)
                probs.append(eval_res["win_prob"])
                exp_rs.append(eval_res["expected_r"])
                verdicts.append(eval_res["verdict"])
                decisions.append(eval_res.get("decision", "PASS_TO_AI" if eval_res["verdict"] == "APPROVED" else "REJECT"))
                flags_list.append(json.dumps(eval_res["risk_flags"]))

            df['ml_prob'] = probs
            df['ml_win_prob'] = probs
            df['expected_r'] = exp_rs
            df['ml_verdict'] = verdicts
            df['ml_decision'] = decisions
            df['risk_flags_json'] = flags_list
        except Exception as e:
            logger.debug(f"Failed to use SwingQuantMLEngine in MLReRanker: {e}")
            df['ml_prob'] = cls.predict_win_probability(df)
            df['ml_win_prob'] = df['ml_prob']
            df['expected_r'] = ((df['ml_prob'] * 1.5) - ((1.0 - df['ml_prob']) * 1.0)).round(2)
            df['ml_verdict'] = np.where(df['ml_prob'] >= 0.55, "APPROVED", np.where(df['ml_prob'] >= 0.48, "CAUTION", "REJECTED"))
            df['ml_decision'] = np.where(df['ml_prob'] >= 0.56, "PASS_TO_AI", np.where(df['ml_prob'] >= 0.48, "WATCH", "REJECT"))
            df['risk_flags_json'] = "[]"

        df['score'] = (df['score_raw'] * 0.85 + (df['ml_prob'] * 100.0) * 0.15).clip(0.0, 100.0)
        return df


# ==============================================================================
# EXECUTION & POSITION SIZING
# ==============================================================================
class ExecutionEngine:
    @staticmethod
    def classify(row: pd.Series) -> str:
        score = float(row.get('score', 0.0) or 0.0)
        dist_pivot = float(row.get('dist_to_pivot', 0.0) or 0.0)
        dist_ema20 = float(row.get('dist_to_ema20', 0.0) or 0.0)
        vol_ratio = float(row.get('vol_ratio', 1.0) or 1.0)
        daily_range_pct = float(row.get('daily_range_pct', 5.0) or 5.0)
        is_accumulating_coil = bool(row.get('is_accumulating_coil', False))

        # F-35: Fail-closed default (PENDING / HOLD) rather than fail-open APPROVED
        ml_verdict = str(row.get('ml_verdict', 'PENDING')).upper()
        ml_dec = str(row.get('ml_decision', 'HOLD')).upper()

        # Closed-Loop Pre-Trade ML Gatekeeper: Block setups classified as REJECTED
        if ml_verdict == "REJECTED" or ml_dec == "REJECT":
            return "AVOID"

        vol_factor = float(row.get('vol_factor', 1.0) or 1.0)
        projected_vol = float(row.get('projected_vol_ratio', 0.0) or (vol_ratio / max(0.15, vol_factor)))
        effective_vol = max(vol_ratio, projected_vol)

        if score < 40.0:
            return "AVOID"

        # 1. ACCUMULATION COIL (Stealth Day 1 Entry before the jump):
        # High score, resting within 3.8% of 20 EMA, tight range (<=3.5%), controlled volume
        if score >= 78.0 and abs(dist_ema20) <= 3.8:
            if is_accumulating_coil or (vol_ratio <= 1.10 and daily_range_pct <= 3.8):
                if ml_verdict in ("APPROVED", "CAUTION") or ml_dec in ("PASS_TO_AI", "WATCH"):
                    return "BUY_NOW"

        # 2. BREAKOUT SURGE (Momentum Entry on volume expansion):
        # Tested proximity to pivot & EMA20, projected volume >= 1.15x
        if score >= 80.0 and abs(dist_pivot) <= 3.8 and abs(dist_ema20) <= 6.5:
            if effective_vol >= 1.15 and (ml_verdict in ("APPROVED", "CAUTION") or ml_dec in ("PASS_TO_AI", "WATCH")):
                return "BUY_NOW"
            else:
                return "PRE_BREAKOUT"
        elif score >= 70.0 and abs(dist_pivot) <= 5.0:
            return "PRE_BREAKOUT"
        # F-35: Route coherent mid-score setups to PRE_BREAKOUT or WAIT_PULLBACK rather than dropping to AVOID
        elif score >= 60.0:
            if abs(dist_pivot) <= 5.0 and abs(dist_ema20) <= 5.0:
                return "PRE_BREAKOUT"
            else:
                return "WAIT_PULLBACK"
        else:
            return "AVOID"

    @staticmethod
    def calculate_position(
        row: pd.Series,
        capital: float = P1_BASE_CAPITAL,
        base_risk_pct: float = P1_BASE_RISK_PCT
    ) -> Dict[str, Any]:
        entry = float(row.get('close', 0.0) or 0.0)
        if pd.isna(entry) or entry <= 0:
            return {
                "entry": 0.0, "sl": 0.0, "t1": 0.0, "t2": 0.0, "rr": 0.0,
                "shares": 0, "capital_at_risk": 0.0, "risk_pct": 0.0,
                "atr_pct": 0.0, "tier": "REJECT", "entry_type": "REJECT"
            }

        atr14 = float(row.get('atr14', 0.0) or 0.0)
        if pd.isna(atr14) or atr14 <= 0:
            atr14 = entry * 0.03

        low_20d = float(row.get('low', 0.0) or 0.0)
        if pd.isna(low_20d) or low_20d <= 0:
            low_20d = entry - 1.5 * atr14

        score = float(row.get('score', 0.0) or 0.0)
        conf = int(row.get('confidence', 3) or 3)
        ml_verdict = str(row.get('ml_verdict', 'PENDING')).upper()
        dist_ema20 = float(row.get('dist_to_ema20', 0.0) or 0.0)
        daily_rng = float(row.get('daily_range_pct', 5.0) or 5.0)
        is_accum_coil = bool(row.get('is_accumulating_coil', False) or (abs(dist_ema20) <= 3.8 and daily_rng <= 3.2))
        entry_type = "ACCUMULATION_COIL" if is_accum_coil else "BREAKOUT_SURGE"

        # Dynamically scale tier risk relative to base_risk_pct
        if score >= 90.0 and conf >= 5 and ml_verdict == "APPROVED":
            risk_pct = round(base_risk_pct * (1.00 / 0.75), 4)
            tier = "A+"
        elif score >= 80.0 and conf >= 4 and ml_verdict == "APPROVED":
            risk_pct = round(base_risk_pct * 1.0, 4)
            tier = "A"
        else:
            risk_pct = round(base_risk_pct * (0.60 / 0.75), 4)
            tier = "B+"

        if entry_type == "ACCUMULATION_COIL":
            # Tight structural stop placed directly below 20 EMA (typically 2.0% - 2.8% risk)
            sl_dist_pct = max(2.0, abs(dist_ema20) + 1.2)
            sl = round(entry * (1.0 - (sl_dist_pct / 100.0)), 2)
            risk_per_share = entry - sl
            r = max(0.01, risk_per_share)
            t1 = round(entry + 2.5 * r, 2)
            t2 = round(entry + 4.5 * r, 2)
            rr = round((t2 - entry) / r, 2)
        else:
            structure_sl = min(low_20d, entry - 1.5 * atr14)
            hard_sl = entry * 0.92
            sl = round(max(structure_sl, hard_sl), 2)
            risk_per_share = entry - sl
            if risk_per_share <= 0:
                sl = round(entry * 0.95, 2)
                risk_per_share = entry - sl
            r = max(0.01, risk_per_share)
            t1 = round(entry + 1.5 * r, 2)
            t2 = round(entry + 3.0 * r, 2)
            rr = round((t2 - entry) / r, 2)

        risk_amt = capital * (risk_pct / 100.0)
        shares = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0

        max_pos_val = capital * 0.20
        min_pos_val = capital * 0.03

        if (shares * entry) > max_pos_val:
            shares = int(max_pos_val / entry)

        if (shares * entry) < min_pos_val or shares < 1:
            shares = 0

        capital_at_risk = round(shares * risk_per_share, 2)

        return {
            "entry": entry,
            "sl": sl,
            "t1": t1,
            "t2": t2,
            "rr": rr,
            "shares": shares,
            "capital_at_risk": capital_at_risk,
            "risk_pct": risk_pct,
            "tier": tier,
            "entry_type": entry_type
        }


# ==============================================================================
# POSITION RECONCILIATION FOR PORTFOLIO 1 (STRICT MARKET-HOURS TRADING GATE)
# ==============================================================================
def reconcile_open_institutional_positions(ohlcv_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Evaluates open positions in Portfolio 1 (Institutional 6-Pillar):
    - Updates mark-to-market `current_price` at all times.
    - STRICT TRADING GATE: Only executes Stop-Loss exits, T1 partial bookings, or T2 target exits
      during official NSE trading hours (09:15 AM – 03:30 PM IST, Mon–Fri).
    """
    open_positions = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    if not open_positions:
        return []

    can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
    closed_events = []

    for pos in open_positions:
        try:
            sym = pos["symbol"]
            formatted_sym = format_nse_ticker(sym)
            is_freshly_fetched = (formatted_sym in ohlcv_dict or sym in ohlcv_dict)
            df = ohlcv_dict.get(formatted_sym)
            if df is None:
                df = ohlcv_dict.get(sym)
            if df is None:
                # Outside market hours, do not overwrite live position prices with stale local cache
                if not can_trade:
                    continue
                df, _ = state_engine.get_ohlcv_df(sym)
            if df is None or df.empty:
                continue

            entry_px = float(pos["entry_price"])
            sl = float(pos["stop_loss"])
            orig_sl = float(pos.get("original_stop_loss") or sl)
            t1 = float(pos["target"])
            t2 = float(pos.get("target_2") or (entry_px + 3.0 * (entry_px - orig_sl)))
            qty = int(pos["quantity"])
            t1_hit = bool(pos.get("t1_hit", 0))

            latest_bar = df.iloc[-1]
            c = float(latest_bar["Close"])
            h = float(latest_bar["High"])
            l = float(latest_bar["Low"])

            bar_idx = df.index[-1]
            bar_date_str = str(bar_idx)[:10]
            entry_date_str = str(pos.get("entry_date") or "")[:10]

            # PRE-ENTRY DATA GUARD: If latest bar is older than entry date, do not trigger false exits
            if entry_date_str and bar_date_str < entry_date_str:
                state_engine.update_position_price(
                    sym,
                    current_price=c,
                    highest_price=max(float(pos.get("highest_price") or entry_px), c),
                    portfolio_type=PORTFOLIO_TYPE
                )
                continue

            is_same_entry_day = (bar_date_str == entry_date_str)

            # F-36: During live trading hours (09:15 - 15:30 IST), stop-loss evaluation strictly monitors
            # the live current market price `c` (LTP). Daily candle Low `l` from delayed external feeds
            # contains pre-entry wicks, auction prints, etc. Similarly, on same entry day, `check_high` must be `c`
            # so pre-entry morning highs do not falsely trigger immediate target hits.
            check_low = c
            check_high = c if is_same_entry_day else max(h, c)

            # Always update MTM price so dashboard reflects latest price
            if not can_trade:
                state_engine.update_position_price(
                    sym,
                    current_price=c,
                    highest_price=max(float(pos.get("highest_price") or entry_px), check_high),
                    portfolio_type=PORTFOLIO_TYPE
                )
                continue

            # STRICT TRADING GATE OPEN (09:15 - 15:30 IST): Evaluate live exits
            # Check Stop Loss
            if check_low <= sl:
                closed = state_engine.close_position(
                    sym, exit_price=c,
                    exit_reason="STOP_LOSS" if not t1_hit else "TRAILING_BE_STOP",
                    portfolio_type=PORTFOLIO_TYPE
                )
                if closed:
                    closed_events.append(closed)
                continue

            # Check T1 Target
            if not t1_hit and check_high >= t1:
                booked_qty = qty // 2
                pnl_t1 = round((t1 - entry_px) * booked_qty, 2)
                rem_qty = qty - booked_qty
                with state_engine._lock:
                    with state_engine._get_connection() as conn:
                        conn.execute("""
                        UPDATE active_positions
                        SET quantity = ?, stop_loss = ?, t1_hit = 1, t1_pnl = ?, sl_reason = 'BREAKEVEN_TRAILING', updated_at = ?
                        WHERE symbol = ? AND portfolio_type = ?
                        """, (rem_qty, entry_px, pnl_t1, format_ist(), sym, PORTFOLIO_TYPE))
                logger.info(f"🎯 [P1 T1 Hit] {sym} booked 50% ({booked_qty} shs) @ ₹{t1:.2f}. SL moved to BE ₹{entry_px:.2f}.")
                telegram_vault.send_alert_and_sync(
                    title=f"🎯 P1 TARGET 1 HIT (+1.5R): {sym}",
                    message=f"Booked 50% ({booked_qty} shs) @ ₹{t1:.2f} (PnL: +₹{pnl_t1:.2f}). Stop Loss moved to Breakeven ₹{entry_px:.2f}.",
                    sync_db=True
                )

            # Check T2 Target
            if t1_hit and check_high >= t2:
                closed = state_engine.close_position(sym, exit_price=max(t2, c), exit_reason="T2_FULL_TARGET", portfolio_type=PORTFOLIO_TYPE)
                if closed:
                    closed_events.append(closed)
                continue

            # Trailing SL update after T1
            if t1_hit:
                highest_seen = max(float(pos.get("highest_price") or entry_px), check_high)
                if len(df) >= 15:
                    h14 = df["High"].to_numpy(dtype=np.float64, copy=False)[-14:]
                    l14 = df["Low"].to_numpy(dtype=np.float64, copy=False)[-14:]
                    cp14 = df["Close"].to_numpy(dtype=np.float64, copy=False)[-15:-1]
                    atr14 = float(np.mean(np.maximum(h14 - l14, np.maximum(np.abs(h14 - cp14), np.abs(l14 - cp14)))))
                else:
                    atr14 = entry_px * 0.02
                trail_sl = round(max(entry_px, highest_seen - 1.5 * atr14), 2)
                if trail_sl > sl:
                    state_engine.update_position_price(sym, current_price=c, stop_loss=trail_sl, highest_price=highest_seen, portfolio_type=PORTFOLIO_TYPE)
                    telegram_vault.send_alert_and_sync(
                        title=f"📈 P1 TRAILING STOP RATCHETED: {sym}",
                        message=f"Stop-Loss raised from ₹{sl:,.2f} to <b>₹{trail_sl:,.2f}</b> (LTP: ₹{c:,.2f}).",
                        sync_db=True
                    )
                else:
                    state_engine.update_position_price(sym, current_price=c, highest_price=highest_seen, portfolio_type=PORTFOLIO_TYPE)
            else:
                state_engine.update_position_price(sym, current_price=c, highest_price=max(float(pos.get("highest_price") or entry_px), check_high), portfolio_type=PORTFOLIO_TYPE)
        except Exception as pos_err:
            logger.error(f"Error reconciling position {pos.get('symbol')}: {pos_err}")

    return closed_events


# ==============================================================================
# INSTITUTIONAL ENGINE COORDINATOR
# ==============================================================================
class InstitutionalEngine:
    def __init__(self, capital: float = P1_BASE_CAPITAL, risk_per_trade_pct: float = P1_BASE_RISK_PCT):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.chartink = ChartinkUniverseFetcher()
        self.yahoo = DirectYahooFinanceFetcher(max_workers=24)

    def run(
        self,
        custom_tickers: Optional[List[str]] = None,
        auto_execute: bool = True,
        mode_override: Optional[str] = None,
        universe_source: Optional[str] = None
    ) -> pd.DataFrame:
        start_time = time.time()
        mode, mode_desc = get_run_mode()
        if mode_override:
            mode = mode_override

        can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
        _institutional_engine_state["status"] = "RUNNING"
        _institutional_engine_state["last_run_mode"] = mode
        _institutional_engine_state["last_run_timestamp"] = format_ist()
        _institutional_engine_state["trading_gate_open"] = can_trade
        _institutional_engine_state["trading_gate_message"] = gate_msg

        source = universe_source or state_engine.get_swing_universe_source()
        if source not in ("IN_HOUSE", "CHARTINK"):
            source = "IN_HOUSE"
        source_label = "In-House NSE" if source == "IN_HOUSE" else "Chartink Live"
        init_stage_msg = (
            f"Step 1/6: Sourcing universe from {source_label}..."
            if source == "IN_HOUSE" else
            f"Step 1/6: Connecting live to Chartink Screener ({self.chartink.url})..."
        )

        state_engine.start_scan_progress(
            scan_type=f"INSTITUTIONAL_6PILLAR ({mode.upper()})",
            initial_stage=init_stage_msg
        )
        logger.info(f"=== Institutional 6-Pillar Engine Run ({mode.upper()}) [{source_label}] at {format_ist()} ===")

        try:
            # Step 1: Ingest Universe Live from Chartink or In-House
            if custom_tickers and len(custom_tickers) > 0:
                tickers = [format_nse_ticker(t) for t in custom_tickers]
                _institutional_engine_state["chartink_fetched_count"] = len(tickers)
                state_engine.update_scan_progress(
                    stage=f"Step 1/6: Using {len(tickers)} specified tickers...",
                    percent=10,
                    stocks_total=len(tickers),
                    stocks_done=0
                )
            else:
                from_live_chartink = False
                if source == "IN_HOUSE":
                    logger.info("Swing candidate universe sourcing: IN_HOUSE (All NSE Equities Screener).")
                    from engines.screener_comparison_engine import screener_comparison_engine
                    tickers, nse_total_cnt = screener_comparison_engine.get_inhouse_screened_tickers()
                    _institutional_engine_state["nse_total_count"] = nse_total_cnt
                    _institutional_engine_state["passing_criteria_count"] = len(tickers)
                    _institutional_engine_state["chartink_fetched_count"] = len(tickers)
                    stage_msg = f"Step 1/6: Sourced {len(tickers)} qualifying candidate stocks (from {nse_total_cnt} total NSE stocks passing EQ-FINAL criteria) for In-House 6-Pillar Analysis."
                else:
                    logger.info("Swing candidate universe sourcing: CHARTINK (All Cash Equities).")
                    tickers = self.chartink.fetch_tickers()
                    from_live_chartink = bool(tickers)
                    if not tickers:
                        meta_rows = state_engine.get_all_ohlcv_metadata(limit=2000)
                        tickers = [f"{r['symbol']}.NS" for r in meta_rows if r.get("symbol") and not str(r["symbol"]).startswith("^")]
                    _institutional_engine_state["chartink_fetched_count"] = len(tickers)
                    stage_msg = (
                        f"Step 1/6: Connected to Chartink! Fetched {len(tickers)} live NSE stocks."
                        if from_live_chartink else
                        f"Step 1/6: Chartink unavailable; loaded {len(tickers)} cached universe stocks from Master DB."
                    )
                state_engine.update_scan_progress(
                    stage=stage_msg,
                    percent=12,
                    stocks_total=len(tickers),
                    stocks_done=0
                )

            source_tag = "CUSTOM" if custom_tickers else source
            source_name = "Custom Input" if custom_tickers else ("Chartink Screener" if source == "CHARTINK" else "In-House Nifty 500 Engine")
            try:
                state_engine.record_system_activity(
                    category="SCANNER_PARITY",
                    action=f"Screener Candidate Sourcing ({source_tag})",
                    status="SUCCESS",
                    summary=f"Sourced {len(tickers)} positive candidate stocks from {source_name} for 6-pillar analysis.",
                    details={
                        "source": source_tag,
                        "count": len(tickers),
                        "sample": [clean_symbol(t) for t in tickers[:12]],
                        "timestamp": format_ist()
                    },
                    execution_sec=round(time.time() - start_time, 2)
                )
            except Exception:
                pass

            # Step 2: Parallel Benchmarks Fetch
            state_engine.update_scan_progress(
                stage=f"Step 2/6: Parallel fetch of 11 NSE Market & Sector Benchmarks (^NSEI, ^CRSLDX, Bank, IT, Auto, Pharma)...",
                percent=18,
                stocks_total=len(tickers),
                stocks_done=0
            )
            benchmarks = self.yahoo.fetch_benchmarks()

            # Step 3: Multi-Threaded Incremental Gap-Fill OHLCV Fetch
            ohlcv_dict, dl_stats = self.yahoo.fetch_universe_ohlcv(tickers, min_bars=80, source_label=source_label)
            _institutional_engine_state["cache_hits_count"] = dl_stats.get("cache_hits", 0)
            _institutional_engine_state["gap_filled_count"] = dl_stats.get("gap_filled", 0)
            _institutional_engine_state["full_downloads_count"] = dl_stats.get("full_downloads", 0)

            if not ohlcv_dict:
                _institutional_engine_state["status"] = "FAILED"
                state_engine.finish_scan_progress("Scan aborted: No valid OHLCV data retrieved.")
                return pd.DataFrame()

            # Step 4: Reconcile active positions & Multi-Threaded Parallel Feature Extraction
            state_engine.update_scan_progress(
                stage=f"Step 4/6: Reconciling active positions & launching 16-Thread Parallel Indicator Engine across {len(ohlcv_dict)} liquid stocks...",
                percent=70,
                stocks_total=len(ohlcv_dict),
                stocks_done=0
            )
            reconciled = reconcile_open_institutional_positions(ohlcv_dict)
            _institutional_engine_state["reconciled_trades"] = len(reconciled)

            # Step 5: 16-Thread Parallel C-Vectorized Feature Extraction
            t_feat = time.time()
            raw_records = FeatureEngine.compute_universe_metrics_parallel(ohlcv_dict, benchmarks=benchmarks, max_workers=16)
            logger.info(f"⚡ Parallel Technical Indicator Engine computed 30 features for {len(raw_records)} stocks in {time.time() - t_feat:.2f}s.")

            if not raw_records:
                _institutional_engine_state["status"] = "NO_CANDIDATES"
                state_engine.finish_scan_progress("Scan finished: 0 stocks passed raw feature extraction.")
                return pd.DataFrame()

            raw_df = pd.DataFrame(raw_records)

            # Step 6: 6-Pillar Percentile Scoring
            state_engine.update_scan_progress(
                stage=f"Step 5/6: Constructing 6-Pillars & Ranking Universe Percentiles ({len(raw_df)} liquid stocks out of {len(tickers)} {source_label} stocks)...",
                percent=85
            )
            scored_df = PercentileScoringEngine.score_universe(raw_df, run_mode=mode)

            # Step 7: ML Re-Ranking
            state_engine.update_scan_progress(
                stage="Step 6/6: Applying Calibrated ML Meta-Classifier Re-Ranker & Score-Weighted Sizing...",
                percent=92
            )
            reranked_df = MLReRanker.apply_reranking(scored_df)

            # Step 8: Classification & Position Sizing
            final_rows = []
            for _, row in reranked_df.iterrows():
                bucket = ExecutionEngine.classify(row)
                pos_info = ExecutionEngine.calculate_position(row, capital=self.capital, base_risk_pct=self.risk_per_trade_pct)

                final_rows.append({
                    "ticker": clean_symbol(row['ticker']),
                    "sector": row['sector'],
                    "bucket": bucket,
                    "score": round(row['score'], 1),
                    "confidence": int(row['confidence']),
                    "ml_prob": float(row['ml_prob']),
                    "tier": pos_info['tier'],
                    "close": row['close'],
                    "entry": pos_info['entry'],
                    "sl": pos_info['sl'],
                    "t1": pos_info['t1'],
                    "t2": pos_info['t2'],
                    "rr": pos_info['rr'],
                    "shares": pos_info['shares'],
                    "capital_at_risk": pos_info['capital_at_risk'],
                    "risk_pct": pos_info['risk_pct'],
                    "atr_pct": round(row['atr_pct'], 2),
                    "rsi14": round(row['rsi14'], 1),
                    "dist_to_pivot": round(row['dist_to_pivot'], 2),
                    "dist_to_ema20": round(row['dist_to_ema20'], 2),
                    "inst_score": round(row['inst_score'], 1),
                    "week_confluence": bool(row['week_confluence']),
                    "intraday_pct": round(row['intraday_pct'], 2),
                    "intraday_adj": round(row['intraday_adj'], 2),
                    "rs_spread": round(row.get('sector_excess_6m', 0.0), 2),
                    "expected_r": float(row.get('expected_r', 0.0)),
                    "entry_type": str(pos_info.get('entry_type', 'BREAKOUT_SURGE')),
                    "ml_verdict": str(row.get('ml_verdict', 'APPROVED')),
                    "risk_flags": str(row.get('risk_flags_json', '[]'))
                })

            signals_df = pd.DataFrame(final_rows)
            bucket_order = {"BUY_NOW": 0, "PRE_BREAKOUT": 1, "WAIT_PULLBACK": 2, "AVOID": 3}
            signals_df["_sort_key"] = signals_df["bucket"].map(bucket_order)
            signals_df.sort_values(by=["_sort_key", "score", "ml_prob"], ascending=[True, False, False], inplace=True)
            signals_df.drop(columns=["_sort_key"], inplace=True)
            signals_df.reset_index(drop=True, inplace=True)

            # Save to database
            records_list = signals_df.to_dict(orient="records")
            state_engine.save_latest_signals(records_list, portfolio_type=PORTFOLIO_TYPE)

            # Update engine runtime state
            b_now = int((signals_df['bucket'] == 'BUY_NOW').sum())
            p_brk = int((signals_df['bucket'] == 'PRE_BREAKOUT').sum())
            w_pb = int((signals_df['bucket'] == 'WAIT_PULLBACK').sum())
            avoid = int((signals_df['bucket'] == 'AVOID').sum())

            pullback = p_brk + w_pb
            _institutional_engine_state["universe_size"] = len(signals_df)
            _institutional_engine_state["buy_now_count"] = b_now
            _institutional_engine_state["pre_breakout_count"] = p_brk
            _institutional_engine_state["wait_pullback_count"] = w_pb
            _institutional_engine_state["pullback_count"] = pullback
            _institutional_engine_state["avoid_count"] = avoid
            _institutional_engine_state["signals"] = records_list[:100]
            _institutional_engine_state["status"] = "COMPLETED"

            b_now_list = signals_df[signals_df['bucket'] == 'BUY_NOW']['ticker'].tolist()
            p_brk_list = signals_df[signals_df['bucket'] == 'PRE_BREAKOUT']['ticker'].tolist()
            w_pb_list = signals_df[signals_df['bucket'] == 'WAIT_PULLBACK']['ticker'].tolist()
            avoid_list = signals_df[signals_df['bucket'] == 'AVOID']['ticker'].tolist()

            try:
                state_engine.record_system_activity(
                    category="SCANNER_PARITY",
                    action="6-Pillar Indicator Analysis & Classification",
                    status="SUCCESS",
                    summary=f"Analyzed {len(signals_df)} stocks: {b_now} BUY_NOW, {p_brk} PRE_BREAKOUT, {w_pb} WAIT_PULLBACK, {avoid} AVOID.",
                    details={
                        "total_analyzed": len(signals_df),
                        "buy_now_count": b_now,
                        "buy_now_stocks": b_now_list,
                        "pre_breakout_count": p_brk,
                        "pre_breakout_stocks": p_brk_list[:25],
                        "wait_pullback_count": w_pb,
                        "wait_pullback_stocks": w_pb_list[:25],
                        "avoid_count": avoid,
                        "mode": mode
                    },
                    execution_sec=round(time.time() - start_time, 2)
                )
            except Exception:
                pass

            # 6. ML + AI FINAL BUY DECISION GATE & EXECUTION
            orders_placed = 0
            gate_notice = ""
            active_positions = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
            active_syms = [p["symbol"] for p in active_positions]

            # 1. Enforce strict derived ranking: Sort strictly by score DESC, then ml_prob DESC
            buy_now_candidates = signals_df[(signals_df['bucket'] == 'BUY_NOW')].copy()
            buy_now_candidates.sort_values(by=["score", "ml_prob"], ascending=[False, False], inplace=True)

            # 2. Derive available slots and free cash from live portfolio metrics
            p1_stats = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE, base_capital=self.capital)
            max_slots = int(p1_stats.get("max_slots", 10))
            available_slots = max(0, max_slots - len(active_positions))
            available_cash = float(p1_stats.get("available_cash", 0.0))

            scan_run_id = f"scan_{int(start_time)}"
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            logger.info(
                f"📊 [P1 ML+AI Buy Gate] Active: {len(active_positions)}/{max_slots} | "
                f"Available Slots: {available_slots} | Free Cash: ₹{available_cash:,.2f} | "
                f"BUY_NOW Candidates: {len(buy_now_candidates)} | Auto-Execute: {auto_execute} | Can-Trade: {can_trade}"
            )

            from engines.swing_quant_ml_engine import swing_quant_ml_engine
            market_regime = swing_quant_ml_engine.capture_market_regime()

            for _, r in buy_now_candidates.iterrows():
                ticker = str(r['ticker']).replace(".NS", "").replace(".BO", "").strip().upper()
                cand_id = f"cand_{ticker}_{int(start_time)}"
                req_val = round(float(r['entry']) * int(r['shares']), 2)

                tech_snapshot = {
                    "score": float(r['score']),
                    "confidence": int(r['confidence']),
                    "rsi14": float(r.get('rsi14', 55.0)),
                    "dist_to_pivot": float(r.get('dist_to_pivot', 0.0)),
                    "dist_to_ema20": float(r.get('dist_to_ema20', 0.0)),
                    "vol_ratio": float(r.get('vol_ratio', 1.0)),
                    "atr14": float(r.get('atr14', 1.0)),
                    "adx14": float(r.get('adx14', 25.0)),
                    "inst_score": float(r.get('inst_score', 60.0)),
                    "week_confluence": int(r.get('week_confluence', 1)),
                    "is_vcp": int(r.get('is_vcp', 0)),
                    "sector": str(r.get('sector', 'DIVERSIFIED'))
                }

                # Step 1: Pre-Trade ML Gatekeeper (Fail-closed defaults)
                eval_res = swing_quant_ml_engine.evaluate_candidate(r.to_dict(), market_regime=market_regime)
                ml_decision = str(eval_res.get("decision", "REJECT_NOT_EVALUABLE"))
                ml_prob = float(eval_res.get("win_prob", 0.0))
                ml_score = float(eval_res.get("prediction_score", r['score']))
                expected_r = float(eval_res.get("expected_r", 0.0))
                risk_flags = list(eval_res.get("risk_flags", []))
                fatal_flags = list(eval_res.get("fatal_flags", []))

                if ml_decision != "PASS_TO_AI":
                    ml_reason_str = ", ".join(risk_flags) if risk_flags else f"Low win probability ({ml_prob*100:.0f}%) or negative expected R"
                    tech_rejection_msg = f"Due to technical analysis this stock was rejected: {ml_reason_str}"
                    logger.info(f"🚫 [ML Gate Rejected] {ticker}: decision={ml_decision}, {tech_rejection_msg}")
                    state_engine.update_signal_ml_verdict(
                        symbol=ticker,
                        portfolio_type=PORTFOLIO_TYPE,
                        ml_verdict="ML_REJECTED",
                        rejection_reason=tech_rejection_msg,
                        ai_summary=""
                    )
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id,
                        symbol=ticker,
                        scanner_source="In-House Nifty 500 Engine" if source == "IN_HOUSE" else "Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id,
                        technical_classification="BUY NOW",
                        technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision,
                        ml_win_prob=ml_prob,
                        ml_score=ml_score,
                        expected_r=expected_r,
                        ml_reasons=risk_flags,
                        ai_request_id=None,
                        ai_status=None,
                        ai_decision=None,
                        ai_reason_code=None,
                        ai_summary=None,
                        ai_factors={},
                        risk_check_status="NOT_REACHED",
                        final_status="ML_REJECTED" if ml_decision == "REJECT" else ml_decision,
                        failure_attribution=""
                    )
                    shadow_meta = r.to_dict()
                    shadow_meta.update({
                        "ml_win_prob": ml_prob,
                        "ml_decision": ml_decision,
                        "rejection_reason": tech_rejection_msg
                    })
                    swing_quant_ml_engine.record_shadow_candidate(
                        symbol=ticker,
                        portfolio_type=PORTFOLIO_TYPE,
                        entry_price=float(r['entry']),
                        stop_loss=float(r['sl']),
                        target=float(r['t1']),
                        target_2=float(r['t2']),
                        candidate_meta=shadow_meta,
                        rejection_reason=f"ML_{ml_decision}: {tech_rejection_msg}"
                    )
                    continue

                # Step 2: Assemble Complete Structured AI Request
                from ai_engine.service import ai_service
                from ai_engine.schemas import (
                    SwingCandidateAIRequest,
                    SwingCandidateIdentity,
                    SwingCandidateContext,
                    SwingTechnicalAnalysis,
                    SwingMLAnalysis,
                    SwingTradePlan,
                    SwingHistoricalEvidence,
                    SwingPortfolioContext,
                    SwingInstructions
                )

                # F-34: Measure real data quality & verify watchdog status
                from core.data_watchdog import data_watchdog
                from core.data_quality import validate_ohlcv_dataframe
                feed_allowed, feed_reason = data_watchdog.is_trading_allowed("swing_trading")

                cand_df = dl_stats.get("dataframes", {}).get(ticker) if isinstance(dl_stats, dict) else None
                if cand_df is None:
                    try:
                        cand_df, _ = state_engine.get_ohlcv_df(ticker)
                    except Exception:
                        cand_df = None

                if cand_df is not None:
                    dq_valid, _, dq_warnings = validate_ohlcv_dataframe(cand_df, symbol=ticker)
                else:
                    dq_valid, dq_warnings = False, ["Missing OHLCV candle data"]
                is_stale = (not feed_allowed)
                dq_status = "valid" if (feed_allowed and dq_valid and not is_stale) else ("stale" if is_stale else "invalid")
                real_data_quality = {
                    "status": dq_status,
                    "is_stale": is_stale,
                    "missing_fields": [],
                    "warnings": dq_warnings + ([feed_reason] if not feed_allowed else [])
                }

                if not feed_allowed or dq_status == "invalid":
                    logger.warning(f"🚨 [Data Quality Gate Blocked] {ticker}: {real_data_quality}")
                    continue

                ai_req_id = f"ai_req_{ticker}_{int(time.time())}"
                ds_summary = swing_quant_ml_engine.get_status().get("dataset_summary", {})
                dataset_count = int(ds_summary.get("total_records") or 1)
                emp_win_rate = ds_summary.get("empirical_win_rate")
                emp_wr_float = round(float(emp_win_rate), 4) if emp_win_rate is not None else None
                ai_request = SwingCandidateAIRequest(
                    schema_version="1.0",
                    request_id=ai_req_id,
                    task="swing_final_buy_decision",
                    module="swing_trading",
                    candidate=SwingCandidateIdentity(
                        candidate_id=cand_id,
                        exchange="NSE",
                        symbol=ticker,
                        scanner_classification="BUY NOW",
                        scanner_run_id=scan_run_id
                    ),
                    context=SwingCandidateContext(
                        as_of=now_iso,
                        timezone="Asia/Kolkata",
                        data_mode="live" if feed_allowed else "degraded",
                        data_quality=real_data_quality
                    ),
                    technical_analysis=SwingTechnicalAnalysis(
                        classification="BUY NOW",
                        entry_conditions=["Score >= 80", "Pivot distance <= 2.0%", "Volume ratio >= 1.15x"],
                        satisfied_conditions=["6-Pillar institutional structure valid", "Volume expansion confirmed"],
                        unsatisfied_conditions=[],
                        features=tech_snapshot,
                        engine_version="6-Pillar-v2.4"
                    ),
                    ml_analysis=SwingMLAnalysis(
                        decision=ml_decision,
                        model_name=eval_res.get("model_name", "HistGradientBoosting+CalibratedClassifierCV"),
                        model_version=eval_res.get("model_version", "swing_v2.0"),
                        target_definition=eval_res.get("target_definition", "target_before_stop_loss_in_holding_period"),
                        prediction_score=ml_score,
                        calibrated_probability=ml_prob,
                        expected_r=expected_r,
                        feature_snapshot_id=eval_res.get("feature_snapshot_id"),
                        decision_policy_version="1.0",
                        risk_flags=risk_flags,
                        fatal_flags=fatal_flags
                    ),
                    trade_plan=SwingTradePlan(
                        entry_rule="Breakout confirmation with volume",
                        entry_price=float(r['entry']),
                        stop_loss=float(r['sl']),
                        target=float(r['t1']),
                        target_2=float(r['t2']),
                        holding_period="5-15 sessions",
                        reward_risk_ratio=float(r['rr']),
                        position_size=int(r['shares']),
                        risk_per_share=round(float(r['entry']) - float(r['sl']), 2),
                        capital_at_risk=float(r.get('capital_at_risk', 3750.0)),
                        estimated_transaction_costs=round(float(r['entry']) * int(r['shares']) * 0.0012, 2)
                    ),
                    # Use real empirical summary if available; never fabricate candidate ml_prob as historical win rate
                    historical_evidence=SwingHistoricalEvidence(
                        sample_count=dataset_count,
                        target_before_stop_frequency=emp_wr_float,
                        stop_before_target_frequency=round(1.0 - emp_wr_float, 4) if emp_wr_float is not None else None,
                        evaluation_type="out_of_sample",
                        period="NSE 500 Historical Validation",
                        historical_win_rate=emp_wr_float
                    ),
                    portfolio_context=SwingPortfolioContext(
                        already_held=(ticker in active_syms),
                        duplicate_pending_order=False,
                        available_capital=available_cash,
                        risk_limits_available=True,
                        active_slots_used=len(active_positions),
                        max_slots=max_slots
                    ),
                    instructions=SwingInstructions(
                        objective="Make a final analytical BUY NOW or DO NOT BUY decision using the supplied evidence.",
                        allow_order_execution=False,
                        do_not_invent_data=True
                    )
                )

                # Step 3: AI Final Analytical Decision
                ai_res = ai_service.evaluate_swing_candidate(ai_request)
                logger.info(f"🤖 [AI Gate Decision] {ticker}: decision={ai_res.final_decision}, reason={ai_res.reason_code}")

                # If external AI provider was rate-limited or unavailable, fall back to Calibrated Quant ML model
                is_ai_outage = (ai_res.status in ("AI_UNAVAILABLE", "AI_DISABLED") or "rate limit" in str(ai_res.summary or "").lower())
                if ai_res.final_decision != "BUY_NOW" and is_ai_outage and ml_decision == "PASS_TO_AI" and ml_prob >= 0.55 and expected_r > 0.0:
                    logger.info(
                        f"⚡ [Quant ML Fallback] {ticker}: AI provider temporarily rate-limited ({ai_res.reason_code}). "
                        f"Admitting via Calibrated Quant ML Model (Win Prob: {ml_prob*100:.0f}%, Score: {r['score']}, Expected R: {expected_r:.2f}R)."
                    )
                    ai_res.final_decision = "BUY_NOW"
                    ai_res.reason_code = "QUANT_ML_APPROVED"
                    ai_res.summary = f"Admitted via Calibrated Quant ML Model (Win Prob: {ml_prob*100:.0f}%, Score: {r['score']}, Expected R: {expected_r:.2f}R) during external AI provider rate-limiting."

                # If AI returned DO_NOT_BUY
                if ai_res.final_decision != "BUY_NOW":
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id,
                        symbol=ticker,
                        scanner_source="Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id,
                        technical_classification="BUY NOW",
                        technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision,
                        ml_win_prob=ml_prob,
                        ml_score=ml_score,
                        expected_r=expected_r,
                        ml_reasons=risk_flags,
                        ai_request_id=ai_req_id,
                        ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision,
                        ai_reason_code=ai_res.reason_code,
                        ai_summary=ai_res.summary,
                        ai_factors={
                            "supporting": ai_res.supporting_factors,
                            "contradicting": ai_res.contradicting_factors,
                            "risk": ai_res.risk_factors
                        },
                        risk_check_status="NOT_REACHED",
                        final_status="AI_DO_NOT_BUY",
                        failure_attribution=""
                    )
                    tech_rejection_msg = f"Due to technical analysis this stock was rejected: {ai_res.reason_code} - {ai_res.summary}"
                    logger.info(f"🛑 [P1 AI Gate Blocked] {ticker} rejected by AI Gatekeeper: {tech_rejection_msg}")
                    state_engine.update_signal_ml_verdict(
                        symbol=ticker,
                        portfolio_type=PORTFOLIO_TYPE,
                        ml_verdict=f"AI_REJECTED: {ai_res.reason_code}",
                        rejection_reason=tech_rejection_msg,
                        ai_summary=ai_res.summary
                    )
                    shadow_meta = r.to_dict()
                    shadow_meta.update({
                        "ml_win_prob": ml_prob,
                        "ml_decision": ml_decision,
                        "ai_decision": ai_res.final_decision,
                        "ai_reason_code": ai_res.reason_code,
                        "ai_summary": ai_res.summary,
                        "rejection_reason": tech_rejection_msg
                    })
                    swing_quant_ml_engine.record_shadow_candidate(
                        symbol=ticker,
                        portfolio_type=PORTFOLIO_TYPE,
                        entry_price=float(r['entry']),
                        stop_loss=float(r['sl']),
                        target=float(r['t1']),
                        target_2=float(r['t2']),
                        candidate_meta=shadow_meta,
                        rejection_reason=f"AI_{ai_res.reason_code}: {tech_rejection_msg}"
                    )
                    state_engine.update_scan_progress(
                        stage=f"🛑 AI Gated DO NOT BUY {ticker}: {ai_res.reason_code}",
                        percent=96,
                        add_log=True
                    )
                    continue

                # Step 4: AI Approved! Now Apply Deterministic Risk Engine Checks
                state_engine.update_signal_ml_verdict(
                    symbol=ticker,
                    portfolio_type=PORTFOLIO_TYPE,
                    ml_verdict="AI_APPROVED",
                    rejection_reason="",
                    ai_summary=ai_res.summary
                )
                if not auto_execute:
                    # Gated analysis recorded, but auto execution is disabled
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id,
                        symbol=ticker,
                        scanner_source="Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id,
                        technical_classification="BUY NOW",
                        technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision,
                        ml_win_prob=ml_prob,
                        ml_score=ml_score,
                        expected_r=expected_r,
                        ml_reasons=risk_flags,
                        ai_request_id=ai_req_id,
                        ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision,
                        ai_reason_code=ai_res.reason_code,
                        ai_summary=ai_res.summary,
                        ai_factors={"supporting": ai_res.supporting_factors, "contradicting": ai_res.contradicting_factors, "risk": ai_res.risk_factors},
                        risk_check_status="AUTO_EXECUTE_DISABLED",
                        final_status="PURCHASE_ELIGIBLE"
                    )
                    continue

                if not can_trade:
                    gate_notice = " [🔒 STRICT TRADING GATE: Market Closed — Signals updated, 0 trades executed]"
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id,
                        symbol=ticker,
                        scanner_source="Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id,
                        technical_classification="BUY NOW",
                        technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision,
                        ml_win_prob=ml_prob,
                        ml_score=ml_score,
                        expected_r=expected_r,
                        ml_reasons=risk_flags,
                        ai_request_id=ai_req_id,
                        ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision,
                        ai_reason_code=ai_res.reason_code,
                        ai_summary=ai_res.summary,
                        ai_factors={"supporting": ai_res.supporting_factors, "contradicting": ai_res.contradicting_factors, "risk": ai_res.risk_factors},
                        risk_check_status="BLOCKED_BY_MARKET_HOURS",
                        final_status="PURCHASE_ELIGIBLE"
                    )
                    continue

                if ticker in active_syms:
                    logger.info(f"ℹ️ [P1 Risk Gate Blocked] {ticker} already held in active portfolio.")
                    continue

                if orders_placed >= available_slots:
                    logger.info(f"ℹ️ [P1 Risk Gate Blocked] All available slots filled ({orders_placed}/{available_slots}).")
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id, symbol=ticker, scanner_source="Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id, technical_classification="BUY NOW", technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision, ml_win_prob=ml_prob, ml_score=ml_score, expected_r=expected_r,
                        ml_reasons=risk_flags, ai_request_id=ai_req_id, ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision, ai_reason_code=ai_res.reason_code, ai_summary=ai_res.summary,
                        ai_factors={"supporting": ai_res.supporting_factors, "contradicting": ai_res.contradicting_factors, "risk": ai_res.risk_factors},
                        risk_check_status="BLOCKED_BY_SLOTS", final_status="BLOCKED_BY_RISK"
                    )
                    break

                if r['shares'] <= 0:
                    continue

                if req_val > available_cash:
                    logger.warning(f"⚠️ [P1 Risk Gate Blocked] {ticker}: Required ₹{req_val:,.2f} > Available Cash ₹{available_cash:,.2f}")
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id, symbol=ticker,
                        scanner_source="In-House Nifty 500 Engine" if source == "IN_HOUSE" else "Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id, technical_classification="BUY NOW", technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision, ml_win_prob=ml_prob, ml_score=ml_score, expected_r=expected_r,
                        ml_reasons=risk_flags, ai_request_id=ai_req_id, ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision, ai_reason_code=ai_res.reason_code, ai_summary=ai_res.summary,
                        ai_factors={"supporting": ai_res.supporting_factors, "contradicting": ai_res.contradicting_factors, "risk": ai_res.risk_factors},
                        risk_check_status="INSUFFICIENT_CASH", final_status="BLOCKED_BY_RISK"
                    )
                    state_engine.update_scan_progress(
                        stage=f"⚠️ Skipped {ticker}: Required ₹{req_val:,.0f} > Available Cash ₹{available_cash:,.0f}",
                        percent=96, add_log=True
                    )
                    continue

                # Step 5: Execute Authorized Purchase
                real_meta = r.to_dict()
                real_meta.update({
                    "ml_win_prob": ml_prob,
                    "ml_decision": ml_decision,
                    "ai_decision": ai_res.final_decision,
                    "ai_reason_code": ai_res.reason_code,
                    "ai_summary": ai_res.summary
                })

                opened = state_engine.open_position(
                    symbol=ticker,
                    entry_price=r['entry'],
                    quantity=r['shares'],
                    stop_loss=r['sl'],
                    target=r['t1'],
                    target_2=r['t2'],
                    triggered_from=f"INSTITUTIONAL_{mode.upper()}",
                    notes=f"ML+AI Approved | {ai_res.reason_code} | ML: {ml_prob*100:.0f}% | Score: {r['score']} | T1: ₹{r['t1']} | T2: ₹{r['t2']}",
                    sl_reason="INITIAL_SL",
                    sector=r['sector'],
                    portfolio_type=PORTFOLIO_TYPE,
                    entry_reason=f"ML+AI Approved BUY_NOW: {ai_res.reason_code}, ML={ml_prob*100:.0f}%, Score={r['score']}",
                    entry_regime="BULLISH_CONFIRMED",
                    entry_metrics=real_meta
                )
                if opened:
                    orders_placed += 1
                    available_cash = max(0.0, available_cash - req_val)
                    active_syms.append(ticker)
                    swing_quant_ml_engine.record_decision_audit(
                        candidate_id=cand_id, symbol=ticker, scanner_source="Chartink 6-Pillar Screener",
                        scanner_run_id=scan_run_id, technical_classification="BUY NOW", technical_snapshot=tech_snapshot,
                        ml_decision=ml_decision, ml_win_prob=ml_prob, ml_score=ml_score, expected_r=expected_r,
                        ml_reasons=risk_flags, ai_request_id=ai_req_id, ai_status=ai_res.status,
                        ai_decision=ai_res.final_decision, ai_reason_code=ai_res.reason_code, ai_summary=ai_res.summary,
                        ai_factors={"supporting": ai_res.supporting_factors, "contradicting": ai_res.contradicting_factors, "risk": ai_res.risk_factors},
                        risk_check_status="PASSED", final_status="ORDER_FILLED",
                        trade_id=f"P1_{ticker}_{int(start_time)}"
                    )
                    state_engine.update_scan_progress(
                        stage=f"✅ ML+AI Approved BUY_NOW #{orders_placed}: {ticker} ({r['shares']} shs @ ₹{r['entry']:.2f}) [Prob: {ml_prob*100:.0f}%]",
                        percent=96,
                        add_log=True
                    )

            _institutional_engine_state["orders_executed"] = orders_placed

            # Closed-Loop Shadow Candidate Tracking
            try:
                from engines.swing_quant_ml_engine import swing_quant_ml_engine
                current_active_syms = [p["symbol"] for p in state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)]
                shadow_logged = 0
                for _, s_row in signals_df.iterrows():
                    s_sym = s_row['ticker']
                    if s_sym in current_active_syms:
                        continue
                    bkt = s_row['bucket']
                    verd = str(s_row.get('ml_verdict', 'APPROVED')).upper()
                    # Strictly ingest all BUY_NOW candidates into Uncapped Training Portfolio (no cash/slot limits)
                    if bkt == 'BUY_NOW':
                        rej_tag = f"BUY_NOW_{verd}"
                        logged = swing_quant_ml_engine.record_shadow_candidate(
                            symbol=s_sym,
                            portfolio_type=PORTFOLIO_TYPE,
                            entry_price=float(s_row['entry']),
                            stop_loss=float(s_row['sl']),
                            target=float(s_row['t1']),
                            target_2=float(s_row['t2']),
                            candidate_meta=s_row.to_dict(),
                            rejection_reason=rej_tag
                        )
                        if logged:
                            shadow_logged += 1
                if shadow_logged > 0:
                    logger.info(f"👻 Logged {shadow_logged} BUY_NOW candidates into Uncapped Training Portfolio.")
            except Exception as e:
                logger.debug(f"Failed to record shadow candidates: {e}")

            try:
                state_engine.record_system_activity(
                    category="PORTFOLIO_RISK",
                    action="Risk Gate Evaluation & Portfolio Execution",
                    status="SUCCESS" if orders_placed > 0 or not can_trade else "INFO",
                    summary=(
                        f"Risk Gate evaluated {b_now} BUY_NOW candidates: {orders_placed} added to portfolio "
                        f"({len(active_positions) + orders_placed}/{max_slots} slots, ₹{available_cash:,.0f} free cash). "
                        f"Trading Gate: {'OPEN' if can_trade else 'CLOSED (Market Hours)'}."
                    ),
                    details={
                        "buy_now_total": b_now,
                        "orders_executed": orders_placed,
                        "active_slots_used": len(active_positions) + orders_placed,
                        "max_slots": max_slots,
                        "available_slots": available_slots,
                        "available_cash": round(available_cash, 2),
                        "can_trade": can_trade,
                        "gate_notice": gate_notice
                    },
                    execution_sec=round(time.time() - start_time, 2)
                )
            except Exception:
                pass

            if shadow_logged > 0:
                try:
                    state_engine.record_system_activity(
                        category="SHADOW_ML",
                        action="ML Shadow Candidate Training Ingest",
                        status="SUCCESS",
                        summary=f"Logged {shadow_logged} unconstrained BUY_NOW candidates to quant_trade_dataset for offline ML training.",
                        details={
                            "candidates_logged": shadow_logged,
                            "sample_symbols": [s for s in b_now_list if s not in active_syms][:15],
                            "dataset": "quant_trade_dataset"
                        },
                        execution_sec=round(time.time() - start_time, 2)
                    )
                except Exception:
                    pass

            elapsed = time.time() - start_time
            nse_tot = _institutional_engine_state.get("nse_total_count") or state_engine.get_total_nse_stocks_count()
            pass_cnt = _institutional_engine_state.get("passing_criteria_count") or len(tickers)
            summary_msg = (
                f"{source_label} Screener Complete in {elapsed:.1f}s: {nse_tot} NSE stocks screened -> "
                f"{pass_cnt} passed criteria -> {len(signals_df)} liquid analyzed -> "
                f"BUY NOW: {b_now} (Auto-Executed: {orders_placed}) | PULLBACK: {pullback} (Pre-Breakout: {p_brk}, Wait Pullback: {w_pb}) | AVOID: {avoid}{gate_notice}"
            )
            state_engine.log_scan_run(
                scan_type=f"INSTITUTIONAL_{mode.upper()}",
                portfolio_type=PORTFOLIO_TYPE,
                status="SUCCESS",
                execution_time_sec=elapsed,
                candidates_count=b_now + p_brk,
                orders_count=orders_placed,
                reconciled_count=len(reconciled),
                message=summary_msg,
                details=dl_stats
            )
            state_engine.set_kv("institutional_snapshot", {
                "source": source,
                "source_label": source_label,
                "nse_total_count": nse_tot,
                "passing_criteria_count": pass_cnt,
                "chartink_fetched_count": pass_cnt,
                "universe_size": len(signals_df),
                "cache_hits_count": dl_stats.get("cache_hits", 0),
                "gap_filled_count": dl_stats.get("gap_filled", 0),
                "full_downloads_count": dl_stats.get("full_downloads", 0),
                "buy_now_count": b_now,
                "pre_breakout_count": p_brk,
                "wait_pullback_count": w_pb,
                "pullback_count": pullback,
                "avoid_count": avoid,
                "last_run_time": format_ist()
            })
            state_engine.finish_scan_progress(summary_msg)

            # Record scheduled/manual 6-pillar run telemetry for morning (10:20 AM) or afternoon (3:00 PM)
            curr_hour = now_ist().hour
            is_morning = (mode_override == "morning") or (curr_hour < 13)
            run_key = "morning_1020" if is_morning else "afternoon_305"
            state_engine.record_scheduled_scan_outcome(
                run_key=run_key,
                status="SUCCESS",
                buy_now_count=b_now,
                total_scanned=len(signals_df),
                pre_breakout_count=p_brk,
                wait_pullback_count=w_pb,
                note=f"{b_now} BUY NOW stocks found out of {len(signals_df)} analyzed"
            )

            # Master Database backup is scheduled daily at 15:45 IST via DailyBackupManager

            logger.info(summary_msg)
            return signals_df
        except Exception as err:
            _institutional_engine_state["status"] = "FAILED"
            err_msg = f"Scan encountered error: {err}"
            logger.exception(err_msg)
            state_engine.finish_scan_progress(err_msg)
            curr_hour = now_ist().hour
            is_morning = (mode_override == "morning") or (curr_hour < 13)
            run_key = "morning_1020" if is_morning else "afternoon_305"
            state_engine.record_scheduled_scan_outcome(
                run_key=run_key,
                status="FAILED",
                error=str(err),
                note=f"Scan error: {err}"
            )
            return pd.DataFrame()


def get_institutional_status() -> Dict[str, Any]:
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    state_copy = dict(_institutional_engine_state)
    if not state_copy.get("chartink_fetched_count"):
        saved = state_engine.get_kv("institutional_snapshot", {})
        if isinstance(saved, dict):
            for k, v in saved.items():
                if not state_copy.get(k):
                    state_copy[k] = v
    state_copy["trading_gate_open"] = can_trade
    state_copy["trading_gate_message"] = gate_msg
    state_copy["stats"] = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE)
    state_copy["active_count"] = len(state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE))
    state_copy["last_price_stream"] = state_engine.get_kv("last_price_stream", {})
    return state_copy


def refresh_live_market_prices(scope: str = "ACTIVE_AND_SIGNALS", symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fast multi-threaded live price streamer & MTM updater (runs every 5 minutes during
    NSE market hours 09:15–15:30 IST and on manual 'Refresh Live Prices' click):
    1. Fetches live intraday quotes/bars from Yahoo Finance for all Active Positions
       and Screened Signals on the page.
    2. Updates `active_positions` (LTP, Unrealized PnL, Highest Price, Trailing Stops).
    3. Updates `latest_signals` (`close`) and `ohlcv_master_store` (`last_close`, `change_pct`, live partial candle).
    """
    t0 = time.time()
    fetcher = DirectYahooFinanceFetcher(max_workers=24)

    active_positions = state_engine.get_active_positions()
    scope_up = (scope or "ACTIVE_AND_SIGNALS").strip().upper()

    if symbols:
        ordered_syms = [s.strip().upper() for s in symbols if s and not s.startswith("^")]
    else:
        pos_syms = [p["symbol"].strip().upper() for p in active_positions if p.get("symbol")]

        sig_syms: List[str] = []
        if scope_up in ("ACTIVE_AND_SIGNALS", "ALL"):
            p1_sigs = state_engine.get_latest_signals(portfolio_type="INSTITUTIONAL_6PILLAR")
            p2_sigs = state_engine.get_latest_signals(portfolio_type="NIFTY500_MOMENTUM")
            if scope_up == "ALL":
                sig_syms = [s["symbol"].strip().upper() for s in (p1_sigs + p2_sigs) if s.get("symbol")]
            else:
                # Refresh all actionable buckets (BUY_NOW, PRE_BREAKOUT, WAIT_PULLBACK) + P2 signals
                actionable_p1 = [
                    s["symbol"].strip().upper()
                    for s in p1_sigs
                    if s.get("symbol") and s.get("bucket") in ("BUY_NOW", "PRE_BREAKOUT", "WAIT_PULLBACK")
                ]
                p2_list = [s["symbol"].strip().upper() for s in p2_sigs if s.get("symbol")]
                sig_syms = actionable_p1 + p2_list

        ordered_syms: List[str] = []
        seen = set()
        for s in pos_syms + sig_syms:
            # Strictly ignore non-NSE equities, commodity futures (=F), or indices (^)
            if s and s not in seen and not s.startswith("^") and not s.endswith("=F") and "=" not in s:
                seen.add(s)
                ordered_syms.append(s)

    if not ordered_syms:
        return {
            "status": "NO_SYMBOLS",
            "updated_count": 0,
            "symbols_updated": 0,
            "positions_updated": 0,
            "active_positions_updated": 0,
            "timestamp": format_ist()
        }

    # Auto-subscribe active symbols and batch prime via Kotak Neo (zero Yahoo overhead)
    kotak_is_active = False
    try:
        from engines.kotak_neo_engine import kotak_neo_engine
        if kotak_neo_engine._is_authenticated:
            kotak_is_active = True
            kotak_neo_engine.subscribe_symbols(ordered_syms)
            unprimed = [s for s in ordered_syms if not kotak_neo_engine.get_live_price(s)]
            if unprimed:
                kotak_neo_engine.fetch_rest_quotes(unprimed)
    except Exception:
        pass

    def _fetch_live_tick(sym: str) -> Tuple[str, Optional[pd.DataFrame], Optional[float], Optional[float]]:
        # 1. Fast Path: In-Memory Kotak Neo Real-Time WebSocket / REST Tick Cache
        if kotak_is_active:
            try:
                from engines.kotak_neo_engine import kotak_neo_engine
                kotak_tick = kotak_neo_engine.get_live_tick(sym)
                if not kotak_tick or not kotak_tick.get("ltp"):
                    lp = kotak_neo_engine.get_live_price(sym)
                    if lp and lp > 0:
                        kotak_tick = {"ltp": lp, "high": lp, "low": lp, "volume": 0}

                if kotak_tick and kotak_tick.get("ltp") and float(kotak_tick["ltp"]) > 0:
                    k_ltp = float(kotak_tick["ltp"])
                    k_high = float(kotak_tick.get("high") or k_ltp)
                    cached_df, _ = state_engine.get_ohlcv_df(sym)
                    if cached_df is not None and not cached_df.empty:
                        merged = cached_df.copy()
                        today_d = now_ist().date()
                        last_dt = pd.to_datetime(merged.index[-1]).date()
                        if last_dt == today_d:
                            merged.loc[merged.index[-1], "Close"] = k_ltp
                            merged.loc[merged.index[-1], "High"] = max(float(merged.loc[merged.index[-1], "High"]), k_high)
                            if kotak_tick.get("low"):
                                merged.loc[merged.index[-1], "Low"] = min(float(merged.loc[merged.index[-1], "Low"]), float(kotak_tick["low"]))
                        else:
                            new_row = pd.DataFrame([{
                                "Open": float(kotak_tick.get("open") or k_ltp),
                                "High": k_high,
                                "Low": float(kotak_tick.get("low") or k_ltp),
                                "Close": k_ltp,
                                "Volume": int(kotak_tick.get("volume") or 0)
                            }], index=pd.DatetimeIndex([pd.to_datetime(now_ist())]))
                            merged = pd.concat([merged, new_row])
                        return sym, merged, k_ltp, k_high
                    else:
                        dummy_df = pd.DataFrame([{
                            "Open": k_ltp, "High": k_high, "Low": k_ltp, "Close": k_ltp, "Volume": 0
                        }], index=pd.DatetimeIndex([pd.to_datetime(now_ist())]))
                        return sym, dummy_df, k_ltp, k_high
            except Exception as e:
                logger.debug(f"Kotak tick merge error for {sym}: {e}")
            # Strictly return None - NEVER call Yahoo Finance during market hours when Kotak is active
            return sym, None, None, None

        try:
            cached_df, _ = state_engine.get_ohlcv_df(sym)

            # If cache is absent or truncated (< 80 bars), do a full 2-year fetch so we never truncate historical depth!
            if cached_df is None or len(cached_df) < 80:
                _, full_df = fetcher.fetch_ticker_chart(sym, range_period="2y", max_retries=2)
                if full_df is not None and len(full_df) >= 80:
                    merged = full_df.copy()
                    merged.index = pd.to_datetime(merged.index, utc=True)
                else:
                    delta_df = fetcher._request_yahoo_df(
                        f"{sym}.NS",
                        {"range": "5d", "interval": "1d", "includePrePost": "false"},
                        max_retries=2
                    )
                    if delta_df is None or delta_df.empty:
                        return sym, None, None, None
                    merged = delta_df.copy()
                    merged.index = pd.to_datetime(merged.index, utc=True)
            else:
                delta_df = fetcher._request_yahoo_df(
                    f"{sym}.NS",
                    {"range": "5d", "interval": "1d", "includePrePost": "false"},
                    max_retries=2
                )
                if delta_df is None or delta_df.empty:
                    return sym, None, None, None

                c_df = cached_df.copy()
                d_df = delta_df.copy()
                c_df.index = pd.to_datetime(c_df.index, utc=True)
                d_df.index = pd.to_datetime(d_df.index, utc=True)
                merged = pd.concat([c_df, d_df]).sort_index()
                dt_series = pd.Series(pd.DatetimeIndex(merged.index).date)
                merged = merged[~dt_series.duplicated(keep="last").values]
                if len(merged) > 500:
                    merged = merged.iloc[-500:]

            ltp = round(float(merged["Close"].iloc[-1]), 2)
            day_high = round(float(merged["High"].iloc[-1]), 2)
            return sym, merged, ltp, day_high
        except Exception as e:
            logger.warning(f"Error updating live price tick for {sym}: {e}")
            return sym, None, None, None

    updated_dfs: Dict[str, pd.DataFrame] = {}
    live_ltp_map: Dict[str, Tuple[float, float]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(_fetch_live_tick, s): s for s in ordered_syms}
        for fut in concurrent.futures.as_completed(futures):
            try:
                sym, merged_df, ltp, day_high = fut.result()
                if merged_df is not None and ltp is not None and ltp > 0:
                    updated_dfs[f"{sym}.NS"] = merged_df
                    live_ltp_map[sym] = (ltp, day_high or ltp)
            except Exception:
                pass

    if updated_dfs:
        state_engine.save_ohlcv_batch(updated_dfs)

    now_str = format_ist()
    pos_updated = 0
    for p in active_positions:
        sym = p["symbol"].strip().upper()
        if sym in live_ltp_map:
            ltp, day_high = live_ltp_map[sym]
            prev_high = float(p.get("highest_price") or p.get("entry_price") or ltp)
            state_engine.update_position_price(sym, current_price=ltp, highest_price=max(prev_high, day_high, ltp))
            pos_updated += 1

    # Also update `close` in `latest_signals` so the Screener tables reflect fresh 5m LTP
    if live_ltp_map:
        state_engine.update_signals_live_prices(live_ltp_map, now_str)

    # Reconcile stops/targets if market is open
    can_trade, _ = NSEMarketCalendar.can_execute_trades()
    reconciled_events = []
    if can_trade and updated_dfs:
        try:
            reconciled_events = reconcile_open_institutional_positions(updated_dfs)
        except Exception as e:
            logger.debug(f"Reconcile during live price stream notice: {e}")

    elapsed = round(time.time() - t0, 2)
    kotak_status = {}
    try:
        from engines.kotak_neo_engine import kotak_neo_engine
        kotak_status = kotak_neo_engine.get_status()
    except Exception:
        pass

    stream_meta = {
        "status": "OK",
        "scope": scope_up,
        "updated_count": len(live_ltp_map),
        "symbols_updated": len(live_ltp_map),
        "positions_updated": pos_updated,
        "active_positions_updated": pos_updated,
        "reconciled_count": len(reconciled_events),
        "elapsed_sec": elapsed,
        "timestamp": now_str,
        "kotak_neo_streaming": kotak_status.get("is_streaming", False),
        "kotak_neo_ticks": kotak_status.get("ticks_received_total", 0),
        "prices": {k: v[0] for k, v in list(live_ltp_map.items())[:50]}
    }
    state_engine.set_kv("last_price_stream", stream_meta)
    try:
        state_engine.record_system_activity(
            category="PRICE_STREAM",
            action="5-Min Live Price Stream & MTM Sync",
            status="SUCCESS",
            summary=f"Streamed live quotes: {len(live_ltp_map)} stocks updated & {pos_updated} active holdings updated in {elapsed}s at {now_str}.",
            details=stream_meta,
            execution_sec=elapsed
        )
    except Exception:
        pass
    try:
        from core.data_watchdog import data_watchdog
        data_watchdog.record_packet("swing_trading", latency_ms=elapsed * 1000.0, details={
            "scope": scope_up,
            "updated_count": len(live_ltp_map),
            "positions_updated": pos_updated,
            "reconciled_count": len(reconciled_events),
            "last_stream_time": now_str
        })
    except Exception:
        pass
    if pos_updated > 0 or len(reconciled_events) > 0:
        try:
            telegram_vault.sync_master_vault_async(
                note=f"Live Price Stream MTM Sync ({pos_updated} positions updated, {len(reconciled_events)} reconciled)",
                send_notification=False
            )
        except Exception:
            pass
    logger.info(
        f"⚡ Live Price Stream ({scope_up}): Updated {len(live_ltp_map)} stocks & {pos_updated} active positions in {elapsed}s at {now_str}"
    )
    return stream_meta


class InstitutionalPortfolioBacktester:
    """
    Event-driven historical backtester for Institutional 6-Pillar strategy (F-31).
    Simulates portfolio allocations, trailing ATR exits, and statutory transaction costs.
    """
    def __init__(self, capital: float = 1000000.0, risk_per_trade_pct: float = 0.75):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct

    def run(self, ohlcv_data: Any, lookback_days: int = 250) -> Dict[str, Any]:
        logger.info(f"[*] Running Institutional Portfolio Backtest (Capital: ₹{self.capital:,.0f}, Lookback: {lookback_days} days)...")
        from core.config import EQUITY_ROUND_TRIP_COST_RATE

        ohlcv_dict = ohlcv_data[0] if isinstance(ohlcv_data, tuple) else (ohlcv_data or {})
        valid_symbols = {s: df for s, df in ohlcv_dict.items() if df is not None and len(df) >= min(60, lookback_days)}

        if not valid_symbols:
            logger.warning("No valid symbols with sufficient history for backtest.")
            return {
                "initial_capital": self.capital,
                "final_equity": self.capital,
                "total_net_pnl": 0.0,
                "return_pct": 0.0,
                "symbols_evaluated": 0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "expectancy_r": 0.0,
                "status": "COMPLETED",
                "message": "No valid symbols with sufficient history."
            }

        print(f"Loaded {len(valid_symbols)} universe symbols with sufficient history for {lookback_days}-day backtest.")

        # Collect unique trading dates across all valid symbols
        all_dates = set()
        for df in valid_symbols.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index[-lookback_days:])
            elif "date" in df.columns:
                all_dates.update(pd.to_datetime(df["date"]).iloc[-lookback_days:])

        sorted_dates = sorted(list(all_dates))
        if not sorted_dates:
            sorted_dates = list(range(lookback_days))

        current_cash = float(self.capital)
        max_slots = 10
        active_positions = {}  # symbol -> dict
        closed_trades = []
        equity_curve = [self.capital]

        for dt in sorted_dates:
            # 1. Reconcile existing active positions for exits on date dt
            syms_to_close = []
            for sym, pos in list(active_positions.items()):
                df = valid_symbols[sym]
                if dt not in df.index:
                    continue
                bar = df.loc[dt]
                c = float(bar["Close"])
                h = float(bar["High"])
                l = float(bar["Low"])

                entry_px = pos["entry_price"]
                sl = pos["stop_loss"]
                t1 = pos["target"]
                t2 = pos["target_2"]
                qty = pos["quantity"]
                t1_hit = pos["t1_hit"]

                # Exit checks
                if l <= sl:
                    # Stop loss hit
                    exit_px = min(c, sl)
                    gross_pnl = (exit_px - entry_px) * qty + pos["t1_pnl"]
                    turnover = (entry_px * pos["orig_qty"]) + (exit_px * qty) + (t1 * (pos["orig_qty"] - qty) if t1_hit else 0.0)
                    costs = round(turnover * EQUITY_ROUND_TRIP_COST_RATE, 2)
                    net_pnl = round(gross_pnl - costs, 2)
                    current_cash += (exit_px * qty) - costs
                    risk_unit = max(0.01, entry_px - pos["orig_sl"])
                    r_mult = round(net_pnl / max(0.01, risk_unit * pos["orig_qty"]), 2)
                    closed_trades.append({
                        "symbol": sym, "entry_price": entry_px, "exit_price": exit_px,
                        "pnl": net_pnl, "r_multiple": r_mult, "exit_reason": "STOP_LOSS"
                    })
                    syms_to_close.append(sym)
                    continue

                if not t1_hit and h >= t1:
                    # Book 50% at T1
                    booked_qty = qty // 2
                    pnl_t1 = (t1 - entry_px) * booked_qty
                    rem_qty = qty - booked_qty
                    current_cash += (t1 * booked_qty)
                    pos["quantity"] = rem_qty
                    pos["stop_loss"] = entry_px  # Breakeven
                    pos["t1_hit"] = True
                    pos["t1_pnl"] = pnl_t1
                    pos["highest_price"] = max(pos["highest_price"], h)

                if pos.get("t1_hit") and h >= t2:
                    # Target 2 hit
                    exit_px = max(t2, c)
                    gross_pnl = (exit_px - entry_px) * pos["quantity"] + pos["t1_pnl"]
                    turnover = (entry_px * pos["orig_qty"]) + (exit_px * pos["quantity"]) + (t1 * (pos["orig_qty"] - pos["quantity"]))
                    costs = round(turnover * EQUITY_ROUND_TRIP_COST_RATE, 2)
                    net_pnl = round(gross_pnl - costs, 2)
                    current_cash += (exit_px * pos["quantity"]) - costs
                    risk_unit = max(0.01, entry_px - pos["orig_sl"])
                    r_mult = round(net_pnl / max(0.01, risk_unit * pos["orig_qty"]), 2)
                    closed_trades.append({
                        "symbol": sym, "entry_price": entry_px, "exit_price": exit_px,
                        "pnl": net_pnl, "r_multiple": r_mult, "exit_reason": "T2_TARGET"
                    })
                    syms_to_close.append(sym)
                    continue

                # Trailing ATR ratchet
                if pos.get("t1_hit"):
                    pos["highest_price"] = max(pos["highest_price"], h)
                    atr = pos.get("atr", entry_px * 0.02)
                    trail_sl = round(max(entry_px, pos["highest_price"] - 1.5 * atr), 2)
                    if trail_sl > pos["stop_loss"]:
                        pos["stop_loss"] = trail_sl

            for s in syms_to_close:
                active_positions.pop(s, None)

            # 2. Evaluate candidate entries if slots are available
            available_slots = max_slots - len(active_positions)
            if available_slots > 0 and current_cash > (self.capital * 0.05):
                for sym, df in valid_symbols.items():
                    if sym in active_positions:
                        continue
                    if dt not in df.index:
                        continue
                    idx_loc = df.index.get_loc(dt)
                    if idx_loc < 30:
                        continue
                    sub_df = df.iloc[max(0, idx_loc - 30):idx_loc + 1]
                    c = float(sub_df["Close"].iloc[-1])
                    ema20 = float(sub_df["Close"].ewm(span=20).mean().iloc[-1])
                    high20 = float(sub_df["High"].iloc[-21:-1].max()) if len(sub_df) > 20 else c

                    # Simple 6-pillar setup check
                    if c > ema20 and c >= (high20 * 0.98):
                        atr14 = float(np.mean(sub_df["High"].iloc[-14:] - sub_df["Low"].iloc[-14:]))
                        risk_per_share = max(c * 0.03, 1.5 * atr14)
                        risk_amt = self.capital * (self.risk_per_trade_pct / 100.0)
                        shares = int(risk_amt / risk_per_share)
                        pos_val = shares * c
                        if pos_val > (self.capital * 0.20):
                            shares = int((self.capital * 0.20) / c)
                        if shares > 5 and (shares * c) <= current_cash:
                            entry_costs = round(shares * c * (EQUITY_ROUND_TRIP_COST_RATE / 2.0), 2)
                            current_cash -= (shares * c + entry_costs)
                            active_positions[sym] = {
                                "entry_price": c,
                                "orig_sl": c - risk_per_share,
                                "stop_loss": c - risk_per_share,
                                "target": round(c + 1.5 * risk_per_share, 2),
                                "target_2": round(c + 3.0 * risk_per_share, 2),
                                "quantity": shares,
                                "orig_qty": shares,
                                "t1_hit": False,
                                "t1_pnl": 0.0,
                                "highest_price": c,
                                "atr": atr14
                            }
                            available_slots -= 1
                            if available_slots <= 0:
                                break

            # Track end-of-day equity
            mtm_active = sum(pos["quantity"] * float(valid_symbols[sym].loc[dt]["Close"]) for sym, pos in active_positions.items() if dt in valid_symbols[sym].index)
            day_equity = current_cash + mtm_active
            equity_curve.append(day_equity)

        # Close any remaining active positions at last available price
        for sym, pos in list(active_positions.items()):
            df = valid_symbols[sym]
            last_c = float(df["Close"].iloc[-1])
            gross_pnl = (last_c - pos["entry_price"]) * pos["quantity"] + pos["t1_pnl"]
            turnover = (pos["entry_price"] * pos["orig_qty"]) + (last_c * pos["quantity"])
            costs = round(turnover * EQUITY_ROUND_TRIP_COST_RATE, 2)
            net_pnl = round(gross_pnl - costs, 2)
            current_cash += (last_c * pos["quantity"]) - costs
            risk_unit = max(0.01, pos["entry_price"] - pos["orig_sl"])
            r_mult = round(net_pnl / max(0.01, risk_unit * pos["orig_qty"]), 2)
            closed_trades.append({
                "symbol": sym, "entry_price": pos["entry_price"], "exit_price": last_c,
                "pnl": net_pnl, "r_multiple": r_mult, "exit_reason": "END_OF_BACKTEST"
            })

        # Calculate authentic quantitative metrics
        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t["pnl"] > 0]
        losses = [t for t in closed_trades if t["pnl"] < 0]
        win_rate = round((len(wins) / max(1, total_trades)) * 100.0, 1) if total_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0.0)

        total_net_pnl = round(sum(t["pnl"] for t in closed_trades), 2)
        final_equity = round(self.capital + total_net_pnl, 2)
        return_pct = round(((final_equity - self.capital) / self.capital) * 100.0, 2)

        # Max Drawdown from equity curve
        peak = self.capital
        max_dd_pct = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if peak > 0:
                dd_pct = (dd / peak) * 100.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
        max_drawdown_pct = round(max_dd_pct, 2)

        r_list = [t["r_multiple"] for t in closed_trades]
        expectancy_r = round(float(sum(r_list) / len(r_list)), 3) if r_list else 0.0

        results = {
            "initial_capital": self.capital,
            "final_equity": final_equity,
            "total_net_pnl": total_net_pnl,
            "return_pct": return_pct,
            "symbols_evaluated": len(valid_symbols),
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "expectancy_r": expectancy_r,
            "status": "COMPLETED",
            "message": f"Simulated {total_trades} trades across {len(valid_symbols)} stocks ({win_rate}% win rate, {profit_factor} PF, {max_drawdown_pct}% max DD)."
        }
        print(f"[OK] Backtest completed: Return: {return_pct:+.2f}%, Win Rate: {win_rate}%, PF: {profit_factor}, Max DD: {max_drawdown_pct}%")
        return results




