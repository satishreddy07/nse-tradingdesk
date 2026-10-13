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

            logger.info(f"Fetched {len(tickers)} stocks dynamically from Chartink screener.")
            return sorted(list(set(tickers)))
        except Exception as e:
            logger.warning(f"Chartink fetch encountered notice: {e}. Falling back to cached seed tickers.")
            return []


class DirectYahooFinanceFetcher:
    """
    High-Speed Multi-Threaded OHLCV Downloader with Smart Incremental Gap-Filling
    and Telegram Cloud Vault Integration:
    - Uses 24 worker threads with pooled keep-alive HTTP connections & dual Yahoo endpoints.
    - Checks local/Telegram-restored cache (`data/historical_data/{SYM}.pkl`).
    - If a stock exists in cache up to a past date (e.g. Oct 08) and reappears later (e.g. 20 days later),
      fetches ONLY the missing delta bars (`period1` -> `period2`), merges & deduplicates cleanly!
    - If brand new or >180 days gap, downloads full `range=2y`.
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
        self.storage_dir = Path(storage_dir) if storage_dir else HISTORICAL_DATA_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_storage_path(self, ticker: str):
        clean = clean_symbol(ticker)
        pkl_path = self.storage_dir / f"{clean}.pkl"
        return pkl_path

    def load_raw_cached(self, ticker: str) -> Tuple[Optional[pd.DataFrame], float]:
        """Returns (cached_df, file_age_hours) regardless of age so gap-filling can inspect last date."""
        pkl_path = self._get_storage_path(ticker)
        if pkl_path.exists():
            try:
                age_hours = (time.time() - pkl_path.stat().st_mtime) / 3600.0
                df = pd.read_pickle(pkl_path)
                if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns and len(df) >= 30:
                    return df, age_hours
            except Exception as e:
                logger.debug(f"Could not read cached pickle for {ticker}: {e}")
        return None, 9999.0

    def save_to_storage(self, ticker: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        try:
            df = df.sort_index()
            df = df[~pd.Series(df.index.date).duplicated(keep="last").values]
            if len(df) > 500:
                df = df.iloc[-500:]
            pkl_path = self._get_storage_path(ticker)
            df.to_pickle(pkl_path)
        except Exception as e:
            logger.debug(f"Failed to persist {ticker} to storage: {e}")

    @staticmethod
    def get_latest_expected_bar_date() -> datetime.date:
        """Determines the latest trading session date that should be present in OHLCV data."""
        now = now_ist()
        d = now.date()
        if NSEMarketCalendar.is_trading_day(d) and now.time() >= datetime.time(9, 15):
            return d
        # Walk back to most recent completed trading day
        d = d - datetime.timedelta(days=1)
        while not NSEMarketCalendar.is_trading_day(d):
            d = d - datetime.timedelta(days=1)
        return d

    def _request_yahoo_df(
        self,
        ticker: str,
        params: Dict[str, Any],
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        yf_ticker = ticker if (ticker.startswith("^") or "." in ticker) else f"{ticker}.NS"
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
                    df = df.sort_index()
                    df = df[~pd.Series(df.index.date).duplicated(keep="last").values]
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
        is_market_open: Optional[bool] = None
    ) -> Tuple[str, Optional[pd.DataFrame], str]:
        """
        Smart Incremental Gap-Filler for a single ticker:
        Returns (ticker, dataframe, fetch_mode) where fetch_mode in ('CACHE_HIT', 'GAP_FILLED', 'FULL_DOWNLOAD', 'FAILED').
        - If cached_df exists up to `last_bar_date`:
          - If `last_bar_date >= target_date` and (not is_market_open or file modified within 20 mins):
            returns cached_df immediately ('CACHE_HIT').
          - If `1 <= gap_days <= 300` (e.g. stock reappeared after 20 days since Oct 08, or needs today's bar):
            fetches only the compact missing delta range (`5d` / `1mo` / `3mo` / `6mo` / `1y`), concatenates with cached_df,
            deduplicates by calendar date, trims to 500 bars, and saves ('GAP_FILLED').
        - Otherwise fetches full `range='2y'` ('FULL_DOWNLOAD').
        """
        if target_date is None:
            target_date = self.get_latest_expected_bar_date()
        if is_market_open is None:
            is_market_open, _ = NSEMarketCalendar.can_execute_trades()
        cached_df, age_hours = self.load_raw_cached(ticker)

        if cached_df is not None and len(cached_df) >= 80:
            last_idx = cached_df.index[-1]
            last_bar_date = last_idx.date() if hasattr(last_idx, "date") else pd.Timestamp(last_idx).date()
            gap_days = (target_date - last_bar_date).days

            # Case 1: Already up to date!
            if gap_days <= 0 and (not is_market_open or age_hours <= 0.35):
                return ticker, cached_df, "CACHE_HIT"

            # Case 2: Cached in cloud/local up to a past date (e.g. Oct 08 -> 20 days gap) or live intraday bar update
            if 0 <= gap_days <= 300:
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
                    if cached_df.index.tz is not None and delta_df.index.tz is None:
                        delta_df.index = delta_df.index.tz_localize(IST)
                    elif cached_df.index.tz is None and delta_df.index.tz is not None:
                        delta_df.index = delta_df.index.tz_localize(None)

                    merged = pd.concat([cached_df, delta_df]).sort_index()
                    merged = merged[~pd.Series(merged.index.date).duplicated(keep="last").values]
                    if len(merged) > 500:
                        merged = merged.iloc[-500:]
                    self.save_to_storage(ticker, merged)
                    return ticker, merged, "GAP_FILLED"
                else:
                    return ticker, cached_df, "CACHE_HIT"

        # Case 3: Brand new stock or gap > 300 days -> Full 2-Year Download
        full_df = self._request_yahoo_df(ticker, {"range": "2y", "interval": "1d", "includePrePost": "false"}, max_retries=3)
        if full_df is not None and not full_df.empty:
            self.save_to_storage(ticker, full_df)
            return ticker, full_df, "FULL_DOWNLOAD"

        return ticker, cached_df, "FAILED"

    def fetch_universe_ohlcv(self, tickers: List[str], min_bars: int = 105) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
        """
        Multi-threaded universe OHLCV fetcher with Telegram Cloud Vault auto-restore & auto-backup:
        1. If local cache has < 50 files, restores OHLCV bundle from Telegram Cloud first.
        2. Processes all tickers across 24 threads using Smart Incremental Gap-Filling.
        3. If any stocks were gap-filled or newly downloaded, syncs the updated OHLCV bundle back to Telegram Cloud.
        """
        total_tickers = len(tickers)
        local_pkl_count = len(list(self.storage_dir.glob("*.pkl")))
        if local_pkl_count < 50:
            state_engine.update_scan_progress(
                stage="Step 2b/6: Restoring cached 2-Yr OHLCV Vault from Telegram Cloud...",
                percent=19,
                stocks_total=total_tickers,
                stocks_done=0,
                add_log=True
            )
            telegram_vault.restore_ohlcv_from_cloud()

        target_date = self.get_latest_expected_bar_date()
        is_mkt_open, _ = NSEMarketCalendar.can_execute_trades()

        results: Dict[str, pd.DataFrame] = {}
        stats = {
            "total_requested": total_tickers,
            "cache_hits": 0,
            "gap_filled": 0,
            "full_downloads": 0,
            "eligible": 0
        }

        state_engine.update_scan_progress(
            stage=f"Step 3/6: Multi-threaded Incremental OHLCV Sync across {total_tickers} Chartink stocks (24 threads)...",
            percent=22,
            stocks_total=total_tickers,
            stocks_done=0,
            add_log=True
        )

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_or_gapfill_ticker, t, target_date, is_mkt_open): t
                for t in tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                try:
                    sym, df, mode = future.result()
                    if mode == "CACHE_HIT":
                        stats["cache_hits"] += 1
                    elif mode == "GAP_FILLED":
                        stats["gap_filled"] += 1
                    elif mode == "FULL_DOWNLOAD":
                        stats["full_downloads"] += 1

                    if df is not None and len(df) >= min_bars:
                        last_close = float(df["Close"].iloc[-1])
                        avg_vol_20d = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else 0
                        if last_close >= 100.0 and avg_vol_20d >= 100000:
                            results[sym] = df
                            stats["eligible"] += 1
                except Exception:
                    pass

                if completed % 40 == 0 or completed == total_tickers:
                    pct = 22 + int((completed / max(1, total_tickers)) * 43)
                    state_engine.update_scan_progress(
                        stage=(
                            f"Step 3/6: OHLCV Sync ({completed}/{total_tickers} Chartink stocks) • "
                            f"Cached: {stats['cache_hits']} | Gap-Filled: {stats['gap_filled']} | New 2Y: {stats['full_downloads']} | "
                            f"Liquid: {stats['eligible']}"
                        ),
                        percent=pct,
                        stocks_total=total_tickers,
                        stocks_done=completed,
                        add_log=(completed % 160 == 0 or completed == total_tickers)
                    )

        # If any stocks were gap-filled or newly downloaded, sync updated OHLCV cache to Telegram Cloud in background
        if stats["gap_filled"] > 0 or stats["full_downloads"] > 0:
            threading.Thread(
                target=telegram_vault.backup_ohlcv_to_cloud,
                kwargs={"custom_note": f"Scan Sync ({stats['gap_filled']} gap-filled, {stats['full_downloads']} new)"},
                daemon=True
            ).start()

        logger.info(
            f"OHLCV Sync Complete: {total_tickers} Chartink stocks -> "
            f"{stats['cache_hits']} cached, {stats['gap_filled']} gap-filled, {stats['full_downloads']} full 2Y -> "
            f"{len(results)} liquid stocks eligible."
        )
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
        bench_data = {}
        for name, ticker in benchmarks.items():
            _, df = self.fetch_ticker_chart(ticker, range_period="1y")
            if df is not None and not df.empty:
                bench_data[name] = df
        return bench_data


# ==============================================================================
# FEATURE ENGINE
# ==============================================================================
class FeatureEngine:
    @staticmethod
    def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        tr_smooth = tr.rolling(period).sum()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).sum() / (tr_smooth + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).sum() / (tr_smooth + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        return dx.rolling(period).mean()

    @staticmethod
    def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def check_minervini_template(df: pd.DataFrame, ema50: pd.Series, ema150: pd.Series, ema200: pd.Series) -> int:
        close = df['Close'].iloc[-1]
        c_50 = ema50.iloc[-1]
        c_150 = ema150.iloc[-1]
        c_200 = ema200.iloc[-1]

        lookback_bars = min(len(df), 250)
        high_52w = df['High'].iloc[-lookback_bars:].max()
        low_52w = df['Low'].iloc[-lookback_bars:].min()

        score = 0
        if close > c_150 and close > c_200: score += 1
        if c_150 > c_200: score += 1
        if len(ema200) >= 22 and c_200 > ema200.iloc[-22]: score += 1
        if c_50 > c_150 and c_50 > c_200: score += 1
        if close > c_50: score += 1
        if close >= low_52w * 1.30: score += 1
        if close >= high_52w * 0.75: score += 1
        if close > df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]: score += 1
        return score

    @classmethod
    def compute_stock_metrics(
        cls,
        ticker: str,
        df: pd.DataFrame,
        benchmarks: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        n = len(df)
        if n < 105:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        sector = get_stock_sector(ticker)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema150 = close.ewm(span=150, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        ema20_slope = (ema20.iloc[-1] - ema20.iloc[-10]) / ema20.iloc[-10] * 100 if n >= 10 else 0.0
        ema50_slope = (ema50.iloc[-1] - ema50.iloc[-10]) / ema50.iloc[-10] * 100 if n >= 10 else 0.0
        adx_series = cls.calc_adx(high, low, close, 14)
        adx14 = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 20.0
        mv_passed = cls.check_minervini_template(df, ema50, ema150, ema200)

        c_now = float(close.iloc[-1])
        ret_1m = (c_now - close.iloc[-21]) / close.iloc[-21] * 100 if n >= 21 else 0.0
        ret_3m = (c_now - close.iloc[-63]) / close.iloc[-63] * 100 if n >= 63 else 0.0
        ret_6m = (c_now - close.iloc[-126]) / close.iloc[-126] * 100 if n >= 126 else 0.0

        ret_3m_ex_1m = ret_3m - ret_1m
        ret_6m_ex_3m = ret_6m - ret_3m

        nifty_df = benchmarks.get("NIFTY_50")
        nifty_ret_3m = 0.0
        if nifty_df is not None and len(nifty_df) >= 63:
            nifty_ret_3m = (nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[-63]) / nifty_df['Close'].iloc[-63] * 100
        excess_3m = ret_3m - nifty_ret_3m

        sector_bench_key = sector if sector in benchmarks else "NIFTY_500"
        sector_df = benchmarks.get(sector_bench_key, nifty_df)
        sector_ret_6m = 0.0
        if sector_df is not None and len(sector_df) >= 126:
            sector_ret_6m = (sector_df['Close'].iloc[-1] - sector_df['Close'].iloc[-126]) / sector_df['Close'].iloc[-126] * 100
        sector_excess_6m = ret_6m - sector_ret_6m

        rsi_series = cls.calc_rsi(close, 14)
        rsi14 = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr14 / c_now) * 100
        atr_distance = abs(atr_pct - 2.5)

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = ((std20 * 4) / (sma20 + 1e-9)).iloc[-1]
        bb_width_series = (std20 * 4) / (sma20 + 1e-9)
        bb_width_min_6m = bb_width_series.iloc[-min(n, 126):].min()
        bb_sq_ratio = bb_width / (bb_width_min_6m + 1e-9)

        vol_50d_avg_prev = volume.shift(1).rolling(50).mean().iloc[-1] if n >= 51 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_50d_avg_prev + 1e-9)

        daily_returns = close.pct_change() * 100
        vol_50_rolling = volume.rolling(min(n, 50)).mean()
        accum_mask = (daily_returns.iloc[-20:] > 0.7) & (volume.iloc[-20:] > vol_50_rolling.iloc[-20:])
        accum_days = int(accum_mask.sum())

        down_vols = volume.iloc[-10:][daily_returns.iloc[-10:] < 0]
        max_down_vol = down_vols.max() if not down_vols.empty else 0
        pocket_pivot = bool(daily_returns.iloc[-1] > 0 and volume.iloc[-1] > max_down_vol)
        vol_dryup = float(volume.iloc[-5:].mean() / (vol_50d_avg_prev + 1e-9))

        pivot_20d = float(high.iloc[-20:].max())
        dist_to_pivot = ((pivot_20d - c_now) / pivot_20d) * 100
        dist_to_ema20 = ((c_now - ema20.iloc[-1]) / ema20.iloc[-1]) * 100

        close_10d = close.iloc[-10:]
        base_tightness = float((close_10d.max() - close_10d.min()) / (close_10d.mean() + 1e-9) * 100)

        try:
            weekly_close = close.resample('W').last().dropna()
            w_10 = weekly_close.ewm(span=10, adjust=False).mean()
            w_30 = weekly_close.ewm(span=30, adjust=False).mean()
            w_rsi = cls.calc_rsi(weekly_close, 14)
            week_confluence = bool(
                weekly_close.iloc[-1] > w_10.iloc[-1] and
                w_10.iloc[-1] > w_30.iloc[-1] and
                w_rsi.iloc[-1] > 50
            )
        except Exception:
            week_confluence = False

        is_vcp = bool(base_tightness < 6.0 and bb_sq_ratio < 1.35)

        inst_score = (
            (accum_days / 6.0 * 35) +
            (25 if pocket_pivot else 0) +
            (min(vol_ratio, 2.5) / 2.5 * 25) +
            (15 if c_now > (low.iloc[-1] + (high.iloc[-1] - low.iloc[-1]) * 0.6) else 0)
        )
        inst_score = float(np.clip(inst_score, 0, 100))

        intraday_pct = float(daily_returns.iloc[-1])
        c_high = float(high.iloc[-1])
        c_low = float(low.iloc[-1])
        range_pos = float((c_now - c_low) / ((c_high - c_low) + 1e-9) * 100)

        vol_factor = get_intraday_volume_factor()
        projected_vol_ratio = vol_ratio / vol_factor
        intraday_score = float(np.clip(
            (50 + intraday_pct * 16.67) * 0.50 +
            range_pos * 0.30 +
            min(projected_vol_ratio, 2.0) / 2.0 * 100 * 0.20,
            0, 100
        ))

        return {
            "ticker": ticker,
            "sector": sector,
            "close": round(c_now, 2),
            "high": round(c_high, 2),
            "low": round(c_low, 2),
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
            "accum_days": accum_days,
            "pocket_pivot": pocket_pivot,
            "vol_dryup": vol_dryup,
            "pivot_20d": pivot_20d,
            "dist_to_pivot": dist_to_pivot,
            "dist_to_ema20": dist_to_ema20,
            "base_tightness": base_tightness,
            "week_confluence": week_confluence,
            "is_vcp": is_vcp,
            "inst_score": inst_score,
            "intraday_pct": intraday_pct,
            "intraday_score": intraday_score,
            "above_50dma": bool(c_now > ema50.iloc[-1])
        }


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

        def get_rsi_mult(rsi):
            if 50.0 <= rsi <= 70.0: return 1.05
            elif (40.0 <= rsi < 50.0) or (70.0 < rsi <= 80.0): return 1.00
            elif 80.0 < rsi <= 85.0: return 0.85
            elif rsi > 85.0: return 0.60
            elif rsi < 30.0: return 0.70
            else: return 0.90

        df['p_momentum'] = p_mom_raw * df['rsi14'].apply(get_rsi_mult)

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
        p_dist_pivot = cls.pct_rank(df['dist_to_pivot'], ascending=False)
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
        df['ml_prob'] = cls.predict_win_probability(df)
        df['score'] = (df['score_raw'] * 0.85 + (df['ml_prob'] * 100.0) * 0.15).clip(0.0, 100.0)
        return df


# ==============================================================================
# EXECUTION & POSITION SIZING
# ==============================================================================
class ExecutionEngine:
    @staticmethod
    def classify(row: pd.Series) -> str:
        score = row['score']
        dist_pivot = row['dist_to_pivot']
        dist_ema20 = row['dist_to_ema20']
        vol_ratio = row.get('vol_ratio', 1.0)

        if score < 40.0:
            return "AVOID"
        elif score >= 80.0 and dist_pivot <= 2.0:
            if vol_ratio >= 1.15:
                return "BUY_NOW"
            else:
                return "PRE_BREAKOUT"
        elif score >= 70.0 and dist_pivot <= 5.0:
            return "PRE_BREAKOUT"
        elif score >= 60.0 and (dist_pivot > 5.0 or dist_ema20 > 5.0 or vol_ratio < 1.15):
            return "WAIT_PULLBACK"
        else:
            return "AVOID"

    @staticmethod
    def calculate_position(
        row: pd.Series,
        capital: float = P1_BASE_CAPITAL,
        base_risk_pct: float = P1_BASE_RISK_PCT
    ) -> Dict[str, Any]:
        entry = row['close']
        atr14 = row['atr14']
        low_20d = row['low']
        score = row['score']
        conf = row['confidence']

        if score >= 90.0 and conf >= 5:
            risk_pct = 1.00
            tier = "A+"
        elif score >= 80.0 and conf >= 4:
            risk_pct = 0.75
            tier = "A"
        else:
            risk_pct = 0.60
            tier = "B+"

        structure_sl = min(low_20d, entry - 1.5 * atr14)
        hard_sl = entry * 0.92
        sl = round(max(structure_sl, hard_sl), 2)

        risk_per_share = entry - sl
        if risk_per_share <= 0:
            sl = round(entry * 0.95, 2)
            risk_per_share = entry - sl

        r = risk_per_share
        t1 = round(entry + 1.5 * r, 2)
        t2 = round(entry + 3.0 * r, 2)
        rr = round(3.0, 2)

        risk_amt = capital * (risk_pct / 100.0)
        shares = int(risk_amt / risk_per_share)

        max_pos_val = capital * 0.20
        min_pos_val = capital * 0.03

        if (shares * entry) > max_pos_val:
            shares = int(max_pos_val / entry)

        if (shares * entry) < min_pos_val or shares < 5:
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
            "tier": tier
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
        sym = pos["symbol"]
        formatted_sym = format_nse_ticker(sym)
        df = ohlcv_dict.get(formatted_sym) or ohlcv_dict.get(sym)
        if df is None or df.empty:
            continue

        latest_bar = df.iloc[-1]
        c = float(latest_bar["Close"])
        h = float(latest_bar["High"])
        l = float(latest_bar["Low"])

        entry_px = float(pos["entry_price"])
        sl = float(pos["stop_loss"])
        orig_sl = float(pos.get("original_stop_loss") or sl)
        t1 = float(pos["target"])
        t2 = float(pos.get("target_2") or (entry_px + 3.0 * (entry_px - orig_sl)))
        qty = int(pos["quantity"])
        t1_hit = bool(pos.get("t1_hit", 0))

        # Always update MTM price so dashboard reflects latest price
        if not can_trade:
            state_engine.update_position_price(
                sym,
                current_price=c,
                highest_price=max(float(pos.get("highest_price") or entry_px), h)
            )
            continue

        # STRICT TRADING GATE OPEN (09:15 - 15:30 IST): Evaluate live exits
        # Check Stop Loss
        if l <= sl or c <= sl:
            closed = state_engine.close_position(
                sym, exit_price=sl,
                exit_reason="STOP_LOSS" if not t1_hit else "TRAILING_BE_STOP"
            )
            if closed:
                closed_events.append(closed)
            continue

        # Check T1 Target
        if not t1_hit and h >= t1:
            booked_qty = qty // 2
            pnl_t1 = round((t1 - entry_px) * booked_qty, 2)
            rem_qty = qty - booked_qty
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    conn.execute("""
                    UPDATE active_positions
                    SET quantity = ?, stop_loss = ?, t1_hit = 1, t1_pnl = ?, sl_reason = 'BREAKEVEN_TRAILING', updated_at = ?
                    WHERE symbol = ?
                    """, (rem_qty, entry_px, pnl_t1, format_ist(), sym))
            logger.info(f"🎯 [P1 T1 Hit] {sym} booked 50% ({booked_qty} shs) @ ₹{t1:.2f}. SL moved to BE ₹{entry_px:.2f}.")
            telegram_vault.send_alert_and_sync(
                title=f"🎯 P1 TARGET 1 HIT (+1.5R): {sym}",
                message=f"Booked 50% ({booked_qty} shs) @ ₹{t1:.2f} (PnL: +₹{pnl_t1:.2f}). Stop Loss moved to Breakeven ₹{entry_px:.2f}.",
                sync_db=True
            )

        # Check T2 Target
        if t1_hit and h >= t2:
            closed = state_engine.close_position(sym, exit_price=t2, exit_reason="T2_FULL_TARGET")
            if closed:
                closed_events.append(closed)
            continue

        # Trailing SL update after T1
        if t1_hit:
            highest_seen = max(float(pos.get("highest_price") or entry_px), h)
            tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else entry_px * 0.02
            trail_sl = round(max(entry_px, highest_seen - 1.5 * atr14), 2)
            if trail_sl > sl:
                state_engine.update_position_price(sym, current_price=c, stop_loss=trail_sl, highest_price=highest_seen)
            else:
                state_engine.update_position_price(sym, current_price=c, highest_price=highest_seen)
        else:
            state_engine.update_position_price(sym, current_price=c, highest_price=max(float(pos.get("highest_price") or entry_px), h))

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
        auto_execute: bool = False,
        mode_override: Optional[str] = None
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

        state_engine.start_scan_progress(
            scan_type=f"INSTITUTIONAL_6PILLAR ({mode.upper()})",
            initial_stage=f"Step 1/6: Connecting live to Chartink Screener ({self.chartink.url})..."
        )
        logger.info(f"=== Institutional 6-Pillar Engine Run ({mode.upper()}) at {format_ist()} ===")

        # Step 1: Ingest Universe Live from Chartink
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
            tickers = self.chartink.fetch_tickers()
            if not tickers:
                cached_files = list(self.yahoo.storage_dir.glob("*.pkl"))
                seen_syms = set()
                for f in cached_files:
                    stem = f.stem.upper()
                    if stem not in seen_syms and not stem.startswith("^"):
                        seen_syms.add(stem)
                        tickers.append(f"{stem}.NS")
            _institutional_engine_state["chartink_fetched_count"] = len(tickers)
            state_engine.update_scan_progress(
                stage=f"Step 1/6: Connected to Chartink! Fetched {len(tickers)} live NSE stocks.",
                percent=12,
                stocks_total=len(tickers),
                stocks_done=0
            )

        # Step 2: Benchmarks
        state_engine.update_scan_progress(
            stage=f"Step 2/6: Fetching NSE Market & Sector Benchmarks (^NSEI, ^CRSLDX, Bank, IT, Auto, Pharma) for {len(tickers)} Chartink stocks...",
            percent=18,
            stocks_total=len(tickers),
            stocks_done=0
        )
        benchmarks = self.yahoo.fetch_benchmarks()

        # Step 3: Multi-Threaded Incremental Gap-Fill OHLCV Fetch
        ohlcv_dict, dl_stats = self.yahoo.fetch_universe_ohlcv(tickers, min_bars=105)
        _institutional_engine_state["cache_hits_count"] = dl_stats.get("cache_hits", 0)
        _institutional_engine_state["gap_filled_count"] = dl_stats.get("gap_filled", 0)
        _institutional_engine_state["full_downloads_count"] = dl_stats.get("full_downloads", 0)

        if not ohlcv_dict:
            _institutional_engine_state["status"] = "FAILED"
            state_engine.finish_scan_progress("Scan aborted: No valid OHLCV data retrieved.")
            return pd.DataFrame()

        # Step 4: Reconcile active positions
        state_engine.update_scan_progress(
            stage=f"Step 4/6: Reconciling active positions & computing 30 raw technical features across {len(ohlcv_dict)} liquid stocks (from {len(tickers)} Chartink stocks)...",
            percent=70,
            stocks_total=len(tickers),
            stocks_done=len(tickers)
        )
        reconciled = reconcile_open_institutional_positions(ohlcv_dict)
        _institutional_engine_state["reconciled_trades"] = len(reconciled)

        # Step 5: Feature Extraction
        raw_records = []
        for sym, df in ohlcv_dict.items():
            metrics = FeatureEngine.compute_stock_metrics(sym, df, benchmarks=benchmarks)
            if metrics is not None:
                raw_records.append(metrics)

        if not raw_records:
            _institutional_engine_state["status"] = "NO_CANDIDATES"
            state_engine.finish_scan_progress("Scan finished: 0 stocks passed raw feature extraction.")
            return pd.DataFrame()

        raw_df = pd.DataFrame(raw_records)

        # Step 6: 6-Pillar Percentile Scoring
        state_engine.update_scan_progress(
            stage=f"Step 5/6: Constructing 6-Pillars & Ranking Universe Percentiles ({len(raw_df)} liquid stocks out of {len(tickers)} Chartink stocks)...",
            percent=82
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
                "rs_spread": round(row.get('sector_excess_6m', 0.0), 2)
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

        _institutional_engine_state["universe_size"] = len(signals_df)
        _institutional_engine_state["buy_now_count"] = b_now
        _institutional_engine_state["pre_breakout_count"] = p_brk
        _institutional_engine_state["wait_pullback_count"] = w_pb
        _institutional_engine_state["avoid_count"] = avoid
        _institutional_engine_state["signals"] = records_list[:100]
        _institutional_engine_state["status"] = "COMPLETED"

        # Auto-execution if enabled AND Strict Market-Hours Trading Gate is OPEN (09:15 - 15:30 IST)
        orders_placed = 0
        gate_notice = ""
        if auto_execute:
            if not can_trade:
                gate_notice = " [🔒 STRICT TRADING GATE: Market Closed — Signals updated, 0 trades executed]"
                logger.warning(f"🔒 [P1 Auto-Execute Blocked] {gate_msg}")
            else:
                active_syms = [p["symbol"] for p in state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)]
                buy_now_candidates = signals_df[(signals_df['bucket'] == 'BUY_NOW') & (~signals_df['ticker'].isin(active_syms))]

                for _, r in buy_now_candidates.iterrows():
                    if r['shares'] > 0 and orders_placed < 3:
                        opened = state_engine.open_position(
                            symbol=r['ticker'],
                            entry_price=r['entry'],
                            quantity=r['shares'],
                            stop_loss=r['sl'],
                            target=r['t1'],
                            target_2=r['t2'],
                            triggered_from=f"INSTITUTIONAL_{mode.upper()}",
                            notes=f"Tier {r['tier']} | Score: {r['score']} | ML P(W): {r['ml_prob']} | T1: ₹{r['t1']} | T2: ₹{r['t2']}",
                            sl_reason="INITIAL_SL",
                            sector=r['sector'],
                            portfolio_type=PORTFOLIO_TYPE,
                            entry_reason=f"6-Pillar BUY_NOW, Score={r['score']}, ML={r['ml_prob']}, RVOL confirmation",
                            entry_regime="BULLISH_CONFIRMED",
                            entry_metrics=r.to_dict()
                        )
                        if opened:
                            orders_placed += 1

        _institutional_engine_state["orders_executed"] = orders_placed

        elapsed = time.time() - start_time
        summary_msg = (
            f"Chartink Live Scan Complete in {elapsed:.1f}s: {len(tickers)} Chartink stocks fetched "
            f"(Cached: {dl_stats['cache_hits']}, Gap-Filled: {dl_stats['gap_filled']}, New: {dl_stats['full_downloads']}) -> "
            f"{len(signals_df)} liquid stocks analyzed -> BUY NOW: {b_now} | PRE BREAKOUT: {p_brk} | WAIT PULLBACK: {w_pb}{gate_notice}"
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
        state_engine.finish_scan_progress(summary_msg)

        # Backup latest signals & DB state to Telegram Cloud Vault in background
        threading.Thread(
            target=telegram_vault.backup_db_to_cloud,
            kwargs={"custom_note": f"P1 Scan Complete ({b_now} BUY_NOW)"},
            daemon=True
        ).start()

        logger.info(summary_msg)
        return signals_df


def get_institutional_status() -> Dict[str, Any]:
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    state_copy = dict(_institutional_engine_state)
    state_copy["trading_gate_open"] = can_trade
    state_copy["trading_gate_message"] = gate_msg
    state_copy["stats"] = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE)
    state_copy["active_count"] = len(state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE))
    return state_copy

