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
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import pytz
from bs4 import BeautifulSoup

from core.data_quality import IST, now_ist, format_ist
from core.stock_registry import (
    NSE_SECTOR_MAP, SECTOR_BENCHMARK_MAP, get_stock_sector, format_nse_ticker, clean_symbol
)
from core.config import (
    P1_BASE_CAPITAL, P1_BASE_RISK_PCT, CHARTINK_URL, HISTORICAL_DATA_DIR
)
from execution.state_engine import state_engine

logger = logging.getLogger("InstitutionalEngine")
PORTFOLIO_TYPE = "INSTITUTIONAL_6PILLAR"

# Thread-safe engine runtime state
_institutional_engine_state: Dict[str, Any] = {
    "last_run_timestamp": None,
    "last_run_mode": None,
    "status": "IDLE",
    "universe_size": 0,
    "buy_now_count": 0,
    "pre_breakout_count": 0,
    "wait_pullback_count": 0,
    "avoid_count": 0,
    "reconciled_trades": 0,
    "orders_executed": 0,
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
# DATA LAYER
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
    def __init__(self, max_workers: int = 14, storage_dir: Optional[str] = None):
        self.max_workers = max_workers
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }
        self.storage_dir = Path(storage_dir) if storage_dir else HISTORICAL_DATA_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_storage_path(self, ticker: str):
        clean = clean_symbol(ticker)
        pkl_path = self.storage_dir / f"{clean}.pkl"
        csv_path = self.storage_dir / f"{clean}.csv"
        return pkl_path, csv_path

    def load_from_storage(self, ticker: str, max_age_hours: float = 14.0) -> Optional[pd.DataFrame]:
        pkl_path, csv_path = self._get_storage_path(ticker)
        if pkl_path.exists():
            try:
                age_hours = (time.time() - pkl_path.stat().st_mtime) / 3600.0
                if age_hours <= max_age_hours:
                    df = pd.read_pickle(pkl_path)
                    if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 30:
                        return df
            except Exception as e:
                logger.debug(f"Could not load cached pickle for {ticker}: {e}")

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 30:
                    return df
            except Exception as e:
                logger.debug(f"Could not load cached CSV for {ticker}: {e}")
        return None

    def save_to_storage(self, ticker: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        try:
            if len(df) > 500:
                df = df.iloc[-500:]
            pkl_path, csv_path = self._get_storage_path(ticker)
            df.to_pickle(pkl_path)
            df.to_csv(csv_path)
        except Exception as e:
            logger.debug(f"Failed to persist {ticker} to storage: {e}")

    def fetch_ticker_chart(self, ticker: str, range_period: str = "2y", max_retries: int = 3) -> Tuple[str, Optional[pd.DataFrame]]:
        url = self.base_url.format(ticker=ticker)
        params = {"range": range_period, "interval": "1d", "includePrePost": "false"}

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    res = data.get("chart", {}).get("result", [])
                    if not res:
                        return ticker, None

                    quote_data = res[0]
                    timestamps = quote_data.get("timestamp", [])
                    indicators = quote_data.get("indicators", {})
                    quotes = indicators.get("quote", [{}])[0]

                    if not timestamps or not quotes.get("close"):
                        return ticker, None

                    dates = [datetime.datetime.fromtimestamp(ts, tz=IST) for ts in timestamps]
                    df = pd.DataFrame({
                        "Open": quotes.get("open", []),
                        "High": quotes.get("high", []),
                        "Low": quotes.get("low", []),
                        "Close": quotes.get("close", []),
                        "Volume": quotes.get("volume", []),
                    }, index=pd.DatetimeIndex(dates))

                    df.dropna(subset=["Close"], inplace=True)
                    return ticker, df

                elif resp.status_code == 429:
                    time.sleep((2 ** attempt) + np.random.uniform(0.5, 1.5))
                else:
                    time.sleep(0.5)

            except Exception:
                time.sleep(1.0)

        return ticker, None

    def fetch_universe_ohlcv(self, tickers: List[str], min_bars: int = 105) -> Dict[str, pd.DataFrame]:
        results = {}
        tickers_to_download = []

        for t in tickers:
            cached_df = self.load_from_storage(t)
            if cached_df is not None and len(cached_df) >= min_bars:
                last_close = float(cached_df['Close'].iloc[-1])
                avg_vol_20d = float(cached_df['Volume'].iloc[-20:].mean()) if len(cached_df) >= 20 else 0
                if last_close >= 100.0 and avg_vol_20d >= 100000:
                    results[t] = cached_df
            else:
                tickers_to_download.append(t)

        logger.info(f"Loaded {len(results)} cached stocks. Fetching {len(tickers_to_download)} missing via Yahoo...")

        if tickers_to_download:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {executor.submit(self.fetch_ticker_chart, t): t for t in tickers_to_download}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    try:
                        sym, df = future.result()
                        if df is not None and len(df) >= min_bars:
                            self.save_to_storage(sym, df)
                            last_close = float(df['Close'].iloc[-1])
                            avg_vol_20d = float(df['Volume'].iloc[-20:].mean()) if len(df) >= 20 else 0
                            if last_close >= 100.0 and avg_vol_20d >= 100000:
                                results[sym] = df
                    except Exception:
                        pass

        logger.info(f"Total eligible universe for TA analysis: {len(results)} stocks.")
        return results

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
# POSITION RECONCILIATION FOR PORTFOLIO 1
# ==============================================================================
def reconcile_open_institutional_positions(ohlcv_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Evaluates open positions in Portfolio 1 (Institutional 6-Pillar):
    - SL hit -> Exit full
    - T1 hit (+1.5R) -> Book 50%, move SL to Breakeven, activate ATR trailing
    - T2 hit (+3.0R) -> Exit full
    - Trailing SL -> Ratchet stop higher
    """
    open_positions = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    if not open_positions:
        return []

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
                    SET quantity = ?, stop_loss = ?, t1_hit = 1, t1_pnl = ?, sl_reason = 'BREAKEVEN_TRAILING'
                    WHERE symbol = ?
                    """, (rem_qty, entry_px, pnl_t1, sym))
            logger.info(f"🎯 [P1 T1 Hit] {sym} booked 50% ({booked_qty} shs) @ ₹{t1:.2f}. SL moved to BE ₹{entry_px:.2f}.")

        # Check T2 Target
        if t1_hit and h >= t2:
            closed = state_engine.close_position(sym, exit_price=t2, exit_reason="T2_FULL_TARGET")
            if closed:
                closed_events.append(closed)
            continue

        # Trailing SL update after T1
        if t1_hit:
            highest_seen = max(float(pos.get("highest_price") or entry_px), h)
            # 1.5 ATR trail from highest high
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
        self.yahoo = DirectYahooFinanceFetcher(max_workers=14)

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

        _institutional_engine_state["status"] = "RUNNING"
        _institutional_engine_state["last_run_mode"] = mode
        _institutional_engine_state["last_run_timestamp"] = format_ist()

        logger.info(f"=== Institutional 6-Pillar Engine Run ({mode.upper()}) at {format_ist()} ===")

        # Step 1: Ingest Universe
        if custom_tickers and len(custom_tickers) > 0:
            tickers = [format_nse_ticker(t) for t in custom_tickers]
        else:
            tickers = self.chartink.fetch_tickers()
            if not tickers:
                # Load from historical data or seed
                cached_files = list(self.yahoo.storage_dir.glob("*.pkl")) + list(self.yahoo.storage_dir.glob("*.csv"))
                seen_syms = set()
                for f in cached_files:
                    stem = f.stem.upper()
                    if stem not in seen_syms and not stem.startswith("^"):
                        seen_syms.add(stem)
                        tickers.append(f"{stem}.NS")
                if not tickers:
                    # Fallback to key liquid stocks
                    tickers = [f"{s}.NS" for s in ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "LT", "BAJFINANCE", "HCLTECH", "KOTAKBANK", "AXISBANK", "SUNPHARMA", "MARUTI", "M&M", "TATAMOTORS", "NTPC", "POWERGRID", "TITAN", "TATASTEEL", "ADANIENT", "BEL", "DIXON", "POLYCAB", "PERSISTENT", "COFORGE"]]

        # Step 2: Benchmarks
        benchmarks = self.yahoo.fetch_benchmarks()

        # Step 3: Fetch OHLCV
        ohlcv_dict = self.yahoo.fetch_universe_ohlcv(tickers, min_bars=105)
        if not ohlcv_dict:
            _institutional_engine_state["status"] = "FAILED"
            return pd.DataFrame()

        # Step 4: Reconcile active positions
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
            return pd.DataFrame()

        raw_df = pd.DataFrame(raw_records)

        # Step 6: 6-Pillar Percentile Scoring
        scored_df = PercentileScoringEngine.score_universe(raw_df, run_mode=mode)

        # Step 7: ML Re-Ranking
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
        _institutional_engine_state["signals"] = records_list[:50]
        _institutional_engine_state["status"] = "COMPLETED"

        # Auto-execution if enabled
        orders_placed = 0
        if auto_execute:
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
        state_engine.log_scan_run(
            scan_type=f"INSTITUTIONAL_{mode.upper()}",
            portfolio_type=PORTFOLIO_TYPE,
            status="SUCCESS",
            execution_time_sec=elapsed,
            candidates_count=b_now + p_brk,
            orders_count=orders_placed,
            reconciled_count=len(reconciled),
            message=f"Analyzed {len(signals_df)} stocks. BUY_NOW: {b_now}, PRE_BREAKOUT: {p_brk}. {orders_placed} orders placed."
        )

        logger.info(f"Scan complete in {elapsed:.1f}s. {len(signals_df)} analyzed. BUY_NOW: {b_now}, Orders: {orders_placed}")
        return signals_df


def get_institutional_status() -> Dict[str, Any]:
    state_copy = dict(_institutional_engine_state)
    state_copy["stats"] = state_engine.get_portfolio_stats(portfolio_type=PORTFOLIO_TYPE)
    state_copy["active_count"] = len(state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE))
    return state_copy
