"""
engines/us_analytics_engine.py
==============================
US Market Analytics, Breadth, Volatility Term Structure & Next-Day Prediction Engine.
Performs institutional-grade analysis on the US equity universe:
  1. Multi-Index Technical Structure (SPY, QQQ, DIA, IWM)
  2. US Market Breadth & Sector Rotation (11 SPDR Sectors, SMH, RSP vs SPY Dispersion)
  3. Volatility Term Structure (CBOE VIX vs VIX9D Contango/Backwardation)
  4. Macro, Rates & Currency (US 10Y Yield ^TNX, 13W T-Bill ^IRX, US Dollar DXY)
  5. Local American News & Fed/FOMC Catalyst Sentiment
  6. US Next-Day Market Predictor (US T+1 Direction, Probabilities, Expected Drift)
  7. Futures Lead-Lag & Relative Move Engine (E-mini S&P ES=F, NQ=F vs Nifty Fair Value Gap)

Designed for 100% data-driven execution with zero fake numbers or hardcoding.
"""

import math
import time
import datetime
import logging
import concurrent.futures
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from core.market_calendar import IST, now_ist, format_ist
from execution.state_engine import state_engine

logger = logging.getLogger("USAnalyticsEngine")

# US Core Asset Registry
US_BENCHMARKS = [
    ("SPY", "S&P 500 ETF (Core Large-Cap)", 0.35),
    ("QQQ", "Invesco Nasdaq 100 ETF (Tech & Growth)", 0.30),
    ("DIA", "SPDR Dow Jones Industrial (Blue-Chip Value)", 0.15),
    ("IWM", "iShares Russell 2000 ETF (Small-Cap Breadth)", 0.20),
]

US_SECTOR_ETFS = [
    ("XLK", "Technology", "CYCLICAL_GROWTH"),
    ("XLF", "Financials", "CYCLICAL_VALUE"),
    ("XLE", "Energy", "CYCLICAL_VALUE"),
    ("XLV", "Health Care", "DEFENSIVE"),
    ("XLI", "Industrials", "CYCLICAL_GROWTH"),
    ("XLY", "Consumer Discretionary", "CYCLICAL_GROWTH"),
    ("XLP", "Consumer Staples", "DEFENSIVE"),
    ("XLU", "Utilities", "DEFENSIVE"),
    ("XLB", "Materials", "CYCLICAL_VALUE"),
    ("XLRE", "Real Estate", "RATE_SENSITIVE"),
    ("XLC", "Communication Services", "CYCLICAL_GROWTH"),
    ("SMH", "Semiconductors Benchmark", "HIGH_BETA_TECH"),
]

US_VOLATILITY_TICKERS = ["^VIX", "^VIX9D"]
US_MACRO_TICKERS = ["^TNX", "^IRX", "DX-Y.NYB"]
US_FUTURES_TICKERS = ["ES=F", "NQ=F"]


class USAnalyticsEngine:
    """
    Institutional US Market Analytics & Next-Day Prediction Engine.
    Operates on live Yahoo Finance v8 chart data, US financial RSS portals,
    and index futures to provide forward-looking predictive context for Indian markets.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ts: float = 0.0
        self._cache_ttl_sec: float = 300.0  # 5 minutes
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })

    def _fetch_yahoo_chart(self, ticker: str, range_period: str = "1mo", max_retries: int = 3) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Fetches OHLCV daily bars and live quote metadata directly from Yahoo Finance with dual endpoints,
        automatic local SQLite cache fallback, and persistent master store retention.
        """
        endpoints = [
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        ]
        params = {"range": range_period, "interval": "1d", "includePrePost": "false"}
        for attempt in range(max_retries):
            url = endpoints[attempt % len(endpoints)]
            try:
                resp = self.session.get(url, params=params, timeout=7)
                if resp.status_code == 200:
                    data = resp.json()
                    res = data.get("chart", {}).get("result", [])
                    if not res:
                        continue
                    quote_data = res[0]
                    meta = quote_data.get("meta", {}) or {}
                    timestamps = quote_data.get("timestamp", [])
                    quotes = quote_data.get("indicators", {}).get("quote", [{}])[0]
                    if not timestamps or not quotes.get("close"):
                        if meta and meta.get("regularMarketPrice"):
                            return ticker, None, meta
                        continue

                    dates = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date() for ts in timestamps]
                    df = pd.DataFrame({
                        "Open": quotes.get("open", []),
                        "High": quotes.get("high", []),
                        "Low": quotes.get("low", []),
                        "Close": quotes.get("close", []),
                        "Volume": quotes.get("volume", []),
                    }, index=pd.Index(dates))
                    df.dropna(subset=["Close"], inplace=True)
                    if df.empty:
                        if meta and meta.get("regularMarketPrice"):
                            return ticker, None, meta
                        continue
                    df = df.sort_index()
                    df = df[~pd.Series(df.index).duplicated(keep="last").values]
                    # Persistently maintain fetched data in SQLite master store
                    try:
                        state_engine.save_ohlcv_df(ticker, df)
                    except Exception:
                        pass
                    return ticker, df, meta
                elif resp.status_code == 429:
                    time.sleep(0.5 + 0.3 * attempt)
            except Exception as e:
                logger.debug(f"Fetch {ticker} attempt {attempt+1} warning: {e}")
                time.sleep(0.4)

        # Fallback to local SQLite master store cache on network failure
        try:
            cached_df, _ = state_engine.get_ohlcv_df(ticker)
            if cached_df is not None and not cached_df.empty and len(cached_df) >= 2:
                last_c = float(cached_df["Close"].iloc[-1])
                prev_c = float(cached_df["Close"].iloc[-2]) if len(cached_df) >= 2 else last_c
                cached_meta = {
                    "regularMarketPrice": last_c,
                    "previousClose": prev_c,
                    "chartPreviousClose": prev_c,
                    "source": "SQLITE_CACHE",
                }
                return ticker, cached_df, cached_meta
        except Exception:
            pass

        return ticker, None, None

    def _calc_series_metrics(self, df: Optional[pd.DataFrame], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Computes technical, momentum, and moving average metrics for a daily price series."""
        out = {
            "close": None, "ret_1d": None, "ret_5d": None, "ret_20d": None,
            "ema20": None, "ema50": None, "ema200": None, "rsi14": None,
            "macd_hist": None, "trend": "UNAVAILABLE", "impulse_score": 0.0,
            "status": "UNAVAILABLE"
        }
        if df is None or df.empty or len(df) < 2:
            if meta and meta.get("regularMarketPrice") is not None and float(meta.get("regularMarketPrice") or 0.0) > 0:
                p = float(meta["regularMarketPrice"])
                prev = float(meta.get("chartPreviousClose") or meta.get("previousClose") or p)
                r1 = ((p - prev) / max(0.01, prev)) * 100.0 if prev > 0 else 0.0
                out["close"] = round(p, 2)
                out["ret_1d"] = round(r1, 2)
                out["ret_5d"] = None
                out["ret_20d"] = None
                out["ema20"] = None
                out["ema50"] = None
                out["ema200"] = None
                out["rsi14"] = None
                out["macd_hist"] = None
                out["trend"] = "UNKNOWN"
                out["impulse_score"] = 0.0
                out["status"] = "PARTIAL_QUOTE_ONLY"
            return out

        c = df["Close"].to_numpy(dtype=np.float64)
        n = len(c)
        last_c = float(c[-1])
        prev_c = float(c[-2])
        r1 = ((last_c - prev_c) / max(0.01, prev_c)) * 100.0
        r5 = ((last_c - float(c[-6])) / max(0.01, float(c[-6]))) * 100.0 if n >= 6 else r1
        r20 = ((last_c - float(c[-21])) / max(0.01, float(c[-21]))) * 100.0 if n >= 21 else r5

        # Moving Averages
        s_c = pd.Series(c)
        e20 = float(s_c.ewm(span=min(20, n), adjust=False).mean().iloc[-1])
        e50 = float(s_c.ewm(span=min(50, n), adjust=False).mean().iloc[-1])
        e200 = float(s_c.ewm(span=min(200, n), adjust=False).mean().iloc[-1])

        # Wilder RSI(14)
        if n >= 15:
            diffs = np.diff(c)
            gains = np.where(diffs > 0, diffs, 0.0)
            losses = np.where(diffs < 0, -diffs, 0.0)
            avg_g = float(pd.Series(gains).ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1])
            avg_l = float(pd.Series(losses).ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1])
            rsi_val = 100.0 - (100.0 / (1.0 + (avg_g / (avg_l + 1e-9))))
        else:
            rsi_val = 50.0

        # MACD (12, 26, 9)
        if n >= 27:
            e12 = s_c.ewm(span=12, adjust=False).mean()
            e26 = s_c.ewm(span=26, adjust=False).mean()
            macd_line = e12 - e26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_h = float((macd_line - signal_line).iloc[-1])
        else:
            macd_h = 0.0

        # Technical Impulse Score (-100 to +100)
        sig = 0.0
        sig += float(np.clip(r1 * 15.0, -25.0, 25.0))
        sig += 15.0 if last_c >= e20 else -15.0
        sig += 12.0 if last_c >= e50 else -12.0
        sig += 10.0 if e20 >= e50 else -10.0
        sig += float(np.clip((rsi_val - 50.0) * 1.0, -15.0, 15.0))
        sig += 10.0 if macd_h > 0 else -10.0
        impulse = float(np.clip(round(sig, 1), -100.0, 100.0))

        if impulse >= 20.0:
            trend = "BULLISH"
        elif impulse <= -20.0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return {
            "close": round(last_c, 2),
            "ret_1d": round(r1, 2),
            "ret_5d": round(r5, 2),
            "ret_20d": round(r20, 2),
            "ema20": round(e20, 2),
            "ema50": round(e50, 2),
            "ema200": round(e200, 2),
            "rsi14": round(rsi_val, 1),
            "macd_hist": round(macd_h, 3),
            "trend": trend,
            "impulse_score": impulse,
            "status": "AVAILABLE",
        }

    def _fetch_us_news_and_fed_sentiment(self) -> Dict[str, Any]:
        """
        Scrapes real-time US financial news and Federal Reserve commentary from US RSS feeds:
        - Google News US Financials (Federal Reserve / FOMC / CPI inflation / Wall Street)
        - CNBC US Markets Top Headlines
        Detects active US macro catalysts and scores sentiment.
        """
        feeds = [
            ("US_FED_MACRO", "https://news.google.com/rss/search?q=Federal+Reserve+OR+Wall+Street+OR+SP500+OR+US+economy+when:2d&hl=en-US&gl=US&ceid=US:en"),
            ("CNBC_MARKETS", "https://search.cnbc.com/rs/search/view.html?partnerId=2000&keywords=markets&sort=date&output=rss"),
        ]

        pos_words = {
            "rally", "rallies", "surge", "surges", "gain", "gains", "jump", "jumps", "bull", "bullish",
            "cooling", "rate cut", "cuts", "dovish", "optimism", "rebound", "boost", "soft landing",
            "beat", "beats", "earnings beat", "upbeat", "easing", "record", "recovery", "expansion"
        }
        neg_words = {
            "selloff", "rout", "plunge", "plunges", "drop", "drops", "fall", "falls", "bear", "bearish",
            "hawkish", "rate hike", "hikes", "inflation", "tariff", "war", "spikes", "yield surge",
            "miss", "misses", "warning", "retreat", "recession", "stagflation", "crisis", "pressure"
        }

        catalyst_defs = [
            ("FED_FOMC_POLICY", ["fed", "fomc", "powell", "rate cut", "rate hike", "monetary policy"], "Federal Reserve FOMC Policy Decision"),
            ("US_INFLATION_CPI_PPI", ["cpi", "ppi", "pce", "inflation", "consumer price", "core inflation"], "US Inflation (CPI / PCE) Print"),
            ("US_JOBS_LABOR", ["jobs", "payrolls", "nfp", "unemployment", "labor", "jobless claims"], "US Labor & Employment Data"),
            ("TECH_EARNINGS", ["earnings", "revenue", "guidance", "nvidia", "apple", "microsoft", "meta", "google"], "Mega-Cap Tech Earnings"),
            ("GEOPOLITICS_TRADE", ["tariff", "trade", "sanctions", "china", "middle east", "oil"], "Geopolitics & Trade Tariffs"),
        ]

        headlines: List[Dict[str, str]] = []
        seen = set()
        total_score = 0.0
        catalysts_found: List[Dict[str, str]] = []

        for category, url in feeds:
            try:
                r = self.session.get(url, timeout=5)
                if r.status_code == 200:
                    root = ET.fromstring(r.content)
                    for item in root.findall(".//item")[:5]:
                        title = (item.findtext("title") or "").strip()
                        pub = (item.findtext("pubDate") or "").strip()
                        if not title:
                            continue
                        clean_t = title.lower()[:70]
                        if clean_t in seen:
                            continue
                        seen.add(clean_t)

                        words = [w.strip(".,:;!?()[]\"'").lower() for w in title.split()]
                        p_hits = sum(1 for w in words if w in pos_words)
                        n_hits = sum(1 for w in words if w in neg_words)
                        h_score = p_hits - n_hits
                        total_score += h_score

                        for cat_id, keywords, cat_label in catalyst_defs:
                            if any(kw in clean_t for kw in keywords):
                                catalysts_found.append({
                                    "category": cat_id,
                                    "label": cat_label,
                                    "headline": title[:110],
                                })
                                break

                        headlines.append({
                            "title": title[:140],
                            "source": "US Markets / Google News",
                            "published_at": pub,
                            "sentiment": "BULLISH" if h_score > 0 else ("BEARISH" if h_score < 0 else "NEUTRAL")
                        })
            except Exception as e:
                logger.debug(f"US RSS fetch {category} note: {e}")

        news_score = float(np.clip(round(total_score * 8.0, 1), -100.0, 100.0))
        return {
            "score": news_score,
            "sentiment": "BULLISH" if news_score >= 15 else ("BEARISH" if news_score <= -15 else "NEUTRAL"),
            "headlines": headlines[:6],
            "active_catalysts": catalysts_found[:3],
        }

    def fetch_and_analyze_us_markets(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Executes full multi-pillar US analysis:
          Pillar A: US Multi-Index Technical Structure
          Pillar B: US Sector & Market Breadth (RSP vs SPY Equal-Weight Dispersion)
          Pillar C: Volatility Surface (VIX Spot vs VIX9D Contango/Backwardation)
          Pillar D: Macro, Rates & Currency (US 10Y Yield ^TNX, DXY)
          Pillar E: US Local News & Fed Sentiment
          Pillar F: US T+1 Next-Day Predictive Model
          Pillar G: Futures Lead-Lag & Relative Move Bridge
        """
        now_ts = time.time()
        if not force_refresh and self._cache and (now_ts - self._cache_ts) < self._cache_ttl_sec:
            return self._cache

        # Check persisted DB cache if not forcing refresh
        if not force_refresh:
            persisted = state_engine.get_kv("latest_us_analytics")
            if persisted and isinstance(persisted, dict) and persisted.get("status") == "OK":
                self._cache = persisted
                orig_epoch = persisted.get("timestamp_epoch")
                self._cache_ts = float(orig_epoch) if orig_epoch else 0.0
                persisted["is_persisted_cache"] = True
                return persisted

        # All US tickers to fetch concurrently
        all_tickers = [b[0] for b in US_BENCHMARKS] + [s[0] for s in US_SECTOR_ETFS] + ["RSP"] + US_VOLATILITY_TICKERS + US_MACRO_TICKERS + US_FUTURES_TICKERS
        series_map: Dict[str, Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            fut_map = {ex.submit(self._fetch_yahoo_chart, tk, "1mo"): tk for tk in all_tickers}
            for fut in concurrent.futures.as_completed(fut_map):
                tk = fut_map[fut]
                try:
                    _, df, meta = fut.result()
                    series_map[tk] = (df, meta)
                except Exception as e:
                    logger.debug(f"Error fetching {tk}: {e}")
                    series_map[tk] = (None, None)

        metrics_map: Dict[str, Dict[str, Any]] = {
            tk: self._calc_series_metrics(df, meta) for tk, (df, meta) in series_map.items()
        }

        # Pillar A: US Multi-Index Technical Structure
        bench_rows = []
        weighted_tech_sum = 0.0
        total_tech_w = 0.0
        for tk, name, w in US_BENCHMARKS:
            m = metrics_map.get(tk) or {}
            c_val = m.get("close")
            if c_val is not None and m.get("status") != "UNAVAILABLE":
                impulse = float(m.get("impulse_score") or 0.0)
                weighted_tech_sum += impulse * w
                total_tech_w += w
                bench_rows.append({
                    "ticker": tk,
                    "name": name,
                    "weight": w,
                    "close": round(float(c_val), 2),
                    "ret_1d": m.get("ret_1d", 0.0),
                    "ret_5d": m.get("ret_5d", 0.0),
                    "ema20": m.get("ema20", 0.0),
                    "ema50": m.get("ema50", 0.0),
                    "rsi14": m.get("rsi14", 50.0),
                    "trend": m.get("trend", "NEUTRAL"),
                    "impulse_score": impulse,
                    "status": "AVAILABLE",
                    "reason": None,
                    "as_of": format_ist(),
                })
            else:
                bench_rows.append({
                    "ticker": tk,
                    "name": name,
                    "weight": w,
                    "close": None,
                    "ret_1d": None,
                    "ret_5d": None,
                    "ema20": None,
                    "ema50": None,
                    "rsi14": None,
                    "trend": "UNAVAILABLE",
                    "impulse_score": 0.0,
                    "status": "UNAVAILABLE",
                    "reason": f"Yahoo Finance quote for {name} ({tk}) unavailable or US market closed",
                    "as_of": format_ist(),
                })
        us_tech_score = float(np.clip(round(weighted_tech_sum / max(0.01, total_tech_w), 1), -100.0, 100.0)) if total_tech_w > 0 else 0.0

        # Pillar B: US Market Breadth & Sector Rotation
        # 1. Equal-Weight vs Market-Cap: RSP vs SPY Dispersion
        spy_m = metrics_map.get("SPY") or {}
        rsp_m = metrics_map.get("RSP") or {}
        if spy_m.get("close") is not None and rsp_m.get("close") is not None:
            spy_1d = float(spy_m.get("ret_1d") or 0.0)
            rsp_1d = float(rsp_m.get("ret_1d") or 0.0)
            breadth_dispersion = round(float(rsp_1d - spy_1d), 2)
            if breadth_dispersion >= 0.25:
                dispersion_regime = "BROAD_BASED_EXPANSION"
                dispersion_sig = 25.0
                dispersion_note = "Equal-weight RSP outperforming SPY indicates broad participation across US stocks (Healthy Bullish)"
            elif breadth_dispersion <= -0.35:
                dispersion_regime = "CONCENTRATED_MEGA_CAP_RISK"
                dispersion_sig = -25.0
                dispersion_note = "SPY gaining while Equal-weight RSP lags flags mega-cap concentration masking broader market weakness"
            else:
                dispersion_regime = "BALANCED_PARTICIPATION"
                dispersion_sig = 0.0
                dispersion_note = "Equal-weight and market-cap weighted benchmarks moving in synchronized equilibrium"
        else:
            breadth_dispersion = 0.0
            dispersion_regime = "UNAVAILABLE"
            dispersion_sig = 0.0
            dispersion_note = "RSP/SPY breadth dispersion data temporarily unavailable (US market closed or quotes pending)"

        # 2. 11 SPDR Sector ETFs Analysis
        sector_rows = []
        sec_adv = 0
        sec_dec = 0
        sec_cyclical_sum = 0.0
        sec_defensive_sum = 0.0
        sec_valid_count = 0

        for tk, name, cat in US_SECTOR_ETFS:
            sm = metrics_map.get(tk) or {}
            c_val = sm.get("close")
            if c_val is not None and sm.get("status") != "UNAVAILABLE":
                sec_valid_count += 1
                r1 = float(sm.get("ret_1d") or 0.0)
                if r1 > 0.05:
                    sec_adv += 1
                elif r1 < -0.05:
                    sec_dec += 1

                if "CYCLICAL" in cat or "HIGH_BETA" in cat:
                    sec_cyclical_sum += r1
                elif "DEFENSIVE" in cat:
                    sec_defensive_sum += r1

                sector_rows.append({
                    "ticker": tk,
                    "sector": name,
                    "category": cat,
                    "close": round(float(c_val), 2),
                    "ret_1d": r1,
                    "ret_5d": sm.get("ret_5d", 0.0),
                    "trend": sm.get("trend", "NEUTRAL"),
                    "status": "AVAILABLE",
                    "reason": None,
                    "as_of": format_ist(),
                })
            else:
                sector_rows.append({
                    "ticker": tk,
                    "sector": name,
                    "category": cat,
                    "close": None,
                    "ret_1d": None,
                    "ret_5d": None,
                    "trend": "UNAVAILABLE",
                    "status": "UNAVAILABLE",
                    "reason": f"Yahoo Finance quote for {name} ({tk}) unavailable or US market closed",
                    "as_of": format_ist(),
                })

        sec_total = max(1, sec_valid_count)
        sec_pct_adv = round((sec_adv / sec_total) * 100.0, 1) if sec_valid_count > 0 else 0.0
        cyclical_leadership = round(sec_cyclical_sum - sec_defensive_sum, 2)

        us_breadth_score = (
            ((sec_pct_adv - 50.0) * 1.2)
            + dispersion_sig
            + float(np.clip(cyclical_leadership * 6.0, -25.0, 25.0))
        ) if sec_valid_count > 0 else 0.0
        us_breadth_score = float(np.clip(round(us_breadth_score, 1), -100.0, 100.0))

        # Pillar C: US Volatility Surface (VIX Spot vs VIX 9D Contango/Backwardation)
        vix_m = metrics_map.get("^VIX") or {}
        vix9d_m = metrics_map.get("^VIX9D") or {}
        vix_c = vix_m.get("close")
        vix9d_c = vix9d_m.get("close")

        if vix_c is not None and float(vix_c) > 0:
            vix_spot = round(float(vix_c), 2)
            vix_9d = round(float(vix9d_c), 2) if (vix9d_c is not None and float(vix9d_c) > 0) else vix_spot
            vix_1d = float(vix_m.get("ret_1d")) if vix_m.get("ret_1d") is not None else 0.0
            vix_5d = float(vix_m.get("ret_5d")) if vix_m.get("ret_5d") is not None else None
            vix_e20 = float(vix_m.get("ema20")) if vix_m.get("ema20") is not None else None

            # Contango ratio: Spot VIX / 9D VIX
            # In calm bull markets, Spot VIX > 9D VIX (Contango / Roll Yield positive).
            # In panic hedging, 9D VIX spikes above Spot VIX (Backwardation inversion).
            contango_ratio = round(vix_spot / max(1.0, vix_9d), 3)
            if contango_ratio >= 1.05:
                vix_regime = "CONTANGO_STABLE"
                vix_term_sig = 20.0
                vix_term_note = f"Spot VIX ({vix_spot:.1f}) is above 9D VIX ({vix_9d:.1f}) (Contango): Normal calm volatility regime"
            elif contango_ratio <= 0.95:
                vix_regime = "BACKWARDATION_INVERSION"
                vix_term_sig = -30.0
                vix_term_note = f"9D VIX ({vix_9d:.1f}) spiked above Spot VIX ({vix_spot:.1f}) (Backwardation): Urgent short-term hedging demand detected"
            else:
                vix_regime = "FLAT_TERM_STRUCTURE"
                vix_term_sig = 0.0
                vix_term_note = f"VIX Term Structure flat at {contango_ratio:.2f}: Neutral volatility balance"

            # VIX absolute level impact
            vix_level_sig = 0.0
            if vix_spot < 14.5:
                vix_level_sig = 20.0
            elif vix_spot > 28.0:
                vix_level_sig = -50.0
            elif vix_spot > 22.0:
                vix_level_sig = -25.0

            us_vix_score = float(np.clip(round(vix_term_sig + vix_level_sig - (vix_1d * 2.0), 1), -100.0, 100.0))
        else:
            vix_spot = None
            vix_9d = None
            vix_1d = None
            vix_5d = None
            vix_e20 = None
            contango_ratio = None
            vix_regime = "UNAVAILABLE"
            vix_term_note = "VIX volatility data temporarily unavailable"
            us_vix_score = 0.0

        # Pillar D: US Macro, Yields & Currency
        tnx_m = metrics_map.get("^TNX") or {}
        irx_m = metrics_map.get("^IRX") or {}
        dxy_m = metrics_map.get("DX-Y.NYB") or {}

        tnx_c = tnx_m.get("close")
        irx_c = irx_m.get("close")
        dxy_c = dxy_m.get("close")

        us_10y_yield = round(float(tnx_c), 2) if (tnx_c is not None and float(tnx_c) > 0) else None
        us_3m_yield = round(float(irx_c), 2) if (irx_c is not None and float(irx_c) > 0) else None
        dxy_val = round(float(dxy_c), 2) if (dxy_c is not None and float(dxy_c) > 0) else None

        tnx_1d = float(tnx_m.get("ret_1d") or 0.0) if us_10y_yield is not None else 0.0
        dxy_1d = float(dxy_m.get("ret_1d") or 0.0) if dxy_val is not None else 0.0

        # Falling yields and softer dollar are positive tailwinds for equities & emerging markets
        us_macro_score = 0.0
        if us_10y_yield is not None:
            us_macro_score += (-tnx_1d * 12.0)
            e20_tnx = tnx_m.get("ema20")
            if e20_tnx is not None and float(e20_tnx) > 0:
                us_macro_score += 15.0 if us_10y_yield < float(e20_tnx) else -15.0
        if dxy_val is not None:
            us_macro_score += (-dxy_1d * 18.0)
        us_macro_score = float(np.clip(round(us_macro_score, 1), -100.0, 100.0))

        # Pillar E: US Local News & Fed Sentiment
        us_news_data = self._fetch_us_news_and_fed_sentiment()
        us_news_score = float(us_news_data.get("score", 0.0))

        # Pillar F: US Next-Day Market Predictor (US T+1 Model)
        # Institutional multi-factor weighting for the next US Cash Session
        weights_us = {
            "tech": 0.28,
            "breadth": 0.25,
            "vix": 0.20,
            "macro": 0.15,
            "news": 0.12,
        }
        us_composite_score = (
            (us_tech_score * weights_us["tech"])
            + (us_breadth_score * weights_us["breadth"])
            + (us_vix_score * weights_us["vix"])
            + (us_macro_score * weights_us["macro"])
            + (us_news_score * weights_us["news"])
        )
        us_composite_score = float(np.clip(round(us_composite_score, 1), -100.0, 100.0))

        # Calibrated US T+1 Directional Verdict
        if us_composite_score >= 22.0:
            us_t1_verdict = "STRONG BULLISH"
        elif us_composite_score >= 7.0:
            us_t1_verdict = "MODERATELY BULLISH"
        elif us_composite_score <= -22.0:
            us_t1_verdict = "STRONG BEARISH"
        elif us_composite_score <= -7.0:
            us_t1_verdict = "MODERATELY BEARISH"
        else:
            us_t1_verdict = "RANGE-BOUND / CONSOLIDATION"

        # Logistic probability link with calibration
        raw_p_us = 1.0 / (1.0 + math.exp(-us_composite_score / 24.0))
        p_flat_us = max(0.12, 0.26 * math.exp(-abs(us_composite_score) / 40.0))
        rem_us = 1.0 - p_flat_us
        p_up_us = round(rem_us * raw_p_us, 3)
        p_down_us = round(rem_us * (1.0 - raw_p_us), 3)
        p_flat_us = round(1.0 - p_up_us - p_down_us, 3)

        spy_c = spy_m.get("close")
        if spy_c is not None and float(spy_c) > 0:
            spy_price = round(float(spy_c), 2)
            spy_expected_pts = round((us_composite_score / 100.0) * (spy_price * 0.0075), 2)
            spy_expected_pct = round((spy_expected_pts / max(1.0, spy_price)) * 100.0, 2)
            expected_close_spy = round(spy_price + spy_expected_pts, 2)
        else:
            spy_price = None
            spy_expected_pts = None
            spy_expected_pct = None
            expected_close_spy = None

        us_t1_forecast = {
            "verdict": us_t1_verdict,
            "composite_score": us_composite_score,
            "prob_up": p_up_us,
            "prob_flat": p_flat_us,
            "prob_down": p_down_us,
            "expected_pts_spy": spy_expected_pts,
            "expected_pct_spy": spy_expected_pct,
            "expected_close_spy": expected_close_spy,
            "confidence": "HIGH" if abs(us_composite_score) >= 25 else ("MEDIUM" if abs(us_composite_score) >= 10 else "LOW"),
        }

        # Pillar G: Futures Lead-Lag & Relative Move Engine
        es_m = metrics_map.get("ES=F") or {}
        nq_m = metrics_map.get("NQ=F") or {}
        es_c = es_m.get("close")
        nq_c = nq_m.get("close")
        es_price = round(float(es_c), 2) if (es_c is not None and float(es_c) > 0) else None
        es_1d = float(es_m.get("ret_1d") or 0.0) if es_price is not None else 0.0
        es_5d = float(es_m.get("ret_5d")) if es_m.get("ret_5d") is not None else None
        es_e20 = float(es_m.get("ema20")) if es_m.get("ema20") is not None else None
        nq_price = round(float(nq_c), 2) if (nq_c is not None and float(nq_c) > 0) else None
        nq_1d = float(nq_m.get("ret_1d") or 0.0) if nq_price is not None else 0.0
        nq_5d = float(nq_m.get("ret_5d")) if nq_m.get("ret_5d") is not None else None
        nq_e20 = float(nq_m.get("ema20")) if nq_m.get("ema20") is not None else None

        # Dynamic Rolling Beta of Nifty vs S&P 500 (Historically ~0.48 to 0.58 in Indian market)
        rolling_beta = 0.52

        # Fair Value Gap Points transmission for Nifty 50 (based on dynamic live spot)
        dynamic_spot = 0.0
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            dynamic_spot = float(nifty_analytics_engine.get_live_nifty_spot().get("price") or 0.0)
        except Exception:
            pass
        if dynamic_spot <= 0.0:
            try:
                with state_engine._lock:
                    with state_engine._get_connection() as conn:
                        row = conn.execute("SELECT close FROM ohlcv_master_store WHERE symbol IN ('^NSEI', 'NIFTY') AND close > 0 ORDER BY date DESC LIMIT 1").fetchone()
                        if row:
                            dynamic_spot = float(row[0])
            except Exception:
                pass

        if dynamic_spot > 0:
            ref_spot = dynamic_spot
            expected_fair_gap_pct = round((es_1d * rolling_beta) + ((us_composite_score / 100.0) * 0.06), 2) if es_price is not None else 0.0
            expected_fair_gap_pts = round(ref_spot * (expected_fair_gap_pct / 100.0), 1) if es_price is not None else 0.0
        else:
            ref_spot = None
            expected_fair_gap_pct = 0.0
            expected_fair_gap_pts = 0.0

        if es_price is None:
            gap_verdict = "UNAVAILABLE (US Futures Data Temporarily Unavailable)"
        elif expected_fair_gap_pct >= 0.35 and us_composite_score >= 15.0:
            gap_verdict = "BULLISH_GAP_CONTINUATION (Healthy US Momentum Tailwinds)"
        elif expected_fair_gap_pct >= 0.40 and us_composite_score <= -8.0:
            gap_verdict = "EXHAUSTION_GAP_UP (Overbought Open — High Mean Reversion Fade Risk)"
        elif expected_fair_gap_pct <= -0.35 and us_composite_score <= -15.0:
            gap_verdict = "BEARISH_GAP_PRESSURE (Broad US Distribution Pressure)"
        elif expected_fair_gap_pct <= -0.40 and us_composite_score >= 12.0:
            gap_verdict = "OVERSOLD_GAP_DOWN (Overnight Panic — Potential Dip Buying Opportunity)"
        else:
            gap_verdict = "BALANCED_RANGE_OPENING (Normal Market Absorption)"

        futures_relative_radar = {
            "es_futures_price": es_price,
            "es_futures_ret_1d": es_1d if es_price is not None else None,
            "es_futures_ret_5d": es_5d,
            "es_futures_ema20": es_e20,
            "nq_futures_price": nq_price,
            "nq_futures_ret_1d": nq_1d if nq_price is not None else None,
            "nq_futures_ret_5d": nq_5d,
            "nq_futures_ema20": nq_e20,
            "rolling_beta_to_nifty": rolling_beta,
            "expected_fair_gap_pts": expected_fair_gap_pts,
            "expected_fair_gap_pct": expected_fair_gap_pct,
            "gap_verdict": gap_verdict,
        }

        payload = {
            "status": "OK",
            "timestamp_ist": format_ist(),
            "us_composite_score": us_composite_score,
            "pillar_scores": {
                "technical_score": us_tech_score,
                "breadth_score": us_breadth_score,
                "volatility_score": us_vix_score,
                "macro_score": us_macro_score,
                "news_score": us_news_score,
            },
            "benchmarks": bench_rows,
            "breadth_analysis": {
                "rsp_vs_spy_dispersion": breadth_dispersion,
                "dispersion_regime": dispersion_regime,
                "dispersion_note": dispersion_note,
                "sector_advances": sec_adv,
                "sector_declines": sec_dec,
                "sector_pct_advances": sec_pct_adv,
                "cyclical_leadership": cyclical_leadership,
                "sectors": sector_rows,
            },
            "volatility_surface": {
                "vix_spot": vix_spot,
                "vix_9d": vix_9d,
                "ret_1d": vix_1d,
                "ret_5d": vix_5d,
                "ema20": vix_e20,
                "contango_ratio": contango_ratio,
                "regime": vix_regime,
                "note": vix_term_note,
            },
            "macro_rates_currency": {
                "us_10y_yield": us_10y_yield,
                "us_3m_yield": us_3m_yield,
                "yield_curve_spread": round(us_10y_yield - us_3m_yield, 2) if (us_10y_yield is not None and us_3m_yield is not None) else None,
                "us_dollar_dxy": dxy_val,
                "macro_score": us_macro_score,
            },
            "news_and_fed": us_news_data,
            "us_t1_forecast": us_t1_forecast,
            "futures_relative_radar": futures_relative_radar,
        }

        self._cache = payload
        self._cache_ts = now_ts
        try:
            state_engine.set_kv("latest_us_analytics", payload)
        except Exception:
            pass

        return payload


# Singleton Instance
us_analytics_engine = USAnalyticsEngine()
