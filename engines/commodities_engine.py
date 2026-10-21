"""
========================================================================================
Kotak Neo MCX Commodities Analytics & Real-Time Stream Engine
========================================================================================
Institutional real-time market data streaming and technical analytics engine for
Multi Commodity Exchange (MCX) contracts powered exclusively by Kotak Neo API v2
(`mcx_fo` exchange segment).

Key Capabilities:
1. Pure Kotak Neo Feed: Streams authentic real-time ticks for Crude Oil, Natural Gas,
   Gold, Silver, Copper, and Zinc. Zero delayed or synthetic scrapers.
2. Dynamic Rolling Expiry Resolver: Automatically maps active near-month futures contracts
   with graceful fallback to pre-validated active exchange tokens.
3. Institutional Analytics: Real-time Floor Pivots (Pivot, R1, S1, R2, S2), Day Range
   Progress, Volume & Open Interest dynamics (Long Buildup, Short Covering, etc.).
4. Market Session Intelligence: Precise MCX trading calendar (09:00 AM to 11:30 PM IST
   Monday-Friday) with session-phase awareness.
5. In-Memory Microsecond Cache & SQLite KV Persistence: Guarantees zero latency on stream
   queries and preserves official settlement quotes across market close/reboots.
========================================================================================
"""

import datetime
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from core.data_quality import now_ist, format_ist
from core import config
from execution.state_engine import state_engine

logger = logging.getLogger("CommoditiesEngine")

# Pre-seeded active MCX contracts for immediate fallback resolution
# Format: Base symbol -> Metadata + Verified Kotak Neo Token & Contract Symbol
COMMODITY_SPECS: Dict[str, Dict[str, Any]] = {
    "CRUDEOIL": {
        "display_name": "Crude Oil",
        "sector": "Energy",
        "unit": "₹ / bbl",
        "lot_size": 100,
        "description": "Light Sweet Crude Oil (WTI / Brent Benchmark)",
        "default_token": "573422",
        "default_symbol": "CRUDEOIL19NOV26FUT",
        "alt_token": "573423",  # Mini
        "alt_symbol": "CRUDEOILM19NOV26FUT",
        "tick_size": 1.0,
    },
    "NATURALGAS": {
        "display_name": "Natural Gas",
        "sector": "Energy",
        "unit": "₹ / mmBtu",
        "lot_size": 1250,
        "description": "Henry Hub Natural Gas Futures",
        "default_token": "570750",
        "default_symbol": "NATURALGAS27OCT26FUT",
        "alt_token": "574319",
        "alt_symbol": "NATURALGAS24NOV26FUT",
        "tick_size": 0.1,
    },
    "GOLD": {
        "display_name": "Gold Mini",
        "sector": "Bullion",
        "unit": "₹ / 10g",
        "lot_size": 100,
        "description": "Gold 995 Purity MCX Benchmark (Mini)",
        "default_token": "571445",
        "default_symbol": "GOLDM05NOV26FUT",
        "alt_token": "571305",
        "alt_symbol": "GOLDGUINEA30OCT26FUT",
        "tick_size": 1.0,
    },
    "SILVER": {
        "display_name": "Silver Mini",
        "sector": "Bullion",
        "unit": "₹ / kg",
        "lot_size": 5,
        "description": "Silver 999 Purity MCX Benchmark (Mini)",
        "default_token": "483080",
        "default_symbol": "SILVERM30NOV26FUT",
        "alt_token": "574825",
        "alt_symbol": "SILVER10030OCT26FUT",
        "tick_size": 1.0,
    },
    "COPPER": {
        "display_name": "Copper",
        "sector": "Base Metals",
        "unit": "₹ / kg",
        "lot_size": 2500,
        "description": "Grade A Copper Cathodes",
        "default_token": "574829",
        "default_symbol": "COPPER30OCT26FUT",
        "alt_token": "578634",
        "alt_symbol": "COPPER30NOV26FUT",
        "tick_size": 0.05,
    },
    "ZINC": {
        "display_name": "Zinc",
        "sector": "Base Metals",
        "unit": "₹ / kg",
        "lot_size": 5000,
        "description": "Special High Grade Zinc",
        "default_token": "574834",
        "default_symbol": "ZINC30OCT26FUT",
        "alt_token": "574835",
        "alt_symbol": "ZINCMINI30OCT26FUT",
        "tick_size": 0.05,
    },
}


class CommoditiesEngine:
    """
    Singleton Manager for MCX Commodities powered by Kotak Neo API.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._active_contracts: Dict[str, Dict[str, Any]] = {}
        self._last_contract_discovery_ts: float = 0.0
        self._contract_discovery_ttl: float = 21600.0  # 6 hours

        # In-memory quotes cache: { "CRUDEOIL": { ltp, change, ... } }
        self._quotes_cache: Dict[str, Dict[str, Any]] = {}
        self._last_quotes_fetch_ts: float = 0.0
        self._quotes_cache_ttl: float = 2.0  # 2.0s cache for fast streaming responses

        # Initial seeding of active contracts
        self._seed_active_contracts()
        self._load_persisted_quotes()

    def _seed_active_contracts(self):
        """Seeds initial contract tokens from verified static dictionary."""
        with self._lock:
            for base_sym, spec in COMMODITY_SPECS.items():
                self._active_contracts[base_sym] = {
                    "base_symbol": base_sym,
                    "display_name": spec["display_name"],
                    "sector": spec["sector"],
                    "unit": spec["unit"],
                    "lot_size": spec["lot_size"],
                    "token": spec["default_token"],
                    "trading_symbol": spec["default_symbol"],
                    "description": spec["description"],
                    "tick_size": spec["tick_size"],
                    "expiry_date": None,
                }

    def _load_persisted_quotes(self):
        """Loads last persisted settlement quotes from SQLite KV on reboot."""
        try:
            persisted = state_engine.get_kv("latest_mcx_commodities_quotes")
            if persisted and isinstance(persisted, dict) and "quotes" in persisted:
                with self._lock:
                    self._quotes_cache = persisted.get("quotes", {})
                    logger.info(f"Loaded {len(self._quotes_cache)} persisted MCX commodity quotes from DB.")
        except Exception as e:
            logger.debug(f"Persisted commodities quotes load notice: {e}")

    # --------------------------------------------------------------------------
    # MARKET HOURS INTELLIGENCE
    # --------------------------------------------------------------------------
    def is_mcx_open(self) -> Tuple[bool, str]:
        """
        Returns whether the MCX commodity market is currently active.
        MCX Trading Sessions (Mon-Fri):
          - Morning Session: 09:00 to 17:00 IST
          - Evening Session: 17:00 to 23:30 IST (Non-agricultural/Energy/Metals)
        Closed on Saturdays and Sundays.
        """
        now = now_ist()
        # Weekday: Monday is 0, Sunday is 6
        if now.weekday() >= 5:
            return False, "MCX CLOSED (Weekend • Reopens Monday 09:00 AM IST)"

        mins = now.hour * 60 + now.minute
        # 09:00 AM = 540 mins, 23:30 PM = 1410 mins
        if mins < 540:
            return False, "MCX PRE-MARKET (Opens 09:00 AM IST)"
        elif mins > 1410:
            return False, "MCX CLOSED (Session Ended 23:30 IST • Reopens 09:00 AM IST)"
        else:
            return True, "MCX ACTIVE (09:00 - 23:30 IST)"

    # --------------------------------------------------------------------------
    # DYNAMIC CONTRACT DISCOVERY
    # --------------------------------------------------------------------------
    def discover_active_contracts(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Queries Kotak Neo API (`search_scrip`) to discover active near-month FUTCOM contracts.
        Ensures smooth rolling when near-month contracts expire.
        """
        now_ts = time.time()
        if not force and self._active_contracts and (now_ts - self._last_contract_discovery_ts) < self._contract_discovery_ttl:
            return self._active_contracts

        try:
            from engines.kotak_neo_engine import kotak_neo_engine
            if not kotak_neo_engine._is_authenticated or not kotak_neo_engine._client:
                # Kotak Neo not authenticated, keep pre-seeded active contracts
                return self._active_contracts

            client = kotak_neo_engine._client
            today_date = now_ist().date()

            updated_contracts = {}
            for base_sym, spec in COMMODITY_SPECS.items():
                try:
                    res = client.search_scrip(exchange_segment="mcx_fo", symbol=base_sym)
                    if isinstance(res, list) and res:
                        candidates = []
                        for item in res:
                            if item.get("pInstType") == "FUTCOM":
                                exp_str = item.get("pExpiryDate")
                                try:
                                    exp_dt = datetime.datetime.strptime(exp_str, "%d%b%Y").date()
                                    if exp_dt >= today_date:
                                        # Prefer mini for Gold/Silver if specified in spec, or standard
                                        trd = item.get("pTrdSymbol", "").upper()
                                        candidates.append((exp_dt, trd, item))
                                except Exception:
                                    pass

                        if candidates:
                            # Sort by nearest unexpired date
                            candidates.sort(key=lambda x: x[0])
                            # If spec specifies Mini preference, filter for Mini if available at nearest expiry
                            best_item = None
                            if "MINI" in spec["display_name"].upper():
                                for c in candidates:
                                    if "M" in c[1] or "MINI" in c[2].get("pDesc", "").upper():
                                        best_item = c
                                        break
                            if not best_item:
                                best_item = candidates[0]

                            exp_date_val, trd_sym_val, raw_item = best_item
                            tok_val = str(raw_item.get("pSymbol", "")).strip()

                            if tok_val:
                                updated_contracts[base_sym] = {
                                    "base_symbol": base_sym,
                                    "display_name": spec["display_name"],
                                    "sector": spec["sector"],
                                    "unit": spec["unit"],
                                    "lot_size": spec["lot_size"],
                                    "token": tok_val,
                                    "trading_symbol": trd_sym_val,
                                    "description": raw_item.get("pDesc") or spec["description"],
                                    "tick_size": spec["tick_size"],
                                    "expiry_date": exp_date_val.strftime("%Y-%m-%d"),
                                }
                except Exception as scrip_err:
                    logger.debug(f"Commodity scrip discovery for {base_sym} notice: {scrip_err}")

            with self._lock:
                for sym, contract in updated_contracts.items():
                    self._active_contracts[sym].update(contract)
                self._last_contract_discovery_ts = now_ts
                logger.info(f"✓ Discovered and resolved {len(updated_contracts)} active MCX contracts via Kotak Neo.")
        except Exception as e:
            logger.warning(f"MCX active contract discovery error: {e}")

        return self._active_contracts

    # --------------------------------------------------------------------------
    # LIVE QUOTES & TECHNICAL ANALYSIS
    # --------------------------------------------------------------------------
    def fetch_live_quotes(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches authentic live quotes from Kotak Neo API for all active MCX commodities.
        Computes floor pivots, range progress, volume, and OI buildup dynamics.
        """
        now_ts = time.time()
        with self._lock:
            if not force_refresh and self._quotes_cache and (now_ts - self._last_quotes_fetch_ts) < self._quotes_cache_ttl:
                return self._build_response_payload()

        try:
            from engines.kotak_neo_engine import kotak_neo_engine
            if not kotak_neo_engine._is_authenticated or not kotak_neo_engine._client:
                # If not authenticated, return cached quotes with offline flag
                return self._build_response_payload(offline_warning="Kotak Neo API not authenticated.")

            # Prepare token query list
            tokens_to_query = []
            token_to_base = {}
            for base_sym, contract in self._active_contracts.items():
                tok = str(contract.get("token", "")).strip()
                if tok:
                    tokens_to_query.append({"instrument_token": tok, "exchange_segment": "mcx_fo"})
                    token_to_base[tok] = base_sym

            if not tokens_to_query:
                return self._build_response_payload(offline_warning="No MCX tokens configured.")

            t0 = time.time()
            quotes_res = kotak_neo_engine._client.quotes(tokens_to_query, quote_type="all")
            latency_ms = round((time.time() - t0) * 1000.0, 1)

            if not isinstance(quotes_res, list):
                logger.warning(f"Unexpected Kotak quotes response: {quotes_res}")
                return self._build_response_payload(offline_warning="Invalid quotes response from Kotak Neo.")

            ts_now_ist = format_ist()
            new_quotes = {}

            for q in quotes_res:
                tok = str(q.get("exchange_token", "")).strip()
                base_sym = token_to_base.get(tok)
                if not base_sym:
                    # Match by symbol if token mapping shifted
                    disp_sym = str(q.get("display_symbol", "")).upper()
                    for b, c in self._active_contracts.items():
                        if c.get("trading_symbol", "").upper() == disp_sym:
                            base_sym = b
                            break

                if not base_sym:
                    continue

                contract = self._active_contracts[base_sym]
                try:
                    ltp = float(q.get("ltp") or 0.0)
                    chg = float(q.get("change") or 0.0)
                    chg_pct = float(q.get("per_change") or 0.0)

                    ohlc = q.get("ohlc", {}) or {}
                    open_px = float(ohlc.get("open") or ltp)
                    high_px = float(ohlc.get("high") or ltp)
                    low_px = float(ohlc.get("low") or ltp)
                    close_px = float(ohlc.get("close") or (ltp - chg if ltp > 0 else ltp))

                    vol = int(float(q.get("last_volume") or 0))
                    oi = int(float(q.get("open_int") or 0))

                    # Floor Pivot Point Analysis
                    pivot = round((high_px + low_px + close_px) / 3.0, 2)
                    r1 = round((2.0 * pivot) - low_px, 2)
                    s1 = round((2.0 * pivot) - high_px, 2)
                    r2 = round(pivot + (high_px - low_px), 2)
                    s2 = round(pivot - (high_px - low_px), 2)

                    # Range progress (0% = at day low, 100% = at day high)
                    rng = high_px - low_px
                    range_pct = round(((ltp - low_px) / rng) * 100.0, 1) if rng > 0 else 50.0

                    # Intraday Trend Regime
                    if rng > 0 and ltp >= high_px * 0.998:
                        trend_regime = "BULLISH_BREAKOUT"
                    elif rng > 0 and ltp <= low_px * 1.002:
                        trend_regime = "BEARISH_BREAKDOWN"
                    elif ltp > pivot and chg_pct > 0.15:
                        trend_regime = "BULLISH_MOMENTUM"
                    elif ltp < pivot and chg_pct < -0.15:
                        trend_regime = "BEARISH_PULLBACK"
                    else:
                        trend_regime = "CONSOLIDATION"

                    # Open Interest Interpretation
                    # Check against previous OI in cache to assess buildup
                    prev_quote = self._quotes_cache.get(base_sym, {})
                    prev_oi = prev_quote.get("open_interest", oi)
                    oi_diff = oi - prev_oi

                    if chg_pct > 0.05 and oi_diff >= 0:
                        oi_interpretation = "LONG_BUILDUP"
                        oi_sentiment = "BULLISH"
                    elif chg_pct < -0.05 and oi_diff >= 0:
                        oi_interpretation = "SHORT_BUILDUP"
                        oi_sentiment = "BEARISH"
                    elif chg_pct < -0.05 and oi_diff < 0:
                        oi_interpretation = "LONG_UNWINDING"
                        oi_sentiment = "BEARISH"
                    elif chg_pct > 0.05 and oi_diff < 0:
                        oi_interpretation = "SHORT_COVERING"
                        oi_sentiment = "BULLISH"
                    else:
                        oi_interpretation = "NEUTRAL"
                        oi_sentiment = "NEUTRAL"

                    # Market Depth (Top 5 bid / ask)
                    depth_data = q.get("depth", {}) or {}
                    bids = depth_data.get("buy", [])[:5] if isinstance(depth_data, dict) else []
                    asks = depth_data.get("sell", [])[:5] if isinstance(depth_data, dict) else []

                    new_quotes[base_sym] = {
                        "base_symbol": base_sym,
                        "display_name": contract["display_name"],
                        "sector": contract["sector"],
                        "unit": contract["unit"],
                        "lot_size": contract["lot_size"],
                        "token": tok,
                        "trading_symbol": q.get("display_symbol") or contract["trading_symbol"],
                        "description": contract.get("description", ""),
                        "ltp": round(ltp, 2),
                        "change": round(chg, 2),
                        "change_pct": round(chg_pct, 2),
                        "open": round(open_px, 2),
                        "high": round(high_px, 2),
                        "low": round(low_px, 2),
                        "close": round(close_px, 2),
                        "volume": vol,
                        "open_interest": oi,
                        "pivot": pivot,
                        "r1": r1,
                        "s1": s1,
                        "r2": r2,
                        "s2": s2,
                        "range_pct": range_pct,
                        "trend_regime": trend_regime,
                        "oi_interpretation": oi_interpretation,
                        "oi_sentiment": oi_sentiment,
                        "updated_at": ts_now_ist,
                        "depth": {
                            "bids": bids,
                            "asks": asks,
                            "total_buy": int(float(q.get("total_buy") or 0)),
                            "total_sell": int(float(q.get("total_sell") or 0)),
                        },
                        "source": "KOTAK_NEO_MCX_LIVE",
                    }
                except Exception as parse_err:
                    logger.debug(f"Quote parse error for {base_sym}: {parse_err}")

            with self._lock:
                self._quotes_cache.update(new_quotes)
                self._last_quotes_fetch_ts = now_ts

            # Persist authentic settlement snapshot in SQLite KV for reboot resilience
            try:
                state_engine.set_kv("latest_mcx_commodities_quotes", {
                    "quotes": self._quotes_cache,
                    "updated_at": ts_now_ist,
                    "latency_ms": latency_ms,
                })
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error fetching MCX commodities quotes: {e}", exc_info=True)

        return self._build_response_payload()

    # --------------------------------------------------------------------------
    # PAYLOAD BUILDER & SECTOR ROLLUPS
    # --------------------------------------------------------------------------
    def _build_response_payload(self, offline_warning: Optional[str] = None) -> Dict[str, Any]:
        """Assembles sector rollups, top movers, and live status into a unified payload."""
        is_open, mkt_text = self.is_mcx_open()

        with self._lock:
            quotes = dict(self._quotes_cache)

        # Sector Rollups
        sector_stats: Dict[str, Dict[str, Any]] = {
            "Energy": {"changes": [], "gainers": 0, "losers": 0, "volume": 0},
            "Bullion": {"changes": [], "gainers": 0, "losers": 0, "volume": 0},
            "Base Metals": {"changes": [], "gainers": 0, "losers": 0, "volume": 0},
        }

        all_items = list(quotes.values())
        for q in all_items:
            sec = q.get("sector")
            if sec in sector_stats:
                chg = q.get("change_pct", 0.0)
                sector_stats[sec]["changes"].append(chg)
                sector_stats[sec]["volume"] += q.get("volume", 0)
                if chg > 0:
                    sector_stats[sec]["gainers"] += 1
                elif chg < 0:
                    sector_stats[sec]["losers"] += 1

        summary = {}
        for sec, stat in sector_stats.items():
            chgs = stat["changes"]
            avg_chg = round(sum(chgs) / len(chgs), 2) if chgs else 0.0
            summary[sec] = {
                "sector": sec,
                "avg_change_pct": avg_chg,
                "gainers": stat["gainers"],
                "losers": stat["losers"],
                "total_volume": stat["volume"],
            }

        # Top Gainer & Top Loser
        top_gainer = max(all_items, key=lambda x: x.get("change_pct", -999.0)) if all_items else None
        top_loser = min(all_items, key=lambda x: x.get("change_pct", 999.0)) if all_items else None

        return {
            "status": "SUCCESS",
            "is_mcx_open": is_open,
            "market_status_text": mkt_text,
            "source": "KOTAK_NEO_MCX",
            "timestamp": format_ist(),
            "warning": offline_warning,
            "summary": summary,
            "top_gainer": top_gainer,
            "top_loser": top_loser,
            "quotes": quotes,
            "contracts_count": len(quotes),
        }

    def get_streaming_payload(self) -> Dict[str, Any]:
        """
        Ultra-fast in-memory retrieval (< 1ms) for continuous live WebSocket/polling stream.
        Triggers an async refresh if the cache is stale and MCX is currently open.
        """
        now_ts = time.time()
        is_open, _ = self.is_mcx_open()

        if is_open and (now_ts - self._last_quotes_fetch_ts) > 3.0:
            # Trigger quote refresh in background to avoid blocking the stream loop
            threading.Thread(
                target=self.fetch_live_quotes,
                args=(False,),
                name="CommoditiesAsyncRefresh",
                daemon=True
            ).start()

        return self._build_response_payload()


# Global Singleton Instance
commodities_engine = CommoditiesEngine()
