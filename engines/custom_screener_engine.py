"""
========================================================================================
Custom Screener & Rule Builder Engine (`engines/custom_screener_engine.py`)
========================================================================================
High-Performance Hybrid Fundamental + Technical Screener Engine:
- Default Preset: "EQ-FINAL" (matching Chartink fundamental + liquidity matrix verbatim):
    #1: Market Cap >= 300 Cr
    #2: Daily SMA(Volume * Close, 50) >= 10,000,000 (>= ₹1 Cr 50D turnover)
    #3: Yearly ROCE >= 10%
    #4: Yearly Return on Net Worth (ROE) >= 10%
    #5: Yearly Debt to Equity <= 1.5
    #6: Quarterly EPS > 0
    #7: Quarterly Net Profit (PAT) > 0
    #8: Daily Close > 75
- Enriches results with 20 EMA distance (e.g. +33.42%, +11.28%, -4.65%), volume formatting,
  and 1-click execution readiness.
- Supports interactive condition building and saving custom screener presets in SQLite.
- Resilient session pooling with auto-refreshing CSRF tokens.
========================================================================================
"""

import time
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.data_quality import now_ist, format_ist
from execution.state_engine import state_engine
from core.stock_registry import format_nse_ticker, clean_symbol

logger = logging.getLogger("CustomScreenerEngine")

# Standard EQ-FINAL Default Conditions
DEFAULT_EQ_FINAL_CONDITIONS = [
    {
        "id": "c1",
        "timeframe": "Overall",
        "field": "Market Cap",
        "operator": ">=",
        "operator_label": "Greater than equal to",
        "value": 300,
        "raw_clause": "Market Cap >= 300"
    },
    {
        "id": "c2",
        "timeframe": "Daily",
        "field": "Daily Sma( Daily Volume * Daily Close , 50 )",
        "operator": ">=",
        "operator_label": "Greater than equal to",
        "value": 10000000,
        "raw_clause": "Daily Sma( Daily Volume * Daily Close , 50 ) >= 10000000"
    },
    {
        "id": "c3",
        "timeframe": "Yearly",
        "field": "Yearly Return on capital employed percentage",
        "operator": ">=",
        "operator_label": "Greater than equal to",
        "value": 10,
        "raw_clause": "Yearly Return on capital employed percentage >= 10"
    },
    {
        "id": "c4",
        "timeframe": "Yearly",
        "field": "Yearly Return on net worth percentage",
        "operator": ">=",
        "operator_label": "Greater than equal to",
        "value": 10,
        "raw_clause": "Yearly Return on net worth percentage >= 10"
    },
    {
        "id": "c5",
        "timeframe": "Yearly",
        "field": "Yearly Debt equity ratio",
        "operator": "<=",
        "operator_label": "Less than equal to",
        "value": 1.5,
        "raw_clause": "Yearly Debt equity ratio <= 1.5"
    },
    {
        "id": "c6",
        "timeframe": "Quarterly",
        "field": "Quarterly Eps after extraordinary items basic",
        "operator": ">",
        "operator_label": "Greater than",
        "value": 0,
        "raw_clause": "Quarterly Eps after extraordinary items basic > 0"
    },
    {
        "id": "c7",
        "timeframe": "Quarterly",
        "field": "Quarterly Net profit/reported profit after tax",
        "operator": ">",
        "operator_label": "Greater than",
        "value": 0,
        "raw_clause": "Quarterly Net profit/reported profit after tax > 0"
    },
    {
        "id": "c8",
        "timeframe": "Daily",
        "field": "Daily Close",
        "operator": ">",
        "operator_label": "Greater than",
        "value": 75,
        "raw_clause": "Daily Close > 75"
    }
]


def format_indian_currency_number(n: Optional[float]) -> str:
    """Formats an integer or float into Indian comma notation: e.g. 2394243 -> 23,94,243."""
    if n is None:
        return "--"
    try:
        val = int(round(float(n)))
        s = str(abs(val))
        if len(s) <= 3:
            res = s
        else:
            last3 = s[-3:]
            rest = s[:-3]
            # split rest into chunks of 2 from right to left
            parts = []
            while len(rest) > 2:
                parts.insert(0, rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.insert(0, rest)
            res = ",".join(parts) + "," + last3
        return f"-{res}" if val < 0 else res
    except Exception:
        return str(n)


class CustomScreenerEngine:
    """
    Engine that manages preset scanners, condition parsing, remote/local execution,
    and fast 60s caching.
    """
    _instance: Optional["CustomScreenerEngine"] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CustomScreenerEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._csrf_token: Optional[str] = None
        self._csrf_expiry: float = 0.0

        # In-memory scan result cache (key -> { "timestamp": float, "data": List[Dict] })
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.init_sqlite_presets()

    def init_sqlite_presets(self):
        """Initializes default presets in SQLite KV store if not already present."""
        presets = state_engine.get_kv("custom_screener_presets", {})
        if "EQ-FINAL" not in presets:
            presets["EQ-FINAL"] = {
                "name": "EQ-FINAL",
                "is_default": True,
                "description": "Fundamental + Liquidity + Technical Matrix (ROCE>=10%, ROE>=10%, D/E<=1.5, PAT>0, Turnover>=1Cr)",
                "conditions": DEFAULT_EQ_FINAL_CONDITIONS,
                "created_at": format_ist()
            }
            state_engine.set_kv("custom_screener_presets", presets)

    def get_presets(self) -> Dict[str, Any]:
        """Returns all configured presets with EQ-FINAL as default."""
        presets = state_engine.get_kv("custom_screener_presets", {})
        if "EQ-FINAL" not in presets:
            self.init_sqlite_presets()
            presets = state_engine.get_kv("custom_screener_presets", {})
        return presets

    def save_preset(self, name: str, conditions: List[Dict[str, Any]], description: str = "") -> bool:
        """Saves a user-defined preset scanner."""
        clean_name = name.strip()
        if not clean_name:
            return False
        presets = self.get_presets()
        presets[clean_name] = {
            "name": clean_name,
            "is_default": False,
            "description": description or f"Custom scan with {len(conditions)} rules",
            "conditions": conditions,
            "updated_at": format_ist()
        }
        state_engine.set_kv("custom_screener_presets", presets)
        return True

    def delete_preset(self, name: str) -> bool:
        """Deletes a preset (cannot delete default EQ-FINAL)."""
        clean_name = name.strip()
        if clean_name == "EQ-FINAL":
            return False
        presets = self.get_presets()
        if clean_name in presets:
            del presets[clean_name]
            state_engine.set_kv("custom_screener_presets", presets)
            return True
        return False

    def _get_csrf_token(self) -> str:
        """Retrieves and caches Chartink CSRF token for 20 minutes."""
        now = time.time()
        if self._csrf_token and now < self._csrf_expiry:
            return self._csrf_token

        url = "https://chartink.com/screener"
        res = self.session.get(url, timeout=12)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        csrf_elem = soup.find("meta", {"name": "csrf-token"})
        if not csrf_elem or "content" not in csrf_elem.attrs:
            raise ValueError("CSRF token not found on Chartink screener page.")

        self._csrf_token = csrf_elem["content"]
        self._csrf_expiry = now + 1200.0  # 20 minutes
        return self._csrf_token

    def build_scan_clause(self, conditions: List[Dict[str, Any]]) -> str:
        """Compiles condition objects into Chartink's scan_clause string."""
        clauses = []
        for c in conditions:
            raw = c.get("raw_clause")
            if raw:
                clauses.append(raw.strip())
            else:
                field = c.get("field", "").strip()
                op = c.get("operator", ">=").strip()
                val = c.get("value", "")
                tf = c.get("timeframe", "").strip()
                if tf and tf not in ("Overall", "") and not field.lower().startswith(tf.lower()):
                    clauses.append(f"{tf} {field} {op} {val}")
                else:
                    clauses.append(f"{field} {op} {val}")

        if not clauses:
            clauses = ["Daily Close > 75"]

        full_inner = " and ".join(clauses)
        return f"( {{cash}} ( {full_inner} ) )"

    def run_scan(
        self,
        preset_name: Optional[str] = "EQ-FINAL",
        custom_conditions: Optional[List[Dict[str, Any]]] = None,
        raw_clause: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the screener query, enriches the data with %_change, volume formatting,
        and 20EMA distance, and returns structured records sorted by %_change descending.
        """
        t0 = time.time()

        # 1. Determine conditions and scan_clause
        if raw_clause:
            scan_clause = raw_clause.strip()
            cache_key = scan_clause
            active_name = "Custom Query"
            conditions_meta = []
        elif custom_conditions:
            scan_clause = self.build_scan_clause(custom_conditions)
            cache_key = scan_clause
            active_name = "Custom Rules"
            conditions_meta = custom_conditions
        else:
            presets = self.get_presets()
            p_data = presets.get(preset_name, presets.get("EQ-FINAL"))
            conditions_meta = p_data.get("conditions", DEFAULT_EQ_FINAL_CONDITIONS)
            scan_clause = self.build_scan_clause(conditions_meta)
            cache_key = f"preset_{preset_name}_{scan_clause}"
            active_name = preset_name or "EQ-FINAL"

        # 2. Check 60-second in-memory cache if not forcing refresh
        now = time.time()
        if not force_refresh and cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["timestamp"] < 60.0:
                logger.info(f"Serving custom scan '{active_name}' from 60s cache ({len(entry['results'])} stocks).")
                return {
                    "status": "SUCCESS",
                    "source": "CACHE",
                    "preset_name": active_name,
                    "count": len(entry["results"]),
                    "results": entry["results"],
                    "conditions": conditions_meta,
                    "scan_clause": scan_clause,
                    "execution_time_ms": round((time.time() - t0) * 1000.0, 1),
                    "timestamp": entry.get("timestamp_ist", format_ist())
                }

        # 3. Request Chartink Process API
        csrf = self._get_csrf_token()
        headers = {
            "X-CSRF-TOKEN": csrf,
            "Referer": "https://chartink.com/screener",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }

        # Column clause requesting close, volume, 1-day % change, and % distance from 20 EMA
        column_clause = "close, volume, (close - 1 day ago close) / 1 day ago close * 100, (close - ema(close, 20)) / ema(close, 20) * 100"
        payload = {
            "scan_clause": scan_clause,
            "column_clause": column_clause
        }

        logger.info(f"Executing Custom Screener Scan: {active_name}...")
        res = self.session.post("https://chartink.com/screener/process", headers=headers, data=payload, timeout=25)
        res.raise_for_status()
        res_json = res.json()

        if res_json.get("scan_error"):
            # Fallback attempt without column_clause if column expression failed
            logger.warning(f"Scan notice with column_clause: {res_json.get('scan_error')}. Retrying with standard payload...")
            res = self.session.post("https://chartink.com/screener/process", headers=headers, data={"scan_clause": scan_clause}, timeout=25)
            res.raise_for_status()
            res_json = res.json()

        raw_stocks = res_json.get("data", [])

        # 4. Normalize and enrich results
        enriched_results = []
        for i, s in enumerate(raw_stocks):
            nsecode = str(s.get("nsecode") or "").strip().upper()
            if not nsecode:
                continue

            name = str(s.get("name") or nsecode).strip()
            close_px = float(s.get("close") or 0.0)
            vol = float(s.get("volume") or s.get(",volume") or 0.0)

            # Extract % change
            per_chg_val = s.get("per_chg")
            if per_chg_val is None:
                # search for custom formula key '(close--1day agoclose)...'
                for k, v in s.items():
                    if "1day" in k:
                        per_chg_val = v
                        break
            try:
                per_chg = float(per_chg_val) if per_chg_val is not None else 0.0
            except Exception:
                per_chg = 0.0

            # Extract 20 EMA distance (as a percentage matching screenshot: e.g. 33.42, 11.28, -4.65)
            ema20_val = s.get("20ema")
            if ema20_val is None:
                for k, v in s.items():
                    if "ema" in k.lower():
                        ema20_val = v
                        break
            try:
                ema20_dist = round(float(ema20_val), 2) if ema20_val is not None else 0.0
            except Exception:
                ema20_dist = 0.0

            enriched_results.append({
                "sr": i + 1,
                "name": name,
                "symbol": nsecode,
                "bsecode": str(s.get("bsecode") or ""),
                "close": round(close_px, 2),
                "per_chg": round(per_chg, 2),
                "volume": int(vol),
                "volume_formatted": format_indian_currency_number(vol),
                "ema20": ema20_dist,
                "ticker_ns": f"{nsecode}.NS"
            })


        # 5. Sort by %_change descending (matching Chartink default layout)
        enriched_results.sort(key=lambda x: x["per_chg"], reverse=True)

        # Re-assign sequential serial numbers after sorting
        for idx, item in enumerate(enriched_results, 1):
            item["sr"] = idx

        # Save to 60s memory cache
        timestamp_ist = format_ist()
        self._cache[cache_key] = {
            "timestamp": now,
            "timestamp_ist": timestamp_ist,
            "results": enriched_results
        }

        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        logger.info(f"Custom Screener Scan '{active_name}' completed in {elapsed_ms}ms ({len(enriched_results)} stocks returned).")

        return {
            "status": "SUCCESS",
            "source": "LIVE",
            "preset_name": active_name,
            "count": len(enriched_results),
            "results": enriched_results,
            "conditions": conditions_meta,
            "scan_clause": scan_clause,
            "execution_time_ms": elapsed_ms,
            "timestamp": timestamp_ist
        }


# Singleton export
custom_screener_engine = CustomScreenerEngine()
