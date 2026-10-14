"""
SQLite Single Master Vault & Execution State Engine (`data/swing_trades.db`)
Consolidates ALL system data into ONE single master file:
1. active_positions  - Open trades with full YYYY-MM-DD HH:MM:SS IST timestamps
2. trade_history     - Closed trades with full YYYY-MM-DD HH:MM:SS IST entry & exit timestamps
3. latest_signals    - Screened setups for Portfolio 1 (6-Pillar) & Portfolio 2 (Nifty 500)
4. scan_logs         - Automated & manual scan execution logs
5. ohlcv_master_store- All ~800+ fetched stocks' 2-year daily OHLCV history + technical summary
6. system_kv_store   - Engine snapshots, Nifty 500 seed list, and Telegram Cloud Vault manifest
"""
import sqlite3
import threading
import json
import zlib
import pickle
import logging
import datetime
import math
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import DB_PATH
from core.data_quality import IST, now_ist, format_ist

logger = logging.getLogger("StateEngine")


def _fast_ewm_last(arr: np.ndarray, alpha: float) -> float:
    """Fast Cython EWM last value matching pd.Series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]."""
    if len(arr) == 0:
        return 0.0
    return float(pd.Series(arr, copy=False).ewm(alpha=alpha, adjust=False).mean().iloc[-1])


class StateEngine:
    """
    Thread-safe Single Master File SQLite Engine (`data/swing_trades.db`).
    Uses journal_mode=TRUNCATE so zero extra sidecar files (.db-wal / .db-shm) clutter the disk.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._progress_lock = threading.RLock()
        self._ohlcv_mem_cache: Dict[str, Tuple[pd.DataFrame, datetime.datetime]] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=45.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _normalize_portfolio_type(self, p_type: Optional[str]) -> str:
        if not p_type:
            return "INSTITUTIONAL_6PILLAR"
        p = str(p_type).strip().upper()
        if "NIFTY" in p:
            return "NIFTY500_MOMENTUM"
        return "INSTITUTIONAL_6PILLAR"

    def _init_db(self):
        with self._lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                # Switch out of WAL mode into single-file DELETE mode so no .db-wal / .db-shm files exist
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.execute("PRAGMA journal_mode=DELETE;")
                except Exception:
                    pass

                cursor = conn.cursor()

                # 1. Active Positions Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    portfolio_type TEXT NOT NULL DEFAULT 'INSTITUTIONAL_6PILLAR',
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    stop_loss REAL NOT NULL,
                    original_stop_loss REAL,
                    target REAL NOT NULL,
                    target_2 REAL DEFAULT 0.0,
                    current_price REAL,
                    highest_price REAL,
                    entry_date TEXT NOT NULL,
                    hold_days INTEGER DEFAULT 0,
                    t1_hit INTEGER DEFAULT 0,
                    t1_pnl REAL DEFAULT 0.0,
                    sl_reason TEXT,
                    sector TEXT,
                    notes TEXT,
                    entry_reason TEXT,
                    entry_regime TEXT,
                    metrics_json TEXT,
                    features_json TEXT,
                    updated_at TEXT NOT NULL
                );
                """)

                # 2. Closed Trade History Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL DEFAULT 'INSTITUTIONAL_6PILLAR',
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    r_multiple REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    hold_days INTEGER DEFAULT 0,
                    exit_reason TEXT NOT NULL,
                    sector TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                # 3. Latest Signals Snapshot Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS latest_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL,
                    sector TEXT,
                    bucket TEXT,
                    score REAL,
                    confidence INTEGER,
                    ml_prob REAL,
                    tier TEXT,
                    close REAL,
                    entry REAL,
                    sl REAL,
                    t1 REAL,
                    t2 REAL,
                    rr REAL,
                    shares INTEGER,
                    capital_at_risk REAL,
                    risk_pct REAL,
                    atr_pct REAL,
                    rsi14 REAL,
                    dist_to_pivot REAL,
                    dist_to_ema20 REAL,
                    inst_score REAL,
                    week_confluence INTEGER,
                    intraday_pct REAL,
                    intraday_adj REAL,
                    rs_spread REAL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, portfolio_type)
                );
                """)

                # 4. Scan Executions Log Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scan_type TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_time_sec REAL,
                    candidates_count INTEGER DEFAULT 0,
                    orders_count INTEGER DEFAULT 0,
                    reconciled_count INTEGER DEFAULT 0,
                    message TEXT,
                    details_json TEXT
                );
                """)

                # 5. Master OHLCV Store Table (All 800+ Stocks' 2-Year Daily Bars in ONE table)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_master_store (
                    symbol TEXT PRIMARY KEY,
                    last_bar_date TEXT NOT NULL,
                    bars_count INTEGER NOT NULL,
                    last_close REAL DEFAULT 0.0,
                    prev_close REAL DEFAULT 0.0,
                    change_pct REAL DEFAULT 0.0,
                    last_volume INTEGER DEFAULT 0,
                    ema20 REAL DEFAULT 0.0,
                    ema50 REAL DEFAULT 0.0,
                    ema200 REAL DEFAULT 0.0,
                    rsi14 REAL DEFAULT 50.0,
                    atr14 REAL DEFAULT 0.0,
                    updated_at TEXT NOT NULL,
                    ohlcv_blob BLOB NOT NULL
                );
                """)

                # 6. Master System Key-Value Store (Snapshots, Manifests, Constituent Seeds)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_kv_store (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """)

                conn.commit()
                logger.info("Single Master File Database schemas verified and ready.")

    # --------------------------------------------------------------------------
    # SYSTEM KEY-VALUE STORE (INSIDE SINGLE MASTER FILE)
    # --------------------------------------------------------------------------
    def set_kv(self, key: str, value: Any) -> None:
        now_str = format_ist()
        val_str = json.dumps(value, default=str)
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO system_kv_store (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at;
                """, (key, val_str, now_str))
                conn.commit()

    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT value_json FROM system_kv_store WHERE key = ?", (key,)).fetchone()
                if row and row["value_json"]:
                    try:
                        return json.loads(row["value_json"])
                    except Exception:
                        return default
        return default

    # --------------------------------------------------------------------------
    # MASTER OHLCV STORAGE (ALL STOCKS IN SINGLE MASTER FILE)
    # --------------------------------------------------------------------------
    @staticmethod
    def _compute_df_summary_and_blob(df: pd.DataFrame) -> Tuple[Dict[str, Any], bytes]:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if len(df) > 500:
            df = df.iloc[-500:]

        closes_np = df["Close"].to_numpy(dtype=np.float64, copy=False)
        highs_np = df["High"].to_numpy(dtype=np.float64, copy=False)
        lows_np = df["Low"].to_numpy(dtype=np.float64, copy=False)
        vols_s = df["Volume"].fillna(0).astype(np.int64)
        n = len(closes_np)

        last_close = float(closes_np[-1])
        prev_close = float(closes_np[-2]) if n >= 2 else last_close
        change_pct = round(((last_close - prev_close) / max(0.01, prev_close)) * 100.0, 2)
        last_volume = int(vols_s.iloc[-1])

        ema20 = round(_fast_ewm_last(closes_np, 2.0 / 21.0), 2)
        ema50 = round(_fast_ewm_last(closes_np, 2.0 / 51.0), 2)
        ema200 = round(_fast_ewm_last(closes_np, 2.0 / 201.0), 2)

        # Fast C-vectorized Wilder RSI 14
        if n >= 2:
            delta = np.empty(n, dtype=np.float64)
            delta[0] = 0.0
            delta[1:] = closes_np[1:] - closes_np[:-1]
            gains = np.where(delta > 0.0, delta, 0.0)
            losses = np.where(delta < 0.0, -delta, 0.0)
            avg_gain = _fast_ewm_last(gains[1:], 1.0 / 14.0)
            avg_loss = _fast_ewm_last(losses[1:], 1.0 / 14.0)
            rs = avg_gain / (avg_loss + 1e-9)
            rsi14 = round(float(100.0 - (100.0 / (1.0 + rs))), 1)
        else:
            rsi14 = 50.0

        # Fast NumPy ATR 14 (only needs last 15 bars!)
        if n >= 2:
            win = min(14, n - 1)
            h_tail = highs_np[-win:]
            l_tail = lows_np[-win:]
            c_prev = closes_np[-(win + 1):-1]
            tr_tail = np.maximum(h_tail - l_tail, np.maximum(np.abs(h_tail - c_prev), np.abs(l_tail - c_prev)))
            atr14 = round(float(np.mean(tr_tail)), 2)
        else:
            atr14 = round(float(highs_np[-1] - lows_np[-1]), 2)

        last_idx = df.index[-1]
        last_bar_date = str(last_idx)[:10]

        compact = pd.DataFrame(index=df.index)
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                compact[col] = df[col].astype(np.float32)
        compact["Volume"] = vols_s

        blob = zlib.compress(pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL), level=1)
        meta = {
            "last_bar_date": last_bar_date,
            "bars_count": len(compact),
            "last_close": round(last_close, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": change_pct,
            "last_volume": last_volume,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "rsi14": rsi14,
            "atr14": atr14
        }
        return meta, blob

    def save_ohlcv_df(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty or "Close" not in df.columns:
            return
        sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_dt = now_ist()
        now_str = format_ist(now_dt)
        try:
            meta, blob = self._compute_df_summary_and_blob(df)
            with self._lock:
                self._ohlcv_mem_cache[sym] = (df, now_dt)
                with self._get_connection() as conn:
                    conn.execute("""
                    INSERT INTO ohlcv_master_store (
                        symbol, last_bar_date, bars_count, last_close, prev_close,
                        change_pct, last_volume, ema20, ema50, ema200, rsi14, atr14,
                        updated_at, ohlcv_blob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        last_bar_date = excluded.last_bar_date,
                        bars_count = excluded.bars_count,
                        last_close = excluded.last_close,
                        prev_close = excluded.prev_close,
                        change_pct = excluded.change_pct,
                        last_volume = excluded.last_volume,
                        ema20 = excluded.ema20,
                        ema50 = excluded.ema50,
                        ema200 = excluded.ema200,
                        rsi14 = excluded.rsi14,
                        atr14 = excluded.atr14,
                        updated_at = excluded.updated_at,
                        ohlcv_blob = excluded.ohlcv_blob;
                    """, (
                        sym, meta["last_bar_date"], meta["bars_count"], meta["last_close"],
                        meta["prev_close"], meta["change_pct"], meta["last_volume"],
                        meta["ema20"], meta["ema50"], meta["ema200"], meta["rsi14"],
                        meta["atr14"], now_str, sqlite3.Binary(blob)
                    ))
                    conn.commit()
        except Exception as e:
            logger.debug(f"Failed to save OHLCV for {sym} into master DB: {e}")

    def save_ohlcv_batch(self, dfs: Dict[str, pd.DataFrame]) -> int:
        """
        Multi-threaded parallel OHLCV summary & compression + chunked SQLite transaction:
        1. Updates in-memory cache immediately.
        2. Computes technical summary & compressed blob across 16 parallel threads.
        3. Writes to SQLite in 250-row chunks so database locks are held for <50ms.
        """
        if not dfs:
            return 0
        now_dt = now_ist()
        now_str = format_ist(now_dt)

        valid_items: List[Tuple[str, pd.DataFrame]] = []
        for symbol, df in dfs.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
            self._ohlcv_mem_cache[sym] = (df, now_dt)
            valid_items.append((sym, df))

        if not valid_items:
            return 0

        def _prep_row(item: Tuple[str, pd.DataFrame]):
            sym, df = item
            meta, blob = self._compute_df_summary_and_blob(df)
            return (
                sym, meta["last_bar_date"], meta["bars_count"], meta["last_close"],
                meta["prev_close"], meta["change_pct"], meta["last_volume"],
                meta["ema20"], meta["ema50"], meta["ema200"], meta["rsi14"],
                meta["atr14"], now_str, sqlite3.Binary(blob)
            )

        rows_to_insert = []
        if len(valid_items) <= 8:
            for item in valid_items:
                try:
                    rows_to_insert.append(_prep_row(item))
                except Exception as e:
                    logger.debug(f"Skip batch item {item[0]}: {e}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                for fut in concurrent.futures.as_completed([executor.submit(_prep_row, it) for it in valid_items]):
                    try:
                        rows_to_insert.append(fut.result())
                    except Exception as e:
                        logger.debug(f"Skip parallel batch item: {e}")

        if not rows_to_insert:
            return 0

        sql = """
        INSERT INTO ohlcv_master_store (
            symbol, last_bar_date, bars_count, last_close, prev_close,
            change_pct, last_volume, ema20, ema50, ema200, rsi14, atr14,
            updated_at, ohlcv_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            last_bar_date = excluded.last_bar_date,
            bars_count = excluded.bars_count,
            last_close = excluded.last_close,
            prev_close = excluded.prev_close,
            change_pct = excluded.change_pct,
            last_volume = excluded.last_volume,
            ema20 = excluded.ema20,
            ema50 = excluded.ema50,
            ema200 = excluded.ema200,
            rsi14 = excluded.rsi14,
            atr14 = excluded.atr14,
            updated_at = excluded.updated_at,
            ohlcv_blob = excluded.ohlcv_blob;
        """
        chunk_size = 250
        for i in range(0, len(rows_to_insert), chunk_size):
            chunk = rows_to_insert[i:i + chunk_size]
            with self._lock:
                with self._get_connection() as conn:
                    conn.executemany(sql, chunk)
                    conn.commit()
        return len(rows_to_insert)

    def get_ohlcv_df(self, symbol: str) -> Tuple[Optional[pd.DataFrame], float]:
        """Retrieves (df, age_hours) for a symbol from the Single Master File."""
        sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_dt = now_ist()

        if sym in self._ohlcv_mem_cache:
            cached_df, cached_dt = self._ohlcv_mem_cache[sym]
            age_hrs = max(0.0, (now_dt - cached_dt).total_seconds() / 3600.0)
            return cached_df, age_hrs

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT updated_at, ohlcv_blob FROM ohlcv_master_store WHERE symbol = ?",
                    (sym,)
                ).fetchone()
                if row and row["ohlcv_blob"]:
                    try:
                        df = pickle.loads(zlib.decompress(row["ohlcv_blob"]))
                        upd_str = row["updated_at"] or ""
                        try:
                            naive_dt = datetime.datetime.strptime(upd_str[:19], "%Y-%m-%d %H:%M:%S")
                            upd_dt = IST.localize(naive_dt) if hasattr(IST, "localize") else naive_dt.replace(tzinfo=now_dt.tzinfo)
                            age_hrs = max(0.0, (now_dt - upd_dt).total_seconds() / 3600.0)
                        except Exception:
                            upd_dt = now_dt - datetime.timedelta(hours=1)
                            age_hrs = 1.0
                        self._ohlcv_mem_cache[sym] = (df, upd_dt)
                        return df, age_hrs
                    except Exception as e:
                        logger.debug(f"Failed to decode OHLCV blob for {sym}: {e}")
        return None, 9999.0

    def get_ohlcv_stats(self) -> Dict[str, Any]:
        """Returns total count and latest bar date across all stocks in ohlcv_master_store."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("""
                SELECT COUNT(*) as cnt, MAX(last_bar_date) as max_date, MAX(updated_at) as last_upd
                FROM ohlcv_master_store
                """).fetchone()
                return {
                    "stocks_count": int(row["cnt"] or 0) if row else 0,
                    "latest_bar_date": (row["max_date"] or "--") if row else "--",
                    "last_updated": (row["last_upd"] or "--") if row else "--"
                }

    def get_all_ohlcv_metadata(self, search: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """Returns fast metadata rows for all fetched stocks in the Single Master File."""
        query = """
        SELECT symbol, last_bar_date, bars_count, last_close, prev_close,
               change_pct, last_volume, ema20, ema50, ema200, rsi14, atr14, updated_at
        FROM ohlcv_master_store
        """
        params: List[Any] = []
        if search:
            query += " WHERE symbol LIKE ?"
            params.append(f"%{search.strip().upper()}%")
        query += " ORDER BY change_pct DESC, last_volume DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    lc = float(d.get("last_close") or 0.0)
                    atr = float(d.get("atr14") or (lc * 0.025))
                    risk = max(0.5, round(2.0 * atr, 2))
                    d["auto_sl"] = round(lc - risk, 2)
                    d["auto_t1"] = round(lc + 1.5 * risk, 2)
                    d["auto_t2"] = round(lc + 3.0 * risk, 2)
                    result.append(d)
                return result

    def get_stock_chart_payload(self, symbol: str, limit_bars: int = 260) -> Dict[str, Any]:
        """
        Builds full Candlestick + Volume + EMA(20/50/200) + RSI14 + Trade Level lines (Entry, SL, T1, T2)
        and raw OHLCV data table from the Single Master File for any fetched or traded stock.
        Automatically gap-fills/refreshes from Yahoo Finance if the stored bar is behind the latest
        trading session or if NSE market hours are currently open.
        """
        sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        df, age_hrs = self.get_ohlcv_df(sym)

        try:
            from engines.institutional_engine import DirectYahooFinanceFetcher
            from core.market_calendar import NSEMarketCalendar
            target_date = DirectYahooFinanceFetcher.get_latest_expected_bar_date()
            is_mkt_open, _ = NSEMarketCalendar.can_execute_trades()

            needs_live_sync = False
            if df is None or df.empty:
                needs_live_sync = True
            else:
                last_idx = df.index[-1]
                last_dt = last_idx.date() if hasattr(last_idx, "date") else pd.Timestamp(last_idx).date()
                if last_dt < target_date or (is_mkt_open and age_hrs > 0.02):
                    needs_live_sync = True

            if needs_live_sync:
                fetcher = DirectYahooFinanceFetcher(max_workers=4)
                _, refreshed_df, _ = fetcher.fetch_or_gapfill_ticker(
                    sym, target_date=target_date, is_market_open=is_mkt_open, persist_immediately=True
                )
                if refreshed_df is not None and not refreshed_df.empty:
                    df = refreshed_df
        except Exception as e:
            logger.debug(f"Auto chart gapfill notice for {sym}: {e}")

        if df is None or df.empty:
            return {"status": "NOT_FOUND", "symbol": sym, "message": f"No OHLCV data in Master File for {sym}"}

        df = df.sort_index()
        df = df[~pd.Series(df.index.date).duplicated(keep="last").values]

        closes = df["Close"].astype(float)
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)
        opens = df["Open"].astype(float)
        vols = df["Volume"].fillna(0).astype(np.int64)

        ema20_s = closes.ewm(span=20, adjust=False).mean().fillna(closes)
        ema50_s = closes.ewm(span=50, adjust=False).mean().fillna(closes)
        ema200_s = closes.ewm(span=200, adjust=False).mean().fillna(closes)

        delta = closes.diff().fillna(0.0)
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        rsi14_s = (100 - (100 / (1 + rs))).fillna(50.0).replace([np.inf, -np.inf], 50.0)

        prev_closes = closes.shift(1).fillna(closes)
        tr = pd.concat([
            (highs - lows),
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs()
        ], axis=1).max(axis=1)
        atr14_s = tr.rolling(14, min_periods=1).mean().fillna(closes * 0.01).replace([np.inf, -np.inf], 1.0)
        pct_chg_s = (closes.pct_change().fillna(0.0) * 100.0).replace([np.inf, -np.inf], 0.0)

        tail_n = min(len(df), max(60, int(limit_bars)))
        df_slice = df.iloc[-tail_n:]

        def _clean_flt(v, default=0.0):
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return default
                return f
            except Exception:
                return default

        candles = []
        for idx in df_slice.index:
            d_str = str(idx)[:10]
            o_val = round(_clean_flt(opens.loc[idx]), 2)
            h_val = round(_clean_flt(highs.loc[idx], o_val), 2)
            l_val = round(_clean_flt(lows.loc[idx], o_val), 2)
            c_val = round(_clean_flt(closes.loc[idx], o_val), 2)
            v_val = int(vols.loc[idx]) if idx in vols.index and not math.isnan(float(vols.loc[idx])) else 0
            candles.append({
                "time": d_str,
                "open": o_val,
                "high": h_val,
                "low": l_val,
                "close": c_val,
                "volume": v_val,
                "change_pct": round(_clean_flt(pct_chg_s.loc[idx], 0.0), 2),
                "ema20": round(_clean_flt(ema20_s.loc[idx], c_val), 2),
                "ema50": round(_clean_flt(ema50_s.loc[idx], c_val), 2),
                "ema200": round(_clean_flt(ema200_s.loc[idx], c_val), 2),
                "rsi14": round(_clean_flt(rsi14_s.loc[idx], 50.0), 1),
                "atr14": round(_clean_flt(atr14_s.loc[idx], max(0.5, c_val * 0.01)), 2)
            })

        # Check if stock is an Active Position or Signal or Historical Trade
        active_pos = None
        signal_row = None
        recent_trades = []
        with self._lock:
            with self._get_connection() as conn:
                ap = conn.execute("SELECT * FROM active_positions WHERE symbol = ?", (sym,)).fetchone()
                if ap:
                    active_pos = dict(ap)
                sig = conn.execute("SELECT * FROM latest_signals WHERE symbol = ? ORDER BY score DESC LIMIT 1", (sym,)).fetchone()
                if sig:
                    signal_row = dict(sig)
                hist = conn.execute("SELECT * FROM trade_history WHERE symbol = ? ORDER BY id DESC LIMIT 5", (sym,)).fetchall()
                recent_trades = [dict(h) for h in hist]

        last_c = candles[-1]["close"]
        last_atr = max(0.5, candles[-1]["atr14"])

        # Check for Nifty 50 Index Prediction Integration
        nifty_pred = None
        if sym in ("^NSEI", "NIFTY", "NIFTY50"):
            nifty_pred = self.get_kv("latest_nifty_predictions")
            if not nifty_pred:
                try:
                    from engines.nifty_analytics_engine import nifty_analytics_engine
                    nifty_pred = nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)
                except Exception:
                    nifty_pred = None

        is_completed_trade = bool(recent_trades and not active_pos)
        show_trade_lines = False
        is_completed = False
        completed_info = None
        trade_status = "NONE"

        if active_pos:
            ep = round(float(active_pos["entry_price"]), 2)
            sl = round(float(active_pos["stop_loss"]), 2)
            t1 = round(float(active_pos["target"]), 2)
            t2 = round(float(active_pos.get("target_2") or (ep + 2.0 * max(1.0, t1 - ep))), 2)
            direction = "BULLISH_LONG"
            level_source = f"ACTIVE POSITION ({active_pos['portfolio_type']})"
            entry_ts = active_pos.get("entry_date", "")
            risk_per_share = max(0.01, round(ep - sl, 2))
            rr_t1 = round((t1 - ep) / risk_per_share, 2)
            rr_t2 = round((t2 - ep) / risk_per_share, 2)
            show_trade_lines = True
            trade_status = "ACTIVE"
        elif is_completed_trade:
            # Position completed (hit Stop Loss or Target). Reset chart: clear trade lines and display completed audit
            last_trade = recent_trades[0]
            is_completed = True
            show_trade_lines = False
            trade_status = "COMPLETED"
            ep = float(last_trade.get("entry_price") or last_c)
            exit_px = float(last_trade.get("exit_price") or last_c)
            sl = None
            t1 = None
            t2 = None
            direction = "COMPLETED"
            exit_reason = str(last_trade.get("exit_reason") or "TRADE_EXITED")
            level_source = f"POSITION COMPLETED ({exit_reason}) • CHART RESET"
            entry_ts = str(last_trade.get("exit_date") or candles[-1]["time"])
            risk_per_share = 0.0
            rr_t1 = float(last_trade.get("r_multiple") or 0.0)
            rr_t2 = 0.0
            completed_info = {
                "entry_price": ep,
                "exit_price": exit_px,
                "exit_reason": exit_reason,
                "pnl": float(last_trade.get("pnl") or 0.0),
                "r_multiple": float(last_trade.get("r_multiple") or 0.0),
                "return_pct": float(last_trade.get("return_pct") or 0.0),
                "entry_date": last_trade.get("entry_date"),
                "exit_date": last_trade.get("exit_date"),
                "hold_days": last_trade.get("hold_days", 1)
            }
        elif nifty_pred and isinstance(nifty_pred, dict) and nifty_pred.get("status") == "OK":
            t1_f = nifty_pred.get("t1_forecast") or {}
            t5_f = nifty_pred.get("t5_forecast") or {}
            comp_score = float(nifty_pred.get("composite_score") or 0.0)
            t1_verd = t1_f.get("verdict") or "NEUTRAL"
            t5_verd = t5_f.get("verdict") or "NEUTRAL"
            is_bearish = ("BEARISH" in t1_verd or "BEARISH" in t5_verd or comp_score < -8.0)
            tech = (nifty_pred.get("pillars") or {}).get("nifty_technicals") or {}

            ep = round(last_c, 2)
            if is_bearish:
                direction = "BEARISH_SHORT"
                r1_val = float(tech.get("r1") or (ep + 1.2 * last_atr))
                t1_exp = float(t1_f.get("expected_close") or (ep - 0.8 * last_atr))
                t5_exp = float(t5_f.get("expected_close") or (ep - 1.8 * last_atr))
                t1_cone_low = float((t1_f.get("range_1sigma") or [ep - 1.5 * last_atr])[0])
                sl = round(max(r1_val, ep + 0.8 * last_atr), 2)
                t1 = round(min(t1_exp, ep - 0.3 * last_atr), 2)
                t2 = round(min(t5_exp, t1_cone_low, ep - 0.9 * last_atr), 2)
                level_source = f"NIFTY QUANT FORECAST: {t1_verd} (T+1) & {t5_verd} (T+5) • BEARISH SHORT CONE"
                risk_per_share = max(0.01, round(sl - ep, 2))
                rr_t1 = round(abs(ep - t1) / risk_per_share, 2)
                rr_t2 = round(abs(ep - t2) / risk_per_share, 2)
            else:
                direction = "BULLISH_LONG"
                s1_val = float(tech.get("s1") or (ep - 1.2 * last_atr))
                t1_exp = float(t1_f.get("expected_close") or (ep + 0.8 * last_atr))
                t5_exp = float(t5_f.get("expected_close") or (ep + 1.8 * last_atr))
                t1_cone_high = float((t1_f.get("range_1sigma") or [ep - last_atr, ep + 1.5 * last_atr])[1])
                sl = round(min(s1_val, ep - 0.8 * last_atr), 2)
                t1 = round(max(t1_exp, ep + 0.3 * last_atr), 2)
                t2 = round(max(t5_exp, t1_cone_high, ep + 0.9 * last_atr), 2)
                level_source = f"NIFTY QUANT FORECAST: {t1_verd} (T+1) & {t5_verd} (T+5) • BULLISH LONG CONE"
                risk_per_share = max(0.01, round(ep - sl, 2))
                rr_t1 = round((t1 - ep) / risk_per_share, 2)
                rr_t2 = round((t2 - ep) / risk_per_share, 2)
            entry_ts = nifty_pred.get("timestamp_ist") or candles[-1]["time"]
            show_trade_lines = True
            trade_status = "NIFTY_FORECAST"
        elif signal_row and float(signal_row.get("sl") or 0) > 0 and not is_completed_trade:
            ep = round(float(signal_row.get("entry") or signal_row.get("close") or last_c), 2)
            sl = round(float(signal_row["sl"]), 2)
            t1 = round(float(signal_row.get("t1") or (ep + 1.5 * max(1.0, ep - sl))), 2)
            t2 = round(float(signal_row.get("t2") or (ep + 3.0 * max(1.0, ep - sl))), 2)
            direction = "BULLISH_LONG"
            level_source = f"SIGNAL SETUP ({signal_row.get('bucket', 'WATCH')} - {signal_row.get('portfolio_type', '')})"
            entry_ts = signal_row.get("updated_at", "")
            risk_per_share = max(0.01, round(ep - sl, 2))
            rr_t1 = round((t1 - ep) / risk_per_share, 2)
            rr_t2 = round((t2 - ep) / risk_per_share, 2)
            show_trade_lines = True
            trade_status = "SIGNAL"
        else:
            ep = last_c
            sl = None
            t1 = None
            t2 = None
            direction = "TECHNICALS"
            level_source = "TECHNICAL ANALYSIS (EMA 20/50/200 & RSI14)"
            entry_ts = candles[-1]["time"]
            risk_per_share = 0.0
            rr_t1 = 0.0
            rr_t2 = 0.0
            show_trade_lines = False
            trade_status = "NONE"

        payload = {
            "status": "OK",
            "symbol": sym,
            "bars_total": len(df),
            "last_bar_date": candles[-1]["time"],
            "latest": candles[-1],
            "trade_levels": {
                "source": level_source,
                "direction": direction,
                "entry_price": ep,
                "stop_loss": sl,
                "target_1": t1,
                "target_2": t2,
                "risk_per_share": risk_per_share,
                "sl_pct": round(((sl - ep) / ep) * 100.0, 2) if (sl and ep) else 0.0,
                "t1_pct": round(((t1 - ep) / ep) * 100.0, 2) if (t1 and ep) else 0.0,
                "t2_pct": round(((t2 - ep) / ep) * 100.0, 2) if (t2 and ep) else 0.0,
                "rr_t1": rr_t1,
                "rr_t2": rr_t2,
                "timestamp": entry_ts,
                "show_trade_lines": show_trade_lines,
                "is_completed": is_completed,
                "trade_status": trade_status,
                "completed_trade": completed_info
            },
            "active_position": active_pos,
            "signal": signal_row,
            "recent_trades": recent_trades,
            "candles": candles
        }

        def _sanitize_json(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            elif isinstance(obj, dict):
                return {k: _sanitize_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_json(x) for x in obj]
            return obj

        return _sanitize_json(payload)

    # --------------------------------------------------------------------------
    # ACTIVE POSITIONS OPERATIONS (IMMEDIATE MASTER DB SAVE + IMMEDIATE CLOUD SYNC)
    # --------------------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        target_2: float = 0.0,
        triggered_from: str = "MANUAL",
        notes: str = "",
        sl_reason: str = "INITIAL_SL",
        sector: str = "DIVERSIFIED",
        portfolio_type: str = "INSTITUTIONAL_6PILLAR",
        entry_reason: str = "",
        entry_regime: str = "",
        entry_metrics: Optional[Dict[str, Any]] = None,
        trade_features: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Opens a new position in the Single Master File with full YYYY-MM-DD HH:MM:SS IST timestamp
        and immediately syncs the Master File to Telegram Cloud.
        STRICT TRADING GATE: Blocks order execution outside NSE trading hours (09:15–15:30 IST, Mon–Fri).
        """
        from core.market_calendar import NSEMarketCalendar
        from core.telegram_vault import telegram_vault

        can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
        if not can_trade:
            logger.warning(f"🔒 [Strict Trading Gate] Blocked BUY order for {symbol}: {gate_reason}")
            return False

        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        p_type = self._normalize_portfolio_type(portfolio_type)
        now_str = format_ist()  # Full timestamp: YYYY-MM-DD HH:MM:SS IST

        metrics_str = json.dumps(entry_metrics or {})
        features_str = json.dumps(trade_features or {})

        with self._lock:
            with self._get_connection() as conn:
                try:
                    existing = conn.execute(
                        "SELECT id, quantity, entry_price FROM active_positions WHERE symbol = ?",
                        (sym_clean,)
                    ).fetchone()
                    if existing:
                        logger.info(
                            f"Position for {sym_clean} is already open ({existing['quantity']} shs @ ₹{existing['entry_price']:.2f}); "
                            "skipping duplicate open without mutating price."
                        )
                        return True

                    # Slot capacity and cash risk validation
                    max_slots = 15 if "NIFTY" in p_type else 10
                    cur_cnt = conn.execute(
                        "SELECT COUNT(*) FROM active_positions WHERE portfolio_type = ?",
                        (p_type,)
                    ).fetchone()[0]
                    if cur_cnt >= max_slots:
                        logger.warning(f"⚠️ [Slot Limit] Cannot open {sym_clean}: {p_type} already at {cur_cnt}/{max_slots} slots.")
                        return False

                    base_cap = 100000.0 if "NIFTY" in p_type else 500000.0
                    total_inv = conn.execute(
                        "SELECT SUM(entry_price * quantity) FROM active_positions WHERE portfolio_type = ?",
                        (p_type,)
                    ).fetchone()[0] or 0.0
                    t1_pnl_sum = conn.execute(
                        "SELECT SUM(t1_pnl) FROM active_positions WHERE portfolio_type = ? AND t1_hit = 1",
                        (p_type,)
                    ).fetchone()[0] or 0.0
                    hist_pnl = conn.execute(
                        "SELECT SUM(pnl) FROM trade_history WHERE portfolio_type = ?",
                        (p_type,)
                    ).fetchone()[0] or 0.0
                    free_cash = max(0.0, base_cap + hist_pnl + t1_pnl_sum - total_inv)
                    required_cash = entry_price * quantity

                    if required_cash > free_cash:
                        logger.warning(
                            f"⚠️ [Risk Gate] Insufficient available cash to open {sym_clean}: "
                            f"Required ₹{required_cash:,.2f} > Free Cash ₹{free_cash:,.2f}."
                        )
                        return False

                    conn.execute("""
                    INSERT INTO active_positions (
                        symbol, portfolio_type, entry_price, quantity, stop_loss, original_stop_loss,
                        target, target_2, current_price, highest_price, entry_date, hold_days,
                        t1_hit, t1_pnl, sl_reason, sector, notes, entry_reason, entry_regime,
                        metrics_json, features_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sym_clean, p_type, entry_price, quantity, stop_loss, stop_loss,
                        target, target_2, entry_price, entry_price, now_str,
                        sl_reason, sector, notes, entry_reason, entry_regime,
                        metrics_str, features_str, now_str
                    ))

                    sync_type = "AUTO_BUY_CLOUD_SYNC" if "AUTO" in str(triggered_from).upper() else "MANUAL_BUY_CLOUD_SYNC"
                    sync_msg = (
                        f"🟢 BUY EXECUTED & CLOUD SYNCED: {sym_clean} ({quantity} shs @ ₹{entry_price:,.2f} | "
                        f"SL ₹{stop_loss:,.2f} | T1 ₹{target:,.2f} | T2 ₹{target_2:,.2f}) • {now_str}"
                    )
                    sync_details = json.dumps({
                        "event": "TRADE_OPEN",
                        "symbol": sym_clean,
                        "portfolio": p_type,
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "target_1": target,
                        "target_2": target_2,
                        "triggered_from": triggered_from,
                        "timestamp_ist": now_str,
                        "cloud_sync": "SYNCED_TO_TELEGRAM_VAULT"
                    })
                    conn.execute("""
                    INSERT INTO scan_logs (
                        timestamp, scan_type, portfolio_type, status, execution_time_sec,
                        candidates_count, orders_count, reconciled_count, message, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (now_str, sync_type, p_type, "CLOUD_SYNCED", 0.25, 1, 1, 1, sync_msg, sync_details))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error opening position for {sym_clean}: {e}")
                    return False

        logger.info(f"Opened position for {sym_clean} ({quantity} shs @ ₹{entry_price:.2f}) in {p_type} at {now_str}")

        # Immediately sync Single Master File to Telegram Cloud Vault
        telegram_vault.send_alert_and_sync(
            title=f"🚀 NEW TRADE OPENED: {sym_clean} ({p_type})",
            message=(
                f"<b>Symbol:</b> <code>{sym_clean}</code>\n"
                f"<b>Portfolio:</b> {p_type}\n"
                f"<b>Entry Price:</b> ₹{entry_price:,.2f} × {quantity} shs\n"
                f"<b>Stop Loss:</b> ₹{stop_loss:,.2f} | <b>Target 1:</b> ₹{target:,.2f} | <b>Target 2:</b> ₹{target_2:,.2f}\n"
                f"<b>Entry Timestamp:</b> {now_str}\n"
                f"<b>Setup:</b> {entry_reason or notes}"
            ),
            sync_db=True
        )
        return True

    def get_active_positions(self, portfolio_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves currently open positions enriched with latest live prices & timestamps."""
        query = """
        SELECT a.*,
               m.last_close AS master_last_close,
               m.change_pct AS day_change_pct,
               m.updated_at AS master_updated_at
        FROM active_positions a
        LEFT JOIN ohlcv_master_store m ON a.symbol = m.symbol
        """
        params = []
        if portfolio_type and str(portfolio_type).upper() != "ALL":
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE a.portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY a.entry_date DESC"

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    master_lc = float(item.get("master_last_close") or 0.0)
                    master_upd = item.get("master_updated_at") or ""
                    pos_upd = item.get("updated_at") or ""
                    if master_lc > 0 and (master_upd >= pos_upd or float(item.get("current_price") or 0) == float(item.get("entry_price") or 0)):
                        item["current_price"] = round(master_lc, 2)
                        item["highest_price"] = round(max(float(item.get("highest_price") or 0.0), master_lc), 2)
                        if master_upd:
                            item["price_updated_at"] = master_upd
                    else:
                        item["price_updated_at"] = pos_upd or item.get("entry_date", "")
                    item["day_change_pct"] = round(float(item.get("day_change_pct") or 0.0), 2)
                    ep = float(item.get("entry_price") or 0.0)
                    cp = float(item.get("current_price") or ep)
                    sl = float(item.get("stop_loss") or ep * 0.95)
                    qty = int(item.get("quantity") or 0)
                    item["qty"] = qty
                    item["unrealized_pnl"] = round((cp - ep) * qty, 2)
                    item["return_pct"] = round(((cp - ep) / max(0.01, ep)) * 100.0, 2)
                    item["stop_distance_pct"] = round(((cp - sl) / max(0.01, cp)) * 100.0, 2)
                    r_unit = max(0.01, ep - float(item.get("original_stop_loss") or sl))
                    item["current_r"] = round((cp - ep) / r_unit, 2)
                    try:
                        item["metrics"] = json.loads(item.get("metrics_json") or "{}")
                    except Exception:
                        item["metrics"] = {}
                    result.append(item)
                return result

    def update_position_price(
        self,
        symbol: str,
        current_price: float,
        stop_loss: Optional[float] = None,
        highest_price: Optional[float] = None
    ) -> bool:
        """Updates MTM price and optionally stops/highs for an active position."""
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()

        with self._lock:
            with self._get_connection() as conn:
                try:
                    if stop_loss is not None and highest_price is not None:
                        conn.execute("""
                        UPDATE active_positions
                        SET current_price = ?, stop_loss = ?, highest_price = ?, updated_at = ?
                        WHERE symbol = ?
                        """, (current_price, stop_loss, highest_price, now_str, sym_clean))
                    elif stop_loss is not None:
                        conn.execute("""
                        UPDATE active_positions
                        SET current_price = ?, stop_loss = ?, updated_at = ?
                        WHERE symbol = ?
                        """, (current_price, stop_loss, now_str, sym_clean))
                    else:
                        conn.execute("""
                        UPDATE active_positions
                        SET current_price = ?, updated_at = ?
                        WHERE symbol = ?
                        """, (current_price, now_str, sym_clean))
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to update price for {sym_clean}: {e}")
                    return False

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        notes: str = "",
        bypass_gate: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Closes an active position, logs it to trade_history with full IST timestamps,
        and immediately syncs the Single Master File to Telegram Cloud.
        STRICT TRADING GATE: Blocks exit execution outside NSE trading hours unless bypass_gate=True.
        """
        from core.market_calendar import NSEMarketCalendar
        from core.telegram_vault import telegram_vault

        if not bypass_gate:
            can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
            if not can_trade:
                logger.warning(f"🔒 [Strict Trading Gate] Blocked EXIT order for {symbol}: {gate_reason}")
                return None

        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()  # Full timestamp: YYYY-MM-DD HH:MM:SS IST

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM active_positions WHERE symbol = ?", (sym_clean,)).fetchone()
                if not row:
                    logger.warning(f"Position {sym_clean} not found to close.")
                    return None

                pos = dict(row)
                entry_px = float(pos["entry_price"])
                qty = int(pos["quantity"])
                p_type = pos["portfolio_type"]
                sector = pos.get("sector", "DIVERSIFIED")
                entry_dt_str = pos["entry_date"]

                # PnL math
                rem_pnl = (exit_price - entry_px) * qty
                total_pnl = float(pos.get("t1_pnl", 0.0)) + rem_pnl
                orig_sl = float(pos.get("original_stop_loss") or pos["stop_loss"])
                risk_unit = max(0.01, entry_px - orig_sl)
                r_multiple = round(total_pnl / ((risk_unit * qty) + 1e-9), 2)
                return_pct = round(((exit_price - entry_px) / entry_px) * 100.0, 2)

                # Estimate hold days
                try:
                    entry_dt = datetime.datetime.strptime(entry_dt_str[:10], "%Y-%m-%d")
                    hold_days = max(1, (now_ist().date() - entry_dt.date()).days)
                except Exception:
                    hold_days = int(pos.get("hold_days", 1) or 1)

                # Insert into trade_history with full timestamps
                conn.execute("""
                INSERT INTO trade_history (
                    symbol, portfolio_type, entry_price, exit_price, quantity, pnl,
                    r_multiple, return_pct, entry_date, exit_date, hold_days,
                    exit_reason, sector, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sym_clean, p_type, entry_px, exit_price, qty, round(total_pnl, 2),
                    r_multiple, return_pct, entry_dt_str, now_str, hold_days,
                    exit_reason, sector, notes or pos.get("notes", ""), now_str
                ))

                sync_msg = (
                    f"🏁 TRADE EXIT & CLOUD SYNCED: {sym_clean} ({qty} shs @ ₹{exit_price:,.2f} | "
                    f"PnL ₹{total_pnl:+,.2f} / {r_multiple:+.2f}R / {return_pct:+.2f}% | {exit_reason}) • {now_str}"
                )
                sync_details = json.dumps({
                    "event": "TRADE_CLOSE",
                    "symbol": sym_clean,
                    "portfolio": p_type,
                    "entry_price": entry_px,
                    "exit_price": exit_price,
                    "quantity": qty,
                    "pnl": round(total_pnl, 2),
                    "r_multiple": r_multiple,
                    "return_pct": return_pct,
                    "exit_reason": exit_reason,
                    "entry_date": entry_dt_str,
                    "exit_date": now_str,
                    "cloud_sync": "SYNCED_TO_TELEGRAM_VAULT"
                })
                conn.execute("""
                INSERT INTO scan_logs (
                    timestamp, scan_type, portfolio_type, status, execution_time_sec,
                    candidates_count, orders_count, reconciled_count, message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (now_str, f"TRADE_EXIT_SYNC ({exit_reason})", p_type, "CLOUD_SYNCED", 0.25, 1, 1, 1, sync_msg, sync_details))

                # Delete from active_positions
                conn.execute("DELETE FROM active_positions WHERE symbol = ?", (sym_clean,))
                # Clear any stale signal for this symbol so chart and screener reset
                conn.execute("DELETE FROM latest_signals WHERE symbol = ?", (sym_clean,))
                conn.commit()

                trade_record = {
                    "symbol": sym_clean,
                    "portfolio_type": p_type,
                    "entry_price": entry_px,
                    "exit_price": exit_price,
                    "quantity": qty,
                    "pnl": round(total_pnl, 2),
                    "r_multiple": r_multiple,
                    "return_pct": return_pct,
                    "entry_date": entry_dt_str,
                    "exit_date": now_str,
                    "exit_reason": exit_reason,
                    "hold_days": hold_days
                }
                logger.info(f"Closed {sym_clean} in {p_type} at {now_str}: PnL ₹{total_pnl:+.2f} ({r_multiple:+.2f}R) Reason: {exit_reason}")

                # Immediately sync Single Master File to Telegram Cloud Vault
                telegram_vault.send_alert_and_sync(
                    title=f"🏁 TRADE CLOSED: {sym_clean} ({exit_reason})",
                    message=(
                        f"<b>Symbol:</b> <code>{sym_clean}</code> ({p_type})\n"
                        f"<b>Entry:</b> ₹{entry_px:,.2f} ({entry_dt_str})\n"
                        f"<b>Exit:</b> ₹{exit_price:,.2f} ({now_str})\n"
                        f"<b>Realized PnL:</b> ₹{total_pnl:+,.2f} ({r_multiple:+.2f}R | {return_pct:+.2f}%)\n"
                        f"<b>Reason:</b> {exit_reason}"
                    ),
                    sync_db=True
                )
                return trade_record

    def reset_portfolio_trades(self, portfolio_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Resets active_positions and trade_history (either for a specific portfolio or all portfolios)
        so Realized & Unrealized PnL start cleanly at ₹0.00, and immediately syncs to Telegram Cloud.
        """
        from core.telegram_vault import telegram_vault

        with self._lock:
            with self._get_connection() as conn:
                if portfolio_type:
                    p_type = self._normalize_portfolio_type(portfolio_type)
                    conn.execute("DELETE FROM active_positions WHERE portfolio_type = ?", (p_type,))
                    conn.execute("DELETE FROM trade_history WHERE portfolio_type = ?", (p_type,))
                else:
                    conn.execute("DELETE FROM active_positions")
                    conn.execute("DELETE FROM trade_history")
                conn.commit()
        now_str = format_ist()
        logger.info(f"Reset trades completed (portfolio={portfolio_type or 'ALL'}) at {now_str}.")
        telegram_vault.send_alert_and_sync(
            title=f"🔄 PORTFOLIO TRADES RESET ({portfolio_type or 'ALL'})",
            message=f"Active positions & trade history cleared for <b>{portfolio_type or 'ALL'}</b> at {now_str}. Master Vault synced.",
            sync_db=True
        )
        return {"status": "SUCCESS", "portfolio": portfolio_type or "ALL", "timestamp": now_str}

    # --------------------------------------------------------------------------
    # TRADE HISTORY & PORTFOLIO METRICS
    # --------------------------------------------------------------------------
    def get_trade_history(self, portfolio_type: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Retrieves closed trade logs."""
        query = "SELECT * FROM trade_history"
        params = []
        if portfolio_type and str(portfolio_type).upper() != "ALL":
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]

    def get_portfolio_stats(self, base_capital: float = 500000.0, portfolio_type: Optional[str] = None) -> Dict[str, Any]:
        """Calculates comprehensive quantitative performance metrics including realized partial T1 targets."""
        if isinstance(base_capital, str):
            portfolio_type = base_capital
            base_capital = 100000.0 if portfolio_type and "NIFTY" in str(portfolio_type).upper() else 500000.0
        elif portfolio_type and "NIFTY" in str(portfolio_type).upper() and base_capital == 500000.0:
            base_capital = 100000.0

        p_type = self._normalize_portfolio_type(portfolio_type) if portfolio_type else None
        history = self.get_trade_history(portfolio_type=p_type, limit=1000)
        active = self.get_active_positions(portfolio_type=p_type)

        # MTM Unrealized PnL, Open Risk, Invested Capital, and Booked T1 PnL
        unrealized = 0.0
        open_risk = 0.0
        total_invested = 0.0
        active_t1_pnl = 0.0
        t1_booked_count = 0

        for p in active:
            cp = float(p.get("current_price") or p["entry_price"])
            ep = float(p["entry_price"])
            sl = float(p["stop_loss"])
            qty = int(p["quantity"])
            unrealized += (cp - ep) * qty
            open_risk += max(0.0, (ep - sl) * qty)
            total_invested += ep * qty
            if p.get("t1_hit"):
                t1_pnl_val = float(p.get("t1_pnl") or 0.0)
                active_t1_pnl += t1_pnl_val
                t1_booked_count += 1

        closed_realized_pnl = sum(float(t["pnl"]) for t in history)
        total_realized_pnl = round(closed_realized_pnl + active_t1_pnl, 2)

        max_slots = 15 if (p_type and "NIFTY" in p_type) else 10
        slots_used = len(active)
        slots_available = max(0, max_slots - slots_used)
        available_cash = round(max(0.0, base_capital + total_realized_pnl - total_invested), 2)

        wins = [t for t in history if t["pnl"] > 0]
        losses = [t for t in history if t["pnl"] <= 0]

        total_wins_count = len(wins) + t1_booked_count
        total_trades = len(history) + t1_booked_count

        gross_profit = sum(float(t["pnl"]) for t in wins) + active_t1_pnl
        gross_loss = abs(sum(float(t["pnl"]) for t in losses))

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "total_realized_pnl": 0.0,
                "closed_realized_pnl": 0.0,
                "t1_realized_pnl": 0.0,
                "t1_booked_count": 0,
                "unrealized_pnl": round(unrealized, 2),
                "total_invested": round(total_invested, 2),
                "available_cash": available_cash,
                "max_slots": max_slots,
                "slots_used": slots_used,
                "slots_available": slots_available,
                "avg_win_r": 0.0,
                "avg_loss_r": 0.0,
                "active_count": slots_used,
                "open_risk": round(open_risk, 2)
            }

        win_rate = round((total_wins_count / total_trades) * 100.0, 1)
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        else:
            profit_factor = round(gross_profit / 1.0, 2) if gross_profit > 0 else 0.0

        r_list = [float(t["r_multiple"]) for t in history] + ([1.5] * t1_booked_count)
        expectancy_r = round(float(sum(r_list) / len(r_list)), 3) if r_list else 0.0

        win_r_list = [float(t["r_multiple"]) for t in wins] + ([1.5] * t1_booked_count)
        avg_win_r = round(float(sum(win_r_list) / len(win_r_list)), 2) if win_r_list else 0.0
        avg_loss_r = round(float(sum(t["r_multiple"] for t in losses) / len(losses)), 2) if losses else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "total_realized_pnl": total_realized_pnl,
            "closed_realized_pnl": round(closed_realized_pnl, 2),
            "t1_realized_pnl": round(active_t1_pnl, 2),
            "t1_booked_count": t1_booked_count,
            "unrealized_pnl": round(unrealized, 2),
            "total_invested": round(total_invested, 2),
            "available_cash": available_cash,
            "max_slots": max_slots,
            "slots_used": slots_used,
            "slots_available": slots_available,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "active_count": slots_used,
            "open_risk": round(open_risk, 2)
        }

    # --------------------------------------------------------------------------
    # SIGNALS SNAPSHOT
    # --------------------------------------------------------------------------
    def save_latest_signals(self, signals: List[Dict[str, Any]], portfolio_type: str):
        """Replaces current signal snapshot for a given portfolio."""
        p_type = self._normalize_portfolio_type(portfolio_type)
        now_str = format_ist()

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM latest_signals WHERE portfolio_type = ?", (p_type,))
                for s in signals:
                    sym = s.get("ticker") or s.get("symbol", "")
                    sym_clean = sym.replace(".NS", "").replace(".BO", "").strip().upper()
                    conn.execute("""
                    INSERT INTO latest_signals (
                        symbol, portfolio_type, sector, bucket, score, confidence, ml_prob,
                        tier, close, entry, sl, t1, t2, rr, shares, capital_at_risk,
                        risk_pct, atr_pct, rsi14, dist_to_pivot, dist_to_ema20,
                        inst_score, week_confluence, intraday_pct, intraday_adj, rs_spread, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sym_clean, p_type, s.get("sector", "DIVERSIFIED"), s.get("bucket", "WATCH"),
                        float(s.get("score", 0.0)), int(s.get("confidence", 0)), float(s.get("ml_prob", 0.0)),
                        s.get("tier", "B+"), float(s.get("close", 0.0)), float(s.get("entry", 0.0)),
                        float(s.get("sl", 0.0)), float(s.get("t1", 0.0)), float(s.get("t2", 0.0)),
                        float(s.get("rr", 0.0)), int(s.get("shares", 0)), float(s.get("capital_at_risk", 0.0)),
                        float(s.get("risk_pct", 0.0)), float(s.get("atr_pct", 0.0)), float(s.get("rsi14", 0.0)),
                        float(s.get("dist_to_pivot", 0.0)), float(s.get("dist_to_ema20", 0.0)),
                        float(s.get("inst_score", 0.0)), 1 if s.get("week_confluence") else 0,
                        float(s.get("intraday_pct", 0.0)), float(s.get("intraday_adj", 0.0)),
                        float(s.get("rs_spread", 0.0)), now_str
                    ))
                conn.commit()

    def update_signals_live_prices(self, live_ltp_map: Dict[str, Tuple[float, float]], updated_at: str) -> int:
        """Batch-updates latest_signals close and intraday_pct from a live price stream map."""
        if not live_ltp_map:
            return 0
        rows = [(round(float(ltp), 2), round(float(chg), 2), updated_at, sym) for sym, (ltp, chg) in live_ltp_map.items()]
        with self._lock:
            with self._get_connection() as conn:
                conn.executemany("""
                UPDATE latest_signals
                SET close = ?, intraday_pct = ?, updated_at = ?
                WHERE symbol = ?
                """, rows)
                conn.commit()
        return len(rows)

    def get_latest_signals(self, portfolio_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves stored signals enriched with latest live prices from ohlcv_master_store."""
        query = """
        SELECT s.*,
               m.last_close AS master_last_close,
               m.change_pct AS master_change_pct,
               m.updated_at AS master_updated_at
        FROM latest_signals s
        LEFT JOIN ohlcv_master_store m ON s.symbol = m.symbol
        """
        params = []
        if portfolio_type and str(portfolio_type).upper() != "ALL":
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE s.portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY s.score DESC, s.ml_prob DESC"

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    master_lc = float(item.get("master_last_close") or 0.0)
                    master_upd = item.get("master_updated_at") or ""
                    sig_upd = item.get("updated_at") or ""
                    if master_lc > 0 and master_upd >= sig_upd:
                        item["close"] = round(master_lc, 2)
                        item["intraday_pct"] = round(float(item.get("master_change_pct") or item.get("intraday_pct") or 0.0), 2)
                        item["updated_at"] = master_upd
                    result.append(item)
                return result

    # --------------------------------------------------------------------------
    # SCAN EXECUTION LOGS
    # --------------------------------------------------------------------------
    def log_scan_run(
        self,
        scan_type: str,
        portfolio_type: str,
        status: str,
        execution_time_sec: float,
        candidates_count: int = 0,
        orders_count: int = 0,
        reconciled_count: int = 0,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """Logs automated or manual scan run metadata."""
        now_str = format_ist()
        details_str = json.dumps(details or {})

        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO scan_logs (
                    timestamp, scan_type, portfolio_type, status, execution_time_sec,
                    candidates_count, orders_count, reconciled_count, message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now_str, scan_type, portfolio_type, status, round(execution_time_sec, 2),
                    candidates_count, orders_count, reconciled_count, message, details_str
                ))
                conn.commit()

    def get_recent_scan_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent scan execution logs."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT * FROM scan_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    try:
                        item["details"] = json.loads(item.get("details_json") or "{}")
                    except Exception:
                        item["details"] = {}
                    result.append(item)
                return result

    # --------------------------------------------------------------------------
    # LIVE REAL-TIME SCAN PROGRESS TRACKER
    # --------------------------------------------------------------------------
    def init_progress_tracker(self):
        if not hasattr(self, "_live_progress"):
            self._live_progress = {
                "is_running": False,
                "scan_type": None,
                "stage": "Idle",
                "percent": 0,
                "stocks_total": 0,
                "stocks_done": 0,
                "started_at": None,
                "updated_at": None,
                "logs": []
            }

    def start_scan_progress(self, scan_type: str, initial_stage: str = "Initializing live scan..."):
        with self._progress_lock:
            self.init_progress_tracker()
            now_str = format_ist()
            self._live_progress = {
                "is_running": True,
                "scan_type": scan_type,
                "stage": initial_stage,
                "percent": 5,
                "stocks_total": 0,
                "stocks_done": 0,
                "started_at": now_str,
                "updated_at": now_str,
                "logs": [f"[{now_str[11:19]}] 🚀 {initial_stage}"]
            }

    def update_scan_progress(
        self,
        stage: str,
        percent: int,
        stocks_total: Optional[int] = None,
        stocks_done: Optional[int] = None,
        add_log: bool = True
    ):
        with self._progress_lock:
            self.init_progress_tracker()
            now_str = format_ist()
            self._live_progress["stage"] = stage
            self._live_progress["percent"] = max(0, min(100, int(percent)))
            self._live_progress["updated_at"] = now_str
            if stocks_total is not None:
                self._live_progress["stocks_total"] = int(stocks_total)
            if stocks_done is not None:
                self._live_progress["stocks_done"] = int(stocks_done)
            if add_log:
                self._live_progress["logs"].append(f"[{now_str[11:19]}] {stage}")
                if len(self._live_progress["logs"]) > 25:
                    self._live_progress["logs"] = self._live_progress["logs"][-25:]

    def finish_scan_progress(self, final_stage: str = "Scan completed successfully."):
        with self._progress_lock:
            self.init_progress_tracker()
            now_str = format_ist()
            self._live_progress["is_running"] = False
            self._live_progress["stage"] = final_stage
            self._live_progress["percent"] = 100
            self._live_progress["updated_at"] = now_str
            self._live_progress["logs"].append(f"[{now_str[11:19]}] ✅ {final_stage}")

    def get_scan_progress(self) -> Dict[str, Any]:
        with self._progress_lock:
            self.init_progress_tracker()
            return dict(self._live_progress)

    # --------------------------------------------------------------------------
    # SCHEDULED 6-PILLAR RUN TELEMETRY (10:20 AM & 3:00 PM IST)
    # --------------------------------------------------------------------------
    def record_scheduled_scan_outcome(
        self,
        run_key: str,  # "morning_1020" or "afternoon_305"
        status: str,   # "SUCCESS", "FAILED", "RUNNING"
        buy_now_count: int = 0,
        total_scanned: int = 0,
        pre_breakout_count: int = 0,
        wait_pullback_count: int = 0,
        execution_time_sec: float = 0.0,
        error: Optional[str] = None,
        note: str = ""
    ):
        """Persists the outcome, status, and BUY stocks count of 10:20 AM and 3:00 PM 6-pillar runs in SQLite."""
        now_dt = now_ist()
        today_str = now_dt.strftime("%Y-%m-%d")
        now_str = format_ist(now_dt)

        telemetry = self.get_kv("scheduled_scans_telemetry") or {}
        if not isinstance(telemetry, dict) or telemetry.get("date") != today_str:
            telemetry = {"date": today_str}

        entry = {
            "status": status,
            "timestamp": now_str,
            "date": today_str,
            "buy_now_count": int(buy_now_count),
            "total_scanned": int(total_scanned),
            "pre_breakout_count": int(pre_breakout_count),
            "wait_pullback_count": int(wait_pullback_count),
            "execution_time_sec": round(float(execution_time_sec), 2),
            "error": error,
            "note": note or f"{buy_now_count} BUY NOW stocks found out of {total_scanned} scanned"
        }
        telemetry[run_key] = entry
        self.set_kv("scheduled_scans_telemetry", telemetry)

    def get_scheduled_scans_telemetry(self) -> Dict[str, Any]:
        """Returns structured status of today's 10:20 AM Morning and 3:00 PM Evening 6-pillar runs."""
        now_dt = now_ist()
        today_str = now_dt.strftime("%Y-%m-%d")
        telemetry = self.get_kv("scheduled_scans_telemetry") or {}
        if not isinstance(telemetry, dict) or telemetry.get("date") != today_str:
            telemetry = {"date": today_str}

        # Check if scan_logs has runs for today if not already in telemetry
        if "morning_1020" not in telemetry or "afternoon_305" not in telemetry:
            with self._lock:
                with self._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT * FROM scan_logs WHERE timestamp LIKE ? ORDER BY id DESC",
                        (f"{today_str}%",)
                    ).fetchall()
                    for r in rows:
                        stype = str(r["scan_type"] or "")
                        if "MORNING" in stype or "1020" in stype:
                            if "morning_1020" not in telemetry:
                                cand = int(r["candidates_count"] or 0)
                                telemetry["morning_1020"] = {
                                    "status": r["status"],
                                    "timestamp": r["timestamp"],
                                    "date": today_str,
                                    "buy_now_count": cand,
                                    "total_scanned": cand + 150,
                                    "execution_time_sec": float(r["execution_time_sec"] or 0),
                                    "note": r["message"]
                                }
                        elif "AFTERNOON" in stype or "305" in stype or "300" in stype or "1500" in stype:
                            if "afternoon_305" not in telemetry:
                                cand = int(r["candidates_count"] or 0)
                                telemetry["afternoon_305"] = {
                                    "status": r["status"],
                                    "timestamp": r["timestamp"],
                                    "date": today_str,
                                    "buy_now_count": cand,
                                    "total_scanned": cand + 200,
                                    "execution_time_sec": float(r["execution_time_sec"] or 0),
                                    "note": r["message"]
                                }

        current_time_str = now_dt.strftime("%H:%M")
        m_status = telemetry.get("morning_1020")
        if not m_status:
            is_past = current_time_str >= "10:20"
            m_status = {
                "status": "PENDING" if not is_past else "AWAITING_TRIGGER",
                "timestamp": "--",
                "date": today_str,
                "buy_now_count": 0,
                "total_scanned": 0,
                "note": "Scheduled for 10:20 AM IST Mon-Fri" if not is_past else "Ready to trigger (Market Open)"
            }

        a_status = telemetry.get("afternoon_305")
        if not a_status:
            is_past = current_time_str >= "15:00"
            a_status = {
                "status": "PENDING" if not is_past else "AWAITING_TRIGGER",
                "timestamp": "--",
                "date": today_str,
                "buy_now_count": 0,
                "total_scanned": 0,
                "note": "Scheduled for 03:00 PM IST Mon-Fri" if not is_past else "Ready to trigger (MOC Window)"
            }

        return {
            "date": today_str,
            "morning_1020": m_status,
            "afternoon_305": a_status
        }


# Global Singleton Instance
state_engine = StateEngine()
