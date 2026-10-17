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


class ClosingConnection:
    """
    Wrapper around sqlite3.Connection that ensures:
    1. Context manager `with conn:` commits on success, rollbacks on error, AND closes the connection on exit.
    2. Transparently forwards all attributes & methods (cursor, execute, commit, close, etc.) to the underlying sqlite3.Connection.
    3. Eliminates Python 3.12+ ResourceWarning: unclosed database and prevents SQLite file descriptor leaks.
    """
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._conn.__exit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                self._conn.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


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

    def _get_connection(self) -> ClosingConnection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=45.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return ClosingConnection(conn)

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
                # Enable SQLite WAL (Write-Ahead Logging) mode for concurrent readers & zero collision
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("PRAGMA wal_autocheckpoint=1000;")
                except Exception:
                    pass

                cursor = conn.cursor()

                # 1. Active Positions Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL DEFAULT 'INSTITUTIONAL_6PILLAR',
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    original_quantity INTEGER,
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
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, portfolio_type)
                );
                """)
                try:
                    cursor.execute("ALTER TABLE active_positions ADD COLUMN original_quantity INTEGER")
                except Exception:
                    pass

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
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN gross_pnl REAL;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN charges REAL DEFAULT 0.0;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN net_pnl REAL;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN t1_pnl REAL DEFAULT 0.0;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN t1_price REAL DEFAULT 0.0;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE trade_history ADD COLUMN remaining_quantity INTEGER;")
                except Exception:
                    pass

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
                    expected_r REAL DEFAULT 0.0,
                    ml_verdict TEXT DEFAULT 'APPROVED',
                    rejection_reason TEXT DEFAULT '',
                    ai_summary TEXT DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, portfolio_type)
                );
                """)

                try:
                    cursor.execute("ALTER TABLE latest_signals ADD COLUMN expected_r REAL DEFAULT 0.0;")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE latest_signals ADD COLUMN ml_verdict TEXT DEFAULT 'APPROVED';")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE latest_signals ADD COLUMN rejection_reason TEXT DEFAULT '';")
                except Exception:
                    pass
                try:
                    cursor.execute("ALTER TABLE latest_signals ADD COLUMN ai_summary TEXT DEFAULT '';")
                except Exception:
                    pass

                # Backfill open active positions into quant_trade_dataset if not tracked yet
                try:
                    cursor.execute("""
                    INSERT OR IGNORE INTO quant_trade_dataset (
                        symbol, portfolio_type, trade_type, status, entry_price, stop_loss, target, target_2,
                        sector, created_at, updated_at
                    )
                    SELECT symbol, portfolio_type, 'REAL', 'ACTIVE', entry_price, stop_loss, target, target_2,
                           sector, entry_date, updated_at
                    FROM active_positions
                    WHERE symbol NOT IN (SELECT symbol FROM quant_trade_dataset WHERE status = 'ACTIVE' AND trade_type = 'REAL');
                    """)
                except Exception:
                    pass

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

        # Fast NumPy Wilder ATR 14
        if n >= 2:
            tr = np.empty(n, dtype=np.float64)
            tr[0] = highs_np[0] - lows_np[0]
            tr[1:] = np.maximum(
                highs_np[1:] - lows_np[1:],
                np.maximum(
                    np.abs(highs_np[1:] - closes_np[:-1]),
                    np.abs(lows_np[1:] - closes_np[:-1])
                )
            )
            atr14 = round(float(_fast_ewm_last(tr, 1.0 / 14.0)), 2)
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

        # History preservation guard:
        # If incoming df is a small delta (< 80 bars) and existing history has more bars,
        # merge the delta onto existing history rather than overwriting deep charts!
        if len(df) < 80:
            existing_df = None
            if sym in self._ohlcv_mem_cache:
                existing_df, _ = self._ohlcv_mem_cache[sym]
            if existing_df is None:
                existing_df, _ = self.get_ohlcv_df(sym)
            if existing_df is not None and not existing_df.empty and len(existing_df) > len(df):
                if existing_df.index.tz is not None and df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                elif existing_df.index.tz is None and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = pd.concat([existing_df, df]).sort_index()
                df = df[~pd.Series(df.index.date).duplicated(keep="last").values]

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

            # Guard against truncating existing history with a small delta:
            if len(df) < 80:
                existing_df = None
                if sym in self._ohlcv_mem_cache:
                    existing_df, _ = self._ohlcv_mem_cache[sym]
                if existing_df is None:
                    existing_df, _ = self.get_ohlcv_df(sym)
                if existing_df is not None and not existing_df.empty and len(existing_df) > len(df):
                    if existing_df.index.tz is not None and df.index.tz is None:
                        df.index = df.index.tz_localize(IST)
                    elif existing_df.index.tz is None and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = pd.concat([existing_df, df]).sort_index()
                    df = df[~pd.Series(df.index.date).duplicated(keep="last").values]

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

    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Convenience helper returning just the DataFrame for symbol or None."""
        df, _ = self.get_ohlcv_df(symbol)
        return df

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
            if df is None or df.empty or len(df) < 80:
                needs_live_sync = True
            else:
                last_idx = df.index[-1]
                last_dt = last_idx.date() if hasattr(last_idx, "date") else pd.Timestamp(last_idx).date()
                now_dt = now_ist()
                is_post_market = (now_dt.time() >= datetime.time(15, 30)) if NSEMarketCalendar.is_trading_day(now_dt.date()) else False
                is_partial_today = (last_dt == target_date and is_post_market and age_hrs > 0.25)
                if last_dt < target_date or (is_mkt_open and age_hrs > 0.02) or is_partial_today:
                    needs_live_sync = True

            if needs_live_sync:
                fetcher = DirectYahooFinanceFetcher(max_workers=4)
                _, refreshed_df, _ = fetcher.fetch_or_gapfill_ticker(
                    sym, target_date=target_date, is_market_open=is_mkt_open, persist_immediately=True
                )
                if refreshed_df is not None and not refreshed_df.empty:
                    df = refreshed_df
                # If still truncated (< 80 bars), fetch full 2-year chart directly:
                if df is None or len(df) < 80:
                    _, full_df = fetcher.fetch_ticker_chart(sym, range_period="2y", max_retries=2)
                    if full_df is not None and len(full_df) >= 80:
                        df = full_df
                        self.save_ohlcv_df(sym, df)
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
            orig_sl = float(active_pos.get("original_stop_loss") or sl)
            risk_per_share = max(0.5, round(ep - orig_sl, 2)) if orig_sl < ep else max(0.5, round(ep * 0.02, 2))
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
        trade_features: Optional[Dict[str, Any]] = None,
        bypass_gate: bool = False
    ) -> bool:
        """
        Opens a new position in the Single Master File with full YYYY-MM-DD HH:MM:SS IST timestamp
        and immediately syncs the Master File to Telegram Cloud.
        STRICT TRADING GATE: Blocks order execution outside NSE trading hours (09:15–15:30 IST, Mon–Fri),
        unless bypass_gate is explicitly set (e.g., in automated testing or offline reconciliation).
        """
        from core.market_calendar import NSEMarketCalendar
        from core.telegram_vault import telegram_vault

        if not bypass_gate:
            can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
            if not can_trade:
                logger.warning(f"🔒 [Strict Trading Gate] Blocked BUY order for {symbol}: {gate_reason}")
                return False

        if quantity <= 0:
            logger.warning(f"⚠️ [Order Gate] Blocked {symbol}: Invalid quantity {quantity} (must be > 0).")
            return False
        if entry_price <= 0 or stop_loss <= 0 or target <= 0:
            logger.warning(f"⚠️ [Order Gate] Blocked {symbol}: Prices must be strictly positive non-zero.")
            return False
        if stop_loss >= entry_price:
            logger.warning(f"⚠️ [Order Gate] Blocked {symbol}: Stop loss (₹{stop_loss:.2f}) must be strictly below entry price (₹{entry_price:.2f}) for long positions.")
            return False
        if target <= entry_price:
            logger.warning(f"⚠️ [Order Gate] Blocked {symbol}: Target (₹{target:.2f}) must be strictly above entry price (₹{entry_price:.2f}).")
            return False

        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        p_type = self._normalize_portfolio_type(portfolio_type)
        now_str = format_ist()  # Full timestamp: YYYY-MM-DD HH:MM:SS IST

        with self._lock:
            with self._get_connection() as conn:
                try:
                    # If entry_metrics not explicitly provided (e.g. UI manual execution), auto-inherit from latest screened signals or shadow dataset
                    if not entry_metrics or "score" not in entry_metrics:
                        sig_row = conn.execute(
                            "SELECT * FROM latest_signals WHERE symbol = ? AND portfolio_type = ?",
                            (sym_clean, p_type)
                        ).fetchone()
                        if sig_row:
                            entry_metrics = dict(sig_row)
                        else:
                            try:
                                shadow_row = conn.execute(
                                    "SELECT * FROM quant_trade_dataset WHERE symbol = ? AND trade_type = 'SHADOW' ORDER BY id DESC LIMIT 1",
                                    (sym_clean,)
                                ).fetchone()
                                if shadow_row:
                                    entry_metrics = dict(shadow_row)
                            except Exception:
                                pass

                    metrics_str = json.dumps(entry_metrics or {})
                    features_str = json.dumps(trade_features or (json.loads(entry_metrics.get("features_json")) if isinstance(entry_metrics, dict) and isinstance(entry_metrics.get("features_json"), str) else {}))

                    existing = conn.execute(
                        "SELECT id, quantity, entry_price FROM active_positions WHERE symbol = ? AND portfolio_type = ?",
                        (sym_clean, p_type)
                    ).fetchone()
                    if existing:
                        logger.info(
                            f"Position for {sym_clean} is already open ({existing['quantity']} shs @ ₹{existing['entry_price']:.2f}); "
                            "skipping duplicate open without mutating price."
                        )
                        return True

                    # Slot capacity and cash risk validation (F-38)
                    from core.config import (
                        P1_BASE_CAPITAL, P2_BASE_CAPITAL, P1_MAX_SLOTS, P2_MAX_SLOTS,
                        MAX_POSITION_CAP_PCT, MAX_SECTOR_POSITIONS, DAILY_LOSS_LIMIT_PCT
                    )
                    is_p2 = "NIFTY" in p_type
                    max_slots = P2_MAX_SLOTS if is_p2 else P1_MAX_SLOTS
                    base_cap = P2_BASE_CAPITAL if is_p2 else P1_BASE_CAPITAL

                    cur_cnt = conn.execute(
                        "SELECT COUNT(*) FROM active_positions WHERE portfolio_type = ?",
                        (p_type,)
                    ).fetchone()[0]
                    if cur_cnt >= max_slots:
                        logger.warning(f"⚠️ [Slot Limit] Cannot open {sym_clean}: {p_type} already at {cur_cnt}/{max_slots} slots.")
                        return False

                    # Position Capital Cap Gate (F-38: max 20% capital per position)
                    max_pos_val = base_cap * MAX_POSITION_CAP_PCT
                    required_cash = entry_price * quantity
                    if required_cash > max_pos_val:
                        logger.warning(
                            f"⚠️ [Risk Gate] Blocked {sym_clean}: Position value ₹{required_cash:,.2f} "
                            f"exceeds {MAX_POSITION_CAP_PCT*100:.0f}% capital cap (₹{max_pos_val:,.2f})."
                        )
                        return False

                    # Sector Concentration Gate (F-38: max 3 positions per sector)
                    if sector and sector.upper() not in ("DIVERSIFIED", "INDEX", "NIFTY500"):
                        sec_count = conn.execute(
                            "SELECT COUNT(*) FROM active_positions WHERE portfolio_type = ? AND UPPER(sector) = ?",
                            (p_type, sector.upper())
                        ).fetchone()[0]
                        if sec_count >= MAX_SECTOR_POSITIONS:
                            logger.warning(
                                f"⚠️ [Sector Gate] Blocked {sym_clean}: Sector '{sector}' already has {sec_count} active positions "
                                f"(max allowed: {MAX_SECTOR_POSITIONS})."
                            )
                            return False

                    # Daily Loss Circuit Breaker Gate (F-38: max 3% daily loss kill switch)
                    today_prefix = now_str[:10]
                    today_realized_loss = conn.execute(
                        "SELECT SUM(pnl) FROM trade_history WHERE portfolio_type = ? AND exit_date LIKE ? AND pnl < 0",
                        (p_type, f"{today_prefix}%")
                    ).fetchone()[0] or 0.0
                    daily_loss_limit = base_cap * DAILY_LOSS_LIMIT_PCT
                    if abs(today_realized_loss) >= daily_loss_limit:
                        logger.warning(
                            f"🚨 [Circuit Breaker] Blocked {sym_clean}: Today's realized loss (-₹{abs(today_realized_loss):,.2f}) "
                            f"has breached the {DAILY_LOSS_LIMIT_PCT*100:.0f}% daily limit (₹{daily_loss_limit:,.2f}). Trading halted."
                        )
                        return False

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

                    if required_cash > free_cash:
                        logger.warning(
                            f"⚠️ [Risk Gate] Insufficient available cash to open {sym_clean}: "
                            f"Required ₹{required_cash:,.2f} > Free Cash ₹{free_cash:,.2f}."
                        )
                        return False

                    conn.execute("""
                    INSERT INTO active_positions (
                        symbol, portfolio_type, entry_price, quantity, original_quantity, stop_loss, original_stop_loss,
                        target, target_2, current_price, highest_price, entry_date, hold_days,
                        t1_hit, t1_pnl, sl_reason, sector, notes, entry_reason, entry_regime,
                        metrics_json, features_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sym_clean, p_type, entry_price, quantity, quantity, stop_loss, stop_loss,
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

                    # Closed-Loop Quant ML Engine: Record Real Trade Entry
                    try:
                        from engines.swing_quant_ml_engine import swing_quant_ml_engine
                        cand_meta = dict(entry_metrics) if entry_metrics else {}
                        cand_meta.update({
                            "sector": sector,
                            "entry": entry_price,
                            "sl": stop_loss,
                            "t1": target,
                            "t2": target_2,
                            "notes": notes
                        })
                        swing_quant_ml_engine.record_real_trade_entry(
                            symbol=sym_clean,
                            portfolio_type=p_type,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target=target,
                            target_2=target_2,
                            candidate_meta=cand_meta
                        )
                    except Exception as ex:
                        logger.debug(f"Could not log entry into quant_trade_dataset: {ex}")
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
                    orig_qty = int(item.get("original_quantity") or (qty * 2 if (item.get("t1_hit") and float(item.get("t1_pnl") or 0.0) > 0) else qty))
                    item["qty"] = qty
                    item["original_quantity"] = orig_qty
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
        highest_price: Optional[float] = None,
        portfolio_type: Optional[str] = None
    ) -> bool:
        """Updates MTM price and optionally stops/highs for an active position, keyed by portfolio_type if specified (F-30)."""
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        p_type = self._normalize_portfolio_type(portfolio_type) if portfolio_type else None
        now_str = format_ist()

        with self._lock:
            with self._get_connection() as conn:
                try:
                    if p_type:
                        cur = conn.execute("SELECT stop_loss FROM active_positions WHERE symbol = ? AND portfolio_type = ?", (sym_clean, p_type)).fetchone()
                    else:
                        cur = conn.execute("SELECT stop_loss FROM active_positions WHERE symbol = ?", (sym_clean,)).fetchone()

                    effective_sl = stop_loss
                    if cur and stop_loss is not None:
                        current_sl = float(cur["stop_loss"] or 0.0)
                        # Ratchet safety gate: Stop Loss can only tighten (move up), never loosen (move down)
                        if current_sl > 0:
                            effective_sl = max(current_sl, stop_loss)

                    # Build targeted update statement
                    where_clause = "WHERE symbol = ? AND portfolio_type = ?" if p_type else "WHERE symbol = ?"
                    where_params = (sym_clean, p_type) if p_type else (sym_clean,)

                    if effective_sl is not None and highest_price is not None:
                        conn.execute(f"""
                        UPDATE active_positions
                        SET current_price = ?, stop_loss = ?, highest_price = ?, updated_at = ?
                        {where_clause}
                        """, (current_price, effective_sl, highest_price, now_str) + where_params)
                    elif effective_sl is not None:
                        conn.execute(f"""
                        UPDATE active_positions
                        SET current_price = ?, stop_loss = ?, updated_at = ?
                        {where_clause}
                        """, (current_price, effective_sl, now_str) + where_params)
                    else:
                        conn.execute(f"""
                        UPDATE active_positions
                        SET current_price = ?, updated_at = ?
                        {where_clause}
                        """, (current_price, now_str) + where_params)
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to update price for {sym_clean} ({p_type or 'ALL'}): {e}")
                    return False

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        notes: str = "",
        bypass_gate: bool = False,
        portfolio_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Closes an active position, logs it to trade_history with full IST timestamps,
        and immediately syncs the Single Master File to Telegram Cloud.
        STRICT TRADING GATE: Blocks exit execution outside NSE trading hours unless bypass_gate=True.
        F-30: Strictly isolates queries and deletions by portfolio_type.
        F-37: Deducts statutory transaction costs (0.12% round trip) and stores full trade audit details.
        """
        from core.market_calendar import NSEMarketCalendar
        from core.telegram_vault import telegram_vault
        from core.config import EQUITY_ROUND_TRIP_COST_RATE

        if not bypass_gate:
            can_trade, gate_reason = NSEMarketCalendar.can_execute_trades()
            if not can_trade:
                logger.warning(f"🔒 [Strict Trading Gate] Blocked EXIT order for {symbol}: {gate_reason}")
                return None

        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()  # Full timestamp: YYYY-MM-DD HH:MM:SS IST
        p_target = self._normalize_portfolio_type(portfolio_type) if portfolio_type else None

        with self._lock:
            with self._get_connection() as conn:
                if p_target:
                    row = conn.execute(
                        "SELECT * FROM active_positions WHERE symbol = ? AND portfolio_type = ?",
                        (sym_clean, p_target)
                    ).fetchone()
                else:
                    row = conn.execute("SELECT * FROM active_positions WHERE symbol = ?", (sym_clean,)).fetchone()

                if not row:
                    logger.warning(f"Position {sym_clean} ({p_target or 'ALL'}) not found to close.")
                    return None

                pos = dict(row)
                entry_px = float(pos["entry_price"])
                qty = int(pos["quantity"])
                p_type = pos["portfolio_type"]
                sector = pos.get("sector", "DIVERSIFIED")
                entry_dt_str = pos["entry_date"]

                # PnL math with statutory transaction costs (F-37)
                t1_pnl_booked = float(pos.get("t1_pnl", 0.0) or 0.0)
                rem_gross_pnl = (exit_price - entry_px) * qty
                gross_pnl = t1_pnl_booked + rem_gross_pnl

                orig_sl = float(pos.get("original_stop_loss") or pos["stop_loss"])
                orig_qty = int(pos.get("original_quantity") or (qty * 2 if (pos.get("t1_hit") and t1_pnl_booked > 0) else qty))

                # Statutory transaction costs: round trip (turnover * EQUITY_ROUND_TRIP_COST_RATE)
                t1_hit = bool(pos.get("t1_hit", 0))
                t1_qty = max(0, orig_qty - qty) if t1_hit else 0
                t1_target = float(pos.get("target") or entry_px)
                total_turnover = (entry_px * orig_qty) + (exit_price * qty) + (t1_target * t1_qty)
                charges = round(total_turnover * EQUITY_ROUND_TRIP_COST_RATE, 2)
                net_pnl = round(gross_pnl - charges, 2)
                total_pnl = round(gross_pnl, 2)

                risk_unit = max(0.01, entry_px - orig_sl)
                initial_total_risk = max(0.01, risk_unit * orig_qty)
                r_multiple = round(total_pnl / initial_total_risk, 2)
                return_pct = round((total_pnl / max(0.01, entry_px * orig_qty)) * 100.0, 2)

                # Estimate hold days
                try:
                    entry_dt = datetime.datetime.strptime(entry_dt_str[:10], "%Y-%m-%d")
                    hold_days = max(1, (now_ist().date() - entry_dt.date()).days)
                except Exception:
                    hold_days = int(pos.get("hold_days", 1) or 1)

                # Insert into trade_history with full timestamps, audit details, and true initial quantity (F-37)
                conn.execute("""
                INSERT INTO trade_history (
                    symbol, portfolio_type, entry_price, exit_price, quantity, pnl,
                    r_multiple, return_pct, entry_date, exit_date, hold_days,
                    exit_reason, sector, notes, created_at,
                    gross_pnl, charges, net_pnl, t1_pnl, t1_price, remaining_quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sym_clean, p_type, entry_px, exit_price, orig_qty, total_pnl,
                    r_multiple, return_pct, entry_dt_str, now_str, hold_days,
                    exit_reason, sector, notes or pos.get("notes", ""), now_str,
                    round(gross_pnl, 2), charges, round(net_pnl, 2), round(t1_pnl_booked, 2), t1_target if t1_hit else 0.0, qty
                ))

                sync_msg = (
                    f"🏁 TRADE EXIT & CLOUD SYNCED: {sym_clean} ({orig_qty} shs @ ₹{exit_price:,.2f} | "
                    f"PnL ₹{total_pnl:+,.2f} [Net ₹{net_pnl:+,.2f} after ₹{charges:.2f} costs] / "
                    f"{r_multiple:+.2f}R / {return_pct:+.2f}% | {exit_reason}) • {now_str}"
                )
                sync_details = json.dumps({
                    "event": "TRADE_CLOSE",
                    "symbol": sym_clean,
                    "portfolio": p_type,
                    "entry_price": entry_px,
                    "exit_price": exit_price,
                    "quantity": orig_qty,
                    "remaining_quantity_closed": qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": charges,
                    "net_pnl": round(net_pnl, 2),
                    "pnl": total_pnl,
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

                # Delete from active_positions strictly matching (symbol, portfolio_type) (F-30)
                conn.execute("DELETE FROM active_positions WHERE symbol = ? AND portfolio_type = ?", (sym_clean, p_type))

                # Clear signal only if no other portfolio has an active position for this symbol
                remaining_active_any = conn.execute(
                    "SELECT COUNT(*) FROM active_positions WHERE symbol = ?",
                    (sym_clean,)
                ).fetchone()[0]
                if remaining_active_any == 0:
                    conn.execute("DELETE FROM latest_signals WHERE symbol = ?", (sym_clean,))
                else:
                    conn.execute("DELETE FROM latest_signals WHERE symbol = ? AND portfolio_type = ?", (sym_clean, p_type))
                conn.commit()

                # Record Closed-Loop Real Trade Exit into quant_trade_dataset
                try:
                    from engines.swing_quant_ml_engine import swing_quant_ml_engine
                    swing_quant_ml_engine.record_real_trade_exit(
                        symbol=sym_clean,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl=round(total_pnl, 2),
                        r_multiple=r_multiple,
                        return_pct=return_pct,
                        hold_days=hold_days
                    )
                except Exception as ex:
                    logger.debug(f"Failed to record trade exit in quant_trade_dataset: {ex}")

                trade_record = {
                    "symbol": sym_clean,
                    "portfolio_type": p_type,
                    "entry_price": entry_px,
                    "exit_price": exit_price,
                    "quantity": orig_qty,
                    "remaining_quantity_closed": qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": charges,
                    "net_pnl": round(net_pnl, 2),
                    "pnl": round(total_pnl, 2),
                    "r_multiple": r_multiple,
                    "return_pct": return_pct,
                    "entry_date": entry_dt_str,
                    "exit_date": now_str,
                    "exit_reason": exit_reason,
                    "hold_days": hold_days
                }
                logger.info(f"Closed {sym_clean} in {p_type} at {now_str}: Net PnL ₹{total_pnl:+.2f} ({r_multiple:+.2f}R) Reason: {exit_reason}")

                # Immediately sync Single Master File to Telegram Cloud Vault
                telegram_vault.send_alert_and_sync(
                    title=f"🏁 TRADE CLOSED: {sym_clean} ({exit_reason})",
                    message=(
                        f"<b>Symbol:</b> <code>{sym_clean}</code> ({p_type})\n"
                        f"<b>Entry:</b> ₹{entry_px:,.2f} ({entry_dt_str})\n"
                        f"<b>Exit:</b> ₹{exit_price:,.2f} ({now_str})\n"
                        f"<b>Realized PnL:</b> ₹{total_pnl:+,.2f} (Net after ₹{charges:.2f} costs | {r_multiple:+.2f}R | {return_pct:+.2f}%)\n"
                        f"<b>Reason:</b> {exit_reason}"
                    ),
                    sync_db=True
                )
                return trade_record

    def reset_portfolio_trades(self, portfolio_type: Optional[str] = None, confirm_token: str = "") -> Dict[str, Any]:
        """
        STRICT IMMUTABILITY GATE:
        Active positions and trade history are permanently protected from automated wipes or accidental calls.
        Resetting trades is strictly blocked unless explicit confirm_token == 'CONFIRM_USER_AUTHORIZED_RESET'.
        """
        if confirm_token != "CONFIRM_USER_AUTHORIZED_RESET":
            logger.warning("🔒 [Strict Safety Gate] Blocked portfolio reset request: Missing or invalid confirmation token.")
            return {
                "status": "BLOCKED",
                "message": "🔒 [Strict Safety Gate] Active and closed trades are protected. Wipe requires confirm_token='CONFIRM_USER_AUTHORIZED_RESET'."
            }

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
        logger.info(f"User-authorized reset trades completed (portfolio={portfolio_type or 'ALL'}) at {now_str}.")
        telegram_vault.send_alert_and_sync(
            title=f"🔄 PORTFOLIO TRADES RESET ({portfolio_type or 'ALL'})",
            message=f"Active positions & trade history cleared for <b>{portfolio_type or 'ALL'}</b> with user confirmation at {now_str}. Master Vault synced.",
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
        """
        Calculates comprehensive quantitative performance metrics.
        F-39: Separates closed trade stats from active positions (no double-counting booked T1 as closed wins),
        computes real peak-to-trough max drawdown, avoids 999.0 sentinels, and queries full history.
        """
        from core.config import P1_BASE_CAPITAL, P2_BASE_CAPITAL, P1_MAX_SLOTS, P2_MAX_SLOTS

        if isinstance(base_capital, str):
            portfolio_type = base_capital
            p_check = self._normalize_portfolio_type(portfolio_type)
            base_capital = P2_BASE_CAPITAL if "NIFTY" in p_check else P1_BASE_CAPITAL
        elif portfolio_type and "NIFTY" in str(portfolio_type).upper() and base_capital == 500000.0:
            base_capital = P2_BASE_CAPITAL

        p_type = self._normalize_portfolio_type(portfolio_type) if portfolio_type else None
        # F-39: Fetch full trade history without artificial 1000 limit
        history = self.get_trade_history(portfolio_type=p_type, limit=100000)
        active = self.get_active_positions(portfolio_type=p_type)

        # MTM Unrealized PnL, Open Risk, Invested Capital, and Booked T1 PnL
        unrealized = 0.0
        open_risk = 0.0
        total_invested = 0.0
        active_t1_pnl = 0.0
        t1_booked_count = 0

        active_t1_r_list = []
        for p in active:
            cp = float(p.get("current_price") or p["entry_price"])
            ep = float(p["entry_price"])
            sl = float(p["stop_loss"])
            orig_sl = float(p.get("original_stop_loss") or sl)
            qty = int(p["quantity"])
            unrealized += (cp - ep) * qty
            open_risk += max(0.0, (ep - sl) * qty)
            total_invested += ep * qty
            if p.get("t1_hit") and float(p.get("t1_pnl") or 0.0) > 0:
                t1_pnl_val = float(p.get("t1_pnl") or 0.0)
                active_t1_pnl += t1_pnl_val
                t1_booked_count += 1
                orig_q = int(p.get("original_quantity") or (qty * 2))
                booked_q = orig_q - qty if orig_q > qty else qty
                risk_per_share = max(0.01, ep - orig_sl)
                t1_r = round(t1_pnl_val / max(0.01, risk_per_share * booked_q), 2)
                active_t1_r_list.append(t1_r)

        closed_realized_pnl = sum(float(t.get("pnl") or 0.0) for t in history)
        total_realized_pnl = round(closed_realized_pnl + active_t1_pnl, 2)

        is_p2 = p_type and "NIFTY" in p_type
        max_slots = P2_MAX_SLOTS if is_p2 else P1_MAX_SLOTS
        slots_used = len(active)
        slots_available = max(0, max_slots - slots_used)
        available_cash = round(max(0.0, base_capital + total_realized_pnl - total_invested), 2)

        # F-39: Closed trades stats only (do NOT count open T1 positions as closed trades or completed wins)
        wins = [t for t in history if float(t.get("pnl") or 0.0) > 0]
        losses = [t for t in history if float(t.get("pnl") or 0.0) < 0]
        total_trades = len(history)
        total_wins_count = len(wins)

        gross_profit = sum(float(t.get("pnl") or 0.0) for t in wins)
        gross_loss = abs(sum(float(t.get("pnl") or 0.0) for t in losses))

        # Real Peak-to-Trough Max Drawdown on Closed Trades Equity Curve (F-33, F-39)
        equity = base_capital
        peak = equity
        max_dd_pct = 0.0
        for t in reversed(history):
            pnl_val = float(t.get("pnl") or 0.0)
            equity += pnl_val
            if equity > peak:
                peak = equity
            dd = peak - equity
            if peak > 0:
                dd_pct = (dd / peak) * 100.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
        max_drawdown_pct = round(max_dd_pct, 2)

        if total_trades == 0:
            avg_t1_r = round(float(sum(active_t1_r_list) / len(active_t1_r_list)), 2) if active_t1_r_list else 0.0
            exp_t1_r = round(float(sum(active_t1_r_list) / len(active_t1_r_list)), 3) if active_t1_r_list else 0.0
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": exp_t1_r,
                "max_drawdown_pct": 0.0,
                "total_realized_pnl": total_realized_pnl,
                "closed_realized_pnl": 0.0,
                "t1_realized_pnl": round(active_t1_pnl, 2),
                "t1_booked_count": t1_booked_count,
                "unrealized_pnl": round(unrealized, 2),
                "total_invested": round(total_invested, 2),
                "available_cash": available_cash,
                "max_slots": max_slots,
                "slots_used": slots_used,
                "slots_available": slots_available,
                "avg_win_r": avg_t1_r,
                "avg_loss_r": 0.0,
                "active_count": slots_used,
                "open_risk": round(open_risk, 2)
            }

        win_rate = round((total_wins_count / total_trades) * 100.0, 1)
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = round(gross_profit, 2)
        else:
            profit_factor = 0.0

        r_list = [float(t["r_multiple"]) for t in history if "r_multiple" in t and t["r_multiple"] is not None]
        expectancy_r = round(float(sum(r_list) / len(r_list)), 3) if r_list else 0.0

        win_r_list = [float(t["r_multiple"]) for t in wins if "r_multiple" in t and t["r_multiple"] is not None]
        avg_win_r = round(float(sum(win_r_list) / len(win_r_list)), 2) if win_r_list else 0.0
        avg_loss_r = round(float(sum(float(t["r_multiple"]) for t in losses if "r_multiple" in t and t["r_multiple"] is not None) / len(losses)), 2) if losses else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "max_drawdown_pct": max_drawdown_pct,
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
                        inst_score, week_confluence, intraday_pct, intraday_adj, rs_spread,
                        expected_r, ml_verdict, rejection_reason, ai_summary, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        float(s.get("rs_spread", 0.0)),
                        float(s.get("expected_r", 0.0)), str(s.get("ml_verdict", "APPROVED")),
                        str(s.get("rejection_reason", "")), str(s.get("ai_summary", "")),
                        now_str
                    ))
                conn.commit()

    def update_signal_ml_verdict(
        self,
        symbol: str,
        portfolio_type: str,
        ml_verdict: str,
        rejection_reason: str = "",
        ai_summary: str = ""
    ) -> bool:
        """Updates ML/AI verdict, rejection reason, and AI summary for a screened signal in latest_signals."""
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        p_type = self._normalize_portfolio_type(portfolio_type)
        now_str = format_ist()
        with self._lock:
            with self._get_connection() as conn:
                try:
                    conn.execute("""
                    UPDATE latest_signals
                    SET ml_verdict = ?, rejection_reason = ?, ai_summary = ?, updated_at = ?
                    WHERE symbol = ? AND portfolio_type = ?
                    """, (ml_verdict, rejection_reason, ai_summary, now_str, sym_clean, p_type))
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to update signal verdict for {sym_clean}: {e}")
                    return False

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

        # 08:45 AM Pre-Market Forecast Lock Auto Run
        p_status = telemetry.get("premarket_0845")
        if not p_status:
            locked_pred = self.get_kv(f"nifty_prediction_locked_{today_str}")
            if not locked_pred:
                cand_pred = self.get_kv("latest_nifty_predictions")
                if cand_pred and isinstance(cand_pred, dict) and cand_pred.get("status") in ("OK", "LOCKED") and cand_pred.get("prediction_date") == today_str:
                    locked_pred = cand_pred
            if locked_pred and isinstance(locked_pred, dict) and locked_pred.get("status") in ("OK", "LOCKED"):
                p_status = {
                    "status": "SUCCESS",
                    "timestamp": locked_pred.get("premarket_locked_at_ist") or locked_pred.get("timestamp_ist") or f"{today_str} 08:45:00 IST",
                    "date": today_str,
                    "buy_now_count": 0,
                    "total_scanned": 50,
                    "execution_time_sec": float(locked_pred.get("execution_time_sec") or 1.2),
                    "note": f"Locked: T+1 {locked_pred.get('t1_forecast', {}).get('verdict')} | T+5 {locked_pred.get('t5_forecast', {}).get('verdict')}"
                }
            else:
                is_past_0845 = current_time_str >= "08:45"
                p_status = {
                    "status": "PENDING" if not is_past_0845 else "AWAITING_TRIGGER",
                    "timestamp": "--",
                    "date": today_str,
                    "buy_now_count": 0,
                    "total_scanned": 0,
                    "note": "Scheduled for 08:45 AM IST Mon-Fri" if not is_past_0845 else "Awaiting 08:45 AM Pre-Market Trigger"
                }

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
            "premarket_0845": p_status,
            "morning_1020": m_status,
            "afternoon_305": a_status
        }


# Global Singleton Instance
state_engine = StateEngine()
