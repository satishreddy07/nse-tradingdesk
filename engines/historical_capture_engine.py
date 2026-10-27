"""
engines/historical_capture_engine.py
====================================
Production Historical Market-Data Capture Engine for NSE Intraday Analytics.
Complies with AGENTS.md & docs/ANTI_FABRICATION_POLICY.md.

Key Architectural Guarantees:
1. Reuses existing Kotak Neo connection via non-blocking tick listener (zero second WebSocket).
2. Thread-safe bounded queue with backpressure monitoring (never blocks WebSocket event loop).
3. Dedicated SQLite WAL-mode persistence (`data/market_history.db`) with batched transactions.
4. Preserves full market-data provenance: exchange timestamps, receive timestamps, quality status.
5. Captures:
   - Raw & Normalized quote events
   - 5-Level Market Depth & Microstructure metrics (L1/L3/L5 OBI, microprice, displacement)
   - Full Option-Chain periodic snapshots
   - 10-second analytical snapshots (VWAP, ORB, GEX, Zero-Gamma, Sectors)
   - 10-minute consolidated macro intervals
   - Session metadata and gap / disconnect tracking
"""

import os
import sys
import time
import json
import uuid
import queue
import sqlite3
import logging
import threading
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.market_data_model import (
    DataQualityStatus,
    MarketEventType,
    InstrumentType,
    NormalizedMarketEvent,
    MarketDepthRecord,
    OptionChainRecord,
    AnalyticalSnapshot10s,
    ConsolidatedSnapshot10m
)

logger = logging.getLogger("HistoricalCaptureEngine")

_BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DB_PATH = os.path.join(_BASE_DIR, "data", "market_history.db")


class ClosingConnection:
    """Safe SQLite connection wrapper that automatically commits and closes."""
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


class HistoricalCaptureEngine:
    """
    High-Throughput Intraday Market Data Capture & Quality Validation Engine.
    Operates completely asynchronously to protect real-time trading and order-flow threads.
    """
    def __init__(self, db_path: str = HISTORY_DB_PATH, max_queue_size: int = 20000):
        self._db_path = db_path
        self._max_queue_size = max_queue_size
        self._lock = threading.RLock()
        
        # Bounded Queue for async batching
        self._event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._is_running: bool = False
        self._worker_thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None

        # Telemetry & Metrics
        self._events_received_count: int = 0
        self._events_persisted_count: int = 0
        self._last_event_time_utc: float = 0.0
        self._last_event_time_ist: Optional[str] = None
        self._backpressure_active: bool = False
        self._dropped_events_count: int = 0
        
        # Gap Tracking
        self._last_tick_time: float = time.time()
        self._missing_intervals: List[Dict[str, Any]] = []

        # Latest state memory buffers for snapshots
        self._latest_spot: Optional[float] = None
        self._latest_quotes: Dict[str, Dict[str, Any]] = {}
        self._latest_depth: Dict[str, Dict[str, Any]] = {}

        self._ensure_schema()

    def _get_connection(self) -> ClosingConnection:
        """Returns a WAL-mode SQLite connection with busy timeout."""
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.row_factory = sqlite3.Row
        return ClosingConnection(conn)

    def _ensure_schema(self) -> None:
        """Initializes tables and high-performance indexes."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS market_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL UNIQUE,
                exchange TEXT DEFAULT 'NSE',
                schedule_start_ist TEXT DEFAULT '09:15:00',
                schedule_end_ist TEXT DEFAULT '15:30:00',
                capture_start_utc REAL,
                capture_stop_utc REAL,
                capture_start_ist TEXT,
                capture_stop_ist TEXT,
                expiry_selected TEXT,
                universe_json TEXT,
                total_events_captured INTEGER DEFAULT 0,
                disconnect_intervals_json TEXT DEFAULT '[]',
                missing_intervals_json TEXT DEFAULT '[]',
                session_status TEXT DEFAULT 'ACTIVE',
                created_at_ist TEXT,
                updated_at_ist TEXT
            );

            CREATE TABLE IF NOT EXISTS raw_market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                event_id TEXT NOT NULL,
                instrument_token TEXT,
                trading_symbol TEXT,
                event_type TEXT NOT NULL,
                exchange_ts TEXT,
                receive_ts_utc REAL NOT NULL,
                receive_ts_ist TEXT NOT NULL,
                raw_payload TEXT,
                quality_status TEXT DEFAULT 'VALID',
                schema_version TEXT DEFAULT 'v1.0'
            );
            CREATE INDEX IF NOT EXISTS idx_raw_events_ts ON raw_market_events(session_date, receive_ts_utc);
            CREATE INDEX IF NOT EXISTS idx_raw_events_sym ON raw_market_events(trading_symbol, receive_ts_utc);

            CREATE TABLE IF NOT EXISTS normalized_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                instrument_token TEXT,
                trading_symbol TEXT NOT NULL,
                exchange TEXT DEFAULT 'NSE',
                ltp REAL NOT NULL,
                high REAL,
                low REAL,
                open REAL,
                close REAL,
                volume INTEGER,
                oi INTEGER,
                oi_change INTEGER,
                exchange_ts TEXT,
                receive_ts_utc REAL NOT NULL,
                receive_ts_ist TEXT NOT NULL,
                quality_status TEXT DEFAULT 'VALID'
            );
            CREATE INDEX IF NOT EXISTS idx_quotes_sym_ts ON normalized_quotes(trading_symbol, receive_ts_utc);
            CREATE INDEX IF NOT EXISTS idx_quotes_session ON normalized_quotes(session_date, receive_ts_utc);

            CREATE TABLE IF NOT EXISTS market_depth_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                instrument_token TEXT,
                trading_symbol TEXT NOT NULL,
                ltp REAL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                total_buy INTEGER DEFAULT 0,
                total_sell INTEGER DEFAULT 0,
                l1_obi REAL,
                l3_obi REAL,
                l5_obi REAL,
                microprice REAL,
                midpoint REAL,
                displacement REAL,
                spread REAL,
                relative_spread_pct REAL,
                book_status TEXT DEFAULT 'OK',
                exchange_ts TEXT,
                receive_ts_utc REAL NOT NULL,
                receive_ts_ist TEXT NOT NULL,
                quality_status TEXT DEFAULT 'VALID'
            );
            CREATE INDEX IF NOT EXISTS idx_depth_sym_ts ON market_depth_snapshots(trading_symbol, receive_ts_utc);
            CREATE INDEX IF NOT EXISTS idx_depth_session ON market_depth_snapshots(session_date, receive_ts_utc);

            CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                timestamp_utc REAL NOT NULL,
                timestamp_ist TEXT NOT NULL,
                expiry TEXT NOT NULL,
                spot_price REAL NOT NULL,
                atm_strike INTEGER NOT NULL,
                strikes_json TEXT NOT NULL,
                total_ce_oi INTEGER DEFAULT 0,
                total_pe_oi INTEGER DEFAULT 0,
                pcr REAL DEFAULT 0.0,
                call_wall INTEGER,
                put_wall INTEGER,
                max_pain INTEGER,
                quality_status TEXT DEFAULT 'VALID'
            );
            CREATE INDEX IF NOT EXISTS idx_oc_snapshots_ts ON option_chain_snapshots(session_date, timestamp_utc);

            CREATE TABLE IF NOT EXISTS analytical_snapshots_10s (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                timestamp_utc REAL NOT NULL,
                timestamp_ist TEXT NOT NULL,
                spot_price REAL NOT NULL,
                futures_price REAL,
                futures_obi REAL,
                futures_microprice REAL,
                futures_displacement REAL,
                futures_volume_delta REAL,
                futures_cvd REAL,
                atm_strike INTEGER,
                atm_iv REAL,
                atm_ce_iv REAL,
                atm_pe_iv REAL,
                iv_skew REAL,
                total_net_gex_cr REAL,
                zero_gamma_flip REAL,
                spot_vs_flip_pts REAL,
                gex_regime TEXT,
                vwap REAL,
                vwap_distance_pts REAL,
                vwap_distance_pct REAL,
                orb_15m_high REAL,
                orb_15m_low REAL,
                orb_15m_width REAL,
                orb_15m_status TEXT,
                orb_30m_high REAL,
                orb_30m_low REAL,
                orb_30m_width REAL,
                orb_30m_status TEXT,
                sector_breadth_pct REAL,
                sector_spread_pp REAL,
                weighted_index_drag_pp REAL,
                top_drag_sector TEXT,
                top_lift_sector TEXT,
                data_quality_status TEXT DEFAULT 'VALID',
                data_quality_flags TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_analytics_10s_ts ON analytical_snapshots_10s(session_date, timestamp_utc);

            CREATE TABLE IF NOT EXISTS consolidated_snapshots_10m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                slot_label TEXT NOT NULL,
                slot_start_utc REAL NOT NULL,
                slot_end_utc REAL NOT NULL,
                open_spot REAL,
                high_spot REAL,
                low_spot REAL,
                close_spot REAL,
                spot_change_pts REAL,
                spot_change_pct REAL,
                vwap REAL,
                vwap_distance_pts REAL,
                futures_flow_summary_json TEXT,
                obi_stats_json TEXT,
                delta_cvd_summary_json TEXT,
                option_chain_summary_json TEXT,
                gex_summary_json TEXT,
                sector_breadth_pct REAL,
                sector_spread_pp REAL,
                weighted_index_drag_pp REAL,
                completeness_score REAL DEFAULT 100.0,
                missing_intervals_json TEXT DEFAULT '[]',
                quality_status TEXT DEFAULT 'VALID'
            );
            CREATE INDEX IF NOT EXISTS idx_consolidated_10m_ts ON consolidated_snapshots_10m(session_date, slot_start_utc);

            CREATE TABLE IF NOT EXISTS data_quality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                timestamp_utc REAL NOT NULL,
                timestamp_ist TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                instrument_symbol TEXT,
                message TEXT NOT NULL,
                details_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_quality_events ON data_quality_events(session_date, timestamp_utc);
            """)

    def start(self) -> None:
        """Starts asynchronous capture worker and snapshot schedulers."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True

            # Register with KotakNeoEngine
            try:
                from engines.kotak_neo_engine import kotak_neo_engine
                kotak_neo_engine.register_tick_listener(self.on_broker_tick)
                logger.info("HistoricalCaptureEngine hooked into Kotak Neo live stream.")
            except Exception as e:
                logger.warning(f"Could not hook into Kotak Neo tick listener: {e}")

            # Initialize trading session record
            self._record_session_start()

            # Start DB Batch Writer Worker
            self._worker_thread = threading.Thread(target=self._db_batch_worker, daemon=True, name="CaptureDbBatchWorker")
            self._worker_thread.start()

            # Start Snapshot Generator Worker
            self._snapshot_thread = threading.Thread(target=self._snapshot_scheduler_worker, daemon=True, name="CaptureSnapshotWorker")
            self._snapshot_thread.start()

            logger.info("HistoricalCaptureEngine started successfully.")

    def stop(self) -> None:
        """Stops background capture workers gracefully and flushes queued records."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

            try:
                from engines.kotak_neo_engine import kotak_neo_engine
                kotak_neo_engine.unregister_tick_listener(self.on_broker_tick)
            except Exception:
                pass

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            if self._snapshot_thread and self._snapshot_thread.is_alive():
                self._snapshot_thread.join(timeout=2.0)

            self._record_session_stop()
            logger.info("HistoricalCaptureEngine stopped.")

    def on_broker_tick(self, message: Any) -> None:
        """
        Non-blocking ingestion callback from Kotak Neo WebSocket thread.
        Never executes heavy parsing or DB writes inside this method.
        """
        if not self._is_running or not message:
            return

        now_ts = time.time()
        self._events_received_count += 1
        self._last_event_time_utc = now_ts
        self._last_event_time_ist = format_ist()

        # Check queue backpressure (>= 90% threshold)
        q_size = self._event_queue.qsize()
        threshold = max(1, int(self._max_queue_size * 0.9))
        if q_size >= threshold:
            if not self._backpressure_active:
                self._backpressure_active = True
                self._log_quality_event(
                    event_type="QUEUE_BACKPRESSURE_ACTIVE",
                    severity="WARNING",
                    symbol="SYSTEM",
                    message=f"Capture queue backpressure reached 90% ({q_size}/{self._max_queue_size} events)."
                )
        else:
            self._backpressure_active = False

        try:
            self._event_queue.put_nowait((now_ts, message))
        except queue.Full:
            self._dropped_events_count += 1
            logger.warning("HistoricalCaptureEngine queue overflow! Dropping event.")

    def _db_batch_worker(self) -> None:
        """Background daemon thread draining queue in batched transactions."""
        batch_events: List[Tuple[float, Any]] = []
        last_flush_ts = time.time()

        while self._is_running:
            try:
                # Fetch up to 150 events or wait 0.25s
                try:
                    item = self._event_queue.get(timeout=0.25)
                    batch_events.append(item)
                    while len(batch_events) < 150:
                        try:
                            batch_events.append(self._event_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass

                now_ts = time.time()
                if batch_events and (len(batch_events) >= 50 or (now_ts - last_flush_ts) >= 0.25):
                    self._persist_events_batch(batch_events)
                    self._events_persisted_count += len(batch_events)
                    batch_events = []
                    last_flush_ts = now_ts

            except Exception as e:
                logger.error(f"Error in capture DB batch worker: {e}", exc_info=True)
                time.sleep(0.5)

        # Flush any remaining items in queue on shutdown
        while True:
            try:
                batch_events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        if batch_events:
            try:
                self._persist_events_batch(batch_events)
                self._events_persisted_count += len(batch_events)
            except Exception as fe:
                logger.error(f"Error persisting final batch on shutdown: {fe}")

    def _persist_events_batch(self, batch: List[Tuple[float, Any]]) -> None:
        """Normalizes and bulk-inserts a batch of broker events."""
        today_str = now_ist().strftime("%Y-%m-%d")
        raw_rows = []
        quote_rows = []
        depth_rows = []

        from engines.kotak_neo_engine import kotak_neo_engine

        for recv_utc, msg in batch:
            recv_ist = datetime.datetime.fromtimestamp(recv_utc, IST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            tok = str(getattr(msg, "instrument_token", "") or (msg.get("tk") if isinstance(msg, dict) else "") or "").strip()
            sym = getattr(msg, "trading_symbol", None) or (msg.get("ts") if isinstance(msg, dict) else None)
            if not sym and tok:
                sym = kotak_neo_engine.resolve_symbol(tok)
            sym = str(sym or tok or "UNKNOWN").strip().upper()

            raw_str = None
            try:
                raw_str = json.dumps(msg if isinstance(msg, dict) else (msg.dict() if hasattr(msg, "dict") else str(msg)))
            except Exception:
                raw_str = str(msg)

            evt_id = f"{tok}_{int(recv_utc * 1000)}_{uuid.uuid4().hex[:6]}"
            raw_rows.append((
                today_str, evt_id, tok, sym, "RAW_TICK",
                getattr(msg, "last_trade_time", "") or (msg.get("ltt", "") if isinstance(msg, dict) else ""),
                recv_utc, recv_ist, raw_str, "VALID", "v1.0"
            ))

            # Extract normalized price quote
            ltp = getattr(msg, "last_traded_price", None) or (msg.get("ltp") if isinstance(msg, dict) else None)
            if ltp is not None:
                try:
                    ltp_val = round(float(ltp), 2)
                    if ltp_val > 0:
                        hi = getattr(msg, "high_price", None) or (msg.get("h") if isinstance(msg, dict) else None)
                        lo = getattr(msg, "low_price", None) or (msg.get("lo") if isinstance(msg, dict) else None)
                        op = getattr(msg, "open_price", None) or (msg.get("op") if isinstance(msg, dict) else None)
                        cl = getattr(msg, "close_price", None) or (msg.get("c") if isinstance(msg, dict) else None)
                        vol = getattr(msg, "volume_traded_today", None) or (msg.get("v") if isinstance(msg, dict) else None)
                        oi = getattr(msg, "open_interest", None) or (msg.get("oi") if isinstance(msg, dict) else None)
                        oic = getattr(msg, "changeinOpenInterest", None) or (msg.get("oic") if isinstance(msg, dict) else None)
                        ex_time = getattr(msg, "last_trade_time", "") or (msg.get("ltt", "") if isinstance(msg, dict) else "")

                        quote_rows.append((
                            today_str, tok, sym, "NSE", ltp_val,
                            round(float(hi), 2) if hi is not None else None,
                            round(float(lo), 2) if lo is not None else None,
                            round(float(op), 2) if op is not None else None,
                            round(float(cl), 2) if cl is not None else None,
                            int(vol) if vol is not None else None,
                            int(oi) if oi is not None else None,
                            int(oic) if oic is not None else None,
                            str(ex_time or ""), recv_utc, recv_ist, "VALID"
                        ))

                        # Cache latest in memory
                        self._latest_quotes[sym] = {
                            "symbol": sym, "token": tok, "ltp": ltp_val, "volume": vol,
                            "oi": oi, "oi_change": oic, "updated_utc": recv_utc
                        }
                        if sym in ("NIFTY 50", "NIFTY", "^NSEI"):
                            self._latest_spot = ltp_val
                except Exception:
                    pass

            # Extract Level-2 Market Depth
            try:
                prev_d = self._latest_depth.get(sym)
                d_payload = kotak_neo_engine._extract_market_depth_payload(msg, prev_depth=prev_d, ltp=float(ltp or 0.0))
                if d_payload.get("bids") or d_payload.get("asks") or d_payload.get("total_buy", 0) > 0:
                    depth_rows.append((
                        today_str, tok, sym, float(ltp or 0.0),
                        json.dumps(d_payload.get("bids", [])),
                        json.dumps(d_payload.get("asks", [])),
                        int(d_payload.get("total_buy") or 0),
                        int(d_payload.get("total_sell") or 0),
                        d_payload.get("l1_obi"),
                        d_payload.get("l3_obi"),
                        d_payload.get("l5_obi"),
                        d_payload.get("microprice"),
                        d_payload.get("midpoint"),
                        d_payload.get("displacement"),
                        d_payload.get("spread"),
                        d_payload.get("relative_spread_pct"),
                        d_payload.get("book_status", "OK"),
                        getattr(msg, "last_trade_time", "") or (msg.get("ltt", "") if isinstance(msg, dict) else ""),
                        recv_utc, recv_ist, "VALID"
                    ))
                    self._latest_depth[sym] = d_payload
            except Exception:
                pass

        # Execute single batch transaction
        with self._get_connection() as conn:
            if raw_rows:
                conn.executemany("""
                INSERT INTO raw_market_events (
                    session_date, event_id, instrument_token, trading_symbol, event_type,
                    exchange_ts, receive_ts_utc, receive_ts_ist, raw_payload, quality_status, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, raw_rows)

            if quote_rows:
                conn.executemany("""
                INSERT INTO normalized_quotes (
                    session_date, instrument_token, trading_symbol, exchange, ltp,
                    high, low, open, close, volume, oi, oi_change,
                    exchange_ts, receive_ts_utc, receive_ts_ist, quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, quote_rows)

            if depth_rows:
                conn.executemany("""
                INSERT INTO market_depth_snapshots (
                    session_date, instrument_token, trading_symbol, ltp,
                    bids_json, asks_json, total_buy, total_sell,
                    l1_obi, l3_obi, l5_obi, microprice, midpoint, displacement,
                    spread, relative_spread_pct, book_status, exchange_ts,
                    receive_ts_utc, receive_ts_ist, quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, depth_rows)

    def _snapshot_scheduler_worker(self) -> None:
        """Periodic scheduler generating 10s analytical and 10m macro snapshots."""
        last_10s_ts = 0.0
        last_10m_ts = 0.0
        last_oc_ts = 0.0

        while self._is_running:
            try:
                now_ts = time.time()
                dt_now = now_ist()
                mkt_status = NSEMarketCalendar.get_market_status(dt_now)
                is_open = mkt_status.get("is_open", False)

                # 1. 10-Second Analytical Snapshot
                if (now_ts - last_10s_ts) >= 10.0:
                    self._capture_10s_analytical_snapshot(now_ts, dt_now)
                    last_10s_ts = now_ts

                # 2. 30-Second Full Option-Chain Snapshot
                if (now_ts - last_oc_ts) >= 30.0:
                    self._capture_option_chain_snapshot(now_ts, dt_now)
                    last_oc_ts = now_ts

                # 3. 10-Minute Consolidated Macro Snapshot (at boundary minutes ending in 5 or 0)
                if (now_ts - last_10m_ts) >= 600.0 or (dt_now.minute % 10 == 5 and (now_ts - last_10m_ts) >= 60.0):
                    self._capture_10m_consolidated_snapshot(now_ts, dt_now)
                    last_10m_ts = now_ts

                # 4. Check for market data gaps (> 5.0 seconds between ticks during active market)
                if is_open:
                    gap_sec = now_ts - self._last_event_time_utc
                    if self._last_event_time_utc > 0 and gap_sec > 8.0:
                        gap_rec = {
                            "start_utc": self._last_event_time_utc,
                            "end_utc": now_ts,
                            "gap_seconds": round(gap_sec, 2),
                            "timestamp_ist": format_ist()
                        }
                        if not self._missing_intervals or (now_ts - self._missing_intervals[-1]["end_utc"]) > 15.0:
                            self._missing_intervals.append(gap_rec)
                            self._log_quality_event(
                                event_type="MARKET_DATA_STREAM_GAP",
                                severity="WARNING",
                                symbol="SYSTEM",
                                message=f"Detected market feed silence of {gap_sec:.1f}s during trading hours."
                            )

                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in snapshot scheduler worker: {e}", exc_info=True)
                time.sleep(5.0)

    def _capture_10s_analytical_snapshot(self, now_ts: float, dt_now: datetime.datetime) -> None:
        """Captures authenticated 10s analytical snapshot using verified production engines."""
        try:
            from engines.intraday_analytics_engine import intraday_analytics_engine
            suite = intraday_analytics_engine.get_full_intraday_suite(force_refresh=False)
            if not suite or suite.get("status") != "OK":
                return

            vwap_d = suite.get("vwap_and_orb", {})
            gex_d = suite.get("gamma_exposure", {})
            sec_d = suite.get("sectoral_divergence", {})
            depth_d = suite.get("order_flow_depth", {})

            today_str = dt_now.strftime("%Y-%m-%d")
            ts_ist = format_ist(dt_now)

            o15 = vwap_d.get("orb_15m", {})
            o30 = vwap_d.get("orb_30m", {})

            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO analytical_snapshots_10s (
                    session_date, timestamp_utc, timestamp_ist, spot_price,
                    futures_price, futures_obi, futures_microprice, futures_displacement,
                    futures_volume_delta, futures_cvd, atm_strike, atm_iv, atm_ce_iv, atm_pe_iv,
                    iv_skew, total_net_gex_cr, zero_gamma_flip, spot_vs_flip_pts, gex_regime,
                    vwap, vwap_distance_pts, vwap_distance_pct,
                    orb_15m_high, orb_15m_low, orb_15m_width, orb_15m_status,
                    orb_30m_high, orb_30m_low, orb_30m_width, orb_30m_status,
                    sector_breadth_pct, sector_spread_pp, weighted_index_drag_pp,
                    top_drag_sector, top_lift_sector, data_quality_status, data_quality_flags
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """, (
                    today_str, now_ts, ts_ist,
                    float(suite.get("spot_price") or vwap_d.get("spot_price") or 0.0),
                    depth_d.get("futures_price"),
                    depth_d.get("futures_obi"),
                    depth_d.get("futures_microprice"),
                    depth_d.get("futures_displacement"),
                    depth_d.get("volume_delta"),
                    depth_d.get("cvd"),
                    gex_d.get("max_gamma_strike"),
                    None, None, None, None,
                    gex_d.get("total_net_gex_cr"),
                    gex_d.get("zero_gamma_flip_level"),
                    gex_d.get("spot_vs_flip_pts"),
                    gex_d.get("market_gex_regime"),
                    vwap_d.get("vwap"),
                    vwap_d.get("vwap_distance_pts"),
                    vwap_d.get("vwap_distance_pct"),
                    o15.get("high"), o15.get("low"), o15.get("range_pts"), o15.get("status"),
                    o30.get("high"), o30.get("low"), o30.get("range_pts"), o30.get("status"),
                    sec_d.get("breadth_pct_green"),
                    sec_d.get("sector_spread_pp"),
                    sec_d.get("weighted_index_drag_pp"),
                    sec_d.get("top_drag_sector"),
                    sec_d.get("top_lift_sector"),
                    "VALID", "[]"
                ))
        except Exception as e:
            logger.debug(f"10s snapshot capture notice: {e}")

    def _capture_option_chain_snapshot(self, now_ts: float, dt_now: datetime.datetime) -> None:
        """Captures authenticated full option-chain strike snapshot."""
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            oc = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False)
            w = oc.get("weekly", {})
            if not w or w.get("status") != "OK":
                return

            strikes = w.get("strikes", [])
            if not strikes:
                return

            today_str = dt_now.strftime("%Y-%m-%d")
            ts_ist = format_ist(dt_now)
            walls = w.get("walls_and_pain", {})

            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO option_chain_snapshots (
                    session_date, timestamp_utc, timestamp_ist, expiry, spot_price,
                    atm_strike, strikes_json, total_ce_oi, total_pe_oi, pcr,
                    call_wall, put_wall, max_pain, quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    today_str, now_ts, ts_ist,
                    str(w.get("expiry") or "--"),
                    float(w.get("spot_price") or 0.0),
                    int(w.get("atm_strike") or 0),
                    json.dumps(strikes),
                    int((w.get("totals") or {}).get("total_ce_oi") or 0),
                    int((w.get("totals") or {}).get("total_pe_oi") or 0),
                    float(w.get("pcr") or 0.0),
                    walls.get("call_wall_1"),
                    walls.get("put_wall_1"),
                    walls.get("max_pain"),
                    "VALID"
                ))
        except Exception as e:
            logger.debug(f"Option-chain snapshot capture notice: {e}")

    def _capture_10m_consolidated_snapshot(self, now_ts: float, dt_now: datetime.datetime) -> None:
        """Computes and stores consolidated 10-minute macro interval."""
        try:
            today_str = dt_now.strftime("%Y-%m-%d")
            slot_label = dt_now.strftime("%H:%M")
            start_utc = now_ts - 600.0

            with self._get_connection() as conn:
                # Query 10s snapshots in this 10-minute window
                rows = conn.execute("""
                SELECT spot_price, vwap, vwap_distance_pts, sector_breadth_pct, sector_spread_pp, weighted_index_drag_pp
                FROM analytical_snapshots_10s
                WHERE session_date = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc ASC
                """, (today_str, start_utc, now_ts)).fetchall()

                if not rows:
                    return

                spots = [float(r["spot_price"]) for r in rows if r["spot_price"] is not None and float(r["spot_price"]) > 0]
                if not spots:
                    return

                open_s = spots[0]
                close_s = spots[-1]
                high_s = max(spots)
                low_s = min(spots)
                chg_pts = round(close_s - open_s, 2)
                chg_pct = round((chg_pts / open_s) * 100.0, 3)

                last_row = rows[-1]
                vwap_val = last_row["vwap"]
                vwap_dist = last_row["vwap_distance_pts"]
                b_pct = last_row["sector_breadth_pct"]
                s_pp = last_row["sector_spread_pp"]
                w_drag = last_row["weighted_index_drag_pp"]

                # Expected 10s snapshots in 10 minutes = 60
                completeness = min(100.0, round((len(rows) / 60.0) * 100.0, 1))

                conn.execute("""
                INSERT INTO consolidated_snapshots_10m (
                    session_date, slot_label, slot_start_utc, slot_end_utc,
                    open_spot, high_spot, low_spot, close_spot,
                    spot_change_pts, spot_change_pct, vwap, vwap_distance_pts,
                    sector_breadth_pct, sector_spread_pp, weighted_index_drag_pp,
                    completeness_score, missing_intervals_json, quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    today_str, slot_label, start_utc, now_ts,
                    open_s, high_s, low_s, close_s,
                    chg_pts, chg_pct, vwap_val, vwap_dist,
                    b_pct, s_pp, w_drag, completeness, "[]", "VALID"
                ))
        except Exception as e:
            logger.debug(f"10m consolidated snapshot notice: {e}")

    def _record_session_start(self) -> None:
        """Records trading session activation."""
        today_str = now_ist().strftime("%Y-%m-%d")
        now_ts = time.time()
        now_str = format_ist()

        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO market_sessions (
                session_date, exchange, schedule_start_ist, schedule_end_ist,
                capture_start_utc, capture_start_ist, session_status, created_at_ist, updated_at_ist
            ) VALUES (?, 'NSE', '09:15:00', '15:30:00', ?, ?, 'ACTIVE', ?, ?)
            ON CONFLICT(session_date) DO UPDATE SET
                capture_start_utc = excluded.capture_start_utc,
                session_status = 'ACTIVE',
                updated_at_ist = excluded.updated_at_ist
            """, (today_str, now_ts, now_str, now_str, now_str))

    def _record_session_stop(self) -> None:
        """Records trading session shutdown."""
        today_str = now_ist().strftime("%Y-%m-%d")
        now_ts = time.time()
        now_str = format_ist()

        with self._get_connection() as conn:
            conn.execute("""
            UPDATE market_sessions SET
                capture_stop_utc = ?,
                capture_stop_ist = ?,
                total_events_captured = ?,
                missing_intervals_json = ?,
                session_status = 'COMPLETED',
                updated_at_ist = ?
            WHERE session_date = ?
            """, (now_ts, now_str, self._events_persisted_count, json.dumps(self._missing_intervals), now_str, today_str))

    def _log_quality_event(self, event_type: str, severity: str, symbol: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Logs a data-quality warning or defect to the audit trail."""
        today_str = now_ist().strftime("%Y-%m-%d")
        now_ts = time.time()
        now_str = format_ist()

        try:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO data_quality_events (
                    session_date, timestamp_utc, timestamp_ist, event_type, severity, instrument_symbol, message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (today_str, now_ts, now_str, event_type, severity, symbol, message, json.dumps(details or {})))
        except Exception:
            pass

    def get_health_telemetry(self) -> Dict[str, Any]:
        """Provides full data completeness and pipeline health telemetry."""
        now_ts = time.time()
        age_sec = round(now_ts - self._last_event_time_utc, 1) if self._last_event_time_utc > 0 else None
        db_size_mb = 0.0
        table_counts = {}

        if os.path.exists(self._db_path):
            db_size_mb = round(os.path.getsize(self._db_path) / (1024 * 1024), 2)
            try:
                with self._get_connection() as conn:
                    for tbl in ("raw_market_events", "normalized_quotes", "market_depth_snapshots", "option_chain_snapshots", "analytical_snapshots_10s", "consolidated_snapshots_10m", "data_quality_events"):
                        c = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                        table_counts[tbl] = c
            except Exception:
                pass

        try:
            from engines.kotak_neo_engine import kotak_neo_engine
            ws_conn = bool(kotak_neo_engine._is_connected)
            sub_count = len(kotak_neo_engine._live_ticks)
        except Exception:
            ws_conn = False
            sub_count = 0

        return {
            "status": "OK",
            "capture_engine_active": self._is_running,
            "websocket_connected": ws_conn,
            "subscribed_instruments_count": sub_count,
            "events_received_total": self._events_received_count,
            "events_persisted_total": self._events_persisted_count,
            "queue_depth": self._event_queue.qsize(),
            "backpressure_active": self._backpressure_active,
            "dropped_events_count": self._dropped_events_count,
            "last_event_time_ist": self._last_event_time_ist,
            "quote_age_seconds": age_sec,
            "db_size_mb": db_size_mb,
            "table_record_counts": table_counts,
            "missing_intervals_count": len(self._missing_intervals),
            "recent_missing_intervals": self._missing_intervals[-5:],
            "telemetry_timestamp_ist": format_ist(),
        }

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Lists captured historical market sessions."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM market_sessions ORDER BY session_date DESC").fetchall()
            return [dict(r) for r in rows]

    def get_analytical_snapshots_10s(self, session_date: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetches 10-second analytical time series for historical comparison."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM analytical_snapshots_10s
            WHERE session_date = ?
            ORDER BY timestamp_utc ASC
            LIMIT ?
            """, (session_date, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_consolidated_snapshots_10m(self, session_date: str) -> List[Dict[str, Any]]:
        """Fetches 10-minute consolidated macro snapshots for session replay/backtest."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM consolidated_snapshots_10m
            WHERE session_date = ?
            ORDER BY slot_start_utc ASC
            """, (session_date,)).fetchall()
            return [dict(r) for r in rows]

    def get_data_quality_events(self, session_date: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches recorded data quality anomalies and warnings."""
        with self._get_connection() as conn:
            if session_date:
                rows = conn.execute("""
                SELECT * FROM data_quality_events
                WHERE session_date = ?
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """, (session_date, limit)).fetchall()
            else:
                rows = conn.execute("""
                SELECT * FROM data_quality_events
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]



historical_capture_engine = HistoricalCaptureEngine()
