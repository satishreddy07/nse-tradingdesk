"""
SQLite State Engine & Execution Journal
Manages portfolio holdings, trade logs, scan runs, and signals for both portfolios:
- INSTITUTIONAL_6PILLAR
- NIFTY500_MOMENTUM
"""
import sqlite3
import threading
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.config import DB_PATH
from core.data_quality import now_ist, format_ist

logger = logging.getLogger("StateEngine")

class StateEngine:
    """Thread-safe SQLite execution state engine with WAL mode."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency read/writes
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
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
            with self._get_connection() as conn:
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

                conn.commit()
                logger.info("Database schemas verified and ready.")

    # --------------------------------------------------------------------------
    # ACTIVE POSITIONS OPERATIONS (WITH STRICT MARKET-HOURS TRADING GATE)
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
        Opens a new position in the database.
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
                    conn.execute("""
                    INSERT INTO active_positions (
                        symbol, portfolio_type, entry_price, quantity, stop_loss, original_stop_loss,
                        target, target_2, current_price, highest_price, entry_date, hold_days,
                        t1_hit, t1_pnl, sl_reason, sector, notes, entry_reason, entry_regime,
                        metrics_json, features_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity = quantity + excluded.quantity,
                        current_price = excluded.current_price,
                        updated_at = excluded.updated_at;
                    """, (
                        sym_clean, p_type, entry_price, quantity, stop_loss, stop_loss,
                        target, target_2, entry_price, entry_price, now_str,
                        sl_reason, sector, notes, entry_reason, entry_regime,
                        metrics_str, features_str, now_str
                    ))
                    conn.commit()
                    logger.info(f"Opened position for {sym_clean} ({quantity} shs @ ₹{entry_price:.2f}) in {p_type} at {now_str}")

                    # Sync to Telegram Cloud Vault
                    telegram_vault.send_alert_and_sync(
                        title=f"🚀 NEW TRADE OPENED: {sym_clean} ({p_type})",
                        message=(
                            f"<b>Symbol:</b> <code>{sym_clean}</code>\n"
                            f"<b>Portfolio:</b> {p_type}\n"
                            f"<b>Entry Price:</b> ₹{entry_price:,.2f} × {quantity} shs\n"
                            f"<b>Stop Loss:</b> ₹{stop_loss:,.2f} | <b>Target 1:</b> ₹{target:,.2f}\n"
                            f"<b>Entry Timestamp:</b> {now_str}\n"
                            f"<b>Setup:</b> {entry_reason or notes}"
                        ),
                        sync_db=True,
                        sync_ohlcv=False
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error opening position for {sym_clean}: {e}")
                    return False

    def get_active_positions(self, portfolio_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves currently open positions, optionally filtered by portfolio."""
        query = "SELECT * FROM active_positions"
        params = []
        if portfolio_type:
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY entry_date DESC"

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
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
        Closes an active position and logs it to trade_history with full IST timestamps.
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

                # Delete from active_positions
                conn.execute("DELETE FROM active_positions WHERE symbol = ?", (sym_clean,))
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

                # Sync to Telegram Cloud Vault
                telegram_vault.send_alert_and_sync(
                    title=f"🏁 TRADE CLOSED: {sym_clean} ({exit_reason})",
                    message=(
                        f"<b>Symbol:</b> <code>{sym_clean}</code> ({p_type})\n"
                        f"<b>Entry:</b> ₹{entry_px:,.2f} ({entry_dt_str})\n"
                        f"<b>Exit:</b> ₹{exit_price:,.2f} ({now_str})\n"
                        f"<b>Realized PnL:</b> ₹{total_pnl:+,.2f} ({r_multiple:+.2f}R | {return_pct:+.2f}%)\n"
                        f"<b>Reason:</b> {exit_reason}"
                    ),
                    sync_db=True,
                    sync_ohlcv=False
                )
                return trade_record

    def reset_portfolio_trades(self, portfolio_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Resets active_positions and trade_history (either for a specific portfolio or all portfolios)
        so Realized & Unrealized PnL start cleanly at ₹0.00.
        """
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
        logger.info(f"Reset trades completed (portfolio={portfolio_type or 'ALL'}).")
        return {"status": "SUCCESS", "portfolio": portfolio_type or "ALL", "timestamp": format_ist()}


    # --------------------------------------------------------------------------
    # TRADE HISTORY & PORTFOLIO METRICS
    # --------------------------------------------------------------------------
    def get_trade_history(self, portfolio_type: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Retrieves closed trade logs."""
        query = "SELECT * FROM trade_history"
        params = []
        if portfolio_type:
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]

    def get_portfolio_stats(self, portfolio_type: Optional[str] = None) -> Dict[str, Any]:
        """Calculates comprehensive quantitative performance metrics."""
        history = self.get_trade_history(portfolio_type=portfolio_type, limit=1000)
        active = self.get_active_positions(portfolio_type=portfolio_type)

        total_trades = len(history)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "total_realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "avg_win_r": 0.0,
                "avg_loss_r": 0.0,
                "active_count": len(active),
                "open_risk": sum(max(0.0, (float(p['entry_price']) - float(p['stop_loss'])) * int(p['quantity'])) for p in active)
            }

        wins = [t for t in history if t["pnl"] > 0]
        losses = [t for t in history if t["pnl"] <= 0]
        
        win_rate = round((len(wins) / total_trades) * 100.0, 1)
        gross_profit = sum(float(t["pnl"]) for t in wins)
        gross_loss = abs(sum(float(t["pnl"]) for t in losses))
        profit_factor = round(gross_profit / (gross_loss + 1e-9), 2)
        
        r_list = [float(t["r_multiple"]) for t in history]
        expectancy_r = round(float(sum(r_list) / len(r_list)), 3)
        avg_win_r = round(float(sum(t["r_multiple"] for t in wins) / len(wins)), 2) if wins else 0.0
        avg_loss_r = round(float(sum(t["r_multiple"] for t in losses) / len(losses)), 2) if losses else 0.0
        total_realized_pnl = round(sum(float(t["pnl"]) for t in history), 2)

        # MTM Unrealized PnL
        unrealized = 0.0
        open_risk = 0.0
        for p in active:
            cp = float(p.get("current_price") or p["entry_price"])
            ep = float(p["entry_price"])
            sl = float(p["stop_loss"])
            qty = int(p["quantity"])
            unrealized += (cp - ep) * qty
            open_risk += max(0.0, (ep - sl) * qty)

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "total_realized_pnl": total_realized_pnl,
            "unrealized_pnl": round(unrealized, 2),
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "active_count": len(active),
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

    def get_latest_signals(self, portfolio_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves stored signals."""
        query = "SELECT * FROM latest_signals"
        params = []
        if portfolio_type:
            p_type = self._normalize_portfolio_type(portfolio_type)
            query += " WHERE portfolio_type = ?"
            params.append(p_type)
        query += " ORDER BY score DESC, ml_prob DESC"

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]

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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self.init_progress_tracker()
            now_str = format_ist()
            self._live_progress["is_running"] = False
            self._live_progress["stage"] = final_stage
            self._live_progress["percent"] = 100
            self._live_progress["updated_at"] = now_str
            self._live_progress["logs"].append(f"[{now_str[11:19]}] ✅ {final_stage}")

    def get_scan_progress(self) -> Dict[str, Any]:
        with self._lock:
            self.init_progress_tracker()
            return dict(self._live_progress)


# Global Singleton Instance
state_engine = StateEngine()

