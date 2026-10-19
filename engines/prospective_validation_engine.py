"""
========================================================================================
Prospective Paper-Trading Validation Engine (`engines/prospective_validation_engine.py`)
========================================================================================
Mission:
Deterministic, persistent prospective paper-trading pipeline for the Pre-Rally Pattern
Discovery & Validation Engine:
- Frozen Baseline Configuration:
    * Pattern Rule: ATR Squeeze <= 0.85 (ATR5 / ATR20 <= 0.85)
    * Profit Target: +5.0%
    * Stop Loss: -3.5%
    * Horizon: 10 Sessions
    * Entry Execution: NEXT_OPEN (also supports SAME_CLOSE)
    * Transaction Cost Schedule: Statutory NSE Cash (or Flat 0.35%)
- ZERO Live Broker Orders:
    * Strictly writes to SQLite `prospective_paper_signals` and `prospective_paper_trades`.
    * Fully decoupled from live portfolio order execution.
- Truthful Accounting:
    * Distinguishes Statistical Outcome Labels from Realized Executable P&L.
    * Transparently compares prospective performance against historical OOS baseline
      (37.2% label win rate, -0.04% simulated net expectancy).
- Idempotent and thread-safe persistence.
========================================================================================
"""

import os
import json
import math
import time
import hashlib
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.data_quality import now_ist, format_ist
from engines.trade_simulator import TradeSimulator, TransactionCostModel, SIMULATOR_VERSION
from execution.state_engine import state_engine

logger = logging.getLogger("ProspectiveValidationEngine")

PROSPECTIVE_ENGINE_VERSION = "1.0.0-PROSPECTIVE-AUDITED"
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

FROZEN_BASELINE_CONFIG = {
    "version": PROSPECTIVE_ENGINE_VERSION,
    "pattern_id": "ATR_SQUEEZE_85",
    "pattern_name": "ATR Squeeze <= 0.85",
    "description": "ATR(5) / ATR(20) <= 0.85 (Pre-rally Volatility Contraction)",
    "target_pct": 5.0,
    "stop_pct": -3.5,
    "horizon_days": 10,
    "entry_model": "NEXT_OPEN",
    "cost_type": "STATUTORY_NSE_CASH",
    "broker_mode": "ZERO_BROKER_PAPER_ONLY"
}


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Computes deterministic hash for prospective configuration."""
    stable_dict = {
        "pattern_id": cfg.get("pattern_id", "ATR_SQUEEZE_85"),
        "target_pct": float(cfg.get("target_pct", 5.0)),
        "stop_pct": float(cfg.get("stop_pct", -3.5)),
        "horizon_days": int(cfg.get("horizon_days", 10)),
        "entry_model": str(cfg.get("entry_model", "NEXT_OPEN")).upper(),
        "cost_type": str(cfg.get("cost_type", "STATUTORY_NSE_CASH"))
    }
    return hashlib.md5(json.dumps(stable_dict, sort_keys=True).encode()).hexdigest()[:12]


class ProspectiveValidationEngine:
    """
    Manages prospective signal discovery, forward trade tracking, and realized
    performance attribution without broker order dispatch.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.environ.get("DB_PATH") or os.path.join(os.path.dirname(MODULE_DIR), "data", "swing_trades.db")

        self._lock = threading.Lock()
        self.config = dict(FROZEN_BASELINE_CONFIG)
        self.config_hash = compute_config_hash(self.config)
        self.cost_model = TransactionCostModel(cost_type=self.config["cost_type"])
        self.simulator = TradeSimulator(
            target_pct=self.config["target_pct"],
            stop_pct=self.config["stop_pct"],
            horizon_days=self.config["horizon_days"],
            entry_model=self.config["entry_model"],
            cost_model=self.cost_model
        )
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Initializes tables for prospective paper-trading."""
        with self._lock:
            with self._get_conn() as conn:
                # 1. Signals Table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS prospective_paper_signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_val REAL NOT NULL,
                    atr5 REAL NOT NULL,
                    atr20 REAL NOT NULL,
                    close_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, signal_date, config_hash)
                );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pps_sym ON prospective_paper_signals(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pps_date ON prospective_paper_signals(signal_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pps_status ON prospective_paper_signals(status)")

                # 2. Trades Table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS prospective_paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_model TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    hold_days INTEGER DEFAULT 0,
                    gross_return_pct REAL,
                    charges_pct REAL,
                    net_return_pct REAL,
                    current_price REAL,
                    unrealized_return_pct REAL,
                    mfe_pct REAL,
                    mae_pct REAL,
                    is_win INTEGER,
                    original_label INTEGER,
                    charges_breakdown_json TEXT,
                    config_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, entry_date, config_hash)
                );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ppt_sym ON prospective_paper_trades(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ppt_status ON prospective_paper_trades(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ppt_entry ON prospective_paper_trades(entry_date)")
                conn.commit()

    def _get_df_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Safely loads OHLCV DataFrame from state_engine, unwrapping (df, age) tuples."""
        try:
            res = state_engine.get_ohlcv_df(symbol)
            if isinstance(res, tuple):
                return res[0]
            return res
        except Exception as e:
            logger.debug(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    # =========================================================================
    # SIGNAL DETECTION
    # =========================================================================

    def detect_signals_in_df(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of_date: Optional[str] = None,
        scan_all_bars: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Scans OHLCV DataFrame for ATR Squeeze <= 0.85 setups.
        Calculates features strictly with zero look-ahead bias.
        """
        if df is None or len(df) < 25:
            return []

        df_sorted = df.sort_index().copy()
        h = df_sorted["High"].astype(float)
        l = df_sorted["Low"].astype(float)
        c = df_sorted["Close"].astype(float)

        high_low = h - l
        high_close_prev = (h - c.shift(1)).abs()
        low_close_prev = (l - c.shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        atr5 = tr.rolling(5).mean()
        atr20 = tr.rolling(20).mean()
        squeeze_ratio = atr5 / (atr20 + 1e-9)

        signals = []
        n_bars = len(df_sorted)

        # Determine target index or scan range
        if scan_all_bars:
            eval_indices = list(range(20, n_bars))
        elif as_of_date:
            date_strs = [str(d)[:10] for d in df_sorted.index]
            if as_of_date in date_strs:
                eval_indices = [date_strs.index(as_of_date)]
            else:
                eval_indices = []
        else:
            # Default to the most recent completed bar
            eval_indices = [n_bars - 1]

        for idx in eval_indices:
            if idx < 20:
                continue

            ratio_val = float(squeeze_ratio.iloc[idx])
            if math.isnan(ratio_val):
                continue

            # Check ATR Squeeze condition
            if ratio_val <= 0.85:
                sig_date = str(df_sorted.index[idx])[:10]
                close_px = float(c.iloc[idx])
                atr5_val = float(atr5.iloc[idx])
                atr20_val = float(atr20.iloc[idx])

                sig_id = f"psig_{symbol}_{sig_date}_{self.config_hash}"
                signals.append({
                    "signal_id": sig_id,
                    "symbol": symbol,
                    "signal_date": sig_date,
                    "feature_name": "ATR_SQUEEZE_85",
                    "feature_val": round(ratio_val, 4),
                    "atr5": round(atr5_val, 2),
                    "atr20": round(atr20_val, 2),
                    "close_price": round(close_px, 2),
                    "status": "PENDING",
                    "config_hash": self.config_hash,
                    "signal_idx": idx,
                    "created_at": format_ist()
                })

        return signals

    # =========================================================================
    # PIPELINE EXECUTION: DISCOVERY -> ENTRY -> LIFECYCLE SYNC
    # =========================================================================

    def _resolve_universe_symbols(self, max_symbols: int = 150) -> List[str]:
        """Resolves active liquid stocks from ohlcv_master_store."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT symbol FROM ohlcv_master_store WHERE bars_count >= 25 AND symbol NOT LIKE '^%' AND symbol NOT LIKE '%=%' ORDER BY bars_count DESC LIMIT ?",
                    (max_symbols,)
                ).fetchall()
                if rows:
                    return [r[0] if isinstance(r, (tuple, list)) else r["symbol"] for r in rows]
        except Exception as e:
            logger.debug(f"Notice reading symbols from ohlcv_master_store: {e}")
        return []

    def scan_and_record_signals(
        self,
        symbols: Optional[List[str]] = None,
        as_of_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scans OHLCV series across universe and records candidate signals idempotently.
        """
        if symbols is None:
            symbols = self._resolve_universe_symbols()

        discovered_count = 0
        new_inserted = 0

        with self._lock:
            with self._get_conn() as conn:
                for sym in symbols:
                    df = self._get_df_for_symbol(sym)
                    if df is None or len(df) < 25:
                        continue

                    sigs = self.detect_signals_in_df(sym, df, as_of_date=as_of_date)
                    for s in sigs:
                        discovered_count += 1
                        try:
                            conn.execute("""
                            INSERT OR IGNORE INTO prospective_paper_signals (
                                signal_id, symbol, signal_date, feature_name,
                                feature_val, atr5, atr20, close_price, status,
                                config_hash, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                s["signal_id"], s["symbol"], s["signal_date"],
                                s["feature_name"], s["feature_val"], s["atr5"],
                                s["atr20"], s["close_price"], s["status"],
                                s["config_hash"], s["created_at"]
                            ))
                            if conn.total_changes > 0:
                                new_inserted += 1
                        except Exception as e:
                            logger.debug(f"Error persisting signal {s['signal_id']}: {e}")

                conn.commit()

        return {
            "status": "SUCCESS",
            "symbols_scanned": len(symbols),
            "signals_discovered": discovered_count,
            "new_signals_persisted": new_inserted,
            "as_of_date": as_of_date or "LATEST_AVAILABLE"
        }

    def convert_pending_signals_to_trades(self) -> Dict[str, Any]:
        """
        Converts PENDING signals into OPEN paper trades once entry bar is available.
        - If NEXT_OPEN: looks for bar T+1 Open. If bar T+1 exists, enters trade.
        - If SAME_CLOSE: enters on signal date Close immediately.
        """
        with self._lock:
            with self._get_conn() as conn:
                pending_rows = conn.execute("""
                SELECT * FROM prospective_paper_signals
                WHERE status = 'PENDING' AND config_hash = ?
                """, (self.config_hash,)).fetchall()

                converted = 0
                for r in pending_rows:
                    sym = r["symbol"]
                    sig_date = r["signal_date"]
                    sig_id = r["signal_id"]

                    df = self._get_df_for_symbol(sym)
                    if df is None or len(df) == 0:
                        continue

                    df_sorted = df.sort_index()
                    date_strs = [str(d)[:10] for d in df_sorted.index]

                    if sig_date not in date_strs:
                        continue

                    sig_idx = date_strs.index(sig_date)

                    if self.config["entry_model"] == "NEXT_OPEN":
                        if sig_idx + 1 >= len(df_sorted):
                            # Next day's candle not yet formed (e.g. signal generated today after market close)
                            continue
                        entry_idx = sig_idx + 1
                        entry_date = date_strs[entry_idx]
                        entry_px = float(df_sorted["Open"].iloc[entry_idx])
                    else: # SAME_CLOSE
                        entry_idx = sig_idx
                        entry_date = sig_date
                        entry_px = float(df_sorted["Close"].iloc[entry_idx])

                    if entry_px <= 0.0 or math.isnan(entry_px):
                        continue

                    target_px = round(entry_px * (1.0 + self.config["target_pct"] / 100.0), 2)
                    stop_px = round(entry_px * (1.0 + self.config["stop_pct"] / 100.0), 2)
                    trade_id = f"ptrd_{sym}_{entry_date}_{self.config_hash}"

                    now_time = format_ist()
                    try:
                        conn.execute("""
                        INSERT OR IGNORE INTO prospective_paper_trades (
                            trade_id, signal_id, symbol, signal_date, entry_date,
                            entry_price, entry_model, target_price, stop_price,
                            status, current_price, unrealized_return_pct, hold_days,
                            mfe_pct, mae_pct, config_hash, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 0.0, 0, 0.0, 0.0, ?, ?, ?)
                        """, (
                            trade_id, sig_id, sym, sig_date, entry_date,
                            entry_px, self.config["entry_model"], target_px, stop_px,
                            entry_px, self.config_hash, now_time, now_time
                        ))

                        conn.execute("""
                        UPDATE prospective_paper_signals
                        SET status = 'CONVERTED'
                        WHERE signal_id = ?
                        """, (sig_id,))
                        converted += 1
                    except Exception as e:
                        logger.error(f"Error creating prospective paper trade {trade_id}: {e}")

                conn.commit()

        return {"status": "SUCCESS", "trades_converted": converted}

    def sync_open_paper_trades(self) -> Dict[str, Any]:
        """
        Advances all OPEN paper trades bar-by-bar using TradeSimulator.
        Resolves exits (target, stop, gap, same-bar conflict, or Day 10 close).
        """
        with self._lock:
            with self._get_conn() as conn:
                open_trades = conn.execute("""
                SELECT * FROM prospective_paper_trades
                WHERE status = 'OPEN' AND config_hash = ?
                """, (self.config_hash,)).fetchall()

                completed_count = 0
                updated_count = 0

                for t in open_trades:
                    sym = t["symbol"]
                    entry_date = t["entry_date"]
                    sig_date = t["signal_date"]
                    trade_id = t["trade_id"]

                    df = self._get_df_for_symbol(sym)
                    if df is None or len(df) == 0:
                        continue

                    df_sorted = df.sort_index()
                    date_strs = [str(d)[:10] for d in df_sorted.index]

                    # Locate the signal or entry index
                    if self.config["entry_model"] == "NEXT_OPEN":
                        if sig_date not in date_strs:
                            continue
                        eval_sig_idx = date_strs.index(sig_date)
                    else:
                        if entry_date not in date_strs:
                            continue
                        eval_sig_idx = date_strs.index(entry_date)

                    sim_res = self.simulator.simulate_trade(
                        df=df_sorted,
                        signal_idx=eval_sig_idx,
                        symbol=sym,
                        run_id=f"PROSPECTIVE_{self.config_hash}"
                    )

                    now_time = format_ist()
                    if sim_res.get("status") == "COMPLETED":
                        conn.execute("""
                        UPDATE prospective_paper_trades
                        SET status = 'COMPLETED',
                            exit_date = ?,
                            exit_price = ?,
                            exit_reason = ?,
                            hold_days = ?,
                            gross_return_pct = ?,
                            charges_pct = ?,
                            net_return_pct = ?,
                            current_price = ?,
                            unrealized_return_pct = 0.0,
                            mfe_pct = ?,
                            mae_pct = ?,
                            is_win = ?,
                            original_label = ?,
                            charges_breakdown_json = ?,
                            updated_at = ?
                        WHERE trade_id = ?
                        """, (
                            sim_res["exit_date"],
                            sim_res["exit_price"],
                            sim_res["exit_reason"],
                            sim_res["hold_days"],
                            sim_res["gross_return_pct"],
                            sim_res["charges_pct"],
                            sim_res["net_return_pct"],
                            sim_res["exit_price"],
                            sim_res["mfe_pct"],
                            sim_res["mae_pct"],
                            sim_res["is_win"],
                            sim_res["original_label"],
                            json.dumps(sim_res.get("charges_breakdown", {})),
                            now_time,
                            trade_id
                        ))
                        completed_count += 1
                    elif sim_res.get("status") == "OPEN":
                        conn.execute("""
                        UPDATE prospective_paper_trades
                        SET current_price = ?,
                            unrealized_return_pct = ?,
                            hold_days = ?,
                            mfe_pct = ?,
                            mae_pct = ?,
                            updated_at = ?
                        WHERE trade_id = ?
                        """, (
                            sim_res.get("current_price"),
                            sim_res.get("unrealized_return_pct", 0.0),
                            sim_res.get("hold_days", 0),
                            sim_res.get("mfe_pct", 0.0),
                            sim_res.get("mae_pct", 0.0),
                            now_time,
                            trade_id
                        ))
                        updated_count += 1
                    elif sim_res.get("status") in ("EXCLUDED", "ERROR"):
                        conn.execute("""
                        UPDATE prospective_paper_trades
                        SET status = ?,
                            exit_reason = ?,
                            updated_at = ?
                        WHERE trade_id = ?
                        """, (
                            sim_res.get("status"),
                            sim_res.get("error_reason", "SIMULATOR_REJECTION"),
                            now_time,
                            trade_id
                        ))

                conn.commit()

        return {
            "status": "SUCCESS",
            "trades_evaluated": len(open_trades),
            "trades_completed": completed_count,
            "trades_updated_open": updated_count
        }

    def run_daily_pipeline(self) -> Dict[str, Any]:
        """Unified method for post-market execution."""
        sig_res = self.scan_and_record_signals()
        conv_res = self.convert_pending_signals_to_trades()
        sync_res = self.sync_open_paper_trades()
        return {
            "status": "SUCCESS",
            "signal_scan": sig_res,
            "conversion": conv_res,
            "trade_sync": sync_res,
            "timestamp": format_ist()
        }

    # =========================================================================
    # STATISTICAL TELEMETRY & ATTRIBUTION REPORTING
    # =========================================================================

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """
        Computes transparent statistical metrics and telemetry for prospective validation.
        """
        with self._lock:
            with self._get_conn() as conn:
                # 1. Total Signals
                sig_count = conn.execute("""
                SELECT count(*) FROM prospective_paper_signals WHERE config_hash = ?
                """, (self.config_hash,)).fetchone()[0]

                # 2. Trades by Status
                trade_rows = conn.execute("""
                SELECT * FROM prospective_paper_trades WHERE config_hash = ?
                """, (self.config_hash,)).fetchall()

        trades = [dict(r) for r in trade_rows]
        open_trades = [t for t in trades if t["status"] == "OPEN"]
        completed_trades = [t for t in trades if t["status"] == "COMPLETED"]
        excluded_trades = [t for t in trades if t["status"] in ("EXCLUDED", "ERROR")]

        n_completed = len(completed_trades)

        if n_completed > 0:
            wins = [t for t in completed_trades if t.get("is_win") == 1]
            label_wins = [t for t in completed_trades if t.get("original_label") == 1]
            realized_win_rate = round((len(wins) / n_completed) * 100.0, 2)
            label_win_rate = round((len(label_wins) / n_completed) * 100.0, 2)

            gross_returns = [float(t.get("gross_return_pct") or 0.0) for t in completed_trades]
            net_returns = [float(t.get("net_return_pct") or 0.0) for t in completed_trades]

            mean_gross_expectancy = round(float(np.mean(gross_returns)), 2)
            mean_net_expectancy = round(float(np.mean(net_returns)), 2)

            win_returns = [r for r in net_returns if r > 0]
            loss_returns = [r for r in net_returns if r <= 0]
            avg_win = round(float(np.mean(win_returns)), 2) if win_returns else 0.0
            avg_loss = round(float(np.mean(loss_returns)), 2) if loss_returns else 0.0

            sum_gains = sum(win_returns)
            sum_losses = abs(sum(loss_returns))
            profit_factor = round(sum_gains / sum_losses, 2) if sum_losses > 0 else (99.0 if sum_gains > 0 else 1.0)

            # Cumulative drawdown calculation
            cum_returns = np.cumsum(net_returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = peak - cum_returns
            max_drawdown = round(float(np.max(drawdown)), 2) if len(drawdown) > 0 else 0.0

            # Exit breakdown
            exit_reasons = {}
            for t in completed_trades:
                r = t.get("exit_reason") or "UNKNOWN"
                exit_reasons[r] = exit_reasons.get(r, 0) + 1

            avg_hold_days = round(float(np.mean([t.get("hold_days", 0) for t in completed_trades])), 1)
        else:
            realized_win_rate = 0.0
            label_win_rate = 0.0
            mean_gross_expectancy = 0.0
            mean_net_expectancy = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 1.0
            max_drawdown = 0.0
            exit_reasons = {}
            avg_hold_days = 0.0

        # Sort open trades by entry date descending
        open_trades.sort(key=lambda x: str(x.get("entry_date", "")), reverse=True)
        completed_trades.sort(key=lambda x: str(x.get("exit_date", "")), reverse=True)

        return {
            "status": "SUCCESS",
            "version": PROSPECTIVE_ENGINE_VERSION,
            "config": self.config,
            "config_hash": self.config_hash,
            "historical_baseline_reference": {
                "dataset_scope": "35 liquid NSE stocks, 2,369 historical events",
                "split_cutoff_date": "2026-03-13",
                "oos_sample_size": 199,
                "oos_label_win_rate_pct": 37.2,
                "oos_baseline_rate_pct": 34.6,
                "oos_lift_ratio": 1.08,
                "binary_net_expectancy_pct": -0.69,
                "simulated_net_expectancy_pct": -0.04,
                "statistical_significance": "Not statistically distinguishable from baseline (95% CI [-5.0%, +10.2%])",
                "verdict": "NO_DEMONSTRATED_EDGE"
            },
            "prospective_performance": {
                "signals_count": sig_count,
                "open_trades_count": len(open_trades),
                "completed_trades_count": n_completed,
                "excluded_trades_count": len(excluded_trades),
                "label_win_rate_pct": label_win_rate,
                "realized_win_rate_pct": realized_win_rate,
                "mean_gross_expectancy_pct": mean_gross_expectancy,
                "mean_net_expectancy_pct": mean_net_expectancy,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_drawdown,
                "avg_hold_days": avg_hold_days,
                "exit_reasons_breakdown": exit_reasons
            },
            "open_trades": open_trades[:25],
            "recent_completed_trades": completed_trades[:25],
            "disclaimer": (
                "ZERO REAL BROKER ORDERS: This is an automated, strictly isolated prospective paper-trading engine. "
                "No live capital is committed. All metrics measure forward out-of-sample behavior to test the "
                "ATR Squeeze <= 0.85 hypothesis without look-ahead bias or overfitting."
            ),
            "generated_at": format_ist()
        }


prospective_validation_engine = ProspectiveValidationEngine()
