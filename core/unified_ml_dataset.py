"""
core/unified_ml_dataset.py
==========================
Unified Trade-Level Learning Dataset & Counterfactual Engine across 4 Trading Horizons:
1. INTRADAY: Scalp & 0-5 DTE options momentum.
2. WEEKLY: 1-sigma credit/debit spreads & strangles (40% stop, 1-time re-entry).
3. MONTHLY: 1-sigma credit structures (50% stop, 1-time re-entry).
4. SWING: 6-pillar institutional momentum & Nifty 500 multi-day compounding.

Features:
- Complete Trade Identification (signal ID, trade ID, horizon, session, live/paper/backtest).
- Pure Entry-Time Market Snapshot (32+ macro, technical, derivative, and regime features).
- Decoupled Structured Rationale (human explanations stored separately, zero lookahead leakage).
- Detailed Trade Lifecycle & Outcomes (fills, slippage, MAE/MFE, re-entry counts, gross/net P&L).
- Counterfactual & Rejected Opportunity Recording (realistic simulation of rejected/risk-blocked setups).
- Thread-safe SQLite persistence in `data/swing_trades.db`.
"""

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import DB_PATH
from core.data_quality import IST, format_ist, now_ist
from execution.state_engine import state_engine

logger = logging.getLogger("UnifiedMLDataset")


UNIFIED_DATASET_TABLE = "unified_trade_learning_dataset"

# 32 Standardized Numerical ML Features for Multi-Horizon Learning
def _finite_feature_value(value: Any) -> Optional[float]:
    """Return a finite number or None; never turn missing/bad telemetry into a numeric default."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


UNIFIED_FEATURE_COLS = [
    # Macro & Benchmark Dynamics
    "nifty_spot",
    "nifty_ret_1d",
    "nifty_ret_5d",
    "composite_score",
    "vix_spot",
    "realized_vol_daily",
    "realized_vol_5d",
    "fii_net_flow_cr",
    "fii_fut_long_pct",
    "breadth_ratio",
    # Technical & Momentum Indicators
    "rsi14",
    "atr_pct",
    "dist_to_ema20_pct",
    "dist_to_ema50_pct",
    "dist_to_pivot_pct",
    "volume_ratio_50d",
    "trend_strength_adx",
    "overnight_gap_pct",
    # Options & Derivatives Telemetry (applicable for options, neutral default for cash)
    "near_atm_pcr",
    "chg_oi_pcr",
    "iv_spot",
    "iv_skew",
    "straddle_decay_pct",
    "dist_to_call_wall_pct",
    "dist_to_put_wall_pct",
    "sigma_distance",
    "time_to_expiry_years",
    # Horizon & Setup Architecture
    "horizon_code",        # 1: INTRADAY, 2: WEEKLY, 3: MONTHLY, 4: SWING
    "regime_code",         # 2: STRONG_BULL, 1: MILD_BULL, 0: RANGE, -1: MILD_BEAR, -2: STRONG_BEAR
    "reward_to_risk_est",
    "sl_threshold_pct",
    "time_of_day_mins"
]


class UnifiedMLDataset:
    """
    Central repository for trade-level learning, feature snapshots, decision rationale,
    and counterfactual outcome tracking across all four horizons.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=45.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Creates the `unified_trade_learning_dataset` table and performance indices."""
        with self._lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {UNIFIED_DATASET_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    trading_horizon TEXT NOT NULL,       -- 'INTRADAY', 'WEEKLY', 'MONTHLY', 'SWING'
                    strategy_name TEXT NOT NULL,
                    strategy_version TEXT NOT NULL DEFAULT 'v1.0',
                    record_type TEXT NOT NULL,           -- 'LIVE', 'PAPER', 'BACKTEST', 'COUNTERFACTUAL'
                    market_session TEXT NOT NULL,        -- 'PRE_MARKET', 'MORNING', 'AFTERNOON', 'POST_MARKET'
                    status TEXT NOT NULL,                -- 'ACTIVE', 'RESOLVED_TARGET', 'RESOLVED_STOP_LOSS', 'EXPIRED', 'REJECTED_SIM'
                    accept_reject_verdict TEXT NOT NULL, -- 'ACCEPTED', 'REJECTED', 'BLOCKED_RISK', 'SKIPPED_STALE'
                    decision_source TEXT NOT NULL,       -- 'RULE_BASED', 'ML_MODEL', 'HYBRID'

                    -- Numerical ML Feature Snapshot (Captured at entry time ONLY)
                    features_json TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    regime_confidence REAL DEFAULT 0.0,
                    composite_score REAL DEFAULT 0.0,
                    iv_at_entry REAL DEFAULT 0.0,
                    spot_at_entry REAL NOT NULL,
                    strike REAL DEFAULT 0.0,
                    expiry_date TEXT,
                    dte_days REAL DEFAULT 0.0,
                    sigma_distance REAL DEFAULT 1.0,

                    -- Risk & Capital Parameters
                    intended_entry REAL NOT NULL,
                    intended_target REAL NOT NULL,
                    intended_sl REAL NOT NULL,
                    sl_threshold_pct REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    capital_at_risk REAL NOT NULL,
                    reward_to_risk_ratio REAL DEFAULT 1.0,

                    -- Decoupled Decision Rationale (Zero lookahead leakage)
                    rationale_json TEXT,

                    -- Lifecycle & Outcome Metrics
                    actual_entry REAL,
                    actual_entry_time TEXT,
                    actual_exit REAL,
                    actual_exit_time TEXT,
                    fill_status TEXT DEFAULT 'FILLED',
                    exit_reason TEXT,
                    hold_duration_mins REAL DEFAULT 0.0,
                    mae_pct REAL DEFAULT 0.0,            -- Maximum Adverse Excursion (%)
                    mfe_pct REAL DEFAULT 0.0,            -- Maximum Favorable Excursion (%)
                    gross_pnl REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    total_costs REAL DEFAULT 0.0,
                    r_multiple REAL DEFAULT 0.0,
                    is_winner INTEGER,                   -- 1: win, 0: loss, NULL: active
                    target_hit_before_sl INTEGER,        -- 1: target hit first, 0: sl hit first, NULL: ambiguous/active
                    re_entry_count INTEGER DEFAULT 0,
                    rules_followed INTEGER DEFAULT 1,

                    model_version TEXT DEFAULT 'prod_v1.0',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(signal_id, trade_id)
                );
                """)
                # Indices for ultra-fast walk-forward extraction
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_utld_horizon ON {UNIFIED_DATASET_TABLE}(trading_horizon);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_utld_status ON {UNIFIED_DATASET_TABLE}(status);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_utld_rec_type ON {UNIFIED_DATASET_TABLE}(record_type);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_utld_created ON {UNIFIED_DATASET_TABLE}(created_at);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_utld_winner ON {UNIFIED_DATASET_TABLE}(is_winner);")
                conn.commit()

    # =========================================================================
    # 1. SIGNAL & TRADE INGESTION
    # =========================================================================
    def record_signal_and_trade(
        self,
        signal_id: str,
        trade_id: str,
        instrument: str,
        trading_horizon: str,
        strategy_name: str,
        record_type: str,
        market_session: str,
        accept_reject_verdict: str,
        decision_source: str,
        features: Dict[str, float],
        regime: str,
        spot_at_entry: float,
        intended_entry: float,
        intended_target: float,
        intended_sl: float,
        sl_threshold_pct: float,
        quantity: int,
        capital_at_risk: float,
        rationale: Optional[Dict[str, Any]] = None,
        strike: float = 0.0,
        expiry_date: Optional[str] = None,
        dte_days: float = 0.0,
        sigma_distance: float = 1.0,
        actual_entry: Optional[float] = None,
        actual_entry_time: Optional[str] = None,
        model_version: str = "prod_v1.0"
    ) -> int:
        """
        Records an entry-time signal and trade setup with an immutable feature snapshot.
        """
        now_ts = format_ist()
        trading_horizon = trading_horizon.upper()
        accept_reject_verdict = accept_reject_verdict.upper()

        status = "ACTIVE" if accept_reject_verdict == "ACCEPTED" else "REJECTED_SIM"
        risk_unit = max(0.01, abs(intended_entry - intended_sl))
        target_diff = abs(intended_target - intended_entry)
        rr = round(target_diff / risk_unit, 2)

        supplied_features = features if isinstance(features, dict) else {}
        feat_clean = {
            key: _finite_feature_value(supplied_features.get(key))
            for key in UNIFIED_FEATURE_COLS
        }
        feat_json = json.dumps(feat_clean, allow_nan=False)
        rat_json = json.dumps(rationale or {})

        entry_time_val = actual_entry_time or (now_ts if status == "ACTIVE" else None)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(f"""
                INSERT OR REPLACE INTO {UNIFIED_DATASET_TABLE} (
                    signal_id, trade_id, instrument, trading_horizon, strategy_name,
                    strategy_version, record_type, market_session, status,
                    accept_reject_verdict, decision_source, features_json, regime,
                    regime_confidence, composite_score, iv_at_entry, spot_at_entry,
                    strike, expiry_date, dte_days, sigma_distance, intended_entry,
                    intended_target, intended_sl, sl_threshold_pct, quantity,
                    capital_at_risk, reward_to_risk_ratio, rationale_json,
                    actual_entry, actual_entry_time, fill_status, model_version,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, 'v1.0', ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                );
                """, (
                    signal_id, trade_id, instrument, trading_horizon, strategy_name,
                    record_type, market_session, status, accept_reject_verdict,
                    decision_source, feat_json, regime,
                    _finite_feature_value(supplied_features.get("regime_confidence")),
                    _finite_feature_value(supplied_features.get("composite_score")),
                    _finite_feature_value(supplied_features.get("iv_spot")),
                    float(spot_at_entry),
                    float(strike), expiry_date, float(dte_days), float(sigma_distance),
                    float(intended_entry), float(intended_target), float(intended_sl),
                    float(sl_threshold_pct), int(quantity), float(capital_at_risk),
                    float(rr), rat_json, actual_entry if (actual_entry and actual_entry > 0) else None,
                    entry_time_val,
                    "FILLED" if (status == "ACTIVE" and actual_entry is not None and actual_entry > 0) else ("PENDING_FILL" if status == "ACTIVE" else "REJECTED"),
                    model_version, entry_time_val or now_ts, now_ts
                ))
                conn.commit()
                return cursor.lastrowid

    # =========================================================================
    # 2. TRADE OUTCOME RESOLUTION (CLOSED-LOOP RECORDING)
    # =========================================================================
    def record_trade_outcome(
        self,
        trade_id: str,
        actual_exit: float,
        exit_reason: str,
        gross_pnl: float,
        net_pnl: float,
        total_costs: float,
        actual_exit_time: Optional[str] = None,
        mae_pct: Optional[float] = None,
        mfe_pct: Optional[float] = None,
        hold_duration_mins: Optional[float] = None,
        re_entry_count: Optional[int] = None,
        rules_followed: Optional[int] = None
    ) -> bool:
        """
        Updates an active or counterfactual trade record with verified terminal outcomes.
        Explicitly distinguishes target-hit vs stop-loss hit with cost accounting.
        """
        exit_time = actual_exit_time or format_ist()
        is_win = 1 if net_pnl > 0 else 0

        # Determine target vs SL event sequence
        if "TARGET" in exit_reason.upper() or "PROFIT" in exit_reason.upper():
            target_before_sl = 1
            status = "RESOLVED_TARGET"
        elif "STOP" in exit_reason.upper() or "SL" in exit_reason.upper():
            target_before_sl = 0
            status = "RESOLVED_STOP_LOSS"
        elif "TIME" in exit_reason.upper() or "EXPIRY" in exit_reason.upper():
            # End-of-period PnL does not establish whether target or stop was touched first.
            target_before_sl = None
            status = "EXPIRED"
        else:
            # Manual/other exits cannot be classified as target-before-stop from PnL sign alone.
            target_before_sl = None
            status = "RESOLVED_MANUAL"

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(f"SELECT intended_entry, intended_sl, capital_at_risk FROM {UNIFIED_DATASET_TABLE} WHERE trade_id = ?", (trade_id,)).fetchone()
                if not row:
                    logger.warning(f"Trade ID {trade_id} not found in {UNIFIED_DATASET_TABLE}.")
                    return False

                ep = float(row["intended_entry"])
                sl = float(row["intended_sl"])
                cap_risk = max(100.0, float(row["capital_at_risk"]))
                risk_pts = max(0.1, abs(ep - sl))

                r_mult = round(net_pnl / cap_risk, 2)

                conn.execute(f"""
                UPDATE {UNIFIED_DATASET_TABLE}
                SET actual_exit = ?,
                    actual_exit_time = ?,
                    exit_reason = ?,
                    status = ?,
                    gross_pnl = ?,
                    net_pnl = ?,
                    total_costs = ?,
                    r_multiple = ?,
                    is_winner = ?,
                    target_hit_before_sl = ?,
                    mae_pct = ?,
                    mfe_pct = ?,
                    hold_duration_mins = ?,
                    re_entry_count = ?,
                    rules_followed = ?,
                    updated_at = ?
                WHERE trade_id = ?
                """, (
                    float(actual_exit), exit_time, exit_reason, status,
                    float(gross_pnl), float(net_pnl), float(total_costs),
                    float(r_mult), int(is_win), (int(target_before_sl) if target_before_sl is not None else None),
                    _finite_feature_value(mae_pct), _finite_feature_value(mfe_pct),
                    _finite_feature_value(hold_duration_mins),
                    int(re_entry_count) if re_entry_count is not None else None,
                    int(rules_followed) if rules_followed is not None else None,
                    format_ist(), trade_id
                ))
                conn.commit()
                return True

    # =========================================================================
    # 3. COUNTERFACTUAL SIMULATOR FOR REJECTED & BLOCKED SIGNALS
    # =========================================================================
    def record_counterfactual_simulation(
        self,
        signal_id: str,
        instrument: str,
        trading_horizon: str,
        strategy_name: str,
        market_session: str,
        rejection_reason: str,
        features: Dict[str, float],
        regime: str,
        spot_at_entry: float,
        intended_entry: float,
        intended_target: float,
        intended_sl: float,
        sl_threshold_pct: float,
        quantity: int,
        capital_at_risk: float,
        bars_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Simulates the counterfactual outcome of a rejected or risk-blocked signal
        using realistic post-signal price bars, realistic bid-ask spread, and exchange costs.
        Eliminates survivorship and selection bias in ML training.
        """
        trade_id = f"CF_{signal_id}_{int(datetime.now(timezone.utc).timestamp())}"

        # 1. Record signal in dataset as REJECTED_SIM
        self.record_signal_and_trade(
            signal_id=signal_id,
            trade_id=trade_id,
            instrument=instrument,
            trading_horizon=trading_horizon,
            strategy_name=strategy_name,
            record_type="COUNTERFACTUAL",
            market_session=market_session,
            accept_reject_verdict="REJECTED",
            decision_source="RULE_BASED",
            features=features,
            regime=regime,
            spot_at_entry=spot_at_entry,
            intended_entry=intended_entry,
            intended_target=intended_target,
            intended_sl=intended_sl,
            sl_threshold_pct=sl_threshold_pct,
            quantity=quantity,
            capital_at_risk=capital_at_risk,
            rationale={"rejection_reason": rejection_reason}
        )

        # 2. Simulate realistic trajectory if bars are provided
        if bars_df is None or bars_df.empty or len(bars_df) < 2:
            # Data unavailable: Do NOT fabricate synthetic loss outcomes. Outcome remains unrecorded until real bars exist.
            logger.debug(f"Counterfactual trajectory simulation skipped for {trade_id}: Insufficient empirical bars.")
            return trade_id

        # Realistic bar-by-bar progression
        is_long = intended_target > intended_entry
        sim_exit = intended_entry
        exit_reason = "COUNTERFACTUAL_TIME_EXIT"
        is_win = 0
        mae = 0.0
        mfe = 0.0

        for _, bar in bars_df.iterrows():
            h = float(bar.get("high") or bar.get("High") or bar.get("close"))
            l = float(bar.get("low") or bar.get("Low") or bar.get("close"))
            c = float(bar.get("close") or bar.get("Close"))

            if is_long:
                mae = max(mae, ((intended_entry - l) / intended_entry) * 100.0)
                mfe = max(mfe, ((h - intended_entry) / intended_entry) * 100.0)
                if l <= intended_sl:
                    sim_exit = intended_sl * 0.998  # Add realistic slippage
                    exit_reason = "COUNTERFACTUAL_STOP_LOSS"
                    break
                elif h >= intended_target:
                    sim_exit = intended_target * 0.998
                    exit_reason = "COUNTERFACTUAL_TARGET"
                    break
            else:
                mae = max(mae, ((h - intended_entry) / intended_entry) * 100.0)
                mfe = max(mfe, ((intended_entry - l) / intended_entry) * 100.0)
                if h >= intended_sl:
                    sim_exit = intended_sl * 1.002
                    exit_reason = "COUNTERFACTUAL_STOP_LOSS"
                    break
                elif l <= intended_target:
                    sim_exit = intended_target * 1.002
                    exit_reason = "COUNTERFACTUAL_TARGET"
                    break

        gross_pts = (sim_exit - intended_entry) if is_long else (intended_entry - sim_exit)
        gross_pnl = round(gross_pts * quantity, 2)
        # Realistic statutory friction: 0.12% roundtrip on notional
        costs = round(intended_entry * quantity * 0.0012, 2)
        sim_pnl = round(gross_pnl - costs, 2)
        is_win = 1 if sim_pnl > 0 else 0

        self.record_trade_outcome(
            trade_id=trade_id,
            actual_exit=sim_exit,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            net_pnl=sim_pnl,
            total_costs=costs,
            mae_pct=mae,
            mfe_pct=mfe,
            hold_duration_mins=75.0,
            re_entry_count=0,
            rules_followed=1
        )

        return {
            "trade_id": trade_id,
            "status": "SIMULATED",
            "net_pnl": sim_pnl,
            "is_winner": is_win,
            "exit_reason": exit_reason
        }

    # =========================================================================
    # 4. DATASET RETRIEVAL FOR SPECIFIC HORIZONS
    # =========================================================================
    def get_dataset_for_horizon(
        self,
        horizon: str,
        include_counterfactuals: bool = False,
        min_resolved_only: bool = True,
        record_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieves resolved training rows for one explicitly selected cohort.
        Production defaults to actual LIVE fills only; PAPER, BACKTEST, and
        COUNTERFACTUAL cohorts must be requested separately.
        """
        horizon = horizon.upper()
        allowed_types = {"LIVE", "PAPER", "BACKTEST", "COUNTERFACTUAL"}
        if record_types is None:
            if include_counterfactuals:
                raise ValueError(
                    "Explicit record_types are required when include_counterfactuals=True "
                    "to prevent accidental mixing of empirical and counterfactual cohorts."
                )
            selected_types = ["LIVE"]
        else:
            selected_types = [str(item).upper() for item in record_types]
            invalid_types = set(selected_types) - allowed_types
            if invalid_types:
                raise ValueError(f"Unsupported ML record_type values: {sorted(invalid_types)}")
            if not selected_types:
                raise ValueError("record_types must contain at least one cohort.")
            if "COUNTERFACTUAL" in selected_types and not include_counterfactuals:
                raise ValueError("Counterfactual training requires include_counterfactuals=True.")

        where_clauses = ["trading_horizon = ?"]
        params: List[Any] = [horizon]
        if min_resolved_only:
            where_clauses.append("status IN ('RESOLVED_TARGET', 'RESOLVED_STOP_LOSS', 'EXPIRED', 'RESOLVED_MANUAL')")
            where_clauses.append("is_winner IS NOT NULL")

        # Always scope to the explicitly selected cohort(s). LIVE rows require a verified actual fill,
        # but this must not accidentally filter out explicitly selected PAPER/BACKTEST/COUNTERFACTUAL rows.
        placeholders = ",".join("?" for _ in selected_types)
        where_clauses.append(f"record_type IN ({placeholders})")
        params.extend(selected_types)
        if "LIVE" in selected_types:
            where_clauses.append("(record_type != 'LIVE' OR (actual_entry IS NOT NULL AND actual_entry > 0))")

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT id, signal_id, trade_id, instrument, trading_horizon, record_type,
                   market_session, status, accept_reject_verdict, decision_source,
                   features_json, regime, spot_at_entry, intended_entry, intended_target,
                   intended_sl, sl_threshold_pct, quantity, capital_at_risk,
                   actual_entry, actual_exit, exit_reason, hold_duration_mins,
                   mae_pct, mfe_pct, gross_pnl, net_pnl, total_costs, r_multiple,
                   is_winner, target_hit_before_sl, re_entry_count, rules_followed,
                   model_version, created_at
            FROM {UNIFIED_DATASET_TABLE}
            WHERE {where_sql}
            ORDER BY created_at ASC
        """

        with self._lock:
            with self._get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        # Decode and validate complete finite entry-time snapshots. One corrupt row must not
        # abort the query, and absent fields must not be silently filled with zero.
        valid_records = []
        valid_indices = []
        for row_index, encoded in zip(df.index, df["features_json"]):
            try:
                decoded = json.loads(encoded) if encoded else None
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(decoded, dict):
                continue

            normalized = {}
            valid = True
            for col in UNIFIED_FEATURE_COLS:
                value = _finite_feature_value(decoded.get(col))
                if value is None:
                    valid = False
                    break
                normalized[col] = value
            if valid:
                valid_indices.append(row_index)
                valid_records.append(normalized)

        if not valid_indices:
            return pd.DataFrame()

        df = df.loc[valid_indices].copy()
        feat_df = pd.DataFrame(valid_records, index=valid_indices, columns=UNIFIED_FEATURE_COLS)
        drop_cols = ["features_json"] + [c for c in UNIFIED_FEATURE_COLS if c in df.columns]
        combined = pd.concat([df.drop(columns=drop_cols, errors="ignore"), feat_df], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    # =========================================================================
    # 5. DATASET AUDIT & STATISTICS
    # =========================================================================
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Provides an executive audit of the unified learning dataset across all 4 horizons."""
        with self._lock:
            with self._get_connection() as conn:
                total_rows = conn.execute(f"SELECT count(*) FROM {UNIFIED_DATASET_TABLE}").fetchone()[0]

                rows = conn.execute(f"""
                SELECT trading_horizon, record_type, status, is_winner, net_pnl
                FROM {UNIFIED_DATASET_TABLE}
                """).fetchall()

        by_horizon: Dict[str, Dict[str, Any]] = {
            "INTRADAY": {"total": 0, "live": 0, "counterfactual": 0, "resolved": 0, "winners": 0, "win_rate": 0.0, "net_pnl": 0.0},
            "WEEKLY": {"total": 0, "live": 0, "counterfactual": 0, "resolved": 0, "winners": 0, "win_rate": 0.0, "net_pnl": 0.0},
            "MONTHLY": {"total": 0, "live": 0, "counterfactual": 0, "resolved": 0, "winners": 0, "win_rate": 0.0, "net_pnl": 0.0},
            "SWING": {"total": 0, "live": 0, "counterfactual": 0, "resolved": 0, "winners": 0, "win_rate": 0.0, "net_pnl": 0.0}
        }

        for r in rows:
            h = str(r["trading_horizon"]).upper()
            if h not in by_horizon:
                continue
            stat = by_horizon[h]
            stat["total"] += 1
            if r["record_type"] == "COUNTERFACTUAL":
                stat["counterfactual"] += 1
            else:
                stat["live"] += 1

            if r["status"] in ("RESOLVED_TARGET", "RESOLVED_STOP_LOSS", "EXPIRED", "REJECTED_SIM"):
                stat["resolved"] += 1
                if r["is_winner"] == 1:
                    stat["winners"] += 1
                stat["net_pnl"] += float(r["net_pnl"] or 0.0)

        for h, s in by_horizon.items():
            if s["resolved"] > 0:
                s["win_rate"] = round((s["winners"] / s["resolved"]) * 100.0, 1)
                s["net_pnl"] = round(s["net_pnl"], 2)

        return {
            "total_records": total_rows,
            "table_name": UNIFIED_DATASET_TABLE,
            "feature_count": len(UNIFIED_FEATURE_COLS),
            "features": UNIFIED_FEATURE_COLS,
            "by_horizon": by_horizon,
            "status": "ACTIVE_ONLINE",
            "last_audited_ist": format_ist()
        }


# Global Singleton Instance
unified_ml_dataset = UnifiedMLDataset()
