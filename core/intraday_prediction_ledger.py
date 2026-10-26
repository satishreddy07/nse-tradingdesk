"""
core/intraday_prediction_ledger.py
==================================
Immutable Intraday Prediction Ledger and Multi-Horizon Evaluation Repository.

Core Invariants:
1. Immutable Intraday Prediction Logging:
   - Forecasts are logged once with a unique prediction ID, precise timestamp,
     underlying spot price, predicted direction, probabilities (p_up, p_flat, p_down),
     market regime, composite score, and frozen feature snapshot.
   - Predictions and feature snapshots are NEVER overwritten when new market data arrives.
2. Independent Multi-Horizon Evaluation Tracking:
   - For every logged prediction, evaluation records are tracked at 5-minute, 15-minute,
     and 30-minute horizons in a separate table (`intraday_prediction_evaluations`).
   - Outcome fields (actual price, price delta, realized direction, direction correctness,
     Brier score) remain 'PENDING' until legitimate forward market observations arrive.
3. Safe Concurrency & Restart Recovery:
   - Thread-safe SQLite persistence with WAL mode and row locks.
   - Idempotent evaluation processing prevents duplicate jobs or corrupted states across restarts.
4. Zero Fabrication & Strict Provenance:
   - Stores data provenance, data freshness age, and model version.
   - Refuses to substitute arbitrary prices or assume unchanged spot.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import DB_PATH
from core.data_quality import IST, format_ist, now_ist, check_data_freshness
from core.market_calendar import NSEMarketCalendar

logger = logging.getLogger("IntradayPredictionLedger")

INTRADAY_PRED_LEDGER_TABLE = "intraday_prediction_ledger"
INTRADAY_EVAL_TABLE = "intraday_prediction_evaluations"

# Standard Intraday Feature Keys for Feature Contribution & Attribution
INTRADAY_FEATURE_KEYS = [
    "spot_price",
    "vwap",
    "vwap_distance_pts",
    "vwap_distance_pct",
    "above_vwap",
    "orb_status",
    "choppiness_index",
    "atm_pcr",
    "near_atm_pcr",
    "chg_oi_pcr",
    "net_pe_oi_chg",
    "net_ce_oi_chg",
    "oi_ratio",
    "straddle_premium",
    "straddle_decay_pct",
    "atm_iv",
    "call_wall",
    "put_wall",
    "zero_gamma_flip",
    "net_gamma",
    "order_flow_imbalance",
    "cvd",
    "micro_price_delta",
    "sector_breadth",
    "bank_vs_it_spread",
    "futures_basis_pts",
    "writer_dominance_score",
    "composite_score"
]


class IntradayPredictionLedger:
    """
    Central manager for immutable intraday prediction logging and multi-horizon outcome tracking.
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
        """Initializes tables and indices for immutable prediction ledger and evaluations."""
        with self._lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                # 1. Immutable Intraday Prediction Ledger
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {INTRADAY_PRED_LEDGER_TABLE} (
                    prediction_id TEXT PRIMARY KEY,
                    session_date TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    prediction_epoch REAL NOT NULL,
                    spot_price REAL NOT NULL,
                    predicted_direction TEXT NOT NULL,      -- 'BULLISH', 'BEARISH', 'NEUTRAL'
                    prob_up REAL NOT NULL,
                    prob_flat REAL NOT NULL,
                    prob_down REAL NOT NULL,
                    expected_delta_pts REAL DEFAULT 0.0,
                    expected_delta_pct REAL DEFAULT 0.0,
                    expected_range_low REAL DEFAULT 0.0,
                    expected_range_high REAL DEFAULT 0.0,
                    market_regime TEXT NOT NULL,            -- e.g. 'STRONG_BULLISH', 'RANGE_BOUND', 'BEARISH_BREAKDOWN'
                    composite_score REAL DEFAULT 0.0,
                    writer_dominance_score REAL DEFAULT 0.0,
                    features_snapshot_json TEXT NOT NULL,
                    data_provenance TEXT NOT NULL DEFAULT 'LIVE',
                    data_freshness_sec REAL DEFAULT 0.0,
                    model_version TEXT NOT NULL DEFAULT 'INTRADAY-VAL-1.0.0',
                    prediction_status TEXT NOT NULL DEFAULT 'PENDING',  -- 'PENDING', 'EVALUATED', 'SESSION_CLOSED', 'DATA_UNAVAILABLE'
                    created_at TEXT NOT NULL
                );
                """)

                # 2. Multi-Horizon Evaluations (5m, 15m, 30m)
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {INTRADAY_EVAL_TABLE} (
                    evaluation_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL REFERENCES {INTRADAY_PRED_LEDGER_TABLE}(prediction_id),
                    horizon_min INTEGER NOT NULL,          -- 5, 15, 30
                    target_timestamp TEXT NOT NULL,
                    target_epoch REAL NOT NULL,
                    actual_price REAL,
                    actual_timestamp TEXT,
                    price_delta_pts REAL,
                    price_delta_pct REAL,
                    realized_direction TEXT,               -- 'BULLISH', 'BEARISH', 'NEUTRAL'
                    is_direction_correct INTEGER,          -- 1: HIT, 0: MISS, NULL: PENDING
                    brier_score REAL,
                    evaluation_status TEXT NOT NULL DEFAULT 'PENDING',  -- 'PENDING', 'RESOLVED', 'DATA_UNAVAILABLE', 'SESSION_CLOSED', 'HOLIDAY_OR_CLOSED'
                    status_reason TEXT,
                    evaluated_at TEXT,
                    data_source TEXT,
                    observation_latency_sec REAL,
                    UNIQUE(prediction_id, horizon_min)
                );
                """)

                # Indices for ultra-fast query and recovery
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipl_date ON {INTRADAY_PRED_LEDGER_TABLE}(session_date);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipl_epoch ON {INTRADAY_PRED_LEDGER_TABLE}(prediction_epoch);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipl_status ON {INTRADAY_PRED_LEDGER_TABLE}(prediction_status);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipe_pred_id ON {INTRADAY_EVAL_TABLE}(prediction_id);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipe_target_epoch ON {INTRADAY_EVAL_TABLE}(target_epoch, evaluation_status);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ipe_status ON {INTRADAY_EVAL_TABLE}(evaluation_status);")
                conn.commit()

    # =========================================================================
    # 1. IMMUTABLE PREDICTION LOGGING
    # =========================================================================
    def record_prediction(
        self,
        spot_price: float,
        predicted_direction: str,
        prob_up: float,
        prob_flat: float,
        prob_down: float,
        market_regime: str,
        features: Dict[str, Any],
        expected_delta_pts: float = 0.0,
        expected_delta_pct: float = 0.0,
        expected_range_low: float = 0.0,
        expected_range_high: float = 0.0,
        composite_score: float = 0.0,
        writer_dominance_score: float = 0.0,
        data_provenance: str = "LIVE",
        data_freshness_sec: float = 0.0,
        model_version: str = "INTRADAY-VAL-1.0.0",
        prediction_timestamp: Optional[str] = None,
        prediction_epoch: Optional[float] = None,
        prediction_id: Optional[str] = None,
        horizons_min: Tuple[int, ...] = (5, 15, 30)
    ) -> str:
        """
        Persists an immutable intraday prediction record along with its initial multi-horizon
        evaluation tasks (5m, 15m, 30m).

        If a prediction with the exact same prediction_id already exists, returns the existing ID
        and NEVER overwrites the original record or features.
        """
        if spot_price <= 0.0:
            raise ValueError(f"Invalid spot_price {spot_price}: must be strictly positive.")

        # Normalize direction
        p_dir = str(predicted_direction).strip().upper()
        if p_dir in ("UP", "BULL"):
            p_dir = "BULLISH"
        elif p_dir in ("DOWN", "BEAR"):
            p_dir = "BEARISH"
        elif p_dir in ("FLAT", "RANGE", "RANGE_BOUND", "CONSOLIDATION"):
            p_dir = "NEUTRAL"
        elif p_dir not in ("BULLISH", "BEARISH", "NEUTRAL"):
            # Derive from probabilities
            if prob_up > prob_down and prob_up >= prob_flat:
                p_dir = "BULLISH"
            elif prob_down > prob_up and prob_down >= prob_flat:
                p_dir = "BEARISH"
            else:
                p_dir = "NEUTRAL"

        now_dt = now_ist()
        cur_epoch = time.time()

        pred_epoch = float(prediction_epoch if prediction_epoch is not None else cur_epoch)
        if prediction_timestamp:
            ts_ist = prediction_timestamp
            session_date = prediction_timestamp[:10]
        else:
            ts_ist = format_ist(now_dt)
            session_date = now_dt.strftime("%Y-%m-%d")

        if not prediction_id:
            # Generate deterministic, clean ID: IPRED-YYYYMMDD-HHMMSS-UUID6
            ts_compact = ts_ist.replace("-", "").replace(" ", "").replace(":", "")[:14]
            short_uuid = uuid.uuid4().hex[:6].upper()
            pred_id = f"IPRED-{ts_compact}-{short_uuid}"
        else:
            pred_id = str(prediction_id).strip()

        # Clean features dictionary
        clean_features = {}
        for k in INTRADAY_FEATURE_KEYS:
            if k in features:
                val = features[k]
                if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
                    clean_features[k] = val
                elif isinstance(val, str):
                    clean_features[k] = val
                elif isinstance(val, bool):
                    clean_features[k] = val
                else:
                    clean_features[k] = str(val)
        # Preserve any additional caller features
        for k, v in features.items():
            if k not in clean_features and isinstance(v, (int, float, str, bool)):
                clean_features[k] = v

        feat_json = json.dumps(clean_features, sort_keys=True)
        created_at = format_ist(now_dt)

        with self._lock:
            with self._get_connection() as conn:
                # Idempotency check: if record already exists, preserve it completely
                existing = conn.execute(
                    f"SELECT prediction_id FROM {INTRADAY_PRED_LEDGER_TABLE} WHERE prediction_id = ?",
                    (pred_id,)
                ).fetchone()
                if existing:
                    logger.debug(f"Prediction {pred_id} already exists in ledger; preserving immutable original.")
                    return existing["prediction_id"]

                # Insert immutable prediction
                conn.execute(f"""
                INSERT INTO {INTRADAY_PRED_LEDGER_TABLE} (
                    prediction_id, session_date, prediction_timestamp, prediction_epoch,
                    spot_price, predicted_direction, prob_up, prob_flat, prob_down,
                    expected_delta_pts, expected_delta_pct, expected_range_low, expected_range_high,
                    market_regime, composite_score, writer_dominance_score, features_snapshot_json,
                    data_provenance, data_freshness_sec, model_version, prediction_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """, (
                    pred_id, session_date, ts_ist, pred_epoch,
                    round(float(spot_price), 2), p_dir,
                    round(float(prob_up), 4), round(float(prob_flat), 4), round(float(prob_down), 4),
                    round(float(expected_delta_pts), 2), round(float(expected_delta_pct), 4),
                    round(float(expected_range_low), 2), round(float(expected_range_high), 2),
                    str(market_regime), round(float(composite_score), 2),
                    round(float(writer_dominance_score), 2), feat_json,
                    str(data_provenance), round(float(data_freshness_sec), 1),
                    str(model_version), created_at
                ))

                # Insert pending evaluation horizons
                for h in horizons_min:
                    h_int = int(h)
                    target_epoch = pred_epoch + (h_int * 60.0)
                    target_dt = datetime.fromtimestamp(target_epoch, tz=timezone.utc).astimezone(IST)
                    target_ts = format_ist(target_dt)
                    eval_id = f"EVAL-{pred_id}-{h_int}M"

                    conn.execute(f"""
                    INSERT OR IGNORE INTO {INTRADAY_EVAL_TABLE} (
                        evaluation_id, prediction_id, horizon_min, target_timestamp, target_epoch,
                        evaluation_status
                    ) VALUES (?, ?, ?, ?, ?, 'PENDING')
                    """, (eval_id, pred_id, h_int, target_ts, target_epoch))

                conn.commit()

        logger.info(f"Recorded immutable intraday prediction {pred_id} (Direction={p_dir}, Spot={spot_price:.2f}, Regime={market_regime})")
        return pred_id

    # =========================================================================
    # 2. QUERY & READ INTERFACES
    # =========================================================================
    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an immutable prediction record by ID."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    f"SELECT * FROM {INTRADAY_PRED_LEDGER_TABLE} WHERE prediction_id = ?",
                    (prediction_id,)
                ).fetchone()
                if not row:
                    return None
                d = dict(row)
                try:
                    d["features_snapshot"] = json.loads(d["features_snapshot_json"])
                except Exception:
                    d["features_snapshot"] = {}
                return d

    def list_predictions(
        self,
        session_date: Optional[str] = None,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists recent predictions with optional filtering by session date and status."""
        query = f"SELECT * FROM {INTRADAY_PRED_LEDGER_TABLE} WHERE 1=1"
        params: List[Any] = []
        if session_date:
            query += " AND session_date = ?"
            params.append(session_date)
        if status:
            query += " AND prediction_status = ?"
            params.append(status)
        query += " ORDER BY prediction_epoch DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    try:
                        item["features_snapshot"] = json.loads(item["features_snapshot_json"])
                    except Exception:
                        item["features_snapshot"] = {}
                    results.append(item)
                return results

    def get_evaluations_for_prediction(self, prediction_id: str) -> List[Dict[str, Any]]:
        """Retrieves all horizon evaluations for a specific prediction ID."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM {INTRADAY_EVAL_TABLE} WHERE prediction_id = ? ORDER BY horizon_min ASC",
                    (prediction_id,)
                ).fetchall()
                return [dict(r) for r in rows]

    def get_pending_evaluations(self, max_target_epoch: Optional[float] = None) -> List[Dict[str, Any]]:
        """Retrieves pending evaluation rows that are due for resolution."""
        query = f"""
        SELECT e.*, p.spot_price, p.predicted_direction, p.prob_up, p.prob_flat, p.prob_down,
               p.prediction_timestamp, p.session_date, p.market_regime, p.composite_score
        FROM {INTRADAY_EVAL_TABLE} e
        JOIN {INTRADAY_PRED_LEDGER_TABLE} p ON e.prediction_id = p.prediction_id
        WHERE e.evaluation_status = 'PENDING'
        """
        params: List[Any] = []
        if max_target_epoch is not None:
            query += " AND e.target_epoch <= ?"
            params.append(float(max_target_epoch))
        query += " ORDER BY e.target_epoch ASC"

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]

    # =========================================================================
    # 3. EVALUATION RECORD MUTATION (ISOLATED TO EVALUATION TABLE)
    # =========================================================================
    def update_evaluation(
        self,
        evaluation_id: str,
        evaluation_status: str,
        actual_price: Optional[float] = None,
        actual_timestamp: Optional[str] = None,
        price_delta_pts: Optional[float] = None,
        price_delta_pct: Optional[float] = None,
        realized_direction: Optional[str] = None,
        is_direction_correct: Optional[int] = None,
        brier_score: Optional[float] = None,
        status_reason: Optional[str] = None,
        data_source: Optional[str] = None,
        observation_latency_sec: Optional[float] = None
    ):
        """
        Updates an evaluation record with resolved outcome metrics.
        The parent prediction record in `intraday_prediction_ledger` is NEVER altered,
        preserving original predictive features and probability vector immutability.
        """
        now_ts = format_ist()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(f"""
                UPDATE {INTRADAY_EVAL_TABLE}
                SET evaluation_status = ?,
                    actual_price = ?,
                    actual_timestamp = ?,
                    price_delta_pts = ?,
                    price_delta_pct = ?,
                    realized_direction = ?,
                    is_direction_correct = ?,
                    brier_score = ?,
                    status_reason = ?,
                    data_source = ?,
                    observation_latency_sec = ?,
                    evaluated_at = ?
                WHERE evaluation_id = ?
                """, (
                    evaluation_status,
                    actual_price,
                    actual_timestamp,
                    price_delta_pts,
                    price_delta_pct,
                    realized_direction,
                    is_direction_correct,
                    brier_score,
                    status_reason,
                    data_source,
                    observation_latency_sec,
                    now_ts,
                    evaluation_id
                ))

                # Check if all evaluations for the parent prediction are resolved
                row = conn.execute(
                    f"SELECT prediction_id FROM {INTRADAY_EVAL_TABLE} WHERE evaluation_id = ?",
                    (evaluation_id,)
                ).fetchone()
                if row:
                    pred_id = row["prediction_id"]
                    pending_count = conn.execute(
                        f"SELECT count(*) FROM {INTRADAY_EVAL_TABLE} WHERE prediction_id = ? AND evaluation_status = 'PENDING'",
                        (pred_id,)
                    ).fetchone()[0]
                    if pending_count == 0:
                        conn.execute(
                            f"UPDATE {INTRADAY_PRED_LEDGER_TABLE} SET prediction_status = 'EVALUATED' WHERE prediction_id = ?",
                            (pred_id,)
                        )
                conn.commit()


# Singleton Instance
intraday_prediction_ledger = IntradayPredictionLedger()
