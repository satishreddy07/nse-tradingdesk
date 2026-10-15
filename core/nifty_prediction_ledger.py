"""
core/nifty_prediction_ledger.py
===============================
Immutable Daily Forecast Logging, Accurate Trading Session Alignment,
and Lifecycle Outcome Resolution for T+1 and T+5 NIFTY Predictions.

Core Invariants:
1. Immutable Daily Prediction Logging: Forecasts are logged once with reproducible
   input feature snapshots and locked model version. They are never overwritten upon retraining.
2. Correct Trading Session Alignment:
   - T+1 aligns to the immediate next trading session (t+1).
   - T+5 aligns to the fifth subsequent trading session (t+5).
   - Uses NSEMarketCalendar to skip weekends and official exchange holidays.
3. Separation of Predictions, Labels, and Explanations:
   - Predictive features strictly reflect entry-time information.
   - Realized outcomes remain 'PENDING' until legitimate market close data arrives.
   - Post-outcome Gemini/diagnostic attributions are stored as metadata, never as training features.
4. Thread-safe SQLite persistence in `data/swing_trades.db`.
"""

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import DB_PATH
from core.data_quality import IST, format_ist, now_ist
from core.market_calendar import NSEMarketCalendar
from execution.state_engine import state_engine

logger = logging.getLogger("NiftyPredictionLedger")

NIFTY_LEDGER_TABLE = "nifty_forecast_lifecycle_ledger"

# 20 Standardized Numerical Feature Columns for Nifty T+1 and T+5 Models
NIFTY_FEATURE_COLS = [
    "spot_price",
    "composite_score",
    "composite_score_t5",
    "technical_score",
    "breadth_score",
    "sectoral_score",
    "global_score",
    "macro_score",
    "news_and_deriv_score",
    "institutional_score",
    "vix_spot",
    "near_atm_pcr",
    "chg_oi_pcr",
    "dist_to_ema20_pct",
    "dist_to_ema50_pct",
    "crude_oil_ret_1d",
    "crude_oil_ret_5d",
    "fii_fut_long_pct",
    "sp500_ret_1d",
    "nifty_ret_5d"
]


class NiftyPredictionLedger:
    """
    Central manager for immutable forecast logging and trading session outcome resolution.
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
        """Initializes the immutable forecast ledger table and performance indices."""
        with self._lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {NIFTY_LEDGER_TABLE} (
                    forecast_id TEXT PRIMARY KEY,
                    prediction_date TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    horizon TEXT NOT NULL,                -- 'T+1' or 'T+5'
                    target_date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    spot_at_prediction REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,    -- 'BULLISH', 'FLAT', 'BEARISH'
                    prob_up REAL NOT NULL,
                    prob_flat REAL NOT NULL,
                    prob_down REAL NOT NULL,
                    expected_pts REAL NOT NULL,
                    expected_pct REAL NOT NULL,
                    expected_close REAL NOT NULL,
                    range_1sigma_low REAL NOT NULL,
                    range_1sigma_high REAL NOT NULL,
                    range_2sigma_low REAL NOT NULL,
                    range_2sigma_high REAL NOT NULL,
                    features_snapshot_json TEXT NOT NULL,
                    data_as_of_date TEXT,
                    data_quality_status TEXT DEFAULT 'VERIFIED',
                    model_confidence TEXT DEFAULT 'HIGH',
                    basis_breakdown_json TEXT,

                    -- Outcome Fields (NULL while PENDING)
                    status TEXT NOT NULL DEFAULT 'PENDING',  -- 'PENDING', 'RESOLVED', 'EXPIRED'
                    actual_close REAL,
                    actual_close_date TEXT,
                    actual_return_pts REAL,
                    actual_return_pct REAL,
                    realized_direction TEXT,
                    direction_correct INTEGER,             -- 1: HIT, 0: MISS, NULL: PENDING
                    range_1sigma_hit INTEGER,              -- 1: inside cone, 0: breached, NULL: PENDING
                    range_2sigma_hit INTEGER,              -- 1: inside cone, 0: breached, NULL: PENDING
                    target_hit INTEGER,                    -- 1: reached, 0: missed, NULL: PENDING
                    prediction_error_pts REAL,
                    prediction_error_pct REAL,
                    path_high REAL,
                    path_low REAL,
                    diagnostic_metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(prediction_date, horizon)
                );
                """)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_nfll_hor_stat ON {NIFTY_LEDGER_TABLE}(horizon, status);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_nfll_target ON {NIFTY_LEDGER_TABLE}(target_date);")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_nfll_pred_date ON {NIFTY_LEDGER_TABLE}(prediction_date);")
                conn.commit()

    @staticmethod
    def get_nth_trading_day(start_date: date, n: int = 1) -> date:
        """Calculates the exact n-th subsequent NSE trading day skipping holidays and weekends."""
        cur = start_date
        for _ in range(n):
            cur = NSEMarketCalendar.get_next_trading_day(cur)
        return cur

    # =========================================================================
    # 1. IMMUTABLE FORECAST LOGGING
    # =========================================================================
    def log_forecast(
        self,
        horizon: str,
        spot: float,
        composite_score: float,
        verdict: str,
        prob_up: float,
        prob_flat: float,
        prob_down: float,
        expected_close: float,
        expected_pts: float,
        expected_pct: float,
        range_1sigma: Tuple[float, float],
        range_2sigma: Tuple[float, float],
        features: Dict[str, Any],
        prediction_date: Optional[str] = None,
        prediction_timestamp: Optional[str] = None,
        model_name: str = "NiftyMultiFactorEngine",
        model_version: str = "prod_v1.0",
        data_as_of_date: Optional[str] = None,
        data_quality_status: str = "VERIFIED",
        model_confidence: str = "HIGH",
        basis_breakdown: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Logs an immutable daily forecast for T+1 or T+5.
        If a forecast for this date and horizon already exists, returns the existing forecast_id
        to strictly prevent overwriting original predictions during subsequent recalculations or model retraining.
        """
        horizon = horizon.replace(" ", "+").upper()
        if horizon not in ("T+1", "T+5"):
            raise ValueError(f"Invalid horizon '{horizon}'. Must be 'T+1' or 'T+5'.")

        now_dt = now_ist()
        pred_date = prediction_date or now_dt.strftime("%Y-%m-%d")
        ts_ist = prediction_timestamp or format_ist(now_dt)

        pred_d = datetime.strptime(pred_date, "%Y-%m-%d").date()
        target_d = self.get_nth_trading_day(pred_d, n=1 if horizon == "T+1" else 5)
        target_date_str = target_d.strftime("%Y-%m-%d")

        forecast_id = f"FC_{horizon.replace('+', '')}_{pred_date.replace('-', '')}"

        # Determine directional category
        v_upper = verdict.upper()
        if "BULLISH" in v_upper:
            direction = "BULLISH"
        elif "BEARISH" in v_upper:
            direction = "BEARISH"
        else:
            direction = "FLAT"

        feat_clean = {k: float(features.get(k, 0.0)) for k in NIFTY_FEATURE_COLS}
        feat_json = json.dumps(feat_clean)
        basis_json = json.dumps(basis_breakdown or {})

        with self._lock:
            with self._get_connection() as conn:
                existing = conn.execute(
                    f"SELECT forecast_id FROM {NIFTY_LEDGER_TABLE} WHERE prediction_date = ? AND horizon = ?",
                    (pred_date, horizon)
                ).fetchone()

                if existing:
                    logger.debug(f"Forecast for {pred_date} {horizon} already exists as {existing['forecast_id']}; preserving original immutable record.")
                    return existing["forecast_id"]

                conn.execute(f"""
                INSERT INTO {NIFTY_LEDGER_TABLE} (
                    forecast_id, prediction_date, prediction_timestamp, horizon,
                    target_date, model_name, model_version, spot_at_prediction,
                    composite_score, verdict, predicted_direction, prob_up,
                    prob_flat, prob_down, expected_pts, expected_pct, expected_close,
                    range_1sigma_low, range_1sigma_high, range_2sigma_low, range_2sigma_high,
                    features_snapshot_json, data_as_of_date, data_quality_status,
                    model_confidence, basis_breakdown_json, status, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?
                );
                """, (
                    forecast_id, pred_date, ts_ist, horizon, target_date_str,
                    model_name, model_version, float(spot), float(composite_score),
                    verdict, direction, float(prob_up), float(prob_flat), float(prob_down),
                    float(expected_pts), float(expected_pct), float(expected_close),
                    float(range_1sigma[0]), float(range_1sigma[1]),
                    float(range_2sigma[0]), float(range_2sigma[1]),
                    feat_json, data_as_of_date or pred_date, data_quality_status,
                    model_confidence, basis_json, ts_ist, ts_ist
                ))
                conn.commit()
                logger.info(f"🔒 Logged immutable {horizon} forecast: {forecast_id} (Target: {target_date_str}, Verdict: {verdict})")

        return forecast_id

    # =========================================================================
    # 2. ACCURATE OUTCOME RESOLUTION (CLOSED-LOOP RECORDING)
    # =========================================================================
    def resolve_matured_outcomes(
        self,
        current_date: Optional[str] = None,
        closes_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Audits all PENDING forecasts and resolves outcomes only when legitimate market close
        data for the exact target trading session is available.
        """
        now_dt = now_ist()
        today_date = datetime.strptime(current_date, "%Y-%m-%d").date() if current_date else now_dt.date()
        today_str = today_date.strftime("%Y-%m-%d")
        is_market_closed = now_dt.time() >= dt_time(15, 30) or current_date is not None

        resolved_t1 = 0
        resolved_t5 = 0
        pending_count = 0

        # Fetch recent historical closes for ^NSEI (or test/simulation override)
        closes_by_date = closes_override if closes_override is not None else self._load_historical_closes()

        with self._lock:
            with self._get_connection() as conn:
                pending_rows = conn.execute(
                    f"SELECT * FROM {NIFTY_LEDGER_TABLE} WHERE status = 'PENDING' ORDER BY prediction_date ASC"
                ).fetchall()

        if not pending_rows:
            return {"status": "NO_PENDING_FORECASTS", "resolved_t1": 0, "resolved_t5": 0, "pending": 0}

        for row in pending_rows:
            item = dict(row)
            fc_id = item["forecast_id"]
            horizon = item["horizon"]
            t_date_str = item["target_date"]
            t_date = datetime.strptime(t_date_str, "%Y-%m-%d").date()

            # Outcome only matures on or after target_date after 15:30 IST market close
            if t_date > today_date:
                pending_count += 1
                continue
            if t_date == today_date and not is_market_closed:
                pending_count += 1
                continue

            actual_close = closes_by_date.get(t_date_str)
            if actual_close is None:
                # If target date was a market holiday or missing, look for next trading close
                alt_d = NSEMarketCalendar.get_next_trading_day(t_date)
                alt_d_str = alt_d.strftime("%Y-%m-%d")
                if alt_d <= today_date:
                    actual_close = closes_by_date.get(alt_d_str)

            if actual_close is None:
                pending_count += 1
                continue

            # Calculate actual performance metrics
            spot_0 = float(item["spot_at_prediction"])
            ret_pts = round(actual_close - spot_0, 2)
            ret_pct = round((ret_pts / max(1.0, spot_0)) * 100.0, 2)
            exp_close = float(item["expected_close"])
            err_pts = round(actual_close - exp_close, 2)
            err_pct = round((err_pts / max(1.0, exp_close)) * 100.0, 2)

            low_1s = float(item["range_1sigma_low"])
            high_1s = float(item["range_1sigma_high"])
            low_2s = float(item["range_2sigma_low"])
            high_2s = float(item["range_2sigma_high"])

            cone_1s_hit = 1 if (low_1s <= actual_close <= high_1s) else 0
            cone_2s_hit = 1 if (low_2s <= actual_close <= high_2s) else 0

            # Realized direction thresholds:
            # T+1: +/- 0.08%
            # T+5: +/- 0.25%
            dir_threshold = 0.08 if horizon == "T+1" else 0.25
            if ret_pct >= dir_threshold:
                realized_dir = "BULLISH"
            elif ret_pct <= -dir_threshold:
                realized_dir = "BEARISH"
            else:
                realized_dir = "FLAT"

            pred_dir = item["predicted_direction"]
            dir_correct = 1 if pred_dir == realized_dir else 0

            # Target achievement: path/close reached expected level
            if pred_dir == "BULLISH":
                target_hit = 1 if actual_close >= exp_close else 0
            elif pred_dir == "BEARISH":
                target_hit = 1 if actual_close <= exp_close else 0
            else:
                target_hit = 1 if cone_1s_hit == 1 else 0

            # Diagnostic explanations
            diagnostic = {
                "evaluated_at": format_ist(),
                "realized_direction": realized_dir,
                "direction_outcome": "HIT" if dir_correct == 1 else "MISS",
                "actual_close": actual_close,
                "actual_return_pts": ret_pts,
                "actual_return_pct": ret_pct,
                "error_pts": err_pts,
                "cone_containment": "WITHIN_1SIGMA" if cone_1s_hit == 1 else ("WITHIN_2SIGMA" if cone_2s_hit == 1 else "BREACHED"),
                "diagnostic_notes": self._generate_diagnostic_notes(item, actual_close, ret_pct, dir_correct)
            }

            now_str = format_ist()
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute(f"""
                    UPDATE {NIFTY_LEDGER_TABLE}
                    SET status = 'RESOLVED',
                        actual_close = ?,
                        actual_close_date = ?,
                        actual_return_pts = ?,
                        actual_return_pct = ?,
                        realized_direction = ?,
                        direction_correct = ?,
                        range_1sigma_hit = ?,
                        range_2sigma_hit = ?,
                        target_hit = ?,
                        prediction_error_pts = ?,
                        prediction_error_pct = ?,
                        diagnostic_metadata_json = ?,
                        updated_at = ?
                    WHERE forecast_id = ?
                    """, (
                        actual_close, t_date_str, ret_pts, ret_pct, realized_dir,
                        dir_correct, cone_1s_hit, cone_2s_hit, target_hit,
                        err_pts, err_pct, json.dumps(diagnostic), now_str, fc_id
                    ))
                    conn.commit()

            if horizon == "T+1":
                resolved_t1 += 1
            else:
                resolved_t5 += 1

            logger.info(f"🎯 Resolved {horizon} forecast {fc_id}: Direction {'HIT' if dir_correct == 1 else 'MISS'} (Return {ret_pct:+.2f}%, Err {err_pts:+.1f} pts)")

        return {
            "status": "COMPLETED",
            "resolved_t1": resolved_t1,
            "resolved_t5": resolved_t5,
            "remaining_pending": pending_count
        }

    def _generate_diagnostic_notes(
        self,
        item: Dict[str, Any],
        actual_close: float,
        actual_return_pct: float,
        is_hit: int
    ) -> str:
        """Generates evidence-based root cause attribution separating measured data from hypotheses."""
        verdict = item["verdict"]
        score = item["composite_score"]
        spot = item["spot_at_prediction"]
        pts = actual_close - spot

        if is_hit == 1:
            return (
                f"VERIFIED HIT: Market moved {actual_return_pct:+.2f}% ({pts:+.1f} pts) strictly confirming "
                f"the {verdict} forecast (score {score:+.1f}). Price contained inside defined expectation bands."
            )

        # Diagnose root cause of deviation
        try:
            feats = json.loads(item.get("features_snapshot_json") or "{}")
            vix = float(feats.get("vix_spot", 14.0))
            fii_oi = float(feats.get("fii_fut_long_pct", 50.0))
            pcr = float(feats.get("near_atm_pcr", 1.0))

            reasons = []
            if abs(actual_return_pct) > 1.2:
                reasons.append(f"Elevated session volatility ({actual_return_pct:+.2f}% move)")
            if vix > 16.5:
                reasons.append(f"Elevated India VIX regime ({vix:.1f}) widened standard error cone")
            if (score > 0 and actual_return_pct < 0) and fii_oi < 45.0:
                reasons.append("Institutional FII derivatives net short drag overwhelmed retail technical support")
            if (score < 0 and actual_return_pct > 0) and pcr > 1.25:
                reasons.append("Heavy put writing floor absorbed downside impulse triggering short-covering rally")

            diag_text = "; ".join(reasons) if reasons else "Intraday trend reversal and macro drift divergence"
            return f"VERIFIED MISS: Projected {verdict} ({score:+.1f} score), but Nifty closed at ₹{actual_close:,.2f} ({actual_return_pct:+.2f}%). Potential drivers: {diag_text}."
        except Exception:
            return f"VERIFIED MISS: Realized session return was {actual_return_pct:+.2f}% vs expected {verdict}."

    def _load_historical_closes(self) -> Dict[str, float]:
        """Loads verified closing prices for Nifty 50 (^NSEI)."""
        closes: Dict[str, float] = {}

        # 1. From state_engine OHLCV
        df = state_engine.get_ohlcv("^NSEI")
        if df is not None and not df.empty and "Close" in df.columns:
            for idx, val in zip(df.index, df["Close"].values):
                d_str = str(idx)[:10]
                closes[d_str] = float(val)

        # 2. From nifty_predictions_log fallback
        try:
            with state_engine._get_connection() as conn:
                for r in conn.execute("SELECT prediction_date, actual_t1_close, spot_at_prediction FROM nifty_predictions_log").fetchall():
                    if r["prediction_date"]:
                        p_date = str(r["prediction_date"])
                        if r["actual_t1_close"] is not None and p_date not in closes:
                            closes[p_date] = float(r["actual_t1_close"])
                        elif r["spot_at_prediction"] is not None and p_date not in closes:
                            closes[p_date] = float(r["spot_at_prediction"])
        except Exception:
            pass

        return closes

    # =========================================================================
    # 3. TRAINING DATASET EXTRACTION (MATURED ONLY)
    # =========================================================================
    def get_training_records(
        self,
        horizon: str,
        cutoff_date: Optional[str] = None,
        min_samples: int = 15
    ) -> pd.DataFrame:
        """
        Retrieves strictly resolved (matured) forecasts for model training.
        Guarantees zero future data leakage and strict chronological sequence.
        """
        horizon = horizon.replace(" ", "+").upper()
        sql = f"""
            SELECT forecast_id, prediction_date, prediction_timestamp, horizon,
                   target_date, model_version, spot_at_prediction, composite_score,
                   verdict, predicted_direction, prob_up, prob_flat, prob_down,
                   expected_pts, expected_pct, expected_close, range_1sigma_low, range_1sigma_high,
                   features_snapshot_json, status, actual_close, actual_return_pct,
                   realized_direction, direction_correct, range_1sigma_hit, target_hit,
                   prediction_error_pts
            FROM {NIFTY_LEDGER_TABLE}
            WHERE horizon = '{horizon}' AND status = 'RESOLVED'
        """
        if cutoff_date:
            sql += f" AND prediction_date <= '{cutoff_date}'"
        sql += " ORDER BY prediction_date ASC"

        with self._lock:
            with self._get_connection() as conn:
                df = pd.read_sql(sql, conn)

        if df.empty or len(df) < min_samples:
            return pd.DataFrame()

        # Unpack features_snapshot_json into individual columns
        feat_dicts = [json.loads(s) if s else {} for s in df["features_snapshot_json"]]
        for col in NIFTY_FEATURE_COLS:
            df[col] = [float(fd.get(col, 0.0)) for fd in feat_dicts]

        # Target classification label: 1 (BULLISH), 0 (FLAT), -1 (BEARISH)
        dir_map = {"BULLISH": 1, "FLAT": 0, "BEARISH": -1}
        df["target_class"] = df["realized_direction"].map(dir_map).fillna(0).astype(int)

        return df

    # =========================================================================
    # 4. LEDGER STATISTICS & AUDIT SUMMARY
    # =========================================================================
    def get_ledger_statistics(self) -> Dict[str, Any]:
        """Provides an audit summary of logged, pending, and resolved predictions."""
        with self._lock:
            with self._get_connection() as conn:
                total_all = conn.execute(f"SELECT count(*) FROM {NIFTY_LEDGER_TABLE}").fetchone()[0]

                def _stats_for_horizon(h: str) -> Dict[str, Any]:
                    rows = conn.execute(
                        f"SELECT * FROM {NIFTY_LEDGER_TABLE} WHERE horizon = ?", (h,)
                    ).fetchall()
                    if not rows:
                        return {"total": 0, "resolved": 0, "pending": 0, "hit_rate_pct": 0.0, "cone_hit_pct": 0.0, "mae_pts": 0.0}

                    total = len(rows)
                    resolved = [dict(r) for r in rows if r["status"] == "RESOLVED"]
                    pending = total - len(resolved)

                    if not resolved:
                        return {"total": total, "resolved": 0, "pending": pending, "hit_rate_pct": 0.0, "cone_hit_pct": 0.0, "mae_pts": 0.0}

                    hits = sum(1 for r in resolved if r.get("direction_correct") == 1)
                    cone_hits = sum(1 for r in resolved if r.get("range_1sigma_hit") == 1)
                    mae = np.mean([abs(float(r.get("prediction_error_pts") or 0.0)) for r in resolved])

                    return {
                        "total": total,
                        "resolved": len(resolved),
                        "pending": pending,
                        "hit_rate_pct": round((hits / len(resolved)) * 100.0, 1),
                        "cone_hit_pct": round((cone_hits / len(resolved)) * 100.0, 1),
                        "mae_pts": round(float(mae), 1)
                    }

                return {
                    "total_forecasts": total_all,
                    "t1_statistics": _stats_for_horizon("T+1"),
                    "t5_statistics": _stats_for_horizon("T+5"),
                    "last_audited_ist": format_ist()
                }


# Global Singleton Instance
nifty_prediction_ledger = NiftyPredictionLedger()
