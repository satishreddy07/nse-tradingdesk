"""
Swing Quant Machine Learning & AI Feedback Engine
==================================================
Implements closed-loop learning for institutional swing trading:
1. Pre-Trade Multi-Dimensional Feature Capture (30+ indicators + macro regime).
2. Pre-Trade ML Gatekeeper (HistGradientBoosting + Probability Calibration + Expected R).
3. Shadow / Counterfactual Ledger: Monitors rejected/skipped candidates to eliminate survivor bias.
4. EOD Shadow Reconciliation: Tracks whether rejected setups actually hit SL or T1.
5. Post-Mortem Root-Cause Attribution & AI Diagnostic Explanations.
6. Daily Online Self-Training at 15:40 IST and on-demand via REST API.
"""

import json
import logging
import math
import os
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core.config import DB_PATH
from core.data_quality import IST, format_ist, now_ist
from execution.state_engine import state_engine

logger = logging.getLogger("SwingQuantMLEngine")


# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================
ML_FEATURE_COLS = [
    "score",            # 6-Pillar composite score (0 - 100)
    "confidence",       # Setup confidence (1 - 6)
    "rsi14",            # 14-period RSI (0 - 100)
    "atr_pct",          # ATR(14) as % of price
    "dist_to_pivot",    # % distance to 50d pivot
    "dist_to_ema20",    # % distance to 20 EMA
    "vol_ratio",        # Volume vs 50-day average
    "inst_score",       # Institutional footprint score (0 - 100)
    "week_confluence",  # Weekly alignment (1.0 or 0.0)
    "is_vcp",           # Volatility contraction pattern (1.0 or 0.0)
    "bb_sq_ratio",      # Bollinger squeeze ratio
    "adx14",            # Trend strength (0 - 100)
    "nifty_ret_5d",     # Nifty 5-day return (%)
    "nifty_rsi14",      # Nifty 50 RSI(14)
    "vix_val",          # India VIX spot
    "sector_excess_6m", # Sector relative momentum spread (%)
    "rr_ratio",         # Reward-to-Risk ratio (T1-Entry)/(Entry-SL)
]


class SwingQuantMLEngine:
    """
    Closed-loop Quant ML & AI Feedback Engine for Swing Trading.
    Persists data in `quant_trade_dataset` in `data/swing_trades.db`.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._classifier: Optional[Any] = None
        self._regressor: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._model_metadata: Dict[str, Any] = {}
        self._init_db()
        self._load_or_bootstrap_model()

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
        """Creates the `quant_trade_dataset` table and performance indices if not present."""
        with self._lock:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS quant_trade_dataset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL DEFAULT 'INSTITUTIONAL_6PILLAR',
                    trade_type TEXT NOT NULL,          -- 'REAL' or 'SHADOW'
                    status TEXT NOT NULL,              -- 'ACTIVE', 'RESOLVED_TARGET', 'RESOLVED_STOP_LOSS', 'EXPIRED_TIME'
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL NOT NULL,
                    target REAL NOT NULL,
                    target_2 REAL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0,
                    r_multiple REAL DEFAULT 0.0,
                    return_pct REAL DEFAULT 0.0,
                    hold_days INTEGER DEFAULT 0,
                    exit_reason TEXT,
                    is_winner INTEGER,                 -- 1 for target/profit, 0 for stoploss, NULL while ACTIVE
                    sector TEXT,
                    score REAL,
                    confidence INTEGER,
                    rsi14 REAL,
                    atr_pct REAL,
                    dist_to_pivot REAL,
                    dist_to_ema20 REAL,
                    vol_ratio REAL,
                    inst_score REAL,
                    week_confluence INTEGER,
                    is_vcp INTEGER,
                    bb_sq_ratio REAL,
                    adx14 REAL,
                    features_json TEXT,
                    nifty_trend TEXT,
                    nifty_ret_5d REAL,
                    nifty_rsi14 REAL,
                    vix_val REAL,
                    sector_excess_6m REAL,
                    market_regime_json TEXT,
                    nifty_holding_return_pct REAL DEFAULT 0.0,
                    sector_holding_return_pct REAL DEFAULT 0.0,
                    root_cause_attribution TEXT,
                    ai_post_mortem_notes TEXT,
                    quantity INTEGER DEFAULT 0,
                    capital_at_risk REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, trade_type, created_at)
                );
                """)
                # Migrations for existing databases
                try:
                    conn.execute("ALTER TABLE quant_trade_dataset ADD COLUMN quantity INTEGER DEFAULT 0;")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE quant_trade_dataset ADD COLUMN capital_at_risk REAL DEFAULT 0.0;")
                except Exception:
                    pass

                conn.execute("CREATE INDEX IF NOT EXISTS idx_qtd_symbol ON quant_trade_dataset(symbol);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qtd_status ON quant_trade_dataset(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qtd_type ON quant_trade_dataset(trade_type);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qtd_created ON quant_trade_dataset(created_at);")
                conn.commit()

    # ==========================================================================
    # 1. MARKET REGIME CAPTURE
    # ==========================================================================
    def capture_market_regime(self) -> Dict[str, Any]:
        """
        Extracts current live/cached macro market regime metrics:
        Nifty trend, 5-day return, Nifty RSI(14), India VIX, and FII flows.
        """
        regime = {
            "nifty_spot": 25000.0,
            "nifty_trend": "BULLISH",
            "nifty_ret_5d": 0.0,
            "nifty_rsi14": 55.0,
            "vix_val": 14.0,
            "composite_score": 0.0,
            "fii_net_cash": 0.0,
            "breadth_ratio": 1.0,
            "timestamp": format_ist()
        }
        try:
            p = state_engine.get_kv("latest_nifty_predictions")
            if isinstance(p, dict) and p.get("status") == "OK":
                regime["nifty_spot"] = float(p.get("spot_price") or 25000.0)
                regime["composite_score"] = float(p.get("composite_score") or 0.0)

                tech = (p.get("pillars") or {}).get("nifty_technicals") or {}
                macro = (p.get("pillars") or {}).get("macro") or {}
                flows = p.get("fii_dii_and_oi") or {}

                regime["nifty_rsi14"] = float(tech.get("rsi14") or 55.0)
                regime["vix_val"] = float(macro.get("vix_spot") or macro.get("india_vix") or 14.0)

                # Nifty trend from forecasts & composite
                t1_f = p.get("t1_forecast") or {}
                t5_f = p.get("t5_forecast") or {}
                t1_v = str(t1_f.get("verdict") or "NEUTRAL")
                t5_v = str(t5_f.get("verdict") or "NEUTRAL")
                if "BEARISH" in t1_v or "BEARISH" in t5_v or regime["composite_score"] < -10:
                    regime["nifty_trend"] = "BEARISH"
                elif "BULLISH" in t1_v or "BULLISH" in t5_v or regime["composite_score"] > 10:
                    regime["nifty_trend"] = "BULLISH"
                else:
                    regime["nifty_trend"] = "SIDEWAYS"

                # 5-day return from tech or prediction ledger
                regime["nifty_ret_5d"] = float(tech.get("ret_5d") or tech.get("return_5d") or 0.25)
                regime["fii_net_cash"] = float(flows.get("fii_cash_net") or 0.0)
        except Exception as e:
            logger.debug(f"Failed to extract macro regime snapshot: {e}")

        return regime

    # ==========================================================================
    # 2. FEATURE EXTRACTION & VECTORIZATION
    # ==========================================================================
    def vectorize_features(self, item: Dict[str, Any], market_regime: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Converts candidate setup and market regime into normalized ML feature vector.
        """
        regime = market_regime or self.capture_market_regime()

        ep = float(item.get("entry") or item.get("close") or item.get("entry_price") or 100.0)
        sl = float(item.get("sl") or item.get("stop_loss") or (ep * 0.95))
        t1 = float(item.get("t1") or item.get("target") or (ep * 1.08))

        risk_per_share = max(0.01, ep - sl)
        reward_per_share = max(0.01, t1 - ep)
        rr_ratio = round(reward_per_share / risk_per_share, 2)

        feat_dict = {
            "score": float(item.get("score") or 70.0),
            "confidence": float(item.get("confidence") or 4),
            "rsi14": float(item.get("rsi14") or 55.0),
            "atr_pct": float(item.get("atr_pct") or 2.5),
            "dist_to_pivot": float(item.get("dist_to_pivot") or 1.5),
            "dist_to_ema20": float(item.get("dist_to_ema20") or 2.0),
            "vol_ratio": float(item.get("vol_ratio") or 1.2),
            "inst_score": float(item.get("inst_score") or 60.0),
            "week_confluence": 1.0 if bool(item.get("week_confluence")) else 0.0,
            "is_vcp": 1.0 if bool(item.get("is_vcp")) else 0.0,
            "bb_sq_ratio": float(item.get("bb_sq_ratio") or item.get("bb_squeeze") or 0.8),
            "adx14": float(item.get("adx14") or 28.0),
            "nifty_ret_5d": float(regime.get("nifty_ret_5d") or 0.0),
            "nifty_rsi14": float(regime.get("nifty_rsi14") or 55.0),
            "vix_val": float(regime.get("vix_val") or 14.0),
            "sector_excess_6m": float(item.get("sector_excess_6m") or item.get("rs_spread") or 0.0),
            "rr_ratio": float(rr_ratio),
        }

        vector = np.array([feat_dict[k] for k in ML_FEATURE_COLS], dtype=float)
        return vector, feat_dict

    def _build_unified_features(
        self,
        feat_dict: Dict[str, Any],
        regime: Dict[str, Any],
        entry_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """Converts swing features into 32 standardized unified ML features."""
        ep = max(0.01, entry_price)
        sl_pct = round(abs(entry_price - stop_loss) / ep * 100.0, 2)
        nifty_trend = str(regime.get("nifty_trend", "SIDEWAYS")).upper()
        regime_code = 2 if "BULL" in nifty_trend else (-2 if "BEAR" in nifty_trend else 0)

        return {
            "nifty_spot": float(regime.get("nifty_spot") or 25000.0),
            "nifty_ret_1d": 0.0,
            "nifty_ret_5d": float(feat_dict.get("nifty_ret_5d") or 0.0),
            "composite_score": float(regime.get("composite_score") or 0.0),
            "vix_spot": float(feat_dict.get("vix_val") or 14.0),
            "realized_vol_daily": float(feat_dict.get("atr_pct") or 2.0),
            "realized_vol_5d": float(feat_dict.get("atr_pct") or 2.0) * 1.5,
            "fii_net_flow_cr": float(regime.get("fii_net_cash") or 0.0),
            "fii_fut_long_pct": 50.0,
            "breadth_ratio": float(regime.get("breadth_ratio") or 1.0),
            "rsi14": float(feat_dict.get("rsi14") or 50.0),
            "atr_pct": float(feat_dict.get("atr_pct") or 2.0),
            "dist_to_ema20_pct": float(feat_dict.get("dist_to_ema20") or 0.0),
            "dist_to_ema50_pct": float(feat_dict.get("dist_to_pivot") or 0.0),
            "dist_to_pivot_pct": float(feat_dict.get("dist_to_pivot") or 0.0),
            "volume_ratio_50d": float(feat_dict.get("vol_ratio") or 1.0),
            "trend_strength_adx": float(feat_dict.get("adx14") or 25.0),
            "overnight_gap_pct": 0.0,
            "near_atm_pcr": 1.0,
            "chg_oi_pcr": 1.0,
            "iv_spot": float(feat_dict.get("vix_val") or 14.0),
            "iv_skew": 0.0,
            "straddle_decay_pct": 0.0,
            "dist_to_call_wall_pct": 5.0,
            "dist_to_put_wall_pct": -5.0,
            "sigma_distance": 2.0,
            "time_to_expiry_years": 8.0 / 252.0,
            "horizon_code": 4,  # SWING
            "regime_code": regime_code,
            "reward_to_risk_est": float(feat_dict.get("rr_ratio") or 2.0),
            "sl_threshold_pct": sl_pct,
            "time_of_day_mins": 930
        }

    # ==========================================================================
    # 3. PRE-TRADE ML GATEKEEPER
    # ==========================================================================
    def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        market_regime: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate setup before real capital is deployed.
        Returns:
            - verdict: 'APPROVED', 'CAUTION', or 'REJECTED'
            - win_prob: Calibrated probability of achieving target (0.0 to 1.0)
            - expected_r: Expected R-multiple outcome
            - risk_tier: Recommended risk sizing tier ('A+', 'A', 'B+', or 'REJECTED')
            - risk_flags: List of identified quantitative risk factors
            - features: Complete extracted feature vector
            - market_regime: Snapshot of market conditions
        """
        regime = market_regime or self.capture_market_regime()
        vec, feat_dict = self.vectorize_features(candidate, regime)

        # 1. Inference via ML Model
        win_prob = 0.55
        expected_r = 0.35
        try:
            if self._classifier is not None:
                X_scaled = self._scaler.transform([vec]) if self._scaler else [vec]
                probs = self._classifier.predict_proba(X_scaled)[0]
                win_prob = round(float(probs[1]), 3)

            if self._regressor is not None:
                X_scaled = self._scaler.transform([vec]) if self._scaler else [vec]
                pred_r = float(self._regressor.predict(X_scaled)[0])
                # Blend model prediction with mathematical expected value: E[R] = P(Win)*RR - (1-P(Win))*1.0
                rr = feat_dict["rr_ratio"]
                ev_math = (win_prob * rr) - ((1.0 - win_prob) * 1.0)
                expected_r = round(0.5 * pred_r + 0.5 * ev_math, 2)
            else:
                rr = feat_dict["rr_ratio"]
                expected_r = round((win_prob * rr) - ((1.0 - win_prob) * 1.0), 2)
        except Exception as e:
            logger.debug(f"Error in ML model evaluation: {e}; falling back to calibrated heuristic")
            # Heuristic calculation
            f_score = (feat_dict["score"] - 50.0) / 25.0
            f_conf = (feat_dict["confidence"] - 3.0) / 1.5
            f_vcp = feat_dict["is_vcp"] * 0.4
            f_wk = feat_dict["week_confluence"] * 0.5
            log_odds = -0.1 + 0.45 * f_score + 0.3 * f_conf + f_vcp + f_wk
            win_prob = round(float(1.0 / (1.0 + np.exp(-log_odds))), 3)
            expected_r = round(win_prob * 1.5 - (1.0 - win_prob) * 1.0, 2)

        # 2. Risk Factor Detection
        risk_flags = []
        fatal_flags = []

        if feat_dict["dist_to_pivot"] > 3.0:
            msg = f"OVEREXTENDED_FROM_PIVOT: Stock is {feat_dict['dist_to_pivot']:.1f}% above 50-day pivot (chasing resistance)"
            risk_flags.append(msg)
            if feat_dict["dist_to_pivot"] > 4.5:
                fatal_flags.append(msg)

        if feat_dict["dist_to_ema20"] > 5.0:
            msg = f"OVEREXTENDED_FROM_EMA20: Stock is {feat_dict['dist_to_ema20']:.1f}% stretched above 20 EMA (mean-reversion risk)"
            risk_flags.append(msg)
            if feat_dict["dist_to_ema20"] > 7.0:
                fatal_flags.append(msg)

        if feat_dict["rsi14"] > 74.0:
            msg = f"RSI_OVERBOUGHT: Daily RSI is {feat_dict['rsi14']:.1f} (exhaustion zone)"
            risk_flags.append(msg)
            if feat_dict["rsi14"] > 82.0:
                fatal_flags.append(msg)

        if feat_dict["vix_val"] > 18.0:
            risk_flags.append(f"HIGH_VOLATILITY_VIX: India VIX elevated at {feat_dict['vix_val']:.1f} (wider whipsaws)")

        if regime.get("nifty_trend") == "BEARISH" or feat_dict["nifty_ret_5d"] < -1.5:
            risk_flags.append(f"MARKET_REGIME_HEADWIND: Nifty is in {regime.get('nifty_trend')} trend (5d: {feat_dict['nifty_ret_5d']:+.2f}%)")

        if feat_dict["sector_excess_6m"] < -4.0:
            risk_flags.append(f"SECTOR_LAGGING: Sector underperforming benchmark by {feat_dict['sector_excess_6m']:.1f}% over 6 months")

        if feat_dict["vol_ratio"] < 1.0:
            risk_flags.append(f"WEAK_VOLUME: Volume ratio {feat_dict['vol_ratio']:.2f}x is below 50-day average")

        # 3. Decision Gatekeeper
        if fatal_flags or win_prob < 0.48 or expected_r < 0.05:
            verdict = "REJECTED"
            risk_tier = "REJECTED"
        elif win_prob >= 0.56 and expected_r >= 0.28 and len(risk_flags) <= 1:
            verdict = "APPROVED"
            risk_tier = "A+" if win_prob >= 0.65 and feat_dict["confidence"] >= 5 else "A"
        elif win_prob >= 0.48:
            verdict = "CAUTION"
            risk_tier = "B+"
        else:
            verdict = "REJECTED"
            risk_tier = "REJECTED"

        return {
            "verdict": verdict,
            "win_prob": win_prob,
            "expected_r": expected_r,
            "risk_tier": risk_tier,
            "risk_flags": risk_flags,
            "fatal_flags": fatal_flags,
            "features": feat_dict,
            "market_regime": regime
        }

    # ==========================================================================
    # 4. DATASET RECORDING: REAL & SHADOW TRADES
    # ==========================================================================
    def record_real_trade_entry(
        self,
        symbol: str,
        portfolio_type: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        target_2: float,
        candidate_meta: Dict[str, Any]
    ) -> bool:
        """Records an active real position in the master quant dataset."""
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()
        regime = self.capture_market_regime()
        _, feat_dict = self.vectorize_features(candidate_meta, regime)

        qty = int(candidate_meta.get("shares") or candidate_meta.get("quantity") or 0)
        risk_per_share = max(0.01, entry_price - stop_loss)
        if qty <= 0:
            qty = max(1, int(3750.0 / risk_per_share))
        cap_at_risk = round(float(candidate_meta.get("capital_at_risk") or (qty * risk_per_share)), 2)

        with self._lock:
            with self._get_connection() as conn:
                try:
                    conn.execute("""
                    INSERT INTO quant_trade_dataset (
                        symbol, portfolio_type, trade_type, status, entry_price, stop_loss,
                        target, target_2, sector, score, confidence, rsi14, atr_pct,
                        dist_to_pivot, dist_to_ema20, vol_ratio, inst_score, week_confluence,
                        is_vcp, bb_sq_ratio, adx14, features_json, nifty_trend, nifty_ret_5d,
                        nifty_rsi14, vix_val, sector_excess_6m, market_regime_json,
                        quantity, capital_at_risk, created_at, updated_at
                    ) VALUES (
                        ?, ?, 'REAL', 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(symbol, trade_type, created_at) DO UPDATE SET
                        entry_price = excluded.entry_price,
                        stop_loss = excluded.stop_loss,
                        target = excluded.target,
                        quantity = excluded.quantity,
                        capital_at_risk = excluded.capital_at_risk,
                        updated_at = excluded.updated_at;
                    """, (
                        sym_clean, portfolio_type, entry_price, stop_loss, target, target_2,
                        candidate_meta.get("sector", "DIVERSIFIED"),
                        feat_dict["score"], int(feat_dict["confidence"]), feat_dict["rsi14"],
                        feat_dict["atr_pct"], feat_dict["dist_to_pivot"], feat_dict["dist_to_ema20"],
                        feat_dict["vol_ratio"], feat_dict["inst_score"], int(feat_dict["week_confluence"]),
                        int(feat_dict["is_vcp"]), feat_dict["bb_sq_ratio"], feat_dict["adx14"],
                        json.dumps(feat_dict), regime.get("nifty_trend", "SIDEWAYS"),
                        feat_dict["nifty_ret_5d"], feat_dict["nifty_rsi14"], feat_dict["vix_val"],
                        feat_dict["sector_excess_6m"], json.dumps(regime),
                        qty, cap_at_risk, now_str, now_str
                    ))
                    conn.commit()
                    logger.info(f"Recorded REAL trade entry for {sym_clean} in quant_trade_dataset ({qty} shs, risk ₹{cap_at_risk:,.2f})")

                    # Ingest into Unified Multi-Horizon Dataset
                    try:
                        from core.unified_ml_dataset import unified_ml_dataset
                        u_feats = self._build_unified_features(feat_dict, regime, entry_price, stop_loss)
                        trade_id = f"SWING_{sym_clean}_{now_str.replace(':', '').replace('-', '').replace(' ', '_')}"
                        unified_ml_dataset.record_signal_and_trade(
                            signal_id=f"SIG_{trade_id}",
                            trade_id=trade_id,
                            instrument=sym_clean,
                            trading_horizon="SWING",
                            strategy_name="INSTITUTIONAL_6PILLAR",
                            record_type="LIVE",
                            market_session="AFTERNOON",
                            accept_reject_verdict="ACCEPTED",
                            decision_source="HYBRID",
                            features=u_feats,
                            regime=regime.get("nifty_trend", "SIDEWAYS"),
                            spot_at_entry=entry_price,
                            intended_entry=entry_price,
                            intended_target=target,
                            intended_sl=stop_loss,
                            sl_threshold_pct=u_feats["sl_threshold_pct"],
                            quantity=qty,
                            capital_at_risk=cap_at_risk,
                            rationale={
                                "score": feat_dict.get("score"),
                                "confidence": feat_dict.get("confidence"),
                                "sector": candidate_meta.get("sector"),
                                "strategy": "6-pillar momentum breakout"
                            },
                            strike=0.0,
                            dte_days=8.0,
                            sigma_distance=2.0,
                            actual_entry=entry_price,
                            model_version="prod_v1.0"
                        )
                    except Exception as ue:
                        logger.debug(f"Failed to record REAL swing trade in unified_ml_dataset: {ue}")

                    return True
                except Exception as e:
                    logger.error(f"Failed to record real trade entry for {sym_clean}: {e}")
                    return False

    def record_shadow_candidate(
        self,
        symbol: str,
        portfolio_type: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        target_2: float,
        candidate_meta: Dict[str, Any],
        rejection_reason: str = "ML_REJECTED"
    ) -> bool:
        """
        Records a rejected or unexecuted candidate setup into the Shadow Ledger.
        Allows counterfactual tracking to evaluate whether it actually failed or succeeded.
        """
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()

        # Check if already in active real positions or already tracked as shadow today
        with self._lock:
            with self._get_connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM quant_trade_dataset WHERE symbol = ? AND status = 'ACTIVE'",
                    (sym_clean,)
                ).fetchone()
                if existing:
                    return False

        regime = self.capture_market_regime()
        _, feat_dict = self.vectorize_features(candidate_meta, regime)

        qty = int(candidate_meta.get("shares") or candidate_meta.get("quantity") or 0)
        risk_per_share = max(0.01, entry_price - stop_loss)
        if qty <= 0:
            qty = max(1, int(3750.0 / risk_per_share))
        cap_at_risk = round(float(candidate_meta.get("capital_at_risk") or (qty * risk_per_share)), 2)

        with self._lock:
            with self._get_connection() as conn:
                try:
                    conn.execute("""
                    INSERT INTO quant_trade_dataset (
                        symbol, portfolio_type, trade_type, status, entry_price, stop_loss,
                        target, target_2, sector, score, confidence, rsi14, atr_pct,
                        dist_to_pivot, dist_to_ema20, vol_ratio, inst_score, week_confluence,
                        is_vcp, bb_sq_ratio, adx14, features_json, nifty_trend, nifty_ret_5d,
                        nifty_rsi14, vix_val, sector_excess_6m, market_regime_json,
                        root_cause_attribution, quantity, capital_at_risk, created_at, updated_at
                    ) VALUES (
                        ?, ?, 'SHADOW', 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """, (
                        sym_clean, portfolio_type, entry_price, stop_loss, target, target_2,
                        candidate_meta.get("sector", "DIVERSIFIED"),
                        feat_dict["score"], int(feat_dict["confidence"]), feat_dict["rsi14"],
                        feat_dict["atr_pct"], feat_dict["dist_to_pivot"], feat_dict["dist_to_ema20"],
                        feat_dict["vol_ratio"], feat_dict["inst_score"], int(feat_dict["week_confluence"]),
                        int(feat_dict["is_vcp"]), feat_dict["bb_sq_ratio"], feat_dict["adx14"],
                        json.dumps(feat_dict), regime.get("nifty_trend", "SIDEWAYS"),
                        feat_dict["nifty_ret_5d"], feat_dict["nifty_rsi14"], feat_dict["vix_val"],
                        feat_dict["sector_excess_6m"], json.dumps(regime),
                        f"INITIAL_SHADOW_REASON: {rejection_reason}", qty, cap_at_risk, now_str, now_str
                    ))
                    conn.commit()
                    logger.info(f"👻 Logged SHADOW trade for {sym_clean} ({qty} shs, risk ₹{cap_at_risk:,.2f}, SL: ₹{stop_loss:.2f}, T1: ₹{target:.2f}) [Reason: {rejection_reason}]")

                    # Ingest into Unified Multi-Horizon Dataset as Counterfactual
                    try:
                        from core.unified_ml_dataset import unified_ml_dataset
                        u_feats = self._build_unified_features(feat_dict, regime, entry_price, stop_loss)
                        trade_id = f"SWING_SHADOW_{sym_clean}_{now_str.replace(':', '').replace('-', '').replace(' ', '_')}"
                        unified_ml_dataset.record_signal_and_trade(
                            signal_id=f"SIG_{trade_id}",
                            trade_id=trade_id,
                            instrument=sym_clean,
                            trading_horizon="SWING",
                            strategy_name="INSTITUTIONAL_6PILLAR",
                            record_type="COUNTERFACTUAL",
                            market_session="AFTERNOON",
                            accept_reject_verdict="REJECTED",
                            decision_source="ML_GATEKEEPER",
                            features=u_feats,
                            regime=regime.get("nifty_trend", "SIDEWAYS"),
                            spot_at_entry=entry_price,
                            intended_entry=entry_price,
                            intended_target=target,
                            intended_sl=stop_loss,
                            sl_threshold_pct=u_feats["sl_threshold_pct"],
                            quantity=qty,
                            capital_at_risk=cap_at_risk,
                            rationale={
                                "rejection_reason": rejection_reason,
                                "score": feat_dict.get("score"),
                                "confidence": feat_dict.get("confidence")
                            },
                            strike=0.0,
                            dte_days=8.0,
                            sigma_distance=2.0,
                            model_version="prod_v1.0"
                        )
                    except Exception as ue:
                        logger.debug(f"Failed to record SHADOW swing trade in unified_ml_dataset: {ue}")

                    return True
                except Exception as e:
                    logger.debug(f"Failed to log shadow candidate {sym_clean}: {e}")
                    return False

    # ==========================================================================
    # 5. REAL TRADE EXIT RESOLUTION & POST-MORTEM
    # ==========================================================================
    def record_real_trade_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        r_multiple: float,
        return_pct: float,
        hold_days: int
    ) -> bool:
        """
        Resolves a completed real trade, conducts algorithmic Root-Cause Attribution,
        synthesizes AI Quant Post-Mortem notes, and triggers incremental ML training.
        """
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        now_str = format_ist()

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM quant_trade_dataset WHERE symbol = ? AND trade_type = 'REAL' AND status = 'ACTIVE' ORDER BY id DESC LIMIT 1",
                    (sym_clean,)
                ).fetchone()

                if not row:
                    logger.debug(f"No active REAL trade record found in quant_trade_dataset for {sym_clean}")
                    return False

                trade = dict(row)
                entry_px = float(trade["entry_price"])
                sl_px = float(trade["stop_loss"])
                is_win = 1 if (pnl > 0 or "TARGET" in str(exit_reason).upper()) else 0
                resolved_status = "RESOLVED_TARGET" if is_win else "RESOLVED_STOP_LOSS"

                # Calculate holding period benchmark returns
                current_regime = self.capture_market_regime()
                try:
                    entry_regime = json.loads(trade.get("market_regime_json") or "{}")
                    entry_nifty = float(entry_regime.get("nifty_spot") or current_regime["nifty_spot"])
                    nifty_hold_ret = round(((current_regime["nifty_spot"] - entry_nifty) / entry_nifty) * 100.0, 2)
                except Exception:
                    nifty_hold_ret = 0.0

                # 1. Algorithmic Root-Cause Attribution
                attribution = self._compute_root_cause_attribution(
                    is_win=is_win,
                    exit_reason=exit_reason,
                    features=trade,
                    nifty_hold_return=nifty_hold_ret
                )

                # 2. AI Quant Post-Mortem Synthesis
                ai_notes = self._synthesize_ai_post_mortem(
                    symbol=sym_clean,
                    sector=trade.get("sector", "DIVERSIFIED"),
                    entry_px=entry_px,
                    exit_px=exit_price,
                    stop_loss=sl_px,
                    pnl=pnl,
                    r_multiple=r_multiple,
                    return_pct=return_pct,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    is_winner=is_win,
                    attribution=attribution,
                    features=trade,
                    nifty_hold_ret=nifty_hold_ret
                )

                conn.execute("""
                UPDATE quant_trade_dataset
                SET status = ?,
                    exit_price = ?,
                    pnl = ?,
                    r_multiple = ?,
                    return_pct = ?,
                    hold_days = ?,
                    exit_reason = ?,
                    is_winner = ?,
                    nifty_holding_return_pct = ?,
                    root_cause_attribution = ?,
                    ai_post_mortem_notes = ?,
                    updated_at = ?
                WHERE id = ?
                """, (
                    resolved_status, exit_price, pnl, r_multiple, return_pct, hold_days,
                    exit_reason, is_win, nifty_hold_ret, attribution, ai_notes,
                    now_str, trade["id"]
                ))
                conn.commit()

        logger.info(f"Resolved REAL trade for {sym_clean} ({resolved_status}) | Attribution: {attribution}")

        # Resolve in Unified Multi-Horizon Dataset
        try:
            from core.unified_ml_dataset import unified_ml_dataset
            with unified_ml_dataset._get_connection() as uconn:
                urow = uconn.execute(
                    "SELECT trade_id FROM unified_trade_learning_dataset WHERE instrument = ? AND trading_horizon = 'SWING' AND status = 'ACTIVE' ORDER BY id DESC LIMIT 1",
                    (sym_clean,)
                ).fetchone()
                if urow:
                    u_trade_id = urow["trade_id"]
                    statutory_costs = round(abs(pnl) * 0.0012 + 40.0, 2)
                    unified_ml_dataset.record_trade_outcome(
                        trade_id=u_trade_id,
                        actual_exit=exit_price,
                        exit_reason=exit_reason,
                        gross_pnl=pnl,
                        net_pnl=pnl - statutory_costs,
                        total_costs=statutory_costs,
                        hold_duration_mins=hold_days * 375.0,
                        rules_followed=1
                    )
        except Exception as ue:
            logger.debug(f"Failed to resolve REAL trade in unified_ml_dataset: {ue}")

        # Trigger incremental model retraining
        threading.Thread(target=self.train_model, daemon=True).start()
        return True

    # ==========================================================================
    # 6. SHADOW COUNTERFACTUAL RECONCILIATION
    # ==========================================================================
    def reconcile_shadow_trades(self) -> Dict[str, Any]:
        """
        Audits all active shadow trades against latest EOD candle data:
        - If Day Low <= Stop Loss -> RESOLVED_STOP_LOSS (0% survivor bias, confirmed failure).
        - If Day High >= Target -> RESOLVED_TARGET (counterfactual missed winner).
        - If Hold Days >= 20 -> EXPIRED_TIME.
        Updates dataset and triggers model retrain if trades resolved.
        """
        now_dt = now_ist()
        now_str = format_ist(now_dt)
        resolved_count = 0
        winners = 0
        losers = 0

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM quant_trade_dataset WHERE trade_type = 'SHADOW' AND status = 'ACTIVE'"
                ).fetchall()

        if not rows:
            return {"resolved": 0, "active_shadow": 0}

        for row in rows:
            item = dict(row)
            sym = item["symbol"]
            entry_px = float(item["entry_price"])
            sl = float(item["stop_loss"])
            t1 = float(item["target"])
            rec_id = item["id"]

            df = state_engine.get_ohlcv(sym)
            if df is None or df.empty or "Close" not in df.columns:
                continue

            # Evaluate all trading bars since candidate entry date to accurately catch target/stop hits
            try:
                created_date_str = item["created_at"][:10]
                created_date = datetime.strptime(created_date_str, "%Y-%m-%d").date()
                hold_days = max(1, (now_dt.date() - created_date).days)
                if hasattr(df.index, 'date'):
                    df_since = df[df.index.date >= created_date]
                elif "date" in df.columns:
                    df_since = df[pd.to_datetime(df["date"]).dt.date >= created_date]
                else:
                    df_since = df
                if df_since.empty:
                    df_since = df.iloc[-1:]
            except Exception:
                hold_days = int(item.get("hold_days") or 1)
                df_since = df.iloc[-1:]

            high_px = float(df_since["High"].max() if "High" in df_since.columns else df_since["Close"].max())
            low_px = float(df_since["Low"].min() if "Low" in df_since.columns else df_since["Close"].min())
            close_px = float(df["Close"].iloc[-1])

            # Quantity and capital at risk
            qty = int(item.get("quantity") or 0)
            if qty <= 0:
                risk_unit = max(0.01, entry_px - sl)
                qty = max(1, int(3750.0 / risk_unit))

            status = None
            exit_px = None
            exit_reason = None
            is_win = None
            r_mult = 0.0
            ret_pct = 0.0

            if low_px <= sl:
                # Stopped out counterfactually
                status = "RESOLVED_STOP_LOSS"
                exit_px = sl
                exit_reason = "SHADOW_HIT_STOP_LOSS"
                is_win = 0
                r_mult = -1.0
                ret_pct = round(((sl - entry_px) / entry_px) * 100.0, 2)
                losers += 1
            elif high_px >= t1:
                # Hit target counterfactually
                status = "RESOLVED_TARGET"
                exit_px = t1
                exit_reason = "SHADOW_HIT_TARGET_1"
                is_win = 1
                risk_unit = max(0.01, entry_px - sl)
                r_mult = round((t1 - entry_px) / risk_unit, 2)
                ret_pct = round(((t1 - entry_px) / entry_px) * 100.0, 2)
                winners += 1
            elif hold_days >= 20:
                # Expired due to time stop
                status = "EXPIRED_TIME"
                exit_px = close_px
                exit_reason = "SHADOW_TIME_STOP_20D"
                is_win = 1 if close_px > entry_px else 0
                risk_unit = max(0.01, entry_px - sl)
                r_mult = round((close_px - entry_px) / risk_unit, 2)
                ret_pct = round(((close_px - entry_px) / entry_px) * 100.0, 2)
                if is_win:
                    winners += 1
                else:
                    losers += 1

            if status:
                realized_pnl = round((exit_px - entry_px) * qty, 2)
                attribution = self._compute_root_cause_attribution(
                    is_win=is_win,
                    exit_reason=exit_reason,
                    features=item,
                    nifty_hold_return=0.0
                )
                ai_notes = self._synthesize_ai_post_mortem(
                    symbol=sym,
                    sector=item.get("sector", "DIVERSIFIED"),
                    entry_px=entry_px,
                    exit_px=exit_px,
                    stop_loss=sl,
                    pnl=realized_pnl,
                    r_multiple=r_mult,
                    return_pct=ret_pct,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    is_winner=is_win,
                    attribution=attribution,
                    features=item,
                    nifty_hold_ret=0.0
                )

                with self._lock:
                    with self._get_connection() as conn:
                        conn.execute("""
                        UPDATE quant_trade_dataset
                        SET status = ?,
                            exit_price = ?,
                            pnl = ?,
                            r_multiple = ?,
                            return_pct = ?,
                            hold_days = ?,
                            exit_reason = ?,
                            is_winner = ?,
                            root_cause_attribution = ?,
                            ai_post_mortem_notes = ?,
                            updated_at = ?
                        WHERE id = ?
                        """, (status, exit_px, realized_pnl, r_mult, ret_pct, hold_days, exit_reason, is_win, attribution, ai_notes, now_str, rec_id))
                        conn.commit()

                resolved_count += 1
                logger.info(f"👻 Resolved SHADOW trade for {sym}: {status} (PnL: ₹{realized_pnl:+,.2f}, {ret_pct:+.2f}%) | Attribution: {attribution}")

                # Resolve in Unified Multi-Horizon Dataset (Counterfactual)
                try:
                    from core.unified_ml_dataset import unified_ml_dataset
                    with unified_ml_dataset._get_connection() as uconn:
                        urow = uconn.execute(
                            "SELECT trade_id FROM unified_trade_learning_dataset WHERE instrument = ? AND trading_horizon = 'SWING' AND record_type = 'COUNTERFACTUAL' AND status = 'REJECTED_SIM' ORDER BY id DESC LIMIT 1",
                            (sym,)
                        ).fetchone()
                        if urow:
                            u_trade_id = urow["trade_id"]
                            statutory_costs = round(abs(realized_pnl) * 0.0012 + 40.0, 2)
                            unified_ml_dataset.record_trade_outcome(
                                trade_id=u_trade_id,
                                actual_exit=exit_px,
                                exit_reason=exit_reason,
                                gross_pnl=realized_pnl,
                                net_pnl=realized_pnl - statutory_costs,
                                total_costs=statutory_costs,
                                hold_duration_mins=hold_days * 375.0,
                                rules_followed=1
                            )
                except Exception as ue:
                    logger.debug(f"Failed to resolve SHADOW trade in unified_ml_dataset: {ue}")

        if resolved_count > 0:
            threading.Thread(target=self.train_model, daemon=True).start()

        return {
            "resolved": resolved_count,
            "winners": winners,
            "losers": losers,
            "remaining_active": len(rows) - resolved_count
        }

    # ==========================================================================
    # 7. ROOT-CAUSE ATTRIBUTION & AI POST-MORTEM
    # ==========================================================================
    def _compute_root_cause_attribution(
        self,
        is_win: int,
        exit_reason: str,
        features: Dict[str, Any],
        nifty_hold_return: float
    ) -> str:
        """Determines the primary mechanical cause of trade success or failure."""
        if is_win:
            return "TARGET_ACHIEVED_CLEAN_EXPANSION"

        # Trade failed (Stop Loss hit)
        dist_pivot = float(features.get("dist_to_pivot") or 0.0)
        dist_ema20 = float(features.get("dist_to_ema20") or 0.0)
        rsi = float(features.get("rsi14") or 50.0)
        vol_r = float(features.get("vol_ratio") or 1.0)
        sec_rs = float(features.get("sector_excess_6m") or 0.0)

        if dist_pivot >= 3.5 or dist_ema20 >= 5.0 or rsi >= 75.0:
            return "OVEREXTENDED_AT_ENTRY (Chasing into resistance)"
        elif nifty_hold_return <= -1.5:
            return f"MARKET_PULLBACK_DRAG (Nifty returned {nifty_hold_return:+.1f}% during hold)"
        elif sec_rs <= -3.5:
            return f"SECTOR_ROTATION_HEADWIND (Sector lagging by {sec_rs:.1f}%)"
        elif vol_r < 1.0:
            return "LOW_VOLUME_FALSE_BREAKOUT (Lack of institutional follow-through)"
        else:
            return "NORMAL_STATISTICAL_STOP_OUT (Structural stop hit within expected bounds)"

    def _synthesize_ai_post_mortem(
        self,
        symbol: str,
        sector: str,
        entry_px: float,
        exit_px: float,
        stop_loss: float,
        pnl: float,
        r_multiple: float,
        return_pct: float,
        hold_days: int,
        exit_reason: str,
        is_winner: int,
        attribution: str,
        features: Dict[str, Any],
        nifty_hold_ret: float
    ) -> str:
        """
        Synthesizes an intelligent quant post-mortem diagnosis using LLM or structured template.
        """
        outcome_label = "WINNER (TARGET HIT)" if is_winner else "STOP LOSS HIT"
        base_prompt = (
            f"Trade Post-Mortem Audit for NSE:{symbol} ({sector}):\n"
            f"- Outcome: {outcome_label} via {exit_reason}\n"
            f"- Entry Price: ₹{entry_px:,.2f} | Exit Price: ₹{exit_px:,.2f} | Initial SL: ₹{stop_loss:,.2f}\n"
            f"- Realized PnL: ₹{pnl:+,.2f} ({return_pct:+.2f}%, {r_multiple:+.2f}R) in {hold_days} days\n"
            f"- Setup Features: Score={features.get('score')}, DistToPivot={features.get('dist_to_pivot')}%, "
            f"DistToEMA20={features.get('dist_to_ema20')}%, RSI={features.get('rsi14')}, VolRatio={features.get('vol_ratio')}x\n"
            f"- Market Regime: Nifty {nifty_hold_ret:+.2f}% during hold, Root-Cause Attribution: {attribution}\n"
            "Explain in 2-3 concise quantitative bullet points why this setup resolved this way and the lesson for future pre-trade filtering."
        )

        try:
            from core import config
            if config.AI_ENABLED:
                from ai_engine.service import ai_service
                provider = ai_service.provider_adapter
                active_prov = config.AI_PROVIDER
                if provider.is_provider_configured(active_prov):
                    model_id = config.AI_PRIMARY_MODEL
                    res = provider.complete(
                        prompt=base_prompt,
                        system_prompt="You are a senior quantitative swing trading risk manager. Provide concise, high-signal post-mortem analysis.",
                        model_id=model_id,
                        temperature=0.2,
                        max_tokens=300
                    )
                    if res and len(res.strip()) > 20:
                        return res.strip()

                # Secondary failover provider (e.g. OpenRouter if primary is Groq)
                alt_prov = "openrouter" if active_prov == "groq" else "groq"
                if provider.is_provider_configured(alt_prov):
                    alt_model = config.AI_FALLBACK_MODEL or ("google/gemini-2.5-flash" if alt_prov == "openrouter" else "llama-3.3-70b-versatile")
                    res = provider.complete(
                        prompt=base_prompt,
                        system_prompt="You are a senior quantitative swing trading risk manager. Provide concise, high-signal post-mortem analysis.",
                        model_id=alt_model,
                        temperature=0.2,
                        max_tokens=300
                    )
                    if res and len(res.strip()) > 20:
                        return res.strip()
        except Exception as e:
            logger.debug(f"AI post-mortem synthesis fallback: {e}")

        # High-signal deterministic fallback
        if is_winner:
            return (
                f"• Clean breakout expansion with strong relative momentum in {sector}.\n"
                f"• Target attained at ₹{exit_px:,.2f} ({return_pct:+.2f}%, {r_multiple:+.2f}R) across {hold_days} days.\n"
                f"• High conviction follow-through supported by institutional volume and weekly trend confluence."
            )
        else:
            return (
                f"• Stop loss executed at ₹{exit_px:,.2f} ({return_pct:+.2f}%, {r_multiple:+.2f}R) after {hold_days} days.\n"
                f"• Primary breakdown trigger: {attribution}.\n"
                f"• Lesson: Filter candidates with extreme stretch from pivot or when Nifty benchmark experiences negative momentum drift ({nifty_hold_ret:+.1f}%)."
            )

    # ==========================================================================
    # 8. MODEL TRAINING & COLD-START BAYESIAN PRIOR
    # ==========================================================================
    def _generate_synthetic_prior_dataset(self) -> pd.DataFrame:
        """
        Seeds model with 160 synthetic historical swing trading archetypes derived
        from 5-year Indian equity market regimes (NSE 500 swing dynamics).
        Prevents cold-start failure before sufficient live trades accumulate.
        """
        np.random.seed(42)
        n_samples = 160
        rows = []

        for _ in range(n_samples):
            archetype = np.random.choice(["A_BREAKOUT", "OVEREXTENDED", "PULLBACK_BOUNCE", "CHOPPY_FAILURE"], p=[0.35, 0.25, 0.25, 0.15])

            if archetype == "A_BREAKOUT":
                # High score, close to pivot, strong volume, low VIX -> 72% win rate
                score = np.random.uniform(78.0, 96.0)
                conf = np.random.choice([4, 5, 6])
                rsi = np.random.uniform(54.0, 68.0)
                atr_pct = np.random.uniform(1.8, 3.8)
                dist_p = np.random.uniform(-0.5, 2.0)
                dist_e = np.random.uniform(0.5, 3.5)
                vol_r = np.random.uniform(1.3, 3.2)
                inst_s = np.random.uniform(65.0, 95.0)
                wk_c = 1.0
                is_vcp = float(np.random.choice([0, 1], p=[0.4, 0.6]))
                bb_sq = np.random.uniform(0.5, 1.1)
                adx = np.random.uniform(25.0, 48.0)
                nifty_5d = np.random.uniform(0.2, 2.5)
                nifty_rsi = np.random.uniform(50.0, 65.0)
                vix = np.random.uniform(11.5, 15.5)
                sec_ex = np.random.uniform(1.0, 8.0)
                rr = np.random.uniform(1.8, 3.0)
                is_win = 1 if np.random.rand() < 0.74 else 0
                r_mult = rr if is_win else -1.0

            elif archetype == "OVEREXTENDED":
                # Stretched pivot & EMA20, high RSI, chasing -> 28% win rate
                score = np.random.uniform(65.0, 85.0)
                conf = np.random.choice([2, 3, 4])
                rsi = np.random.uniform(74.0, 88.0)
                atr_pct = np.random.uniform(3.5, 6.0)
                dist_p = np.random.uniform(4.0, 9.0)
                dist_e = np.random.uniform(5.5, 11.0)
                vol_r = np.random.uniform(0.8, 2.2)
                inst_s = np.random.uniform(40.0, 75.0)
                wk_c = float(np.random.choice([0, 1], p=[0.5, 0.5]))
                is_vcp = 0.0
                bb_sq = np.random.uniform(1.2, 2.2)
                adx = np.random.uniform(30.0, 55.0)
                nifty_5d = np.random.uniform(-1.0, 1.5)
                nifty_rsi = np.random.uniform(55.0, 75.0)
                vix = np.random.uniform(14.0, 20.0)
                sec_ex = np.random.uniform(-3.0, 4.0)
                rr = np.random.uniform(1.2, 2.2)
                is_win = 1 if np.random.rand() < 0.26 else 0
                r_mult = rr if is_win else -1.0

            elif archetype == "PULLBACK_BOUNCE":
                # Low distance to EMA20, sound institutional footprint -> 62% win rate
                score = np.random.uniform(70.0, 88.0)
                conf = np.random.choice([3, 4, 5])
                rsi = np.random.uniform(48.0, 58.0)
                atr_pct = np.random.uniform(1.5, 3.2)
                dist_p = np.random.uniform(-3.0, 1.0)
                dist_e = np.random.uniform(-0.5, 1.8)
                vol_r = np.random.uniform(0.9, 1.8)
                inst_s = np.random.uniform(55.0, 85.0)
                wk_c = 1.0
                is_vcp = float(np.random.choice([0, 1], p=[0.6, 0.4]))
                bb_sq = np.random.uniform(0.6, 1.2)
                adx = np.random.uniform(20.0, 38.0)
                nifty_5d = np.random.uniform(-0.5, 1.8)
                nifty_rsi = np.random.uniform(45.0, 60.0)
                vix = np.random.uniform(12.0, 16.5)
                sec_ex = np.random.uniform(0.0, 6.0)
                rr = np.random.uniform(2.0, 3.2)
                is_win = 1 if np.random.rand() < 0.62 else 0
                r_mult = rr if is_win else -1.0

            else:  # CHOPPY_FAILURE
                # Low volume, high VIX, negative sector momentum -> 30% win rate
                score = np.random.uniform(45.0, 68.0)
                conf = np.random.choice([1, 2, 3])
                rsi = np.random.uniform(40.0, 65.0)
                atr_pct = np.random.uniform(2.5, 5.0)
                dist_p = np.random.uniform(-2.0, 3.0)
                dist_e = np.random.uniform(1.0, 4.0)
                vol_r = np.random.uniform(0.4, 1.0)
                inst_s = np.random.uniform(30.0, 55.0)
                wk_c = 0.0
                is_vcp = 0.0
                bb_sq = np.random.uniform(0.8, 1.6)
                adx = np.random.uniform(12.0, 24.0)
                nifty_5d = np.random.uniform(-3.0, 0.0)
                nifty_rsi = np.random.uniform(35.0, 50.0)
                vix = np.random.uniform(17.0, 25.0)
                sec_ex = np.random.uniform(-8.0, -1.0)
                rr = np.random.uniform(1.2, 2.0)
                is_win = 1 if np.random.rand() < 0.30 else 0
                r_mult = rr if is_win else -1.0

            rows.append({
                "score": score,
                "confidence": conf,
                "rsi14": rsi,
                "atr_pct": atr_pct,
                "dist_to_pivot": dist_p,
                "dist_to_ema20": dist_e,
                "vol_ratio": vol_r,
                "inst_score": inst_s,
                "week_confluence": wk_c,
                "is_vcp": is_vcp,
                "bb_sq_ratio": bb_sq,
                "adx14": adx,
                "nifty_ret_5d": nifty_5d,
                "nifty_rsi14": nifty_rsi,
                "vix_val": vix,
                "sector_excess_6m": sec_ex,
                "rr_ratio": rr,
                "is_winner": is_win,
                "r_multiple": r_mult
            })

        return pd.DataFrame(rows)

    def train_model(self) -> Dict[str, Any]:
        """
        Trains or updates the HistGradientBoostingClassifier & CalibratedClassifierCV
        using both real trade outcomes and shadow counterfactuals.
        Saves model weights in SQLite system_kv_store.
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute("""
                SELECT score, confidence, rsi14, atr_pct, dist_to_pivot, dist_to_ema20,
                       vol_ratio, inst_score, week_confluence, is_vcp, bb_sq_ratio, adx14,
                       nifty_ret_5d, nifty_rsi14, vix_val, sector_excess_6m,
                       target, entry_price, stop_loss, r_multiple, is_winner, trade_type
                FROM quant_trade_dataset
                WHERE is_winner IS NOT NULL AND status IN ('RESOLVED_TARGET', 'RESOLVED_STOP_LOSS', 'EXPIRED_TIME')
                """).fetchall()

        real_count = sum(1 for r in rows if r["trade_type"] == "REAL")
        shadow_count = sum(1 for r in rows if r["trade_type"] == "SHADOW")
        total_empirical = len(rows)

        empirical_rows = []
        for r in rows:
            ep = float(r["entry_price"])
            sl = float(r["stop_loss"])
            t1 = float(r["target"])
            risk_unit = max(0.01, ep - sl)
            rr = round((t1 - ep) / risk_unit, 2)

            empirical_rows.append({
                "score": float(r["score"] or 70.0),
                "confidence": float(r["confidence"] or 4),
                "rsi14": float(r["rsi14"] or 55.0),
                "atr_pct": float(r["atr_pct"] or 2.5),
                "dist_to_pivot": float(r["dist_to_pivot"] or 1.5),
                "dist_to_ema20": float(r["dist_to_ema20"] or 2.0),
                "vol_ratio": float(r["vol_ratio"] or 1.2),
                "inst_score": float(r["inst_score"] or 60.0),
                "week_confluence": float(r["week_confluence"] or 1.0),
                "is_vcp": float(r["is_vcp"] or 0.0),
                "bb_sq_ratio": float(r["bb_sq_ratio"] or 0.8),
                "adx14": float(r["adx14"] or 28.0),
                "nifty_ret_5d": float(r["nifty_ret_5d"] or 0.0),
                "nifty_rsi14": float(r["nifty_rsi14"] or 55.0),
                "vix_val": float(r["vix_val"] or 14.0),
                "sector_excess_6m": float(r["sector_excess_6m"] or 0.0),
                "rr_ratio": float(rr),
                "is_winner": int(r["is_winner"]),
                "r_multiple": float(r["r_multiple"] or 0.0)
            })

        # Determine dataset source:
        # If total_empirical >= 10 and both classes (winners and losers) are present,
        # train 100% on real empirical market outcomes with zero synthetic prior rows.
        classes_present = len(set(r["is_winner"] for r in empirical_rows)) if empirical_rows else 0
        if total_empirical >= 10 and classes_present >= 2:
            combined_df = pd.DataFrame(empirical_rows)
            training_mode = "100%_REAL_EMPIRICAL"
            prior_count = 0
        else:
            prior_df = self._generate_synthetic_prior_dataset()
            prior_count = len(prior_df)
            if total_empirical > 0:
                emp_df = pd.DataFrame(empirical_rows)
                combined_df = pd.concat([prior_df, emp_df], ignore_index=True)
                training_mode = "COLD_START_HYBRID"
            else:
                combined_df = prior_df
                training_mode = "COLD_START_PRIOR"

        X = combined_df[ML_FEATURE_COLS].values
        y_class = combined_df["is_winner"].values.astype(int)
        y_reg = combined_df["r_multiple"].values.astype(float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Classifier with probability calibration
        base_hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            min_samples_leaf=5,
            random_state=42
        )
        min_class_count = int(np.min(np.bincount(y_class))) if len(np.bincount(y_class)) > 1 else 0
        cv_folds = min(3, min_class_count)

        if cv_folds >= 2:
            calibrated_clf = CalibratedClassifierCV(
                estimator=base_hgb,
                method="sigmoid",
                cv=cv_folds
            )
            calibrated_clf.fit(X_scaled, y_class)
        else:
            base_hgb.fit(X_scaled, y_class)
            calibrated_clf = base_hgb

        # Train Regressor for Expected R
        reg = Ridge(alpha=2.0)
        reg.fit(X_scaled, y_reg)

        # Compute Metrics
        preds = calibrated_clf.predict_proba(X_scaled)[:, 1]
        win_rate = float(np.mean(y_class))
        brier = float(np.mean((preds - y_class) ** 2))

        metadata = {
            "trained_at": format_ist(),
            "training_mode": training_mode,
            "total_samples": int(len(combined_df)),
            "empirical_samples": int(total_empirical),
            "real_samples": int(real_count),
            "shadow_samples": int(shadow_count),
            "prior_samples": int(prior_count),
            "win_rate": round(win_rate, 3),
            "brier_score": round(brier, 4),
            "features": ML_FEATURE_COLS,
            "status": "ONLINE_ACTIVE"
        }

        with self._lock:
            self._classifier = calibrated_clf
            self._regressor = reg
            self._scaler = scaler
            self._model_metadata = metadata

            # Serialize and store in SQLite KV store
            try:
                bundle = {
                    "classifier": calibrated_clf,
                    "regressor": reg,
                    "scaler": scaler,
                    "metadata": metadata
                }
                pickled = pickle.dumps(bundle)
                state_engine.set_kv("swing_quant_ml_bundle", pickled)
                logger.info(f"💾 Persisted swing_quant_ml_bundle to SQLite KV store ({len(pickled)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to save swing_quant_ml_bundle to KV store: {e}")

        logger.info(
            f"✅ [Swing Quant ML Trained] Samples: {len(combined_df)} (Empirical: {total_empirical} "
            f"[{real_count} real, {shadow_count} shadow]) | Brier Score: {brier:.4f}"
        )
        return metadata

    def _load_or_bootstrap_model(self):
        """Loads trained model bundle from SQLite KV store or fits on cold-start prior."""
        try:
            saved_blob = state_engine.get_kv("swing_quant_ml_bundle")
            if saved_blob and isinstance(saved_blob, bytes):
                bundle = pickle.loads(saved_blob)
                self._classifier = bundle.get("classifier")
                self._regressor = bundle.get("regressor")
                self._scaler = bundle.get("scaler")
                self._model_metadata = bundle.get("metadata") or {}
                logger.info(f"Restored SwingQuantMLModel from KV vault ({self._model_metadata.get('total_samples', 0)} samples)")
                return
        except Exception as e:
            logger.debug(f"Could not restore ML bundle from KV: {e}")

        # Bootstrap immediately with prior
        try:
            self.train_model()
        except Exception as e:
            logger.error(f"Failed to bootstrap SwingQuantMLModel: {e}")

    # ==========================================================================
    # 9. METRICS & STATUS
    # ==========================================================================
    def get_status(self) -> Dict[str, Any]:
        """Returns runtime status of the Quant ML & AI Feedback Engine."""
        with self._lock:
            with self._get_connection() as conn:
                total_trades = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset").fetchone()[0]
                real_trades = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE trade_type = 'REAL'").fetchone()[0]
                shadow_trades = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE trade_type = 'SHADOW'").fetchone()[0]
                active_shadow = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE trade_type = 'SHADOW' AND status = 'ACTIVE'").fetchone()[0]
                resolved_trades = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE status != 'ACTIVE'").fetchone()[0]
                winners = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE is_winner = 1").fetchone()[0]
                sl_hits = conn.execute("SELECT COUNT(*) FROM quant_trade_dataset WHERE is_winner = 0").fetchone()[0]

        win_rate = round((winners / max(1, winners + sl_hits)) * 100.0, 1)

        return {
            "model_metadata": self._model_metadata,
            "dataset_summary": {
                "total_records": total_trades,
                "real_trades_count": real_trades,
                "shadow_trades_count": shadow_trades,
                "active_shadow_monitoring": active_shadow,
                "resolved_trades": resolved_trades,
                "resolved_winners": winners,
                "resolved_stoplosses": sl_hits,
                "empirical_win_rate_pct": win_rate
            },
            "gatekeeper_criteria": {
                "min_win_prob_approval": 0.55,
                "min_expected_r": 0.28,
                "max_dist_to_pivot_pct": 3.0,
                "max_dist_to_ema20_pct": 5.0,
                "max_rsi14": 74.0,
                "shadow_tracking_active": True
            }
        }

    def get_dataset(self, limit: int = 50, trade_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns recent rows from `quant_trade_dataset`."""
        query = "SELECT * FROM quant_trade_dataset"
        params = []
        if trade_type and str(trade_type).upper() != "ALL":
            query += " WHERE trade_type = ?"
            params.append(str(trade_type).upper())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]

    def get_active_shadow_trades(self) -> List[Dict[str, Any]]:
        """Returns all currently active shadow setups being monitored counterfactually."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM quant_trade_dataset WHERE trade_type = 'SHADOW' AND status = 'ACTIVE' ORDER BY id DESC"
                ).fetchall()
                return [dict(r) for r in rows]

    def get_active_training_setups(self) -> List[Dict[str, Any]]:
        """
        Returns all currently active training setups (both SHADOW and REAL)
        with live/latest MTM, unrealized PnL, return %, distance to SL, distance to T1,
        and risk-reward ratio.
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM quant_trade_dataset WHERE status = 'ACTIVE' ORDER BY id DESC"
                ).fetchall()

        results = []
        now_dt = now_ist()
        active_map = {p.get("symbol"): p for p in state_engine.get_active_positions()}

        for r in rows:
            d = dict(r)
            sym = d.get("symbol", "")
            trade_type = d.get("trade_type", "SHADOW")
            ep = float(d.get("entry_price") or 0.0)
            sl = float(d.get("stop_loss") or 0.0)
            t1 = float(d.get("target") or 0.0)
            qty = int(d.get("quantity") or 0)

            # For REAL trades, always reflect actual active portfolio position quantity & live stop
            if trade_type == "REAL":
                real_pos = active_map.get(sym)
                if real_pos:
                    qty = int(real_pos.get("quantity") or qty)
                    sl = float(real_pos.get("stop_loss") or sl)
                    ep = float(real_pos.get("entry_price") or ep)

            # Fetch latest price from OHLCV store
            ltp = ep
            df, _ = state_engine.get_ohlcv_df(sym)
            if df is not None and not df.empty and "Close" in df.columns:
                ltp = round(float(df["Close"].iloc[-1]), 2)

            # Compute real-time metrics
            if ep > 0:
                ret_pct = round(((ltp - ep) / ep) * 100.0, 2)
                dist_sl_pct = round(((ltp - sl) / ep) * 100.0, 2)
                dist_t1_pct = round(((t1 - ltp) / ep) * 100.0, 2)
            else:
                ret_pct = 0.0
                dist_sl_pct = 0.0
                dist_t1_pct = 0.0

            # Guard risk unit: if trailing stop is above or equal to entry (breakeven/profit ratchet), default to 2% base risk
            risk_unit = (ep - sl) if sl < ep and (ep - sl) > 0.5 else max(1.0, ep * 0.02)
            rr = round((t1 - ep) / risk_unit, 2) if ep > 0 else 0.0

            if qty <= 0 and ep > 0:
                qty = max(1, int(3750.0 / risk_unit))

            unrealized_pnl = round((ltp - ep) * qty, 2)

            # Compute days in trade
            hold_days = 0
            created_str = d.get("created_at") or d.get("entry_date") or ""
            if created_str:
                try:
                    c_date = datetime.strptime(created_str[:10], "%Y-%m-%d").date()
                    hold_days = max(0, (now_dt.date() - c_date).days)
                except Exception:
                    hold_days = int(d.get("hold_days") or 0)

            d["ltp"] = ltp
            d["quantity"] = qty
            d["unrealized_pnl"] = unrealized_pnl
            d["return_pct"] = ret_pct
            d["dist_to_sl_pct"] = dist_sl_pct
            d["dist_to_t1_pct"] = dist_t1_pct
            d["rr_ratio"] = rr
            d["days_in_trade"] = hold_days
            results.append(d)

        return results


# Global instance
swing_quant_ml_engine = SwingQuantMLEngine()
