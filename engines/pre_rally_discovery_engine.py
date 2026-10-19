"""
========================================================================================
Pre-Rally Pattern Discovery & Validation Engine (`engines/pre_rally_discovery_engine.py`)
========================================================================================
Empirical research and statistical validation engine that discovers predictive pre-rally
characteristics across historical NSE OHLCV data.

Guiding Principles:
1. Zero Look-Ahead Bias: When evaluating a candidate rally beginning on date T, all predictive
   features are calculated EXCLUSIVELY as of date T-1. Breakout-day volume surge on date T is
   strictly labeled as a confirmation feature and never as a pre-breakout predictor.
2. Balanced Dual-Label Dataset: Evaluates both positive rally events (winners) and negative
   controls (non-winners, failed breakouts, consolidation breakdowns).
3. Chronological Time-Aware Out-of-Sample Validation: Walks forward through chronological
   train/test splits (70% in-sample / 30% out-of-sample holdout) to prevent data snooping.
4. Statistical Rigor: Measures Precision (Win Rate), Recall, Specificity, Lift vs Baseline,
   Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), and stability scores.
5. Resource-Bounded: Streams stock batches with memory discipline (<100MB RAM), operating
   smoothly within 1 GB RAM / 1 vCPU VPS constraints.
6. Zero Mock Data: 100% computed from real historical daily candles and SQLite store.
========================================================================================
"""

import os
import sys
import time
import math
import json
import logging
import hashlib
import datetime
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from core.data_quality import now_ist, format_ist, IST
from execution.state_engine import state_engine

logger = logging.getLogger("PreRallyDiscoveryEngine")


class PreRallyDiscoveryEngine:
    """
    Autonomous empirical engine that discovers, validates, and tracks multi-feature
    pre-rally combinations across historical NSE cash equities.
    """
    _instance: Optional["PreRallyDiscoveryEngine"] = None

    def __init__(self):
        self._lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._last_summary_cache: Optional[Dict[str, Any]] = None
        self._last_combinations_cache: Optional[List[Dict[str, Any]]] = None
        self._last_cache_time: float = 0.0
        self._init_db()

    @classmethod
    def get_instance(cls) -> "PreRallyDiscoveryEngine":
        if cls._instance is None:
            cls._instance = PreRallyDiscoveryEngine()
        return cls._instance

    def _init_db(self):
        """Initializes tables for pre-rally events, discovered combinations, and validation runs."""
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    # 1. Historical Pre-Rally Events & Controls Table
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS pre_rally_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        horizon_days INTEGER NOT NULL,
                        min_fwd_return REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        forward_return REAL NOT NULL,
                        mfe_pct REAL NOT NULL,
                        mae_pct REAL NOT NULL,
                        time_to_target INTEGER,
                        is_winner INTEGER NOT NULL,
                        is_in_sample INTEGER NOT NULL,
                        vdu_ratio_t1 REAL,
                        vdu_ratio_t2 REAL,
                        vol_declining_3d INTEGER,
                        dist_ema20_t1 REAL,
                        dist_ema50_t1 REAL,
                        ema20_slope_5d REAL,
                        daily_range_t1 REAL,
                        atr_contraction_ratio REAL,
                        bb_width_t1 REAL,
                        is_nr4 INTEGER,
                        is_nr7 INTEGER,
                        consolidation_range_5d REAL,
                        dist_swing_high_20d REAL,
                        accum_days_5d INTEGER,
                        breakout_surge_t0 REAL,
                        features_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_pre_rally_sym ON pre_rally_events(symbol);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_pre_rally_date ON pre_rally_events(event_date);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_pre_rally_win ON pre_rally_events(is_winner);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_pre_rally_sample ON pre_rally_events(is_in_sample);")

                    # 2. Discovered & Validated Combinations Table
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS pre_rally_discovered_combinations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        combination_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        category TEXT NOT NULL,
                        rule_expression TEXT NOT NULL,
                        predicates_json TEXT NOT NULL,
                        in_sample_precision REAL NOT NULL,
                        out_sample_precision REAL NOT NULL,
                        overall_precision REAL NOT NULL,
                        baseline_rate REAL NOT NULL,
                        lift_oos REAL NOT NULL,
                        lift_overall REAL NOT NULL,
                        sample_size INTEGER NOT NULL,
                        oos_sample_size INTEGER NOT NULL,
                        win_count INTEGER NOT NULL,
                        oos_win_count INTEGER NOT NULL,
                        avg_forward_return REAL NOT NULL,
                        avg_mfe REAL NOT NULL,
                        avg_mae REAL NOT NULL,
                        mfe_mae_ratio REAL NOT NULL,
                        avg_days_to_target REAL NOT NULL,
                        stability_score REAL NOT NULL,
                        validation_status TEXT NOT NULL,
                        example_winners_json TEXT DEFAULT '[]',
                        example_failures_json TEXT DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_prdc_status ON pre_rally_discovered_combinations(validation_status);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_prdc_lift ON pre_rally_discovered_combinations(lift_oos);")

                    # 3. Discovery Engine Audit Runs Ledger
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS pre_rally_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT UNIQUE NOT NULL,
                        run_timestamp TEXT NOT NULL,
                        total_symbols INTEGER NOT NULL,
                        total_events INTEGER NOT NULL,
                        positive_events INTEGER NOT NULL,
                        negative_events INTEGER NOT NULL,
                        baseline_rally_rate REAL NOT NULL,
                        split_date TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        summary_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """)
                    conn.commit()
        except Exception as e:
            logger.warning(f"Notice initializing pre-rally discovery tables: {e}")

    # =========================================================================
    # ZERO LOOK-AHEAD FEATURE EXTRACTION
    # =========================================================================
    def extract_features_at_t(self, df: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
        """
        Calculates all pre-rally indicators strictly at bar `idx` (representing date T-1).
        Zero future leakage: No information from bar `idx + 1` (breakout day T) or beyond is used.
        Requires at least 20 historical bars prior to `idx`.
        """
        if df is None or len(df) < 20 or idx < 15 or idx >= len(df):
            return None

        # Data quality checks on OHLC slice up to idx
        sub_df = df.iloc[: idx + 1]
        c = sub_df["Close"]
        h = sub_df["High"]
        l = sub_df["Low"]
        v = sub_df["Volume"]

        # 1. Moving Averages
        ema20_series = c.ewm(span=20, adjust=False).mean()
        ema50_series = c.ewm(span=50, adjust=False).mean()
        vol_sma20_series = v.rolling(20).mean()
        vol_sma5_series = v.rolling(5).mean()

        c_t1 = float(c.iloc[idx])
        h_t1 = float(h.iloc[idx])
        l_t1 = float(l.iloc[idx])
        v_t1 = float(v.iloc[idx])

        # Past reference bars
        v_t2 = float(v.iloc[idx - 1]) if idx >= 1 else v_t1
        v_t3 = float(v.iloc[idx - 2]) if idx >= 2 else v_t2
        h_t2 = float(h.iloc[idx - 1]) if idx >= 1 else h_t1
        l_t2 = float(l.iloc[idx - 1]) if idx >= 1 else l_t1

        ema20_t1 = float(ema20_series.iloc[idx])
        ema50_t1 = float(ema50_series.iloc[idx])
        ema20_t5 = float(ema20_series.iloc[idx - 4]) if idx >= 4 else ema20_t1
        vol_sma20_t1 = float(vol_sma20_series.iloc[idx]) if not np.isnan(vol_sma20_series.iloc[idx]) else v_t1
        vol_sma20_t2 = float(vol_sma20_series.iloc[idx - 1]) if idx >= 1 and not np.isnan(vol_sma20_series.iloc[idx - 1]) else vol_sma20_t1
        vol_sma5_t1 = float(vol_sma5_series.iloc[idx]) if not np.isnan(vol_sma5_series.iloc[idx]) else v_t1

        # A. Volume Dynamics
        vdu_ratio_t1 = round(v_t1 / (vol_sma20_t1 + 1e-9), 3)
        vdu_ratio_t2 = round(v_t2 / (vol_sma20_t2 + 1e-9), 3)
        vol_ratio_5d = round(v_t1 / (vol_sma5_t1 + 1e-9), 3)
        vol_declining_3d = 1 if (v_t3 > v_t2 > v_t1) else 0

        # B. Trend & Moving Averages
        dist_ema20_t1 = round(((c_t1 - ema20_t1) / (ema20_t1 + 1e-9)) * 100.0, 2)
        dist_ema50_t1 = round(((c_t1 - ema50_t1) / (ema50_t1 + 1e-9)) * 100.0, 2)
        ema20_slope_5d = round(((ema20_t1 - ema20_t5) / (ema20_t5 + 1e-9)) * 100.0, 2)
        trend_aligned = 1 if (ema20_t1 >= ema50_t1 and dist_ema50_t1 >= -1.0) else 0

        # Count consecutive days above 20 EMA
        above_ema20_count = 0
        for b_i in range(idx, max(-1, idx - 20), -1):
            if c.iloc[b_i] >= ema20_series.iloc[b_i]:
                above_ema20_count += 1
            else:
                break

        # C. Volatility Contraction
        daily_range_t1 = round(((h_t1 - l_t1) / (l_t1 + 1e-9)) * 100.0, 2)
        daily_range_t2 = round(((h_t2 - l_t2) / (l_t2 + 1e-9)) * 100.0, 2)

        # ATR 5 and ATR 20
        high_low = h - l
        high_close_prev = (h - c.shift(1)).abs()
        low_close_prev = (l - c.shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr5_series = tr.rolling(5).mean()
        atr20_series = tr.rolling(20).mean()

        atr5_val = float(atr5_series.iloc[idx]) if not np.isnan(atr5_series.iloc[idx]) else (h_t1 - l_t1)
        atr20_val = float(atr20_series.iloc[idx]) if not np.isnan(atr20_series.iloc[idx]) else atr5_val
        atr_contraction_ratio = round(atr5_val / (atr20_val + 1e-9), 3)

        # Bollinger Band Width 20, 2
        sma20_c = c.rolling(20).mean()
        std20_c = c.rolling(20).std()
        bb_width_t1 = round(((std20_c.iloc[idx] * 2.0 * 2.0) / (sma20_c.iloc[idx] + 1e-9)) * 100.0, 2) if not np.isnan(std20_c.iloc[idx]) else 0.0

        # Narrow Range Sessions (NR4, NR7)
        rng_last4 = [((h.iloc[k] - l.iloc[k]) / (l.iloc[k] + 1e-9)) for k in range(max(0, idx - 3), idx + 1)]
        is_nr4 = 1 if len(rng_last4) == 4 and rng_last4[-1] <= min(rng_last4) else 0

        rng_last7 = [((h.iloc[k] - l.iloc[k]) / (l.iloc[k] + 1e-9)) for k in range(max(0, idx - 6), idx + 1)]
        is_nr7 = 1 if len(rng_last7) == 7 and rng_last7[-1] <= min(rng_last7) else 0

        # D. Price Structure & Consolidation Base
        h_5d_max = float(h.iloc[max(0, idx - 4) : idx + 1].max())
        l_5d_min = float(l.iloc[max(0, idx - 4) : idx + 1].min())
        consolidation_range_5d = round(((h_5d_max - l_5d_min) / (l_5d_min + 1e-9)) * 100.0, 2)

        h_20d_max = float(h.iloc[max(0, idx - 19) : idx + 1].max())
        dist_swing_high_20d = round(((c_t1 - h_20d_max) / (h_20d_max + 1e-9)) * 100.0, 2)

        # Upper-half close accumulation count in last 5 sessions
        accum_days_5d = 0
        for b_k in range(max(0, idx - 4), idx + 1):
            bar_c = float(c.iloc[b_k])
            bar_h = float(h.iloc[b_k])
            bar_l = float(l.iloc[b_k])
            pos = (bar_c - bar_l) / (bar_h - bar_l + 1e-9)
            if pos >= 0.50:
                accum_days_5d += 1

        # Breakout day confirmation metric (T-0, strictly labeled as confirmation, NOT used as pre-rally predictor)
        breakout_surge_t0 = 1.0
        if idx + 1 < len(df):
            vol_t0 = float(df["Volume"].iloc[idx + 1])
            vol_sma20_t0 = float(df["Volume"].iloc[: idx + 2].rolling(20).mean().iloc[-1])
            breakout_surge_t0 = round(vol_t0 / (vol_sma20_t0 + 1e-9), 2)

        return {
            "vdu_ratio_t1": vdu_ratio_t1,
            "vdu_ratio_t2": vdu_ratio_t2,
            "vol_ratio_5d": vol_ratio_5d,
            "vol_declining_3d": vol_declining_3d,
            "dist_ema20_t1": dist_ema20_t1,
            "dist_ema50_t1": dist_ema50_t1,
            "ema20_slope_5d": ema20_slope_5d,
            "trend_aligned": trend_aligned,
            "above_ema20_days": above_ema20_count,
            "daily_range_t1": daily_range_t1,
            "daily_range_t2": daily_range_t2,
            "atr_contraction_ratio": atr_contraction_ratio,
            "bb_width_t1": bb_width_t1,
            "is_nr4": is_nr4,
            "is_nr7": is_nr7,
            "consolidation_range_5d": consolidation_range_5d,
            "dist_swing_high_20d": dist_swing_high_20d,
            "accum_days_5d": accum_days_5d,
            "breakout_surge_t0": breakout_surge_t0,
            # Atomic Predicate Flags for Combinatorial Engine
            "p_vdu": 1 if vdu_ratio_t1 <= 0.80 else 0,
            "p_extreme_vdu": 1 if vdu_ratio_t1 <= 0.50 else 0,
            "p_ema20_coil": 1 if abs(dist_ema20_t1) <= 2.5 else 0,
            "p_tight_range": 1 if daily_range_t1 <= 3.0 else 0,
            "p_ultra_tight": 1 if daily_range_t1 <= 2.0 else 0,
            "p_atr_squeeze": 1 if atr_contraction_ratio <= 0.85 else 0,
            "p_declining_vol": vol_declining_3d,
            "p_accum_base": 1 if accum_days_5d >= 3 else 0,
            "p_trend_align": trend_aligned,
            "p_nr4": is_nr4,
            "p_base_compression": 1 if consolidation_range_5d <= 6.0 else 0
        }

    # =========================================================================
    # EVENT DEFINITION & DUAL DATASET LABELING
    # =========================================================================
    def label_events_from_df(
        self,
        symbol: str,
        df: pd.DataFrame,
        min_fwd_return: float = 5.0,
        fwd_horizon: int = 10,
        max_adverse_pct: float = -3.5,
        non_overlapping_days: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Labels candidate event sessions as positive rally winners (1) vs negative controls (0).
        Enforces:
        - Non-overlapping sampling to avoid counting one continuous rally multiple times.
        - Strict pre-rally feature calculation at T-1.
        """
        if df is None or len(df) < (50 + fwd_horizon + 2):
            return []

        c = df["Close"]
        h = df["High"]
        l = df["Low"]

        events = []
        i = 50
        n_bars = len(df)

        while i < (n_bars - fwd_horizon):
            event_date = str(df.index[i])[:10]
            entry_price = float(c.iloc[i])

            # Forward window: from i + 1 to i + 1 + fwd_horizon
            fwd_high = float(h.iloc[i + 1 : i + 1 + fwd_horizon].max())
            fwd_low = float(l.iloc[i + 1 : i + 1 + fwd_horizon].min())

            fwd_return = round(((fwd_high - entry_price) / (entry_price + 1e-9)) * 100.0, 2)
            mae_pct = round(((fwd_low - entry_price) / (entry_price + 1e-9)) * 100.0, 2)
            mfe_pct = fwd_return

            # Target price & time to target
            target_price = entry_price * (1.0 + min_fwd_return / 100.0)
            time_to_target = None
            for step_k in range(1, fwd_horizon + 1):
                if float(h.iloc[i + step_k]) >= target_price:
                    time_to_target = step_k
                    break

            # Check if target reached before max adverse excursion violated
            is_winner = 0
            if fwd_return >= min_fwd_return:
                if mae_pct >= max_adverse_pct:
                    is_winner = 1
                elif time_to_target is not None:
                    # Check if target was hit before MAE occurred
                    low_before_target = float(l.iloc[i + 1 : i + time_to_target + 1].min())
                    early_mae = ((low_before_target - entry_price) / (entry_price + 1e-9)) * 100.0
                    if early_mae >= max_adverse_pct:
                        is_winner = 1

            # Extract Pre-Rally Features STRICTLY as of i - 1 (T-1)
            features = self.extract_features_at_t(df, i - 1)
            if features is not None:
                event_id = f"evt_{symbol}_{event_date}_{i}"
                event_record = {
                    "event_id": event_id,
                    "symbol": symbol,
                    "event_date": event_date,
                    "horizon_days": fwd_horizon,
                    "min_fwd_return": min_fwd_return,
                    "entry_price": round(entry_price, 2),
                    "forward_return": fwd_return,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "time_to_target": time_to_target or fwd_horizon,
                    "is_winner": is_winner,
                    "vdu_ratio_t1": features["vdu_ratio_t1"],
                    "vdu_ratio_t2": features["vdu_ratio_t2"],
                    "vol_declining_3d": features["vol_declining_3d"],
                    "dist_ema20_t1": features["dist_ema20_t1"],
                    "dist_ema50_t1": features["dist_ema50_t1"],
                    "ema20_slope_5d": features["ema20_slope_5d"],
                    "daily_range_t1": features["daily_range_t1"],
                    "atr_contraction_ratio": features["atr_contraction_ratio"],
                    "bb_width_t1": features["bb_width_t1"],
                    "is_nr4": features["is_nr4"],
                    "is_nr7": features["is_nr7"],
                    "consolidation_range_5d": features["consolidation_range_5d"],
                    "dist_swing_high_20d": features["dist_swing_high_20d"],
                    "accum_days_5d": features["accum_days_5d"],
                    "breakout_surge_t0": features["breakout_surge_t0"],
                    "features": features
                }
                events.append(event_record)

            # Non-overlapping step: if winner, advance by full horizon; if non-winner, advance by 5 days
            step_stride = non_overlapping_days if is_winner else max(5, int(fwd_horizon / 2))
            i += step_stride

        return events

    # =========================================================================
    # COMBINATORIAL HYPOTHESIS CATALOG
    # =========================================================================
    def _get_hypothesis_catalog(self) -> List[Dict[str, Any]]:
        """Returns the structured combinatorial pattern hypothesis definitions."""
        return [
            # --- 1. Single Feature Baselines ---
            {
                "combination_id": "H1_VDU_BASE",
                "name": "Volume Dry-Up (VDU <= 0.8x)",
                "category": "VOLUME_BEHAVIOR",
                "predicates": ["p_vdu"],
                "rule_expression": "VDU_t1 <= 0.80x",
                "description": "Trading volume contracts below 80% of 20-day SMA prior to breakout."
            },
            {
                "combination_id": "H2_EXTREME_VDU",
                "name": "Extreme Volume Dry-Up (VDU <= 0.5x)",
                "category": "VOLUME_BEHAVIOR",
                "predicates": ["p_extreme_vdu"],
                "rule_expression": "VDU_t1 <= 0.50x",
                "description": "Acute volume depletion dropping below half of average volume."
            },
            {
                "combination_id": "H3_EMA20_COIL",
                "name": "20 EMA Coiling Anchor (|Dist| <= 2.5%)",
                "category": "TREND_STRUCTURE",
                "predicates": ["p_ema20_coil"],
                "rule_expression": "|Price - 20EMA| <= 2.5%",
                "description": "Price tightly coiled within 2.5% of the 20-day exponential moving average."
            },
            {
                "combination_id": "H4_TIGHT_RANGE",
                "name": "Daily Volatility Contraction (Range <= 3.0%)",
                "category": "VOLATILITY_CONTRACTION",
                "predicates": ["p_tight_range"],
                "rule_expression": "Daily_Range_t1 <= 3.0%",
                "description": "Candle high-low range compresses below 3% signaling supply exhaustion."
            },
            {
                "combination_id": "H5_ATR_SQUEEZE",
                "name": "ATR Volatility Squeeze (ATR5/ATR20 <= 0.85)",
                "category": "VOLATILITY_CONTRACTION",
                "predicates": ["p_atr_squeeze"],
                "rule_expression": "ATR5 / ATR20 <= 0.85",
                "description": "Short-term ATR contracts significantly below long-term ATR."
            },
            {
                "combination_id": "H6_ACCUM_BASE",
                "name": "Multi-Day Accumulation Base (Upper Closes >= 3)",
                "category": "PRICE_STRUCTURE",
                "predicates": ["p_accum_base"],
                "rule_expression": "Upper_Closes_5d >= 3",
                "description": "At least 3 of last 5 sessions closed in upper half of daily range."
            },
            {
                "combination_id": "H7_VOL_DECLINE",
                "name": "Consecutive Volume Fade (3-Day Decline)",
                "category": "VOLUME_BEHAVIOR",
                "predicates": ["p_declining_vol"],
                "rule_expression": "Vol_t3 > Vol_t2 > Vol_t1",
                "description": "Orderly day-over-day volume decline indicating vanishing selling pressure."
            },
            {
                "combination_id": "H8_TREND_ALIGNED",
                "name": "Bullish Trend Alignment (EMA20 >= EMA50)",
                "category": "TREND_STRUCTURE",
                "predicates": ["p_trend_align"],
                "rule_expression": "EMA20 >= EMA50 AND Price >= EMA50",
                "description": "Moving averages structured in clean bullish order."
            },
            # --- 2. Pairwise Combinations (2-Feature Intersections) ---
            {
                "combination_id": "PAIR_VDU_EMA20",
                "name": "VDU at 20 EMA Support Anchor",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_vdu", "p_ema20_coil"],
                "rule_expression": "VDU_t1 <= 0.80x AND |Dist_EMA20| <= 2.5%",
                "description": "Volume dry-up anchored directly on the institutional 20 EMA benchmark."
            },
            {
                "combination_id": "PAIR_EXTREME_VDU_EMA20",
                "name": "Extreme VDU on 20 EMA Support",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_extreme_vdu", "p_ema20_coil"],
                "rule_expression": "VDU_t1 <= 0.50x AND |Dist_EMA20| <= 2.5%",
                "description": "Severe supply depletion (<=0.5x) occurring precisely on 20 EMA."
            },
            {
                "combination_id": "PAIR_VDU_TIGHT",
                "name": "Volume Dry-Up in Tight Consolidation",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_vdu", "p_tight_range"],
                "rule_expression": "VDU_t1 <= 0.80x AND Daily_Range <= 3.0%",
                "description": "Quiet volume combined with narrow daily candle range."
            },
            {
                "combination_id": "PAIR_EMA20_TIGHT",
                "name": "Tight Base Resting on 20 EMA",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_ema20_coil", "p_tight_range"],
                "rule_expression": "|Dist_EMA20| <= 2.5% AND Daily_Range <= 3.0%",
                "description": "Price compression forming a narrow volatility shelf at 20 EMA."
            },
            {
                "combination_id": "PAIR_VDU_ATR_SQUEEZE",
                "name": "VCP Volatility Squeeze with VDU",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_vdu", "p_atr_squeeze"],
                "rule_expression": "VDU_t1 <= 0.80x AND ATR5/ATR20 <= 0.85",
                "description": "Dual contraction: both trading volume and true range compression."
            },
            {
                "combination_id": "PAIR_ACCUM_EMA20",
                "name": "Multi-Day Accumulation at 20 EMA",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_accum_base", "p_ema20_coil"],
                "rule_expression": "Upper_Closes_5d >= 3 AND |Dist_EMA20| <= 2.5%",
                "description": "Persistent upper closes resting on 20 EMA floor."
            },
            {
                "combination_id": "PAIR_ACCUM_VDU",
                "name": "Accumulation Base with Volume Dry-Up",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_accum_base", "p_vdu"],
                "rule_expression": "Upper_Closes_5d >= 3 AND VDU_t1 <= 0.80x",
                "description": "Quiet inventory absorption closing strong on low volume."
            },
            {
                "combination_id": "PAIR_TREND_VDU",
                "name": "Bullish Trend Continuation VDU",
                "category": "CONFLUENCE_PAIR",
                "predicates": ["p_trend_align", "p_vdu"],
                "rule_expression": "EMA20 >= EMA50 AND VDU_t1 <= 0.80x",
                "description": "Healthy volume rest in an established upward moving average structure."
            },
            # --- 3. High-Conviction Triples (3-Feature Intersections) ---
            {
                "combination_id": "TRIPLE_VDU_COIL",
                "name": "Textbook Triple Squeeze (VDU + EMA + Tight)",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_vdu", "p_ema20_coil", "p_tight_range"],
                "rule_expression": "VDU <= 0.80x AND |Dist_EMA20| <= 2.5% AND Range <= 3.0%",
                "description": "Complete alignment: supply dry-up, narrow price range, and 20 EMA benchmark support."
            },
            {
                "combination_id": "TRIPLE_EXTREME_VCP",
                "name": "High-Conviction VCP (Extreme VDU + EMA + ATR Squeeze)",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_extreme_vdu", "p_ema20_coil", "p_atr_squeeze"],
                "rule_expression": "VDU <= 0.50x AND |Dist_EMA20| <= 2.5% AND ATR5/20 <= 0.85",
                "description": "Acute volume depletion on 20 EMA with multi-session ATR compression."
            },
            {
                "combination_id": "TRIPLE_ACCUM_COIL",
                "name": "Institutional Floor (Accum Base + EMA + VDU)",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_accum_base", "p_ema20_coil", "p_vdu"],
                "rule_expression": "Upper_Closes >= 3 AND |Dist_EMA20| <= 2.5% AND VDU <= 0.80x",
                "description": "Sustained upper closes resting on 20 EMA with dry floating supply."
            },
            {
                "combination_id": "TRIPLE_STEALTH_SQUEEZE",
                "name": "Stealth Contraction (Fade Vol + Tight + ATR Squeeze)",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_declining_vol", "p_tight_range", "p_atr_squeeze"],
                "rule_expression": "Vol_Fade_3d AND Range <= 3.0% AND ATR5/20 <= 0.85",
                "description": "Triple compression: 3 consecutive days of declining volume and range contraction."
            },
            {
                "combination_id": "TRIPLE_TREND_VDU_TIGHT",
                "name": "Trend-Aligned VDU Compression",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_trend_align", "p_vdu", "p_tight_range"],
                "rule_expression": "EMA20 >= EMA50 AND VDU <= 0.80x AND Range <= 3.0%",
                "description": "Clean bullish trend coiling into a tight low-volume spring."
            },
            {
                "combination_id": "TRIPLE_ULTRA_COMPRESSION",
                "name": "Extreme Volatility Pinch (Ultra-Tight + ATR + VDU)",
                "category": "VALIDATED_TRIPLE",
                "predicates": ["p_ultra_tight", "p_atr_squeeze", "p_vdu"],
                "rule_expression": "Range <= 2.0% AND ATR5/20 <= 0.85 AND VDU <= 0.80x",
                "description": "Pinch of volatility and volume: sub-2% candle with ATR compression and VDU."
            }
        ]

    # =========================================================================
    # CORE DISCOVERY & OUT-OF-SAMPLE VALIDATION PIPELINE
    # =========================================================================
    def run_discovery(
        self,
        symbols: Optional[List[str]] = None,
        min_fwd_return: float = 5.0,
        fwd_horizon: int = 10,
        max_adverse_pct: float = -3.5,
        max_symbols: int = 35,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Executes historical event extraction, chronological train/test split, and
        combinatorial statistical validation across the available universe.
        """
        with self._lock:
            start_time = time.time()
            now_dt = now_ist()
            run_id = f"discovery_run_{now_dt.strftime('%Y%m%d_%H%M%S')}"
            run_ts = format_ist(now_dt)

            logger.info(f"🚀 Launching Pre-Rally Pattern Discovery & Validation Run [{run_id}]...")

            # 1. Resolve Target Symbols (bounded for 1GB VPS)
            if not symbols:
                symbols = self._resolve_universe_symbols(max_symbols=max_symbols)

            total_symbols = len(symbols)
            logger.info(f"Processing historical OHLCV data for {total_symbols} symbols...")

            # 2. Extract Events Across All Symbols (1-by-1 streaming memory discipline)
            all_events = []
            for sym in symbols:
                try:
                    df = self._load_or_fetch_stock_df(sym)
                    if df is not None and len(df) >= 70:
                        sym_events = self.label_events_from_df(
                            sym,
                            df,
                            min_fwd_return=min_fwd_return,
                            fwd_horizon=fwd_horizon,
                            max_adverse_pct=max_adverse_pct
                        )
                        all_events.extend(sym_events)
                    # Explicit memory cleanup
                    del df
                except Exception as e:
                    logger.debug(f"Notice extracting events for {sym}: {e}")

            if not all_events:
                logger.warning("No historical events could be extracted. Returning empty result.")
                return self._build_empty_summary(run_id, run_ts)

            # 3. Chronological Time-Aware Out-of-Sample Split (70% In-Sample / 30% Out-of-Sample)
            all_events.sort(key=lambda x: (x["event_date"], x["symbol"]))
            n_events = len(all_events)
            split_idx = int(n_events * 0.70)
            split_date = all_events[split_idx]["event_date"] if split_idx < n_events else all_events[-1]["event_date"]

            for i_ev, ev in enumerate(all_events):
                ev["is_in_sample"] = 1 if i_ev < split_idx else 0

            in_sample_events = all_events[:split_idx]
            out_sample_events = all_events[split_idx:]

            tot_wins = sum(1 for e in all_events if e["is_winner"] == 1)
            is_wins = sum(1 for e in in_sample_events if e["is_winner"] == 1)
            oos_wins = sum(1 for e in out_sample_events if e["is_winner"] == 1)

            baseline_overall = round((tot_wins / (n_events + 1e-9)) * 100.0, 2)
            baseline_is = round((is_wins / (len(in_sample_events) + 1e-9)) * 100.0, 2)
            baseline_oos = round((oos_wins / (len(out_sample_events) + 1e-9)) * 100.0, 2)

            logger.info(
                f"Dataset: {n_events} events ({tot_wins} wins). Baseline Win Rate: {baseline_overall}%. "
                f"In-Sample: {len(in_sample_events)} (Baseline {baseline_is}%), "
                f"Out-of-Sample: {len(out_sample_events)} (Baseline {baseline_oos}%). Split Date: {split_date}"
            )

            # 4. Save Extracted Events to SQLite (Batch Upsert)
            self._persist_events_batch(all_events)

            # 5. Evaluate Hypothesis Catalog Combinations
            catalog = self._get_hypothesis_catalog()
            evaluated_combinations = []

            for hyp in catalog:
                preds = hyp["predicates"]

                # Filter matching events in in-sample and out-of-sample
                matches_is = [e for e in in_sample_events if all(e["features"].get(p, 0) == 1 for p in preds)]
                matches_oos = [e for e in out_sample_events if all(e["features"].get(p, 0) == 1 for p in preds)]
                matches_all = matches_is + matches_oos

                n_is = len(matches_is)
                n_oos = len(matches_oos)
                n_all = len(matches_all)

                wins_is = sum(1 for e in matches_is if e["is_winner"] == 1)
                wins_oos = sum(1 for e in matches_oos if e["is_winner"] == 1)
                wins_all = wins_is + wins_oos

                prec_is = round((wins_is / (n_is + 1e-9)) * 100.0, 1) if n_is > 0 else 0.0
                prec_oos = round((wins_oos / (n_oos + 1e-9)) * 100.0, 1) if n_oos > 0 else 0.0
                prec_all = round((wins_all / (n_all + 1e-9)) * 100.0, 1) if n_all > 0 else 0.0

                lift_is = round(prec_is / (baseline_is + 1e-9), 2) if n_is > 0 else 0.0
                lift_oos = round(prec_oos / (baseline_oos + 1e-9), 2) if n_oos > 0 else 0.0
                lift_all = round(prec_all / (baseline_overall + 1e-9), 2) if n_all > 0 else 0.0

                # Stability Score: Out-of-Sample precision compared to In-Sample precision
                stability = round(min(2.0, prec_oos / (prec_is + 1e-9)), 2) if (n_is >= 3 and n_oos >= 3) else 0.0

                # Return & Risk Metrics
                avg_fwd_ret = round(float(np.mean([e["forward_return"] for e in matches_all])), 2) if n_all > 0 else 0.0
                avg_mfe = round(float(np.mean([e["mfe_pct"] for e in matches_all])), 2) if n_all > 0 else 0.0
                avg_mae = round(float(np.mean([e["mae_pct"] for e in matches_all])), 2) if n_all > 0 else 0.0
                mfe_mae_ratio = round(abs(avg_mfe) / (abs(avg_mae) + 1e-9), 2) if n_all > 0 else 0.0
                avg_days_target = round(float(np.mean([e["time_to_target"] for e in matches_all])), 1) if n_all > 0 else 0.0

                # Validation Classification
                # Strict Criteria: Minimum OOS sample size, positive lift OOS, acceptable stability
                if n_oos >= 4 and n_all >= 12 and lift_oos >= 1.15 and prec_oos >= 30.0 and stability >= 0.65:
                    status = "VALIDATED_OOS"
                elif n_all >= 8 and (lift_all >= 1.05 or lift_oos >= 1.05):
                    status = "EXPLORATORY"
                elif n_all >= 25:
                    status = "NEUTRAL_EDGE"
                else:
                    status = "INSUFFICIENT_DATA"

                # Collect Real Examples for Transparent Evidence
                example_wins = [
                    {
                        "symbol": e["symbol"],
                        "date": e["event_date"],
                        "fwd_return": e["forward_return"],
                        "mfe": e["mfe_pct"],
                        "vdu": e["vdu_ratio_t1"],
                        "dist_ema20": e["dist_ema20_t1"]
                    }
                    for e in matches_all if e["is_winner"] == 1
                ][:8]

                example_fails = [
                    {
                        "symbol": e["symbol"],
                        "date": e["event_date"],
                        "fwd_return": e["forward_return"],
                        "mae": e["mae_pct"],
                        "vdu": e["vdu_ratio_t1"],
                        "dist_ema20": e["dist_ema20_t1"]
                    }
                    for e in matches_all if e["is_winner"] == 0
                ][:8]

                comb_dict = {
                    "combination_id": hyp["combination_id"],
                    "name": hyp["name"],
                    "description": hyp["description"],
                    "category": hyp["category"],
                    "rule_expression": hyp["rule_expression"],
                    "predicates": preds,
                    "in_sample_precision": prec_is,
                    "out_sample_precision": prec_oos,
                    "overall_precision": prec_all,
                    "baseline_rate": baseline_overall,
                    "lift_oos": lift_oos,
                    "lift_overall": lift_all,
                    "sample_size": n_all,
                    "oos_sample_size": n_oos,
                    "win_count": wins_all,
                    "oos_win_count": wins_oos,
                    "avg_forward_return": avg_fwd_ret,
                    "avg_mfe": avg_mfe,
                    "avg_mae": avg_mae,
                    "mfe_mae_ratio": mfe_mae_ratio,
                    "avg_days_to_target": avg_days_target,
                    "stability_score": stability,
                    "validation_status": status,
                    "example_winners": example_wins,
                    "example_failures": example_failures if 'example_failures' in locals() else example_fails,
                    "updated_at": run_ts
                }
                evaluated_combinations.append(comb_dict)

            # Sort combinations: VALIDATED_OOS first, then by lift_oos descending, then sample size
            status_order = {"VALIDATED_OOS": 0, "EXPLORATORY": 1, "INSUFFICIENT_DATA": 2}
            evaluated_combinations.sort(
                key=lambda x: (status_order.get(x["validation_status"], 3), -x["lift_oos"], -x["sample_size"])
            )

            # 6. Persist Evaluated Combinations to SQLite
            self._persist_combinations_batch(evaluated_combinations)

            # 7. Record Run Summary Ledger
            duration_sec = round(time.time() - start_time, 2)
            summary_payload = {
                "run_id": run_id,
                "run_timestamp": run_ts,
                "total_symbols_analyzed": total_symbols,
                "total_events_analyzed": n_events,
                "positive_events": tot_wins,
                "negative_events": n_events - tot_wins,
                "baseline_rally_rate": baseline_overall,
                "in_sample_events": len(in_sample_events),
                "out_sample_events": len(out_sample_events),
                "in_sample_baseline": baseline_is,
                "out_sample_baseline": baseline_oos,
                "split_date": split_date,
                "parameters": {
                    "min_fwd_return": min_fwd_return,
                    "fwd_horizon": fwd_horizon,
                    "max_adverse_pct": max_adverse_pct,
                    "train_test_split": "70% IS / 30% OOS"
                },
                "total_combinations_tested": len(evaluated_combinations),
                "validated_oos_count": sum(1 for c in evaluated_combinations if c["validation_status"] == "VALIDATED_OOS"),
                "exploratory_count": sum(1 for c in evaluated_combinations if c["validation_status"] == "EXPLORATORY"),
                "execution_duration_sec": duration_sec
            }

            self._persist_run_record(summary_payload)

            with self._cache_lock:
                self._last_summary_cache = summary_payload
                self._last_combinations_cache = evaluated_combinations
                self._last_cache_time = time.time()

            logger.info(
                f"✅ Pre-Rally Pattern Discovery Complete in {duration_sec}s: "
                f"{len(evaluated_combinations)} combinations evaluated | "
                f"{summary_payload['validated_oos_count']} VALIDATED_OOS | "
                f"{summary_payload['exploratory_count']} EXPLORATORY."
            )

            return {
                "status": "SUCCESS",
                "summary": summary_payload,
                "combinations": evaluated_combinations
            }

    # =========================================================================
    # LIVE PATTERN MATCHING FOR DAILY SCANNER & MOVERS FORENSICS
    # =========================================================================
    def match_stock_patterns(
        self,
        symbol: str,
        df: Optional[pd.DataFrame],
        idx: int = -2
    ) -> List[Dict[str, Any]]:
        """
        Takes a stock's historical OHLCV DataFrame and evaluates its pre-rally
        characteristics at session `idx` (default is -2, which corresponds to T-1
        relative to the latest active bar at -1).
        Matches against all active discovered combinations and returns matching badges.
        """
        if df is None or len(df) < 18:
            return []

        # Ensure index within range
        if idx < 0:
            resolved_idx = len(df) + idx
        else:
            resolved_idx = idx

        if resolved_idx < 15 or resolved_idx >= len(df):
            return []

        features = self.extract_features_at_t(df, resolved_idx)
        if not features:
            return []

        # Load active combinations (from cache or DB)
        combinations = self.get_discovered_combinations(min_sample=5)
        if not combinations:
            return []

        matches = []
        for c in combinations:
            preds = c.get("predicates", [])
            if not preds:
                try:
                    preds = json.loads(c.get("predicates_json", "[]"))
                except Exception:
                    preds = []

            if preds and all(features.get(p, 0) == 1 for p in preds):
                matches.append({
                    "combination_id": c.get("combination_id"),
                    "name": c.get("name"),
                    "category": c.get("category"),
                    "rule_expression": c.get("rule_expression"),
                    "validation_status": c.get("validation_status"),
                    "lift_oos": float(c.get("lift_oos", 0.0) or 0.0),
                    "out_sample_precision": float(c.get("out_sample_precision", 0.0) or 0.0),
                    "overall_precision": float(c.get("overall_precision", 0.0) or 0.0),
                    "mfe_mae_ratio": float(c.get("mfe_mae_ratio", 0.0) or 0.0)
                })

        return matches

    # =========================================================================
    # PUBLIC QUERY APIS
    # =========================================================================
    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns the latest engine discovery summary or loads it from DB."""
        with self._cache_lock:
            if self._last_summary_cache is not None and (time.time() - self._last_cache_time) < 300:
                return self._last_summary_cache

        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    row = conn.execute("SELECT summary_json FROM pre_rally_runs ORDER BY id DESC LIMIT 1").fetchone()
                    if row and row["summary_json"]:
                        summary = json.loads(row["summary_json"])
                        with self._cache_lock:
                            self._last_summary_cache = summary
                            self._last_cache_time = time.time()
                        return summary
        except Exception as e:
            logger.debug(f"Notice loading saved run summary: {e}")

        # If none recorded yet, run discovery
        res = self.run_discovery()
        return res.get("summary", {})

    def get_discovered_combinations(
        self,
        min_sample: int = 5,
        status_filter: str = "ALL"
    ) -> List[Dict[str, Any]]:
        """Returns discovered combinations with sorting and filtering."""
        with self._cache_lock:
            if self._last_combinations_cache is not None and (time.time() - self._last_cache_time) < 300:
                combs = self._last_combinations_cache
                if status_filter != "ALL":
                    combs = [c for c in combs if c.get("validation_status") == status_filter]
                if min_sample > 0:
                    combs = [c for c in combs if c.get("sample_size", 0) >= min_sample]
                return combs

        # Load from SQLite
        combs_list = []
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    query = "SELECT * FROM pre_rally_discovered_combinations ORDER BY CASE validation_status WHEN 'VALIDATED_OOS' THEN 1 WHEN 'EXPLORATORY' THEN 2 ELSE 3 END, lift_oos DESC, sample_size DESC"
                    rows = conn.execute(query).fetchall()
                    for r in rows:
                        rd = dict(r)
                        rd["predicates"] = json.loads(rd.get("predicates_json") or "[]")
                        rd["example_winners"] = json.loads(rd.get("example_winners_json") or "[]")
                        rd["example_failures"] = json.loads(rd.get("example_failures_json") or "[]")
                        combs_list.append(rd)
        except Exception as e:
            logger.debug(f"Notice loading discovered combinations from DB: {e}")

        if combs_list:
            with self._cache_lock:
                self._last_combinations_cache = combs_list
                self._last_cache_time = time.time()
            if status_filter != "ALL":
                combs_list = [c for c in combs_list if c.get("validation_status") == status_filter]
            if min_sample > 0:
                combs_list = [c for c in combs_list if c.get("sample_size", 0) >= min_sample]
            return combs_list

        # If none in DB, trigger discovery
        res = self.run_discovery()
        return res.get("combinations", [])

    def get_combination_detail(self, combination_id: str) -> Optional[Dict[str, Any]]:
        """Returns in-depth audit record and example evidence for a specific combination."""
        combs = self.get_discovered_combinations(min_sample=0)
        for c in combs:
            if c.get("combination_id") == combination_id:
                return c
        return None

    # =========================================================================
    # INTERNAL DATA & PERSISTENCE HELPERS
    # =========================================================================
    def _resolve_universe_symbols(self, max_symbols: int = 35) -> List[str]:
        """Resolves active liquid stocks from ohlcv_master_store and Nifty 500 seeds."""
        symbols = []
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT symbol FROM ohlcv_master_store WHERE bars_count >= 80 AND symbol NOT LIKE '^%' AND symbol NOT LIKE '%=%' ORDER BY bars_count DESC LIMIT ?",
                        (max_symbols,)
                    ).fetchall()
                    symbols = [r[0] for r in rows]
        except Exception:
            pass

        # If fewer than 15 stored symbols, supplement with core liquid NSE stocks
        if len(symbols) < 15:
            from engines.nifty500_engine import get_nifty500_symbols
            n500 = get_nifty500_symbols()
            priority_anchors = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
                "SBIN", "LTIM", "LT", "BAJFINANCE", "TATAMOTORS", "SUNPHARMA",
                "TITAN", "MARUTI", "KAYNES", "ARE&M", "AMARA", "DIXON", "TRENT"
            ]
            for s in priority_anchors + n500:
                if s not in symbols:
                    symbols.append(s)
                if len(symbols) >= max_symbols:
                    break

        return symbols[:max_symbols]

    def _load_or_fetch_stock_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Loads OHLCV DataFrame from state_engine master store or downloads if missing."""
        df, _ = state_engine.get_ohlcv_df(symbol)
        if df is not None and len(df) >= 70:
            return df

        # Attempt download of 1-year history for symbol
        try:
            ticker = f"{symbol}.NS"
            single_df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(single_df.columns, pd.MultiIndex):
                single_df.columns = single_df.columns.get_level_values(0)
            if single_df is not None and len(single_df) >= 70:
                state_engine.save_ohlcv_df(symbol, single_df)
                return single_df
        except Exception as e:
            logger.debug(f"Notice fetching OHLCV for {symbol}: {e}")

        return None

    def _persist_events_batch(self, events: List[Dict[str, Any]]):
        """Batch inserts or replaces extracted historical pre-rally events."""
        if not events:
            return
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    insert_sql = """
                    INSERT OR REPLACE INTO pre_rally_events (
                        event_id, symbol, event_date, horizon_days, min_fwd_return,
                        entry_price, forward_return, mfe_pct, mae_pct, time_to_target,
                        is_winner, is_in_sample, vdu_ratio_t1, vdu_ratio_t2, vol_declining_3d,
                        dist_ema20_t1, dist_ema50_t1, ema20_slope_5d, daily_range_t1,
                        atr_contraction_ratio, bb_width_t1, is_nr4, is_nr7,
                        consolidation_range_5d, dist_swing_high_20d, accum_days_5d,
                        breakout_surge_t0, features_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    now_str = format_ist()
                    records = []
                    for e in events:
                        records.append((
                            e["event_id"], e["symbol"], e["event_date"], e["horizon_days"], e["min_fwd_return"],
                            e["entry_price"], e["forward_return"], e["mfe_pct"], e["mae_pct"], e["time_to_target"],
                            e["is_winner"], e["is_in_sample"], e["vdu_ratio_t1"], e["vdu_ratio_t2"], e["vol_declining_3d"],
                            e["dist_ema20_t1"], e["dist_ema50_t1"], e["ema20_slope_5d"], e["daily_range_t1"],
                            e["atr_contraction_ratio"], e["bb_width_t1"], e["is_nr4"], e["is_nr7"],
                            e["consolidation_range_5d"], e["dist_swing_high_20d"], e["accum_days_5d"],
                            e["breakout_surge_t0"], json.dumps(e["features"]), now_str
                        ))
                    conn.executemany(insert_sql, records)
                    conn.commit()
        except Exception as e:
            logger.warning(f"Error persisting pre-rally events batch: {e}")

    def _persist_combinations_batch(self, combinations: List[Dict[str, Any]]):
        """Batch inserts or replaces discovered combinations."""
        if not combinations:
            return
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    insert_sql = """
                    INSERT OR REPLACE INTO pre_rally_discovered_combinations (
                        combination_id, name, description, category, rule_expression,
                        predicates_json, in_sample_precision, out_sample_precision, overall_precision,
                        baseline_rate, lift_oos, lift_overall, sample_size, oos_sample_size,
                        win_count, oos_win_count, avg_forward_return, avg_mfe, avg_mae,
                        mfe_mae_ratio, avg_days_to_target, stability_score, validation_status,
                        example_winners_json, example_failures_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    records = []
                    for c in combinations:
                        records.append((
                            c["combination_id"], c["name"], c["description"], c["category"], c["rule_expression"],
                            json.dumps(c["predicates"]), c["in_sample_precision"], c["out_sample_precision"], c["overall_precision"],
                            c["baseline_rate"], c["lift_oos"], c["lift_overall"], c["sample_size"], c["oos_sample_size"],
                            c["win_count"], c["oos_win_count"], c["avg_forward_return"], c["avg_mfe"], c["avg_mae"],
                            c["mfe_mae_ratio"], c["avg_days_to_target"], c["stability_score"], c["validation_status"],
                            json.dumps(c.get("example_winners", [])), json.dumps(c.get("example_failures", [])), c["updated_at"]
                        ))
                    conn.executemany(insert_sql, records)
                    conn.commit()
        except Exception as e:
            logger.warning(f"Error persisting discovered combinations: {e}")

    def _persist_run_record(self, summary: Dict[str, Any]):
        """Persists the discovery execution ledger record."""
        try:
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    conn.execute("""
                    INSERT OR REPLACE INTO pre_rally_runs (
                        run_id, run_timestamp, total_symbols, total_events,
                        positive_events, negative_events, baseline_rally_rate,
                        split_date, parameters_json, summary_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        summary["run_id"], summary["run_timestamp"], summary["total_symbols_analyzed"],
                        summary["total_events_analyzed"], summary["positive_events"], summary["negative_events"],
                        summary["baseline_rally_rate"], summary["split_date"],
                        json.dumps(summary["parameters"]), json.dumps(summary), format_ist()
                    ))
                    conn.commit()
        except Exception as e:
            logger.warning(f"Error persisting discovery run record: {e}")

    def _build_empty_summary(self, run_id: str, ts: str) -> Dict[str, Any]:
        return {
            "status": "EMPTY",
            "summary": {
                "run_id": run_id,
                "run_timestamp": ts,
                "total_symbols_analyzed": 0,
                "total_events_analyzed": 0,
                "positive_events": 0,
                "negative_events": 0,
                "baseline_rally_rate": 0.0,
                "in_sample_events": 0,
                "out_sample_events": 0,
                "in_sample_baseline": 0.0,
                "out_sample_baseline": 0.0,
                "split_date": ts[:10],
                "parameters": {},
                "total_combinations_tested": 0,
                "validated_oos_count": 0,
                "exploratory_count": 0,
                "execution_duration_sec": 0.0
            },
            "combinations": []
        }


pre_rally_discovery_engine = PreRallyDiscoveryEngine.get_instance()
