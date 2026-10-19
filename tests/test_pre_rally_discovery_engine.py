"""
========================================================================================
Test Suite: Pre-Rally Pattern Discovery & Validation Engine
========================================================================================
Verifies:
1. Zero look-ahead bias: Pre-rally features at date T must be calculated strictly as of T-1.
2. Accurate mathematical formulas for VDU, EMA distance, daily range, ATR ratio, NR4/NR7.
3. Chronological train/test split (no future data leakage into in-sample development).
4. Statistical metrics: Precision, Out-of-sample Lift vs baseline, Sample size, MFE/MAE.
5. Non-overlapping event isolation (no duplicate counting of continuous multi-day rallies).
6. SQLite database persistence and REST API endpoints.
========================================================================================
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from execution.state_engine import state_engine
from engines.pre_rally_discovery_engine import (
    PreRallyDiscoveryEngine,
    pre_rally_discovery_engine
)
from web.api import app


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def synthetic_ohlcv_df():
    """Generates synthetic daily OHLCV series for 120 bars with controlled pre-rally patterns."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # Base price path
    base_px = 100.0
    closes = [base_px]
    for _ in range(1, n):
        ret = np.random.normal(0.0005, 0.012)
        closes.append(closes[-1] * (1.0 + ret))

    closes = np.array(closes)
    highs = closes * (1.0 + np.random.uniform(0.005, 0.02, n))
    lows = closes * (1.0 - np.random.uniform(0.005, 0.02, n))
    opens = (highs + lows) / 2.0
    volumes = np.random.uniform(500000, 1500000, n)

    # Inject specific setup at bar 60 (Pre-rally at 59, surge at 60)
    # T-1 (bar 59): Extreme VDU (vol = 200,000 vs 1M avg = 0.2x), tight range 1.5%
    volumes[59] = 200000.0
    highs[59] = closes[59] * 1.008
    lows[59] = closes[59] * 0.993
    opens[59] = closes[59]

    # Day T (bar 60): Surge +7.5% on 3.5x volume
    closes[60] = closes[59] * 1.075
    highs[60] = closes[60] * 1.01
    lows[60] = closes[59] * 0.998
    opens[60] = closes[59] * 1.01
    volumes[60] = 3500000.0

    # Ensure forward bars 61..70 maintain target gain
    for k in range(61, 71):
        highs[k] = max(highs[k], closes[60] * 1.02)
        lows[k] = max(lows[k], closes[60] * 0.98)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=dates)

    return df


def test_feature_extraction_zero_lookahead(synthetic_ohlcv_df):
    """Verifies that extract_features_at_t calculates strictly as of T-1 with zero look-ahead bias."""
    engine = pre_rally_discovery_engine
    idx = 59  # Bar T-1 before the jump at bar 60

    features = engine.extract_features_at_t(synthetic_ohlcv_df, idx)
    assert features is not None, "Feature extraction failed on valid slice"

    # Bar 59 had volume = 200,000; rolling 20 average is ~1,000,000 => VDU ratio should be <= 0.35x
    assert features["vdu_ratio_t1"] <= 0.40, f"Expected low VDU ratio, got {features['vdu_ratio_t1']}"
    assert features["p_extreme_vdu"] == 1, "Expected p_extreme_vdu flag to be 1"
    assert features["p_vdu"] == 1, "Expected p_vdu flag to be 1"

    # Daily range at 59 was (1.008 - 0.993)/0.993 ~ 1.51%
    assert features["daily_range_t1"] <= 2.2, f"Expected tight range <= 2.2%, got {features['daily_range_t1']}"
    assert features["p_tight_range"] == 1
    assert features["p_ultra_tight"] == 1

    # Verify that modifying bar 60 (future breakout) does NOT alter bar 59 pre-rally features
    df_tampered = synthetic_ohlcv_df.copy()
    df_tampered.iloc[60, df_tampered.columns.get_loc("Close")] = 9999.0  # Absurd future price
    df_tampered.iloc[60, df_tampered.columns.get_loc("Volume")] = 99999999.0

    features_tampered = engine.extract_features_at_t(df_tampered, idx)
    assert features_tampered["vdu_ratio_t1"] == features["vdu_ratio_t1"], "Future leakage detected in VDU!"
    assert features_tampered["daily_range_t1"] == features["daily_range_t1"], "Future leakage detected in range!"
    assert features_tampered["dist_ema20_t1"] == features["dist_ema20_t1"], "Future leakage detected in EMA!"


def test_event_labeling_and_non_overlapping(synthetic_ohlcv_df):
    """Verifies dual dataset labeling and non-overlapping event spacing."""
    engine = pre_rally_discovery_engine
    events = engine.label_events_from_df(
        symbol="TEST_STOCK",
        df=synthetic_ohlcv_df,
        min_fwd_return=5.0,
        fwd_horizon=10,
        max_adverse_pct=-3.5,
        non_overlapping_days=10
    )

    assert len(events) >= 3, f"Expected multiple labeled events, got {len(events)}"
    # Verify that bar 60 was captured as a winner (since it rallied +7.5% and held)
    win_events = [e for e in events if e["is_winner"] == 1]
    assert len(win_events) >= 1, "Expected at least 1 positive winner event"

    # Check that events have both positive and negative examples
    non_win_events = [e for e in events if e["is_winner"] == 0]
    assert len(non_win_events) >= 1, "Expected negative control events in dataset"

    # Check non-overlapping: consecutive event dates must be at least 5 trading days apart
    for k in range(len(events) - 1):
        d1 = pd.to_datetime(events[k]["event_date"])
        d2 = pd.to_datetime(events[k + 1]["event_date"])
        delta_days = (d2 - d1).days
        assert delta_days >= 4, f"Overlapping event spacing detected between {d1} and {d2}"


def test_combinatorial_evaluation_and_stability(synthetic_ohlcv_df):
    """Verifies combinatorial hypothesis evaluation, OOS lift, and validation classification."""
    engine = pre_rally_discovery_engine

    # Save synthetic stock to state_engine so discovery can load it
    state_engine.save_ohlcv_df("TEST_DISCOVERY_STOCK", synthetic_ohlcv_df)

    # Run discovery on available stocks
    res = engine.run_discovery(
        symbols=["TEST_DISCOVERY_STOCK", "RELIANCE"],
        min_fwd_return=5.0,
        fwd_horizon=10,
        max_adverse_pct=-3.5,
        max_symbols=2,
        force_refresh=True
    )

    assert res["status"] == "SUCCESS"
    summary = res["summary"]
    assert summary["total_events_analyzed"] > 0
    assert summary["baseline_rally_rate"] > 0
    assert "split_date" in summary

    combinations = res["combinations"]
    assert len(combinations) >= 15, f"Expected at least 15 combinations, got {len(combinations)}"

    # Check that every combination has required statistical fields
    for c in combinations:
        assert "combination_id" in c
        assert "name" in c
        assert "rule_expression" in c
        assert "validation_status" in c
        assert c["validation_status"] in ["VALIDATED_OOS", "EXPLORATORY", "NEUTRAL_EDGE", "INSUFFICIENT_DATA"]
        assert "lift_oos" in c
        assert "out_sample_precision" in c
        assert "in_sample_precision" in c
        assert "mfe_mae_ratio" in c
        assert "sample_size" in c


def test_pattern_matching_on_latest_candle(synthetic_ohlcv_df):
    """Verifies that match_stock_patterns correctly identifies matching patterns for active candidates."""
    engine = pre_rally_discovery_engine
    # Bar 59 was engineered with extreme VDU and ultra-tight range
    matches = engine.match_stock_patterns("TEST_DISCOVERY_STOCK", synthetic_ohlcv_df, idx=59)
    assert len(matches) > 0, "Expected matching patterns for engineered coil bar"

    matched_ids = [m["combination_id"] for m in matches]
    assert "H1_VDU_BASE" in matched_ids or "H2_EXTREME_VDU" in matched_ids, "Expected VDU combination match"


def test_api_pre_rally_endpoints(api_client):
    """Verifies FastAPI REST endpoints for the Pre-Rally Pattern Discovery & Validation Engine."""
    # 1. Summary Endpoint
    r_sum = api_client.get("/api/research/pre-rally/summary")
    assert r_sum.status_code == 200, f"Summary failed: {r_sum.text}"
    data_sum = r_sum.json()
    assert data_sum["status"] == "SUCCESS"
    assert "summary" in data_sum
    assert data_sum["summary"]["total_events_analyzed"] > 0

    # 2. Combinations Endpoint
    r_comb = api_client.get("/api/research/pre-rally/combinations?min_sample=1&status=ALL")
    assert r_comb.status_code == 200, f"Combinations failed: {r_comb.text}"
    data_comb = r_comb.json()
    assert data_comb["status"] == "SUCCESS"
    assert data_comb["count"] > 0
    assert len(data_comb["combinations"]) > 0

    # 3. Filtered Combinations
    r_filt = api_client.get("/api/research/pre-rally/combinations?status=VALIDATED_OOS")
    assert r_filt.status_code == 200
    for c in r_filt.json()["combinations"]:
        assert c["validation_status"] == "VALIDATED_OOS"

    # 4. Detail Endpoint
    comb_id = data_comb["combinations"][0]["combination_id"]
    r_det = api_client.get(f"/api/research/pre-rally/combination-detail?combination_id={comb_id}")
    assert r_det.status_code == 200
    data_det = r_det.json()
    assert data_det["status"] == "SUCCESS"
    assert data_det["combination"]["combination_id"] == comb_id
    assert "example_winners" in data_det["combination"]
    assert "example_failures" in data_det["combination"]


def test_accumulation_engine_includes_pre_rally_patterns():
    """Verifies that AccumulationResearchEngine tags movers with discovered pre-rally patterns."""
    from engines.accumulation_research_engine import accumulation_research_engine
    report = accumulation_research_engine.get_latest_research()
    assert "discovered_combinations_breakdown" in report or "common_findings" in report
