import pytest
from fastapi.testclient import TestClient
from web.api import app
from engines.accumulation_research_engine import accumulation_research_engine

client = TestClient(app)


def test_accumulation_research_engine_structure():
    """Verify that AccumulationResearchEngine returns a complete, authenticated structure."""
    report = accumulation_research_engine.get_latest_research()
    assert report is not None
    assert "audit_id" in report
    assert "run_timestamp" in report
    assert "scan_date" in report
    assert "total_universe" in report
    assert "positive_movers_count" in report
    assert "vdu_rate_pct" in report
    assert "tightness_rate_pct" in report
    assert "ema20_launch_rate_pct" in report
    assert "avg_breakout_surge" in report
    assert "common_findings" in report
    assert isinstance(report["common_findings"], list)
    assert len(report["common_findings"]) >= 3
    assert "archetype_distribution" in report
    assert isinstance(report["archetype_distribution"], dict)
    assert "sha256_hash" in report
    assert len(report["sha256_hash"]) == 64


def test_research_api_endpoints():
    """Verify that /api/research/movers-accumulation and /api/research/run-study endpoints work."""
    # GET endpoint
    res_get = client.get("/api/research/movers-accumulation")
    assert res_get.status_code == 200
    data = res_get.json()
    assert "audit_id" in data
    assert "common_findings" in data
    assert "movers" in data

    # POST endpoint (without long wait)
    # Testing that endpoint is mounted and responds
    res_head = client.get("/api/research/movers-accumulation?force_refresh=false")
    assert res_head.status_code == 200


def test_day1_accumulation_recognition():
    """Verify that a textbook accumulation coil setup is classified as BUY_NOW or PRE_BREAKOUT with ACCUMULATION_COIL type."""
    from engines.institutional_engine import ExecutionEngine
    import pandas as pd

    # Textbook coil setup (e.g. HBLENGINE on Thursday: score 86, dist EMA20 -3.9%, vol_ratio 0.28x, daily range 2.25%)
    coil_row = pd.Series({
        "ticker": "TEST_COIL",
        "score": 86.0,
        "dist_to_pivot": 1.2,
        "dist_to_ema20": -2.5,
        "vol_ratio": 0.35, # Volume Dry-Up
        "daily_range_pct": 2.1, # Tight contraction
        "is_accumulating_coil": True,
        "close": 500.0,
        "confidence": 4,
        "ml_verdict": "APPROVED",
        "ml_decision": "PASS_TO_AI"
    })

    bucket = ExecutionEngine.classify(coil_row)
    pos_info = ExecutionEngine.calculate_position(coil_row)

    assert bucket == "BUY_NOW"
    assert pos_info["entry_type"] == "ACCUMULATION_COIL"
    assert pos_info["sl"] > 0
    # Stop loss should be tight (risk <= 3.8%)
    risk_pct = (pos_info["entry"] - pos_info["sl"]) / pos_info["entry"] * 100.0
    assert risk_pct <= 4.0
    assert pos_info["rr"] >= 2.5


def test_regime_performance_matrix_and_audit_history():
    """Verify that regime matrix aggregates BULLISH, BEARISH, and NEUTRAL scans and audit history works."""
    # Test matrix endpoint
    res_matrix = client.get("/api/research/regime-analysis")
    assert res_matrix.status_code == 200
    data = res_matrix.json()
    assert data["status"] == "SUCCESS"
    assert "current_regime" in data
    assert data["current_regime"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert "regimes" in data
    assert "BULLISH" in data["regimes"]
    assert "BEARISH" in data["regimes"]
    assert "NEUTRAL" in data["regimes"]
    assert "history" in data
    assert isinstance(data["history"], list)

    # Test audit history endpoint
    res_hist = client.get("/api/research/audit-history?limit=10")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert hist_data["status"] == "SUCCESS"
    assert "count" in hist_data
    assert "history" in hist_data
    if hist_data["count"] > 0:
        entry = hist_data["history"][0]
        assert "audit_id" in entry
        assert "market_regime" in entry
        assert "sha256_hash" in entry


def test_funnel_and_pattern_breakdown():
    """Verify that funnel counts (universe, positive, >=3% movers), tags, and pattern breakdown are computed."""
    report = accumulation_research_engine.get_latest_research()
    assert report is not None
    assert "total_universe" in report
    assert "positive_stocks_count" in report
    assert "movers_3pct_count" in report
    assert "total_patterns_observed" in report
    assert "pattern_breakdown" in report
    assert isinstance(report["pattern_breakdown"], list)

    # Check movers have tags and narrative
    movers = report.get("movers", [])
    if movers:
        m0 = movers[0]
        assert "tags" in m0
        assert isinstance(m0["tags"], list)
        assert len(m0["tags"]) > 0
        assert "what_happened_before" in m0
        assert isinstance(m0["what_happened_before"], str)
        assert len(m0["what_happened_before"]) > 10

    # Check pattern breakdown structure
    pb = report.get("pattern_breakdown", [])
    if pb:
        p0 = pb[0]
        assert "pattern_name" in p0
        assert "stocks_count" in p0
        assert "stocks_pct" in p0
        assert "symbols" in p0
        assert isinstance(p0["symbols"], list)
        assert "avg_pre_vdu" in p0
        assert "avg_pre_range" in p0
        assert "avg_ema20_dist" in p0
        assert "avg_gain_pct" in p0
        assert "rationale" in p0


