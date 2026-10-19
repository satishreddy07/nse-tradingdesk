"""
Unit tests for Institutional Stock Test Analyzer & Diagnostic Lab (`engines/stock_test_analyzer.py`)
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from engines.stock_test_analyzer import stock_test_analyzer
from web.api import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_setup_benchmarks_structure():
    """Verifies standard criteria benchmarks for BUY NOW, BREAKOUT, and PULLBACK."""
    benchmarks = stock_test_analyzer.get_setup_benchmarks()
    assert "BUY_NOW" in benchmarks
    assert "BREAKOUT" in benchmarks
    assert "PULLBACK" in benchmarks

    bn = benchmarks["BUY_NOW"]
    assert "criteria" in bn
    assert len(bn["criteria"]) >= 10
    fields = [c["field"] for c in bn["criteria"]]
    assert "Market Capitalization" in fields
    assert "50D Daily Turnover" in fields
    assert "Trend Alignment" in fields
    assert "Pre-Breakout VDU (T-1)" in fields

    bo = benchmarks["BREAKOUT"]
    assert len(bo["criteria"]) >= 5

    pb = benchmarks["PULLBACK"]
    assert len(pb["criteria"]) >= 5


def test_search_nse_stocks():
    """Verifies symbol search returns results and handles queries."""
    res = stock_test_analyzer.search_nse_stocks("TCS")
    assert isinstance(res, list)
    assert len(res) >= 1
    found_tcs = any(r["symbol"] == "TCS" for r in res)
    assert found_tcs

    # Search for arbitrary ticker
    res_arb = stock_test_analyzer.search_nse_stocks("ARBITRARY")
    assert len(res_arb) >= 1
    assert res_arb[0]["symbol"] == "ARBITRARY"


def test_analyze_stock_known_symbol():
    """Verifies complete diagnostic analysis on a known stock."""
    report = stock_test_analyzer.analyze_stock("RELIANCE")
    assert report["status"] == "SUCCESS"
    assert report["symbol"] == "RELIANCE"
    assert report["current_price"] > 0
    assert "overall_score" in report
    assert "buy_now_verdict" in report
    assert "breakout_verdict" in report
    assert "pullback_verdict" in report
    assert "archetype" in report
    assert "trade_plan" in report
    assert "candles" in report
    assert len(report["candles"]) > 0
    assert "fundamental_source_url" in report
    assert "https://www.screener.in/company/RELIANCE/consolidated/" == report["fundamental_source_url"]


def test_api_endpoints(api_client):
    """Verifies FastAPI endpoints for test mode."""
    # 1. Benchmarks
    r1 = api_client.get("/api/test-mode/benchmarks")
    assert r1.status_code == 200
    b_data = r1.json()
    assert "BUY_NOW" in b_data
    assert "BREAKOUT" in b_data
    assert "PULLBACK" in b_data

    # 2. Search
    r2 = api_client.get("/api/test-mode/search?q=INFY")
    assert r2.status_code == 200
    s_data = r2.json()
    assert s_data["status"] == "SUCCESS"
    assert len(s_data["results"]) >= 1

    # 3. Analyze
    r3 = api_client.get("/api/test-mode/analyze/RELIANCE")
    assert r3.status_code == 200
    a_data = r3.json()
    assert a_data["status"] == "SUCCESS"
    assert a_data["symbol"] == "RELIANCE"
    assert "buy_now_audit" in a_data
    assert "breakout_audit" in a_data
    assert "pullback_audit" in a_data


def test_tatamotors_alias_resolution():
    """Verifies that TATAMOTORS resolves to demerged entities (TMCV / TMPV) with genuine data."""
    report = stock_test_analyzer.analyze_stock("TATAMOTORS")
    assert report["status"] == "SUCCESS"
    assert report["symbol"] == "TATAMOTORS"
    assert report["resolved_symbol"] in ["TMCV", "TMPV", "TATAMOTORS"]
    assert report["current_price"] > 0
    assert len(report["candles"]) > 0
    assert "trade_plan" in report
    assert report["trade_plan"]["entry_price"] > 0
    assert report["fundamentals"]["market_cap_cr"] is not None
    assert report["fundamentals"]["market_cap_cr"] > 1000.0  # Tata Motors > ₹1,000 Cr
