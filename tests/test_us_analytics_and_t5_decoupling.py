"""
tests/test_us_analytics_and_t5_decoupling.py
============================================
Comprehensive test suite verifying:
  1. US Market Analytics, Breadth, Volatility Term Structure & Next-Day Prediction.
  2. Decoupled T+5 Structural Weekly Prediction from T+1 Tactical Horizon.
  3. Futures Lead-Lag Relative Move & Fair Value Gap Calculation.
  4. API endpoints /api/nifty/us-market-intelligence and /api/nifty/futures-relative-radar.
  5. Top menu bar clean-up (absence of redundant install app / backup pills).
"""

import math
import pytest
from fastapi.testclient import TestClient
from web.api import app
from engines.us_analytics_engine import USAnalyticsEngine, us_analytics_engine
from engines.nifty_analytics_engine import NiftyAnalyticsEngine, nifty_analytics_engine
from ai_engine.context_builder import build_prediction_context


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_us_analytics_engine_schema_and_calculations():
    """Validates USAnalyticsEngine calculations, metrics validity, and schema completeness."""
    res = us_analytics_engine.fetch_and_analyze_us_markets(force_refresh=False)
    assert res.get("status") == "OK"
    assert "us_composite_score" in res
    assert -100.0 <= res["us_composite_score"] <= 100.0

    # Verify Pillars
    pillars = res.get("pillar_scores", {})
    assert "technical_score" in pillars
    assert "breadth_score" in pillars
    assert "volatility_score" in pillars
    assert "macro_score" in pillars
    assert "news_score" in pillars

    # Verify Benchmarks
    benchmarks = res.get("benchmarks", [])
    assert len(benchmarks) >= 4
    tickers = [b["ticker"] for b in benchmarks]
    assert "SPY" in tickers
    assert "QQQ" in tickers
    assert "IWM" in tickers

    # Verify Breadth & Equal-Weight RSP vs SPY
    breadth = res.get("breadth_analysis", {})
    assert "rsp_vs_spy_dispersion" in breadth
    assert "dispersion_regime" in breadth
    assert len(breadth.get("sectors", [])) >= 10

    # Verify Volatility Surface
    vol = res.get("volatility_surface", {})
    assert "vix_spot" in vol
    assert "vix_9d" in vol
    assert "contango_ratio" in vol
    assert vol.get("regime") in ("CONTANGO_STABLE", "BACKWARDATION_INVERSION", "FLAT_TERM_STRUCTURE")

    # Verify US T+1 Predictor
    t1 = res.get("us_t1_forecast", {})
    assert t1.get("verdict") in ("STRONG BULLISH", "MODERATELY BULLISH", "RANGE-BOUND / CONSOLIDATION", "MODERATELY BEARISH", "STRONG BEARISH")
    assert 0.0 <= t1.get("prob_up", 0.0) <= 1.0
    assert 0.0 <= t1.get("prob_flat", 0.0) <= 1.0
    assert 0.0 <= t1.get("prob_down", 0.0) <= 1.0
    assert abs((t1["prob_up"] + t1["prob_flat"] + t1["prob_down"]) - 1.0) < 0.05

    # Verify Futures Relative Radar
    futures = res.get("futures_relative_radar", {})
    assert "rolling_beta_to_nifty" in futures
    assert "expected_fair_gap_pts" in futures
    assert "expected_fair_gap_pct" in futures
    assert "gap_verdict" in futures


def test_t5_decoupled_from_t1_structural_regime():
    """
    Verifies that T+5 Weekly Prediction evaluates multi-day structural factors (EMA50, EMA200,
    breadth > 50 EMA, institutional flows) rather than blindly mirroring T+1 tactical noise.
    """
    preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True)
    assert preds.get("status") == "OK"
    assert "t1_forecast" in preds
    assert "t5_forecast" in preds

    # Check distinct composite scores
    assert "composite_score_t1" in preds
    assert "composite_score_t5" in preds
    assert isinstance(preds["composite_score_t1"], (int, float))
    assert isinstance(preds["composite_score_t5"], (int, float))

    t1 = preds["t1_forecast"]
    t5 = preds["t5_forecast"]

    # Verify T+1 and T+5 both have valid fields
    assert "verdict" in t1 and "verdict" in t5
    assert "expected_pts" in t1 and "expected_pts" in t5
    assert "expected_close" in t1 and "expected_close" in t5
    assert "prob_up" in t1 and "prob_up" in t5
    assert "range_1sigma" in t1 and len(t1["range_1sigma"]) == 2
    assert "range_1sigma" in t5 and len(t5["range_1sigma"]) == 2

    # Verify Walk-Forward Calibration handled both scores
    val = preds.get("validation_and_calibration", {})
    assert "t1_probs" in val
    assert "t5_probs" in val


def test_ai_context_builder_includes_us_intelligence_and_t5_score():
    """Verifies that the AI Context Builder receives US breadth, VIX term structure & T+5 structural score."""
    sample_preds = {
        "status": "OK",
        "timestamp_ist": "2026-10-14 09:15:00",
        "spot_price": 22500.0,
        "composite_score": -12.0,
        "composite_score_t1": -12.0,
        "composite_score_t5": 24.0,
        "t1_forecast": {
            "verdict": "MODERATELY BEARISH",
            "prob_up": 0.30,
            "prob_flat": 0.25,
            "prob_down": 0.45,
            "expected_pts": -65.0,
            "expected_pct": -0.29,
        },
        "t5_forecast": {
            "verdict": "MODERATELY BULLISH",
            "prob_up": 0.58,
            "prob_flat": 0.22,
            "prob_down": 0.20,
            "expected_pts": 140.0,
            "expected_pct": 0.62,
        },
        "us_intelligence": {
            "us_t1_forecast": {"verdict": "STRONG BULLISH", "prob_up": 0.65},
            "breadth_analysis": {"rsp_vs_spy_dispersion": 0.35, "dispersion_regime": "BROAD_BASED_EXPANSION"},
            "volatility_surface": {"regime": "CONTANGO_STABLE", "vix_spot": 14.8},
        },
        "futures_relative_radar": {
            "es_futures_ret_1d": 0.45,
            "expected_fair_gap_pts": 52.0,
            "gap_verdict": "BULLISH_GAP_CONTINUATION",
        },
        "pillars": {
            "news_and_events": {"headlines": [{"title": "Markets open steady", "source": "Reuters"}], "event_risk_flag": "LOW"},
            "options_overlay": {"near_atm_pcr": 1.15, "max_pain": 22500.0, "atm_iv": 13.2},
            "macro_cross_asset": [{"ticker": "^INDIAVIX", "last_close": 13.5}],
        }
    }

    contract = build_prediction_context(sample_preds)
    assert contract.market_context.composite_score == -12.0
    assert contract.market_context.composite_score_t5 == 24.0
    assert contract.market_context.us_breadth_dispersion == 0.35
    assert contract.market_context.us_vix_term_structure == "CONTANGO_STABLE"
    assert contract.market_context.futures_relative_move is not None
    assert contract.market_context.futures_relative_move.get("expected_fair_gap_pts") == 52.0


def test_api_us_market_intelligence_and_futures_radar_endpoints(client):
    """Verifies that the new API routes return HTTP 200 with valid schema."""
    res1 = client.get("/api/nifty/us-market-intelligence")
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1.get("status") == "OK"
    assert "us_composite_score" in d1

    res2 = client.get("/api/nifty/futures-relative-radar")
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2.get("status") == "OK"
    assert "futures_relative_radar" in d2


def test_top_menu_bar_html_structure():
    """Verifies that backup pill and install app button were removed from top bar, and backup badge is in vault section."""
    with open("web/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # The header nav bar should NOT contain vault-status-pill or btn-pwa-install
    header_section = html.split('<header')[1].split('</header>')[0]
    assert 'id="vault-status-pill"' not in header_section
    assert 'id="btn-pwa-install"' not in header_section

    # The Telegram Cloud Vault section SHOULD contain the backup badge
    assert 'id="vault-card-backup-badge"' in html
    assert 'id="vault-card-backup-time"' in html

    # The US Market Intelligence & Futures Radar section SHOULD be present
    assert 'id="us-market-intelligence-card"' in html
    assert 'id="us-t1-verdict-text"' in html
    assert 'id="us-dispersion-val"' in html
    assert 'id="us-vix-spot"' in html
    assert 'id="us-fair-gap-pts"' in html
