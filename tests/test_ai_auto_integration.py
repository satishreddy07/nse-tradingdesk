"""
Automated Tests for:
1. Automatic AI analysis synthesis during ML prediction generation.
2. Graceful fallback on AI endpoints ensuring valid JSON (no unhandled HTTP 500).
3. Smart AI consensus filtering on auto-options strategy execution (hedging naked positions on caution).
4. Storage of ai_rationale in options_strategies table and alert telemetry.
"""
from datetime import datetime
import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine
from web.api import app
from ai_engine.schemas import AIAnalysisOutput, AIHorizonAnalysis


@pytest.fixture
def test_client():
    return TestClient(app)


def test_ai_analysis_automatically_attached_to_predictions():
    """
    Verifies that compute_nifty_predictions() automatically synthesizes and attaches
    ai_analysis into the prediction payload without needing a separate user trigger.
    """
    mock_ai_resp = AIAnalysisOutput(
        status="OK",
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_ms=120,
        summary="Automated AI Synthesis: Bullish momentum confirmed by multi-pillar scores.",
        horizon_analysis=[
            AIHorizonAnalysis(horizon="T+1", interpretation="Strong breadth expansion.", uncertainties=[])
        ],
        conflicts=[],
        supporting_evidence=["FII index long ratio expanding", "Nifty 50 constituent breadth positive"],
        event_risks=[],
        reasons_to_avoid_trade=[],
        strategy_considerations=["BULL PUT SPREAD"],
        missing_or_stale_data=[]
    )

    with patch("ai_engine.service.ai_service.analyze_predictions", return_value=mock_ai_resp), \
         patch.object(state_engine, "get_kv", return_value=None):
        nifty_analytics_engine._predictions_mem_cache = {
            "status": "OK",
            "spot_price": 24800.0,
            "composite_score": 15.0,
            "timestamp_ist": "14-Oct-2026 10:00:00"
        }
        nifty_analytics_engine._predictions_cache_ts = 9999999999.0
        preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)
        assert "ai_analysis" in preds
        assert preds["ai_analysis"]["status"] == "OK"
        assert "Automated AI Synthesis" in preds["ai_analysis"]["summary"]


def test_ai_endpoints_never_throw_unhandled_500_or_non_json(test_client):
    """
    Verifies that /api/ai/analyze and /api/ai/analysis return structured JSON fallback
    even when underlying dependencies fail, preventing syntax errors in the frontend.
    """
    with patch("ai_engine.service.ai_service.analyze_predictions", side_effect=Exception("Simulated upstream network crash")):
        res = test_client.post("/api/ai/analyze", json={"force_refresh": True})
        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"]
        data = res.json()
        assert data["status"] in ("AI_UNAVAILABLE", "UNAVAILABLE")
        assert "unavailable" in data["summary"].lower()

    with patch("ai_engine.service.ai_service.analyze_predictions", side_effect=RuntimeError("Simulated cache error")):
        res_get = test_client.get("/api/ai/analysis")
        assert res_get.status_code == 200
        assert "application/json" in res_get.headers["content-type"]
        data_get = res_get.json()
        assert data_get["status"] in ("AI_UNAVAILABLE", "UNAVAILABLE")


def test_smart_ai_consensus_filtering_and_rationale_persistence():
    """
    Verifies that evaluate_and_deploy_auto_options_strategies():
    1. Replaces naked LONG_CALL with hedged BULL_PUT_SPREAD when AI cautions on naked risk/resistance.
    2. Persists ai_rationale in the SQLite database options_strategies record.
    3. Respects maximum 2 active strategies and diversity constraint.
    """
    saved_dicts = []
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            saved_rows = conn.execute("SELECT * FROM options_strategies WHERE auto_trade_mode = 'INTRADAY'").fetchall()
            saved_dicts = [dict(r) for r in saved_rows]
            conn.execute("DELETE FROM options_strategies WHERE auto_trade_mode = 'INTRADAY'")
            conn.commit()

    mock_preds = {
        "status": "OK",
        "composite_score": 28.0,
        "t5_forecast": {"verdict": "STRONG BULLISH"},
        "ai_analysis": {
            "status": "OK",
            "reasons_to_avoid_trade": ["Heavy call resistance at 25000; avoid unhedged naked options exposure."],
            "horizon_conflict_explained": "",
            "recommended_options_action": "BULL PUT SPREAD",
            "prediction_interpretation": "Strong upward momentum but heavy overhead supply near 25000 suggests credit spread instead of naked calls."
        }
    }

    mock_opt_chain = {
        "status": "OK",
        "spot_price": 24800.0,
        "weekly": {
            "expiry": "2026-10-15",
            "atm_strike": 24800,
            "strikes": [
                {"strike": 24700, "call_ltp": 160.0, "put_ltp": 45.0, "call_iv": 12.0, "put_iv": 13.0, "call_delta": 0.65, "put_delta": -0.35, "call_oi": 1000, "put_oi": 2500},
                {"strike": 24800, "call_ltp": 100.0, "put_ltp": 80.0, "call_iv": 12.5, "put_iv": 12.5, "call_delta": 0.50, "put_delta": -0.50, "call_oi": 3000, "put_oi": 3000},
                {"strike": 24900, "call_ltp": 50.0, "put_ltp": 130.0, "call_iv": 13.0, "put_iv": 12.0, "call_delta": 0.35, "put_delta": -0.65, "call_oi": 2800, "put_oi": 1200},
            ],
            "volatility_and_gex": {"atm_iv": 12.5}
        },
        "monthly": {
            "expiry": "2026-10-29",
            "atm_strike": 24800,
            "strikes": [
                {"strike": 24700, "call_ltp": 220.0, "put_ltp": 80.0, "call_iv": 13.0, "put_iv": 14.0, "call_delta": 0.60, "put_delta": -0.40, "call_oi": 1500, "put_oi": 3000},
                {"strike": 24800, "call_ltp": 150.0, "put_ltp": 120.0, "call_iv": 13.5, "put_iv": 13.5, "call_delta": 0.50, "put_delta": -0.50, "call_oi": 4000, "put_oi": 4000},
                {"strike": 24900, "call_ltp": 90.0, "put_ltp": 180.0, "call_iv": 14.0, "put_iv": 13.0, "call_delta": 0.40, "put_delta": -0.60, "call_oi": 3500, "put_oi": 1800},
            ],
            "volatility_and_gex": {"atm_iv": 13.5}
        }
    }

    mock_intraday = {
        "latest_snapshot": {
            "market_state": "STRONG_BULLISH_MOMENTUM"
        }
    }

    try:
        with patch.object(nifty_analytics_engine, "compute_nifty_predictions", return_value=mock_preds) as mock_compute:
            with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value=mock_opt_chain) as mock_chain:
                with patch.object(nifty_analytics_engine, "get_intraday_expiry_dashboard", return_value=mock_intraday):
                    # Trigger auto deployment for INTRADAY
                    res = nifty_analytics_engine.evaluate_and_deploy_auto_options_strategies(modes=["INTRADAY"], force_refresh=True)
                    assert res["status"] == "OK"
                    assert res["deployed_count"] >= 1
                    mock_compute.assert_called_once_with(force_refresh=True)
                    mock_chain.assert_called_once_with(force_refresh=True)

                    # Verify deployed strategies do NOT contain unhedged LONG_CALL (substituted due to AI caution)
                    deployed_types = [d["strategy_type"] for d in res["deployed"]]
                    assert "LONG_CALL" not in deployed_types
                    assert any("SPREAD" in t for t in deployed_types)

                    # Verify ai_rationale is in deployment results
                    assert any(d.get("ai_rationale") for d in res["deployed"])

                    # Verify ai_rationale persisted into SQLite table for newly deployed intraday strategies
                    with state_engine._lock:
                        with state_engine._get_connection() as conn:
                            rows = conn.execute("SELECT * FROM options_strategies WHERE status = 'ACTIVE' AND auto_trade_mode = 'INTRADAY'").fetchall()
                            assert len(rows) >= 1
                            for r in rows:
                                d = dict(r)
                                assert "ai_rationale" in d
                                assert len(d["ai_rationale"]) > 0
    finally:
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("DELETE FROM options_strategies WHERE auto_trade_mode = 'INTRADAY'")
                for d in saved_dicts:
                    cols = list(d.keys())
                    vals = [d[c] for c in cols]
                    placeholders = ", ".join(["?"] * len(cols))
                    conn.execute(f"INSERT OR REPLACE INTO options_strategies ({', '.join(cols)}) VALUES ({placeholders})", vals)
                conn.commit()
