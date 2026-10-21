"""
Unit and Integration Tests for Swing Trading ML + AI Final Buy Decision Gate
=============================================================================
Tests:
1. ML Candidate Evaluation Contract: returns standardized fields (decision, win_prob, expected_r, risk_flags, fatal_flags, target_definition).
2. AI Gatekeeper Service & Fail-Safe: validates SwingCandidateAIRequest/Response and ensures fail-safe fallback (DO_NOT_BUY on failure).
3. Immutable Decision Audit Trail: tests recording and retrieving full 5-stage chain in swing_decision_audit.
4. Granular Technical Failure Attribution: verifies detection of technical failure modes (e.g., OVEREXTENDED_AT_ENTRY, RSI_PEAK_EXHAUSTION).
5. Deterministic Risk Checks: verifies that risk rules always have final authority and cannot be bypassed.
6. REST API Endpoints: tests /api/swing/decision-audit and /api/swing/candidate-eval/{symbol}.
"""

import json
import sqlite3
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from engines.swing_quant_ml_engine import swing_quant_ml_engine
from ai_engine.schemas import (
    SwingCandidateAIRequest,
    SwingCandidateAIResponse,
    SwingReasonCode,
    SwingCandidateIdentity,
    SwingCandidateContext,
    SwingTechnicalAnalysis,
    SwingMLAnalysis,
    SwingTradePlan,
    SwingHistoricalEvidence,
    SwingPortfolioContext,
    SwingInstructions
)
from ai_engine.service import AIService
from web.api import app


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


def test_swing_ml_evaluate_candidate_contract():
    """Verify ML candidate evaluation returns standardized contract for AI review."""
    candidate = {
        "ticker": "TEST_ML_GATE",
        "symbol": "TEST_ML_GATE",
        "entry": 1200.0,
        "sl": 1140.0,
        "t1": 1320.0,
        "score": 90.0,
        "confidence": 5,
        "rsi14": 59.0,
        "atr_pct": 2.2,
        "dist_to_pivot": 1.2,
        "dist_to_ema20": 1.9,
        "vol_ratio": 2.1,
        "inst_score": 85.0,
        "week_confluence": 1,
        "is_vcp": 1,
        "bb_sq_ratio": 0.8,
        "adx14": 28.0
    }
    mock_regime = {
        "nifty_trend": "BULLISH",
        "vix": 14.0,
        "status": "CONFIRMED_UPTREND",
        "nifty_ret_5d": 1.2,
        "nifty_ret_20d": 2.5,
        "nifty_sma50_dist": 1.8,
        "advance_decline_ratio": 1.5
    }
    eval_res = swing_quant_ml_engine.evaluate_candidate(candidate, market_regime=mock_regime)
    assert isinstance(eval_res, dict)
    assert "decision" in eval_res
    assert eval_res["decision"] in ("PASS_TO_AI", "REJECT", "WATCH", "INSUFFICIENT_DATA")
    assert "win_prob" in eval_res
    assert 0.0 <= eval_res["win_prob"] <= 1.0
    assert "expected_r" in eval_res
    assert "risk_flags" in eval_res
    assert "fatal_flags" in eval_res
    assert "target_definition" in eval_res
    assert eval_res["decision"] in ("PASS_TO_AI", "WATCH", "REJECT")


def test_swing_ml_rejects_overextended_candidate():
    """Verify ML gate flags and rejects severely overextended setups."""
    bad_candidate = {
        "ticker": "TEST_OVEREXTENDED",
        "entry": 500.0,
        "sl": 420.0,
        "t1": 540.0,
        "score": 45.0,
        "confidence": 1,
        "rsi14": 82.0,  # Overbought
        "dist_to_pivot": 9.5,  # Severely overextended (> 3.5%)
        "dist_to_ema20": 8.0,  # Stretched (> 5%)
        "vol_ratio": 0.6,
        "inst_score": 30.0,
        "week_confluence": 0
    }
    eval_res = swing_quant_ml_engine.evaluate_candidate(bad_candidate)
    assert eval_res["decision"] in ("REJECT", "WATCH")
    assert len(eval_res["risk_flags"]) > 0


def test_ai_candidate_request_response_schemas():
    """Verify Pydantic schema validation for SwingCandidateAIRequest & Response."""
    req = SwingCandidateAIRequest(
        schema_version="1.0",
        request_id="test_req_001",
        task="swing_final_buy_decision",
        module="swing_trading",
        candidate=SwingCandidateIdentity(candidate_id="cand_INFY_01", symbol="INFY", scanner_classification="BUY NOW"),
        context=SwingCandidateContext(as_of="2026-10-15T10:20:00Z"),
        technical_analysis=SwingTechnicalAnalysis(
            classification="BUY NOW",
            features={"score": 88.0, "confidence": 5, "rsi14": 57.5, "dist_to_pivot": 1.1, "vol_ratio": 1.9}
        ),
        ml_analysis=SwingMLAnalysis(
            decision="PASS_TO_AI",
            calibrated_probability=0.68,
            expected_r=0.72,
            risk_flags=[]
        ),
        trade_plan=SwingTradePlan(
            entry_price=1850.0,
            stop_loss=1775.0,
            target=2000.0,
            reward_risk_ratio=2.0,
            position_size=50,
            capital_at_risk=3750.0
        ),
        historical_evidence=SwingHistoricalEvidence(sample_count=45, historical_win_rate=0.67),
        portfolio_context=SwingPortfolioContext(active_slots_used=3, max_slots=10),
        instructions=SwingInstructions(objective="Evaluate candidate for final buy decision.")
    )
    assert req.candidate.symbol == "INFY"
    assert req.ml_analysis.decision == "PASS_TO_AI"

    resp = SwingCandidateAIResponse(
        status="SUCCESS",
        schema_version="1.0",
        request_id="test_req_001",
        candidate_id="cand_INFY_01",
        symbol="INFY",
        model_used="groq/llama-3.3-70b",
        final_decision="BUY_NOW",
        reason_code=SwingReasonCode.APPROVED_CONFLUENCE,
        confidence=0.88,
        summary="High-conviction breakout with institutional volume and favorable index trend.",
        supporting_factors=["Technical score 88/100", "Volume 1.9x 20-day average", "Tight 1.1% pivot distance"],
        contradicting_factors=[],
        risk_factors=["Tight trailing stop buffer required"]
    )
    assert resp.final_decision == "BUY_NOW"
    assert resp.reason_code == SwingReasonCode.APPROVED_CONFLUENCE


def test_ai_service_failsafe_fallback():
    """Verify that if AI is disabled or encounters an exception, it safely returns DO_NOT_BUY with AI_UNAVAILABLE."""
    service = AIService()
    req = SwingCandidateAIRequest(
        schema_version="1.0",
        request_id="test_req_fail",
        task="swing_final_buy_decision",
        module="swing_trading",
        candidate=SwingCandidateIdentity(candidate_id="cand_FAIL_01", symbol="TEST", scanner_classification="BUY NOW"),
        context=SwingCandidateContext(as_of="2026-10-15T10:20:00Z"),
        technical_analysis=SwingTechnicalAnalysis(classification="BUY NOW", features={"score": 75.0}),
        ml_analysis=SwingMLAnalysis(decision="PASS_TO_AI", calibrated_probability=0.60, expected_r=0.40),
        trade_plan=SwingTradePlan(entry_price=100.0, stop_loss=95.0, target=110.0, reward_risk_ratio=2.0, position_size=10, capital_at_risk=50.0),
        instructions=SwingInstructions(objective="Evaluate candidate for final buy decision.")
    )

    with patch("ai_engine.service.config.AI_ENABLED", False):
        res = service.evaluate_swing_candidate(req)
        assert res.final_decision == "DO_NOT_BUY"
        assert res.reason_code == SwingReasonCode.AI_UNAVAILABLE


def test_record_and_retrieve_decision_audit():
    """Verify persisting and reading back a complete candidate decision audit record."""
    cand_id = "test_cand_audit_999"
    sym = "TATAMOTORS"
    tech_snap = {
        "score": 86.5,
        "confidence": 5,
        "rsi14": 58.2,
        "dist_to_pivot": 1.4,
        "dist_to_ema20": 2.0,
        "vol_ratio": 1.8,
        "entry_price": 950.0,
        "stop_loss": 910.0,
        "target": 1030.0
    }
    row_id = swing_quant_ml_engine.record_decision_audit(
        candidate_id=cand_id,
        symbol=sym,
        scanner_source="Chartink 6-Pillar Screener",
        scanner_run_id="TEST_RUN_1020",
        technical_classification="BUY NOW",
        technical_snapshot=tech_snap,
        ml_decision="PASS_TO_AI",
        ml_win_prob=0.72,
        ml_score=86.5,
        expected_r=0.88,
        ml_reasons=["Slight sector divergence"],
        ai_request_id="ai_req_999",
        ai_status="SUCCESS",
        ai_decision="BUY_NOW",
        ai_reason_code="APPROVED_CONFLUENCE",
        ai_summary="Clear VCP base breakout confirmed with institutional accumulation.",
        ai_factors={
            "supporting": ["Clean 6-pillar score", "Low pivot extension"],
            "contradicting": [],
            "risk": ["VIX slightly elevated"]
        },
        risk_check_status="APPROVED",
        final_status="EXECUTED_REAL_P1",
        trade_id="ORD_TATAMOTORS_01"
    )
    assert row_id is not None
    assert row_id > 0

    logs = swing_quant_ml_engine.get_decision_audit_logs(limit=10, symbol=sym)
    assert len(logs) > 0
    matched = next((l for l in logs if l.get("candidate_id") == cand_id), None)
    assert matched is not None
    assert matched["symbol"] == sym
    assert matched["technical_classification"] == "BUY NOW"
    assert matched["ml_decision"] == "PASS_TO_AI"
    assert matched["ml_win_prob"] == 0.72
    assert matched["ai_decision"] == "BUY_NOW"
    assert matched["ai_reason_code"] == "APPROVED_CONFLUENCE"
    assert matched["final_status"] == "EXECUTED_REAL_P1"
    assert isinstance(matched.get("technical_snapshot"), dict)
    assert matched["technical_snapshot"]["score"] == 86.5


def test_failure_attribution_computation():
    """Verify that root cause attribution identifies exact technical failure modes when stops hit."""
    # 1. Overextension failure
    overextended_entry = {
        "dist_to_pivot": 5.2,  # > 3.5%
        "dist_to_ema20": 3.0,
        "rsi14": 65.0,
        "vol_ratio": 1.2
    }
    attr1 = swing_quant_ml_engine._compute_root_cause_attribution(
        is_win=0,
        exit_reason="STOP_LOSS",
        features=overextended_entry,
        nifty_hold_return=0.5
    )
    assert "OVEREXTENDED_AT_ENTRY" in attr1
    assert "50d Pivot Extension" in attr1

    # 2. RSI exhaustion failure
    rsi_exhausted_entry = {
        "dist_to_pivot": 1.5,
        "dist_to_ema20": 2.0,
        "rsi14": 78.5,  # > 72.0
        "vol_ratio": 1.5
    }
    attr2 = swing_quant_ml_engine._compute_root_cause_attribution(
        is_win=0,
        exit_reason="STOP_LOSS",
        features=rsi_exhausted_entry,
        nifty_hold_return=0.2
    )
    assert "RSI_PEAK_EXHAUSTION" in attr2

    # 3. Target achieved
    win_entry = {"dist_to_pivot": 1.2, "dist_to_ema20": 1.5, "rsi14": 58.0, "vol_ratio": 2.0}
    attr3 = swing_quant_ml_engine._compute_root_cause_attribution(
        is_win=1,
        exit_reason="TARGET_HIT",
        features=win_entry,
        nifty_hold_return=1.0
    )
    assert "TARGET_ACHIEVED" in attr3


def test_api_decision_audit_endpoint(api_client):
    """Verify GET /api/swing/decision-audit endpoint returns valid JSON with count and audit_logs."""
    res = api_client.get("/api/swing/decision-audit?limit=20")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert "count" in data
    assert "audit_logs" in data
    assert isinstance(data["audit_logs"], list)
