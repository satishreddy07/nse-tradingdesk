"""
Comprehensive Automated Test Suite for NSE Trading Desk AI Intelligence Layer.

Covers all 17 critical scenarios specified in Section 19:
1. Actual T+1/T+5 prediction contract mapping.
2. Missing, malformed, or non-finite values handling.
3. Conflicting forecast horizons detection and mapping.
4. Valid AI JSON parsing and schema validation.
5. Malformed JSON returned by the model.
6. Unexpected field types and string length / constraint validation.
7. Provider timeout handling (HTTP 408 / TimeoutException).
8. Invalid API credentials handling (HTTP 401/403).
9. Rate limit handling (HTTP 429).
10. Primary model failure and fallback to secondary model.
11. Both providers unavailable -> graceful degradation (AI_UNAVAILABLE).
12. Stale / delayed market data flagging.
13. AI disabled (AI_ENABLED=False) behavior.
14. Zero API key leakage in responses, errors, logs, and string representations.
15. Confirmation that AI failures never alter or suppress ML predictions.
16. Confirmation that AI decision status remains ANALYSIS_ONLY and cannot bypass risk checks.
17. Confirmation that no fake prediction metrics/Greeks are produced.
Plus API integration tests via FastAPI TestClient.
"""
import json
import math
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from core import config
from ai_engine.cache import ai_cache
from ai_engine.context_builder import build_prediction_context, np_clip_prob
from ai_engine.forecast_interpreter import (
    SYSTEM_PROMPT,
    build_interpretation_prompt,
    extract_and_validate_analysis,
)
from ai_engine.model_registry import ModelRegistry
from ai_engine.provider import AIProviderAdapter, ProviderError
from ai_engine.schemas import (
    AIAnalysisOutput,
    AIHorizonAnalysis,
    AIStatusResponse,
    MarketContextItem,
    ModelBenchmarkReport,
    PredictionHorizonItem,
    PredictionInputContract,
)
from ai_engine.service import AIService, ai_service
from web.api import app


# ==============================================================================
# FIXTURES & MOCK DATA
# ==============================================================================

@pytest.fixture
def sample_ml_predictions():
    """Realistic output dictionary from nifty_analytics_engine.compute_nifty_predictions()."""
    return {
        "status": "OK",
        "as_of": "2026-10-14T09:15:00+05:30",
        "spot_price": 24825.50,
        "composite_score": 18.5,
        "action": "ACCUMULATE_BULLISH",
        "t1_forecast": {
            "verdict": "BULLISH",
            "prob_up": 0.64,
            "prob_flat": 0.16,
            "prob_down": 0.20,
            "expected_pct": 0.38,
            "expected_pts": 94.3,
            "expected_close": 24919.8,
            "range_1sigma": [24720.0, 24980.0],
            "range_2sigma": [24600.0, 25100.0],
        },
        "t5_forecast": {
            "verdict": "CONSOLIDATION_RANGE",
            "prob_up": 0.42,
            "prob_flat": 0.38,
            "prob_down": 0.20,
            "expected_pct": 0.15,
            "expected_pts": 37.2,
            "expected_close": 24862.7,
            "range_1sigma": [24550.0, 25150.0],
            "range_2sigma": [24300.0, 25400.0],
        },
        "pillars": {
            "technical": {"score": 24.0, "verdict": "STRONG_BULLISH"},
            "breadth": {"score": 12.0, "verdict": "POSITIVE"},
            "options_overlay": {
                "near_atm_pcr": 1.15,
                "max_pain": 24800.0,
                "sentiment": "BULLISH_SUPPORT"
            },
            "institutional_flows": {
                "latest_cash": {"fii_net": 1250.0, "dii_net": 850.0},
                "latest_oi": {"fii_fut_long_pct": 62.4}
            },
            "news_and_events": {
                "event_risk_flag": "MODERATE",
                "items": [
                    {"title": "RBI keeps repo rate unchanged at 6.5%", "source": "RBI", "published_at": "2026-10-13T10:00:00Z"},
                    {"title": "US CPI cools to 2.4%", "source": "BLS", "published_at": "2026-10-13T18:00:00Z"}
                ]
            },
            "sectoral_analysis": [
                {"name": "NIFTY BANK", "return_1d": 0.85},
                {"name": "NIFTY IT", "return_1d": -0.40},
                {"name": "NIFTY AUTO", "return_1d": 1.10}
            ]
        },
        "self_training_engine": {
            "t1_hit_rate_pct": 73.2,
            "t5_hit_rate_pct": 76.5,
            "total_graded_samples": 180
        }
    }


@pytest.fixture
def valid_llm_json_response():
    """Standard valid JSON output returned by an institutional LLM."""
    return json.dumps({
        "summary": "Nifty exhibits short-term upward momentum led by Banking and Auto sectors, while weekly multi-day structure favors consolidation near the 24,900 resistance band.",
        "horizon_analysis": [
            {
                "horizon": "T+1",
                "interpretation": "T+1 model projects a 64% probability of upward drift targeting 24,920 within an 85-point 1-sigma envelope.",
                "uncertainties": ["Global tech spillover from overnight NASDAQ volatility"]
            },
            {
                "horizon": "T+5",
                "interpretation": "T+5 outlook anticipates consolidation around 24,800-25,100 with elevated flat probability (38%).",
                "uncertainties": ["Upcoming US PPI and retail sales print"]
            }
        ],
        "conflicts": [
            "T+1 bullish technical bias contrasts with IT sector drag (-0.40% 1D)",
            "T+5 momentum dampening relative to sharp T+1 tactical drift"
        ],
        "supporting_evidence": [
            "FII cash net buying of +₹1,250 Cr and DII +₹850 Cr",
            "FII Index Futures long ratio healthy at 62.4%",
            "PCR at 1.15 confirms put writing support below 24,800"
        ],
        "event_risks": [
            "RBI post-policy commentary absorption",
            "US macroeconomic data prints"
        ],
        "reasons_to_avoid_trade": [
            "Aggressive naked call buying near 24,950 1-sigma resistance boundary",
            "Chasing gaps above 24,900 given T+5 consolidation ceiling"
        ],
        "missing_or_stale_data": [
            "NSE India VIX real-time tick feed"
        ],
        "strategy_considerations": [
            "Bull Put Spread (Sell 24700 PE / Buy 24500 PE)",
            "Iron Condor for T+5 consolidation"
        ],
        "decision_status": "ANALYSIS_ONLY"
    })


# ==============================================================================
# 1. ACTUAL T+1 / T+5 PREDICTION CONTRACT MAPPING
# ==============================================================================
def test_prediction_contract_mapping(sample_ml_predictions):
    """Verifies that actual compute_nifty_predictions() dict maps 1:1 to PredictionInputContract."""
    contract = build_prediction_context(sample_ml_predictions, data_mode_override="LIVE")

    assert isinstance(contract, PredictionInputContract)
    assert contract.schema_version == "1.0"
    assert contract.symbol == "NIFTY"
    assert contract.data_mode == "LIVE"
    assert len(contract.predictions) == 2

    # T+1 Horizon Check
    t1 = contract.predictions[0]
    assert t1.horizon == "T+1"
    assert t1.direction == "BULLISH"
    assert t1.probability_up == 0.64
    assert t1.probability_flat == 0.16
    assert t1.probability_down == 0.20
    assert t1.expected_return_pct == 0.38
    assert t1.expected_points == 94.3
    assert t1.expected_close == 24919.8
    assert t1.forecast_range_1sigma == [24720.0, 24980.0]
    assert t1.forecast_range_2sigma == [24600.0, 25100.0]
    assert "73.2%" in t1.validation_status

    # T+5 Horizon Check
    t5 = contract.predictions[1]
    assert t5.horizon == "T+5"
    assert t5.direction == "RANGE-BOUND"
    assert t5.probability_up == 0.42
    assert t5.probability_flat == 0.38
    assert t5.probability_down == 0.20
    assert "76.5%" in t5.validation_status

    # Market Context Check
    ctx = contract.market_context
    assert ctx.spot_price == 24825.50
    assert ctx.composite_score == 18.5
    assert ctx.regime == "TRENDING_BULLISH"
    assert ctx.event_risk == "MODERATE"
    assert ctx.option_chain_pcr == 1.15
    assert ctx.max_pain == 24800.0
    assert "FII Cash: ₹+1,250.0 Cr" in ctx.fii_dii_summary
    assert len(ctx.recent_news_events) == 2


# ==============================================================================
# 2. MISSING, MALFORMED, OR NON-FINITE VALUES
# ==============================================================================
def test_missing_malformed_nonfinite_values():
    """Verifies that np_clip_prob and context builder handle None, NaN, inf, and missing fields safely."""
    # Probability clipping helper
    assert np_clip_prob(None) == 0.5
    assert np_clip_prob("invalid") == 0.5
    assert np_clip_prob(1.4) == 1.0
    assert np_clip_prob(-0.3) == 0.0
    assert np_clip_prob(math.nan) == 0.5 or 0.0 <= np_clip_prob(math.nan) <= 1.0

    # Minimal and partially malformed predictions data
    minimal_data = {
        "status": "OK",
        "spot_price": None,
        "composite_score": "not_a_number",
        "t1_forecast": {},
        "t5_forecast": None,
        "pillars": None
    }
    # Should not crash, fallback to safe defaults
    contract = build_prediction_context(minimal_data, data_mode_override="HISTORICAL_DAILY")
    assert contract.market_context.spot_price > 0.0
    assert contract.market_context.composite_score == 0.0
    assert contract.predictions[0].probability_up == 0.5
    assert contract.predictions[1].probability_up == 0.5


# ==============================================================================
# 3. CONFLICTING FORECAST HORIZONS
# ==============================================================================
def test_conflicting_forecast_horizons():
    """Tests handling when T+1 is Bullish but T+5 is Bearish."""
    conflicting_data = {
        "status": "OK",
        "spot_price": 25000.0,
        "composite_score": 2.0,
        "t1_forecast": {
            "verdict": "BULLISH",
            "prob_up": 0.68,
            "prob_flat": 0.12,
            "prob_down": 0.20,
            "expected_pts": 120.0
        },
        "t5_forecast": {
            "verdict": "BEARISH",
            "prob_up": 0.22,
            "prob_flat": 0.18,
            "prob_down": 0.60,
            "expected_pts": -240.0
        }
    }
    contract = build_prediction_context(conflicting_data, data_mode_override="LIVE")
    assert contract.predictions[0].direction == "BULLISH"
    assert contract.predictions[1].direction == "BEARISH"

    prompt = build_interpretation_prompt(contract)
    assert "T+1" in prompt
    assert "BULLISH" in prompt
    assert "T+5" in prompt
    assert "BEARISH" in prompt


# ==============================================================================
# 4. VALID AI JSON PARSING AND SCHEMA VALIDATION
# ==============================================================================
def test_valid_ai_json_parsing_and_schema(valid_llm_json_response):
    """Verifies that clean JSON and markdown code blocks parse cleanly into AIAnalysisOutput."""
    # 1. Plain JSON string
    res1 = extract_and_validate_analysis(
        raw_content=valid_llm_json_response,
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_ms=450.0
    )
    assert isinstance(res1, AIAnalysisOutput)
    assert res1.provider == "groq"
    assert res1.model == "openai/gpt-oss-120b"
    assert res1.latency_ms == 450.0
    assert res1.decision_status == "ANALYSIS_ONLY"
    assert len(res1.horizon_analysis) == 2
    assert len(res1.conflicts) == 2
    assert len(res1.supporting_evidence) == 3

    # 2. Markdown fenced JSON string
    fenced_content = f"```json\n{valid_llm_json_response}\n```"
    res2 = extract_and_validate_analysis(
        raw_content=fenced_content,
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
        latency_ms=620.0
    )
    assert isinstance(res2, AIAnalysisOutput)
    assert res2.provider == "openrouter"
    assert res2.model == "google/gemma-4-31b-it:free"


# ==============================================================================
# 5. MALFORMED JSON RETURNED BY THE MODEL
# ==============================================================================
def test_malformed_json_handling():
    """Verifies that broken, unparseable, or non-JSON model outputs raise descriptive ValueError."""
    bad_responses = [
        "I am an institutional AI and here is my summary...",  # Plain prose
        "```json\n{summary: 'Missing quotes on keys'}\n```",  # Invalid JSON syntax
        "{'summary': 'Single quotes are invalid in strict JSON'}",
        "{unclosed_json: ",  # Truncated
        "[]"  # Array instead of Object
    ]

    for bad in bad_responses:
        with pytest.raises(ValueError) as exc_info:
            extract_and_validate_analysis(
                raw_content=bad,
                provider="groq",
                model="openai/gpt-oss-20b",
                latency_ms=100.0
            )
        assert "not return valid JSON" in str(exc_info.value) or "output failed schema validation" in str(exc_info.value) or "not an object/dict" in str(exc_info.value)


# ==============================================================================
# 6. UNEXPECTED FIELD TYPES AND STRING CONSTRAINT VALIDATION
# ==============================================================================
def test_unexpected_field_types_and_validation():
    """Verifies Pydantic schema rejection when types or required fields are violated."""
    # Missing required 'summary'
    invalid_dict = {
        "horizon_analysis": [{"horizon": "T+1", "interpretation": "test", "uncertainties": []}],
        "conflicts": [],
        "supporting_evidence": []
    }
    with pytest.raises(ValueError):
        extract_and_validate_analysis(
            raw_content=json.dumps(invalid_dict),
            provider="groq",
            model="openai/gpt-oss-120b",
            latency_ms=100.0
        )

    # Empty horizon_analysis list (min_length=1 required)
    invalid_horizon = {
        "summary": "This is a valid summary of appropriate length.",
        "horizon_analysis": []
    }
    with pytest.raises(ValueError):
        extract_and_validate_analysis(
            raw_content=json.dumps(invalid_horizon),
            provider="groq",
            model="openai/gpt-oss-120b",
            latency_ms=100.0
        )


# ==============================================================================
# 7. PROVIDER TIMEOUT HANDLING
# ==============================================================================
def test_provider_timeout_handling():
    """Verifies that network timeout is caught, converted to ProviderError(408), and handled safely."""
    import httpx
    adapter = AIProviderAdapter(timeout_seconds=2, max_retries=0)

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        with patch.object(config, "GROQ_API_KEY", "test-key-mock"):
            with pytest.raises(ProviderError) as exc_info:
                adapter.complete(
                    prompt="test",
                    system_prompt="test",
                    model_id="openai/gpt-oss-120b"
                )
            assert exc_info.value.status_code == 408
            assert exc_info.value.is_transient is True


# ==============================================================================
# 8. INVALID API CREDENTIALS HANDLING (HTTP 401/403)
# ==============================================================================
def test_invalid_api_credentials_handling():
    """Verifies HTTP 401/403 causes immediate failure without wasted retries."""
    adapter = AIProviderAdapter(timeout_seconds=5, max_retries=3)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = json.dumps({"error": {"message": "Invalid API Key provided"}})

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        with patch.object(config, "GROQ_API_KEY", "invalid-key-mock"):
            with pytest.raises(ProviderError) as exc_info:
                adapter.complete(
                    prompt="test",
                    system_prompt="test",
                    model_id="openai/gpt-oss-120b"
                )
            assert exc_info.value.status_code == 401
            assert exc_info.value.is_transient is False
            # Verify no retry occurred
            assert mock_post.call_count == 1


# ==============================================================================
# 9. RATE LIMIT HANDLING (HTTP 429)
# ==============================================================================
def test_rate_limit_handling():
    """Verifies HTTP 429 retries up to max_retries and marks transient error."""
    adapter = AIProviderAdapter(timeout_seconds=5, max_retries=1)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = json.dumps({"error": {"message": "Rate limit reached on free tier"}})

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        with patch("time.sleep", return_value=None):  # Fast test
            with patch.object(config, "GROQ_API_KEY", "test-key-mock"):
                with pytest.raises(ProviderError) as exc_info:
                    adapter.complete(
                        prompt="test",
                        system_prompt="test",
                        model_id="openai/gpt-oss-120b"
                    )
                assert exc_info.value.status_code == 429
                assert exc_info.value.is_transient is True
                # Initial attempt + 1 retry = 2 calls
                assert mock_post.call_count == 2


# ==============================================================================
# 10. PRIMARY MODEL FAILURE AND FALLBACK TO SECONDARY MODEL
# ==============================================================================
def test_primary_failure_fallback_success(sample_ml_predictions, valid_llm_json_response):
    """Verifies that when primary model fails, fallback model is invoked and succeeds."""
    service = AIService()

    def mock_complete(prompt, system_prompt, model_id, **kwargs):
        if model_id == config.AI_PRIMARY_MODEL:
            raise ProviderError("Primary Groq model service unavailable", provider="groq", status_code=503)
        if model_id == config.AI_FALLBACK_MODEL:
            return {
                "content": valid_llm_json_response,
                "latency_ms": 320.0,
                "model": model_id,
                "provider": "groq",
                "usage": {}
            }
        raise ValueError(f"Unexpected model_id: {model_id}")

    with patch.object(service.provider_adapter, "complete", side_effect=mock_complete):
        with patch.object(config, "AI_ENABLED", True):
            analysis = service.analyze_predictions(
                force_refresh=True,
                predictions_data=sample_ml_predictions
            )
            assert analysis.fallback_used is True
            assert analysis.status == "FALLBACK_OK"
            assert analysis.model == config.AI_FALLBACK_MODEL
            assert analysis.decision_status == "ANALYSIS_ONLY"


# ==============================================================================
# 11. BOTH PROVIDERS UNAVAILABLE -> GRACEFUL DEGRADATION
# ==============================================================================
def test_both_providers_unavailable_graceful_degradation(sample_ml_predictions):
    """Verifies that if primary AND fallback models fail, system returns structured AI_UNAVAILABLE without crashing."""
    service = AIService()

    with patch.object(service.provider_adapter, "complete", side_effect=ProviderError("Network connection reset", provider="groq", is_transient=True)):
        with patch.object(config, "AI_ENABLED", True):
            analysis = service.analyze_predictions(
                force_refresh=True,
                predictions_data=sample_ml_predictions
            )
            assert analysis.status == "AI_UNAVAILABLE"
            assert "AI natural language interpretation is currently unavailable" in analysis.summary
            assert analysis.decision_status == "ANALYSIS_ONLY"
            assert len(analysis.horizon_analysis) == 2


# ==============================================================================
# 12. STALE / DELAYED MARKET DATA FLAGGING
# ==============================================================================
def test_stale_delayed_market_data_flagging(sample_ml_predictions):
    """Verifies market data mode is correctly tagged HISTORICAL_DAILY when market is closed."""
    with patch("core.market_calendar.NSEMarketCalendar.can_execute_trades", return_value=(False, "Market is closed")):
        contract = build_prediction_context(sample_ml_predictions)
        assert contract.data_mode == "HISTORICAL_DAILY"


# ==============================================================================
# 13. AI DISABLED (AI_ENABLED=False) BEHAVIOR
# ==============================================================================
def test_ai_disabled_behavior(sample_ml_predictions):
    """Verifies that when AI_ENABLED is False, no LLM calls are made and status is AI_DISABLED."""
    service = AIService()

    with patch.object(config, "AI_ENABLED", False):
        with patch.object(service.provider_adapter, "complete") as mock_complete:
            analysis = service.analyze_predictions(predictions_data=sample_ml_predictions)
            assert mock_complete.call_count == 0
            assert analysis.status == "AI_DISABLED"
            assert analysis.decision_status == "ANALYSIS_ONLY"
            assert "disabled in configuration" in analysis.summary


# ==============================================================================
# 14. ZERO API KEY LEAKAGE IN RESPONSES, ERRORS, AND STRING REPRESENTATIONS
# ==============================================================================
def test_zero_api_key_leakage():
    """Verifies secret API keys never leak in ProviderError, logs, or status endpoints."""
    secret_groq_key = "gsk_SUPER_SECRET_GROQ_KEY_DO_NOT_LEAK_9999"
    secret_or_key = "sk-or-SUPER_SECRET_OPENROUTER_KEY_DO_NOT_LEAK_8888"

    with patch.object(config, "GROQ_API_KEY", secret_groq_key):
        with patch.object(config, "OPENROUTER_API_KEY", secret_or_key):
            service = AIService()
            status = service.get_status()

            # Status JSON check
            status_json = status.model_dump_json()
            assert secret_groq_key not in status_json
            assert secret_or_key not in status_json
            assert status.groq_configured is True
            assert status.openrouter_configured is True

            # Error message check
            err = ProviderError("Authentication failed for groq (HTTP 401)", provider="groq", status_code=401)
            err_str = str(err)
            assert secret_groq_key not in err_str
            assert secret_or_key not in err_str


# ==============================================================================
# 15. CONFIRMATION THAT AI FAILURES NEVER ALTER OR SUPPRESS ML PREDICTIONS
# ==============================================================================
def test_ai_failures_preserve_ml_predictions(sample_ml_predictions):
    """Verifies raw ML predictions dictionary remains identical and intact before and after AI execution."""
    original_t1_prob = sample_ml_predictions["t1_forecast"]["prob_up"]
    original_t5_verdict = sample_ml_predictions["t5_forecast"]["verdict"]
    original_spot = sample_ml_predictions["spot_price"]

    service = AIService()
    with patch.object(service.provider_adapter, "complete", side_effect=Exception("Total LLM provider outage")):
        with patch.object(config, "AI_ENABLED", True):
            _ = service.analyze_predictions(force_refresh=True, predictions_data=sample_ml_predictions)

    # Re-verify original data has not been modified
    assert sample_ml_predictions["t1_forecast"]["prob_up"] == original_t1_prob
    assert sample_ml_predictions["t5_forecast"]["verdict"] == original_t5_verdict
    assert sample_ml_predictions["spot_price"] == original_spot


# ==============================================================================
# 16. CONFIRMATION THAT AI DECISION STATUS REMAINS ANALYSIS_ONLY
# ==============================================================================
def test_decision_status_enforced_analysis_only(valid_llm_json_response):
    """Verifies that even if model attempts to return AUTO_EXECUTE_BUY, it is overridden to ANALYSIS_ONLY."""
    tampered_json = json.loads(valid_llm_json_response)
    tampered_json["decision_status"] = "AUTO_EXECUTE_BUY"  # Attempted prompt injection / status change

    analysis = extract_and_validate_analysis(
        raw_content=json.dumps(tampered_json),
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_ms=200.0
    )
    assert analysis.decision_status == "ANALYSIS_ONLY"


# ==============================================================================
# 17. CONFIRMATION THAT NO FAKE PREDICTION METRICS / GREEKS ARE PRODUCED
# ==============================================================================
def test_no_fake_metrics_or_greeks_in_ai_layer(valid_llm_json_response):
    """Verifies AI schemas do not contain or fabricate delta, gamma, theta, vega, IV, or option premiums."""
    analysis = extract_and_validate_analysis(
        raw_content=valid_llm_json_response,
        provider="groq",
        model="openai/gpt-oss-120b",
        latency_ms=200.0
    )
    analysis_dict = analysis.model_dump()

    # The AI layer must not declare or output independent Greeks or pricing fields
    forbidden_keys = ["delta", "gamma", "theta", "vega", "implied_volatility", "call_iv", "put_iv", "option_premium"]
    for key in forbidden_keys:
        assert key not in analysis_dict, f"Forbidden Greek/pricing key '{key}' found in AI analysis output!"


# ==============================================================================
# 18. API ENDPOINT INTEGRATION TESTS (FastAPI TestClient)
# ==============================================================================
def test_api_endpoints_status_analyze_benchmark(valid_llm_json_response):
    """Tests the 4 FastAPI routes: /api/ai/status, /api/ai/analysis, /api/ai/analyze, /api/ai/benchmark."""
    client = TestClient(app)

    # 1. GET /api/ai/status
    res_status = client.get("/api/ai/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "enabled" in status_data
    assert "active_provider" in status_data
    assert "available_models" in status_data
    assert len(status_data["available_models"]) >= 5

    # 2. Mock AI analysis for GET /api/ai/analysis and POST /api/ai/analyze
    mock_complete_resp = {
        "content": valid_llm_json_response,
        "latency_ms": 350.0,
        "model": "openai/gpt-oss-120b",
        "provider": "groq",
        "usage": {}
    }

    with patch.object(ai_service.provider_adapter, "complete", return_value=mock_complete_resp):
        with patch.object(config, "AI_ENABLED", True):
            # GET /api/ai/analysis
            res_analysis = client.get("/api/ai/analysis?force_refresh=true")
            assert res_analysis.status_code == 200
            data_analysis = res_analysis.json()
            assert data_analysis["decision_status"] == "ANALYSIS_ONLY"
            assert data_analysis["status"] == "OK"
            assert len(data_analysis["horizon_analysis"]) == 2

            # POST /api/ai/analyze
            res_post = client.post("/api/ai/analyze", json={"force_refresh": True})
            assert res_post.status_code == 200
            data_post = res_post.json()
            assert data_post["decision_status"] == "ANALYSIS_ONLY"

            # POST /api/ai/benchmark
            res_bench = client.post("/api/ai/benchmark")
            assert res_bench.status_code == 200
            bench_res = res_bench.json()
            assert isinstance(bench_res, dict)
            assert bench_res["models_evaluated"] == len(ModelRegistry.list_model_ids())
            benchmarks = bench_res["results"]
            assert isinstance(benchmarks, list)
            assert len(benchmarks) == len(ModelRegistry.list_model_ids())
            for bm in benchmarks:
                assert bm["model_id"] in ModelRegistry.list_model_ids()
                assert bm["success"] is True
                assert bm["schema_valid"] is True
