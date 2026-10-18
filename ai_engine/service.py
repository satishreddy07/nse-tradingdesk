"""
AI Intelligence Orchestrator Service: Manages prediction context ingestion,
provider fallback, validation, caching, and model benchmarking.
"""
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional

from core import config
from ai_engine.cache import ai_cache
from ai_engine.context_builder import build_prediction_context
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
    ModelBenchmarkReport,
    PredictionInputContract,
    SwingCandidateAIRequest,
    SwingCandidateAIResponse,
    SwingReasonCode,
)

logger = logging.getLogger("ai_engine.service")


class AIService:
    """Orchestration service for LLM-driven prediction interpretation."""

    def __init__(self):
        self.provider_adapter = AIProviderAdapter(
            timeout_seconds=config.AI_TIMEOUT_SECONDS,
            max_retries=config.AI_MAX_RETRIES
        )

    def get_status(self) -> AIStatusResponse:
        """Returns safe configuration, credentials availability, and cache state (no secret exposure)."""
        latest = ai_cache.get_latest()
        last_ts = latest.generated_at if latest else None

        return AIStatusResponse(
            enabled=config.AI_ENABLED,
            active_provider=config.AI_PROVIDER,
            primary_model=config.AI_PRIMARY_MODEL,
            fallback_model=config.AI_FALLBACK_MODEL,
            available_models=ModelRegistry.list_model_ids(),
            groq_configured=bool(config.GROQ_API_KEY.strip()),
            openrouter_configured=bool(config.OPENROUTER_API_KEY.strip()),
            last_analysis_timestamp=last_ts,
            cached_analyses_count=ai_cache.count()
        )

    def analyze_predictions(
        self,
        force_refresh: bool = False,
        model_override: Optional[str] = None,
        predictions_data: Optional[Dict[str, Any]] = None
    ) -> AIAnalysisOutput:
        """
        Interprets current ML predictions using configured LLM with fallback and caching.
        Preserves deterministic ML predictions regardless of AI state.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Check if AI layer is disabled
        if not config.AI_ENABLED:
            return AIAnalysisOutput(
                summary="AI intelligence layer is disabled in configuration. Quantitative 6-Pillar ML predictions and deterministic risk controls remain active and authoritative.",
                horizon_analysis=[
                    AIHorizonAnalysis(
                        horizon="T+1",
                        interpretation="ML quantitative prediction active. Enable AI_ENABLED in environment to view natural language analysis.",
                        uncertainties=["AI Layer Disabled"]
                    ),
                    AIHorizonAnalysis(
                        horizon="T+5",
                        interpretation="ML quantitative prediction active. Enable AI_ENABLED in environment to view natural language analysis.",
                        uncertainties=["AI Layer Disabled"]
                    )
                ],
                conflicts=[],
                supporting_evidence=[],
                event_risks=[],
                reasons_to_avoid_trade=[],
                missing_or_stale_data=[],
                strategy_considerations=[],
                decision_status="ANALYSIS_ONLY",
                status="AI_DISABLED",
                generated_at=now_iso
            )

        # 2. Ingest ML predictions from nifty_analytics_engine if not provided
        if predictions_data is None:
            try:
                from engines.nifty_analytics_engine import nifty_analytics_engine
                predictions_data = nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)
            except Exception as pe:
                logger.error(f"Failed to fetch ML predictions: {pe}")
                return self._build_unavailable_fallback(
                    reason=f"Failed to fetch ML predictions: {str(pe)}",
                    now_iso=now_iso
                )

        # 3. Build normalized PredictionInputContract
        try:
            contract = build_prediction_context(predictions_data)
        except Exception as exc:
            logger.error(f"Failed to build prediction context contract: {exc}")
            return self._build_unavailable_fallback(
                reason=f"Failed to assemble prediction context: {str(exc)}",
                now_iso=now_iso
            )

        # 4. Check cache if not forcing refresh
        if not force_refresh and not model_override:
            cached = ai_cache.get(contract)
            if cached:
                return cached

        # 5. Resolve target models
        primary_model = model_override or config.AI_PRIMARY_MODEL
        fallback_model = config.AI_FALLBACK_MODEL if not model_override else None

        prompt = build_interpretation_prompt(contract)

        # 6. Attempt primary model
        primary_err = None
        try:
            res = self.provider_adapter.complete(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model_id=primary_model,
                temperature=0.2,
                max_tokens=2048
            )
            analysis = extract_and_validate_analysis(
                raw_content=res["content"],
                provider=res["provider"],
                model=primary_model,
                latency_ms=res["latency_ms"]
            )
            analysis.generated_at = now_iso
            analysis.fallback_used = False
            ai_cache.set(contract, analysis)
            return analysis
        except Exception as exc:
            primary_err = str(exc)
            logger.warning(f"Primary model {primary_model} failed: {exc}")

        # 7. Attempt fallback model if configured and different from primary
        if fallback_model and fallback_model != primary_model:
            try:
                logger.info(f"Attempting fallback model: {fallback_model}")
                res = self.provider_adapter.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    model_id=fallback_model,
                    temperature=0.2,
                    max_tokens=2048
                )
                analysis = extract_and_validate_analysis(
                    raw_content=res["content"],
                    provider=res["provider"],
                    model=fallback_model,
                    latency_ms=res["latency_ms"]
                )
                analysis.generated_at = now_iso
                analysis.fallback_used = True
                analysis.status = "FALLBACK_OK"
                ai_cache.set(contract, analysis)
                return analysis
            except Exception as fb_exc:
                logger.error(f"Fallback model {fallback_model} also failed: {fb_exc}")

        # 8. Both providers/models failed: return graceful AI_UNAVAILABLE without breaking ML
        return self._build_unavailable_fallback(
            reason=f"Primary model ({primary_model}) error: {primary_err}",
            now_iso=now_iso
        )

    def _build_unavailable_fallback(self, reason: str, now_iso: str) -> AIAnalysisOutput:
        """Constructs safe fallback response adhering to schema when external LLMs are unreachable."""
        return AIAnalysisOutput(
            summary=f"AI natural language interpretation is currently unavailable ({reason}). Deterministic ML predictions, options matrices, and risk safeguards remain 100% authoritative and active.",
            horizon_analysis=[
                AIHorizonAnalysis(
                    horizon="T+1",
                    interpretation="Deterministic quantitative prediction active. AI interpretation temporarily unavailable.",
                    uncertainties=["Provider Unreachable or Quota Exhausted"]
                ),
                AIHorizonAnalysis(
                    horizon="T+5",
                    interpretation="Deterministic quantitative prediction active. AI interpretation temporarily unavailable.",
                    uncertainties=["Provider Unreachable or Quota Exhausted"]
                )
            ],
            conflicts=["AI service temporarily unavailable"],
            supporting_evidence=["Quantitative ML engine running deterministically"],
            event_risks=["External AI provider connectivity degraded"],
            reasons_to_avoid_trade=["Rely solely on quantitative models until AI layer reconnects"],
            missing_or_stale_data=["LLM API response missing"],
            strategy_considerations=["Maintain standard position sizing per quantitative risk engine"],
            decision_status="ANALYSIS_ONLY",
            status="AI_UNAVAILABLE",
            generated_at=now_iso
        )

    def run_model_benchmark(self, sample_contract: Optional[PredictionInputContract] = None) -> List[ModelBenchmarkReport]:
        """
        Runs standardized benchmark comparing all 5 candidate models against identical input.
        Protected administrative function.
        """
        if sample_contract is None:
            # Synthetic standard fixture with conflicting horizons for rigorous testing
            from ai_engine.schemas import MarketContextItem, PredictionHorizonItem
            sample_contract = PredictionInputContract(
                schema_version="1.0",
                symbol="NIFTY",
                as_of=datetime.now(timezone.utc).isoformat(),
                data_mode="HISTORICAL_DAILY",
                predictions=[
                    PredictionHorizonItem(
                        horizon="T+1",
                        direction="BULLISH",
                        probability_up=0.62,
                        probability_flat=0.18,
                        probability_down=0.20,
                        expected_return_pct=0.35,
                        expected_points=85.0,
                        expected_close=24800.0,
                        model_version="6-Pillar-Empirical",
                        validation_status="BENCHMARK_SYNTHETIC_FIXTURE"
                    ),
                    PredictionHorizonItem(
                        horizon="T+5",
                        direction="BEARISH",
                        probability_up=0.38,
                        probability_flat=0.15,
                        probability_down=0.47,
                        expected_return_pct=-0.80,
                        expected_points=-195.0,
                        expected_close=24520.0,
                        model_version="6-Pillar-Empirical",
                        validation_status="BENCHMARK_SYNTHETIC_FIXTURE"
                    )
                ],
                market_context=MarketContextItem(
                    spot_price=24715.0,
                    composite_score=8.5,
                    regime="CONSOLIDATION_RANGE",
                    event_risk="RBI MPC Meeting on Wednesday",
                    data_quality="VALID",
                    pillar_scores={"technical": 18.0, "breadth": -12.0, "global": 10.0, "macro": -8.0},
                    fii_dii_summary="FII Cash: -₹1,850 Cr | DII Cash: +₹2,100 Cr",
                    option_chain_pcr=0.92,
                    max_pain=24700.0
                )
            )

        prompt = build_interpretation_prompt(sample_contract)
        reports: List[ModelBenchmarkReport] = []

        for model_id in ModelRegistry.list_model_ids():
            provider = ModelRegistry.resolve_provider_for_model(model_id)
            start_t = time.perf_counter()
            try:
                res = self.provider_adapter.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    model_id=model_id,
                    temperature=0.2,
                    max_tokens=2048
                )
                latency_ms = res["latency_ms"]
                raw_text = res["content"]

                # Validate JSON and schema
                analysis = extract_and_validate_analysis(
                    raw_content=raw_text,
                    provider=provider,
                    model=model_id,
                    latency_ms=latency_ms
                )

                # Check conflict detection (T+1 Bullish vs T+5 Bearish)
                conflicts_found = len(analysis.conflicts) > 0 or any("diverg" in c.lower() or "conflict" in c.lower() or "t+1" in c.lower() for c in analysis.conflicts)

                reports.append(ModelBenchmarkReport(
                    model_id=model_id,
                    provider=provider,
                    success=True,
                    latency_ms=latency_ms,
                    valid_json=True,
                    schema_valid=True,
                    missing_fields=[],
                    conflicts_detected=conflicts_found,
                    missing_data_flagged=len(analysis.missing_or_stale_data) > 0,
                    summary_preview=analysis.summary[:120] + "..."
                ))
            except Exception as exc:
                latency_ms = round((time.perf_counter() - start_t) * 1000.0, 1)
                reports.append(ModelBenchmarkReport(
                    model_id=model_id,
                    provider=provider,
                    success=False,
                    latency_ms=latency_ms,
                    valid_json=False,
                    schema_valid=False,
                    missing_fields=[],
                    error_message=str(exc)[:200]
                ))

        return reports

    def evaluate_swing_candidate(
        self,
        candidate_req: SwingCandidateAIRequest,
        model_override: Optional[str] = None
    ) -> SwingCandidateAIResponse:
        """
        Evaluates a swing trade candidate that passed the ML gatekeeper (PASS_TO_AI).
        Acts as the final analytical approval gatekeeper before deterministic risk checks.
        Returns a validated SwingCandidateAIResponse with BUY_NOW or DO_NOT_BUY.
        Preserves deterministic safety: if AI is disabled or fails, defaults to DO_NOT_BUY.
        """
        req_id = candidate_req.request_id
        cand_id = candidate_req.candidate.candidate_id
        sym = candidate_req.candidate.symbol
        snapshot_id = candidate_req.ml_analysis.feature_snapshot_id or "unknown"
        ml_ver = candidate_req.ml_analysis.model_version

        # 1. Check if AI layer is disabled
        if not config.AI_ENABLED:
            logger.info(f"AI disabled: returning safe DO_NOT_BUY for {sym}")
            return SwingCandidateAIResponse(
                schema_version="1.0",
                request_id=req_id,
                status="AI_DISABLED",
                candidate_id=cand_id,
                symbol=sym,
                final_decision="DO_NOT_BUY",
                reason_code=SwingReasonCode.AI_UNAVAILABLE,
                summary="AI intelligence layer is disabled in configuration. Automated buy gated for risk safety.",
                supporting_factors=[],
                contradicting_factors=["AI gatekeeper disabled in system environment"],
                risk_factors=["Cannot verify analytical confluence without active AI gatekeeper"],
                missing_information=[],
                conditions_for_reconsideration=["Enable AI_ENABLED in system configuration"],
                model_context={"ml_model_version": ml_ver, "input_snapshot_id": snapshot_id}
            )

        # 2. Check if credentials exist for active provider
        active_provider = config.AI_PROVIDER
        if not self.provider_adapter.is_provider_configured(active_provider):
            # Check fallback provider
            alt_prov = "openrouter" if active_provider == "groq" else "groq"
            if not self.provider_adapter.is_provider_configured(alt_prov):
                logger.warning(f"No configured AI providers available for swing gate on {sym}")
                return SwingCandidateAIResponse(
                    schema_version="1.0",
                    request_id=req_id,
                    status="AI_UNAVAILABLE",
                    candidate_id=cand_id,
                    symbol=sym,
                    final_decision="DO_NOT_BUY",
                    reason_code=SwingReasonCode.AI_UNAVAILABLE,
                    summary="No AI provider API keys configured. Safe gatekeeper blocks automatic entry.",
                    supporting_factors=[],
                    contradicting_factors=["Provider API keys missing"],
                    risk_factors=["Missing analytical gatekeeper approval"],
                    missing_information=["GROQ_API_KEY / OPENROUTER_API_KEY"],
                    conditions_for_reconsideration=["Configure valid AI provider credentials"],
                    model_context={"ml_model_version": ml_ver, "input_snapshot_id": snapshot_id}
                )

        # 3. Construct structured Prompt
        system_prompt = (
            "You are the Chief Quantitative Risk Manager and Final Analytical Approval Gatekeeper for an Institutional Indian Equity Swing Trading Desk.\n"
            "Your objective is to make a final analytical BUY_NOW or DO_NOT_BUY decision using ONLY the supplied candidate evidence.\n"
            "Rules:\n"
            "1. DO NOT BUY is a valid, normal, and preferred outcome when technical evidence conflicts, setup is overextended, volume is low, or risk/reward is unfavorable.\n"
            "2. Never assume ML is always correct. You are permitted and expected to reject candidates that ML passed if technical structure, regime, or overextension present asymmetric risk.\n"
            "3. Ground all supporting, contradicting, and risk factors strictly in the provided JSON metrics. Never fabricate data or prices.\n"
            "4. Respond with ONLY valid JSON adhering strictly to the requested schema. No conversational prose or markdown wrap outside JSON."
        )

        user_prompt = (
            f"Please conduct the final analytical swing trade review for the following candidate:\n\n"
            f"```json\n{candidate_req.model_dump_json(indent=2)}\n```\n\n"
            "Return a strictly valid JSON object matching this schema:\n"
            "{\n"
            '  "schema_version": "1.0",\n'
            f'  "request_id": "{req_id}",\n'
            '  "status": "success",\n'
            f'  "candidate_id": "{cand_id}",\n'
            f'  "symbol": "{sym}",\n'
            '  "final_decision": "BUY_NOW" or "DO_NOT_BUY",\n'
            '  "reason_code": "APPROVED_CONFLUENCE" | "INSUFFICIENT_RISK_REWARD" | "CONFLICTING_TECHNICALS" | "ML_EVIDENCE_INSUFFICIENT" | "EXCESSIVE_VOLATILITY" | "PORTFOLIO_EXPOSURE_LIMIT" | "INVALID_OR_STALE_DATA" | "STRATEGY_CONDITIONS_NOT_SATISFIED" | "MISSING_REQUIRED_INFO" | "OTHER",\n'
            '  "summary": "Clear, grounded 2-3 sentence analytical synthesis of why this setup was approved or rejected",\n'
            '  "supporting_factors": ["Factor 1", ...],\n'
            '  "contradicting_factors": ["Factor 1", ...],\n'
            '  "risk_factors": ["Risk 1", ...],\n'
            '  "missing_information": [],\n'
            '  "conditions_for_reconsideration": ["Condition 1", ...],\n'
            f'  "model_context": {{"ml_model_version": "{ml_ver}", "input_snapshot_id": "{snapshot_id}"}}\n'
            "}"
        )

        # 4. Invoke LLM with fallback
        primary_model = model_override or config.AI_PRIMARY_MODEL
        fallback_model = config.AI_FALLBACK_MODEL if not model_override else None

        raw_res = None
        for target_model in [primary_model, fallback_model]:
            if not target_model:
                continue
            try:
                res = self.provider_adapter.complete(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_id=target_model,
                    temperature=0.1,
                    max_tokens=3000
                )
                if res and res.get("content"):
                    raw_res = res["content"]
                    break
            except Exception as e:
                logger.warning(f"AI candidate review failed on {target_model}: {e}")

        if not raw_res:
            return SwingCandidateAIResponse(
                schema_version="1.0",
                request_id=req_id,
                status="AI_UNAVAILABLE",
                candidate_id=cand_id,
                symbol=sym,
                final_decision="DO_NOT_BUY",
                reason_code=SwingReasonCode.AI_UNAVAILABLE,
                summary="AI providers timed out or failed to return response. Failsafe defaults to DO_NOT_BUY.",
                supporting_factors=[],
                contradicting_factors=["Inference timeout across configured providers"],
                risk_factors=["Execution blocked due to missing gatekeeper verification"],
                missing_information=[],
                conditions_for_reconsideration=["Retry when network/provider connectivity is restored"],
                model_context={"ml_model_version": ml_ver, "input_snapshot_id": snapshot_id}
            )

        # 5. Extract and validate JSON
        try:
            cleaned = raw_res.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

            data = json.loads(cleaned)
            # Ensure required identifiers match
            data["request_id"] = req_id
            data["candidate_id"] = cand_id
            data["symbol"] = sym
            if not data.get("model_context"):
                data["model_context"] = {"ml_model_version": ml_ver, "input_snapshot_id": snapshot_id}

            return SwingCandidateAIResponse.model_validate(data)
        except Exception as ve:
            logger.error(f"Malformed AI response for {sym}: {ve} | Raw: {raw_res[:200]}")
            return SwingCandidateAIResponse(
                schema_version="1.0",
                request_id=req_id,
                status="error",
                candidate_id=cand_id,
                symbol=sym,
                final_decision="DO_NOT_BUY",
                reason_code=SwingReasonCode.STRATEGY_CONDITIONS_NOT_SATISFIED,
                summary=f"AI output failed schema validation ({str(ve)[:100]}). Strict safety gate rejects entry.",
                supporting_factors=[],
                contradicting_factors=["Malformed AI output JSON"],
                risk_factors=["Unverified analytical output"],
                missing_information=[],
                conditions_for_reconsideration=[],
                model_context={"ml_model_version": ml_ver, "input_snapshot_id": snapshot_id}
            )


# Global service instance
ai_service = AIService()

