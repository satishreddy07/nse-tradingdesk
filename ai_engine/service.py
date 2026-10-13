"""
AI Intelligence Orchestrator Service: Manages prediction context ingestion,
provider fallback, validation, caching, and model benchmarking.
"""
from datetime import datetime, timezone
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
            from engines.nifty_analytics_engine import nifty_analytics_engine
            predictions_data = nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)

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
                        model_version="6-Pillar-v2.4",
                        validation_status="WALK_FORWARD_VALIDATED (Hit Rate: 72.0%)"
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
                        model_version="6-Pillar-v2.4",
                        validation_status="WALK_FORWARD_VALIDATED (Hit Rate: 74.5%)"
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


# Global service instance
ai_service = AIService()
