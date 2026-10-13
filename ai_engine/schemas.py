"""
Pydantic Schemas for AI Input & Output Contracts.
Enforces strict machine-readable contracts and validation for the AI intelligence layer.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# 1. STANDARD PREDICTION INPUT CONTRACT (ML -> AI Adapter)
# ==============================================================================
class PredictionHorizonItem(BaseModel):
    horizon: str = Field(..., description="Forecast horizon, e.g., 'T+1' or 'T+5'")
    direction: str = Field(..., description="Directional label, e.g., 'BULLISH', 'BEARISH', 'RANGE-BOUND'")
    probability_up: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of upward drift")
    probability_flat: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated probability of consolidation")
    probability_down: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated probability of downward drift")
    expected_return_pct: Optional[float] = Field(None, description="Expected percentage return based on ATR/drift model")
    expected_points: Optional[float] = Field(None, description="Expected drift points")
    expected_close: Optional[float] = Field(None, description="Expected closing price level")
    forecast_range_1sigma: Optional[List[float]] = Field(None, description="[low_1s, high_1s] expected price band")
    forecast_range_2sigma: Optional[List[float]] = Field(None, description="[low_2s, high_2s] tail risk band")
    model_version: str = Field("6-Pillar-v2.4", description="Underlying quantitative ML engine version")
    validation_status: str = Field("WALK_FORWARD_VALIDATED", description="Attribution/validation state")


class MarketContextItem(BaseModel):
    spot_price: float = Field(..., gt=0.0)
    composite_score: float = Field(..., ge=-100.0, le=100.0)
    regime: Optional[str] = Field(None, description="Macro or volatility regime")
    event_risk: Optional[str] = Field(None, description="Central bank / economic event flag")
    data_quality: str = Field("VALID", description="'VALID', 'DELAYED', or 'INSUFFICIENT_DATA'")
    pillar_scores: Optional[Dict[str, float]] = None
    sectoral_leaders: Optional[List[str]] = None
    fii_dii_summary: Optional[str] = None
    option_chain_pcr: Optional[float] = None
    max_pain: Optional[float] = None
    india_vix: Optional[float] = Field(None, description="India VIX current market level")
    atm_iv: Optional[float] = Field(None, description="ATM implied volatility (%) from live option chain")
    recent_news_events: Optional[List[Dict[str, str]]] = None


class PredictionInputContract(BaseModel):
    schema_version: str = Field("1.0", description="Contract schema version")
    symbol: str = Field("NIFTY", description="Underlying index symbol")
    as_of: str = Field(..., description="ISO-8601 timestamp with timezone")
    data_mode: str = Field(..., description="'LIVE', 'DELAYED', or 'HISTORICAL_DAILY'")
    predictions: List[PredictionHorizonItem] = Field(..., min_length=1)
    market_context: MarketContextItem


# ==============================================================================
# 2. STRICT AI OUTPUT SCHEMA (LLM -> Backend Validator)
# ==============================================================================
class AIHorizonAnalysis(BaseModel):
    horizon: str = Field(..., description="'T+1' or 'T+5'")
    interpretation: str = Field(..., max_length=1200, description="Clear, understandable interpretation of the forecast")
    uncertainties: List[str] = Field(default_factory=list, description="Specific elements that remain uncertain")


class AIAnalysisOutput(BaseModel):
    summary: str = Field(..., min_length=10, max_length=2000, description="High-level synthesis of T+1 and T+5 predictions")
    horizon_analysis: List[AIHorizonAnalysis] = Field(..., min_length=1, description="Horizon-by-horizon explanations")
    conflicts: List[str] = Field(default_factory=list, description="Any directional, momentum, or cross-asset conflicts identified")
    supporting_evidence: List[str] = Field(default_factory=list, description="Verified evidence supporting the directional drift")
    event_risks: List[str] = Field(default_factory=list, description="Known economic/corporate event risks that could derail thesis")
    reasons_to_avoid_trade: List[str] = Field(default_factory=list, description="Specific caution flags or reasons a trader should stand aside")
    missing_or_stale_data: List[str] = Field(default_factory=list, description="Any missing inputs or indicators not present in context")
    strategy_considerations: List[str] = Field(default_factory=list, description="Strategy playbook considerations from existing matrix")
    decision_status: str = Field("ANALYSIS_ONLY", description="Always ANALYSIS_ONLY; AI cannot execute orders")

    # Execution telemetry metadata
    provider: Optional[str] = Field(None, description="Provider that performed inference, e.g., 'groq' or 'openrouter'")
    model: Optional[str] = Field(None, description="Model ID that produced this analysis")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    generated_at: Optional[str] = Field(None, description="ISO timestamp of generation")
    fallback_used: bool = Field(False, description="Whether fallback model had to be invoked")
    status: str = Field("OK", description="'OK', 'FALLBACK_OK', 'AI_DISABLED', or 'AI_UNAVAILABLE'")

    @field_validator("decision_status")
    @classmethod
    def enforce_analysis_only(cls, v: str) -> str:
        if v != "ANALYSIS_ONLY":
            return "ANALYSIS_ONLY"
        return v


# ==============================================================================
# 3. AI CONFIGURATION & BENCHMARK STATUS SCHEMAS
# ==============================================================================
class AIStatusResponse(BaseModel):
    enabled: bool
    active_provider: str
    primary_model: str
    fallback_model: str
    available_models: List[str]
    groq_configured: bool
    openrouter_configured: bool
    last_analysis_timestamp: Optional[str] = None
    cached_analyses_count: int = 0


class ModelBenchmarkReport(BaseModel):
    model_id: str
    provider: str
    success: bool
    latency_ms: float
    valid_json: bool
    schema_valid: bool
    missing_fields: List[str] = Field(default_factory=list)
    conflicts_detected: bool = False
    missing_data_flagged: bool = False
    error_message: Optional[str] = None
    summary_preview: Optional[str] = None
