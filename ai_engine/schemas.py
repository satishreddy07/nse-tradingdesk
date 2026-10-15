"""
Pydantic Schemas for AI Input & Output Contracts.
Enforces strict machine-readable contracts and validation for the AI intelligence layer.
"""
from typing import Dict, List, Optional, Any
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
    composite_score_t5: Optional[float] = Field(None, description="Decoupled structural 5-day weekly composite score")
    us_market_t1_forecast: Optional[Dict[str, Any]] = Field(None, description="US next-day market forecast (SPY/QQQ)")
    us_breadth_dispersion: Optional[float] = Field(None, description="US RSP vs SPY equal-weight breadth dispersion (%)")
    us_vix_term_structure: Optional[str] = Field(None, description="US VIX term structure (CONTANGO_STABLE / BACKWARDATION)")
    futures_relative_move: Optional[Dict[str, Any]] = Field(None, description="E-mini futures momentum & Nifty fair value gap")
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


# ==============================================================================
# 4. SWING TRADING ML + AI FINAL BUY DECISION GATE SCHEMAS
# ==============================================================================
class SwingReasonCode:
    APPROVED_CONFLUENCE = "APPROVED_CONFLUENCE"
    INSUFFICIENT_RISK_REWARD = "INSUFFICIENT_RISK_REWARD"
    CONFLICTING_TECHNICALS = "CONFLICTING_TECHNICALS"
    ML_EVIDENCE_INSUFFICIENT = "ML_EVIDENCE_INSUFFICIENT"
    EXCESSIVE_VOLATILITY = "EXCESSIVE_VOLATILITY"
    PORTFOLIO_EXPOSURE_LIMIT = "PORTFOLIO_EXPOSURE_LIMIT"
    INVALID_OR_STALE_DATA = "INVALID_OR_STALE_DATA"
    STRATEGY_CONDITIONS_NOT_SATISFIED = "STRATEGY_CONDITIONS_NOT_SATISFIED"
    MISSING_REQUIRED_INFO = "MISSING_REQUIRED_INFO"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"
    OTHER = "OTHER"


class SwingCandidateIdentity(BaseModel):
    candidate_id: str
    exchange: str = "NSE"
    symbol: str
    scanner_classification: str = "BUY NOW"
    scanner_run_id: Optional[str] = None


class SwingCandidateContext(BaseModel):
    as_of: str
    timezone: str = "Asia/Kolkata"
    data_mode: str = "live"
    data_quality: Dict[str, Any] = Field(default_factory=lambda: {
        "status": "valid",
        "is_stale": False,
        "missing_fields": [],
        "warnings": []
    })


class SwingTechnicalAnalysis(BaseModel):
    classification: str = "BUY NOW"
    entry_conditions: List[str] = Field(default_factory=list)
    satisfied_conditions: List[str] = Field(default_factory=list)
    unsatisfied_conditions: List[str] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    engine_version: str = "6-Pillar-v2.4"


class SwingMLAnalysis(BaseModel):
    decision: str = "PASS_TO_AI"
    model_name: str = "HistGradientBoosting+CalibratedClassifierCV"
    model_version: str = "swing_v2.0"
    target_definition: str = "target_before_stop_loss_in_holding_period"
    prediction_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    expected_r: Optional[float] = None
    feature_snapshot_id: Optional[str] = None
    decision_policy_version: str = "1.0"
    risk_flags: List[str] = Field(default_factory=list)
    fatal_flags: List[str] = Field(default_factory=list)


class SwingTradePlan(BaseModel):
    entry_rule: str = "Breakout above pivot with volume confirmation"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    target_2: Optional[float] = None
    holding_period: str = "5-15 sessions"
    reward_risk_ratio: Optional[float] = None
    position_size: Optional[int] = None
    risk_per_share: Optional[float] = None
    capital_at_risk: Optional[float] = None
    estimated_transaction_costs: Optional[float] = None


class SwingHistoricalEvidence(BaseModel):
    sample_count: Optional[int] = None
    target_before_stop_frequency: Optional[float] = None
    stop_before_target_frequency: Optional[float] = None
    evaluation_type: str = "out_of_sample"
    period: Optional[str] = None
    historical_win_rate: Optional[float] = None


class SwingPortfolioContext(BaseModel):
    already_held: bool = False
    duplicate_pending_order: bool = False
    available_capital: Optional[float] = None
    risk_limits_available: bool = True
    active_slots_used: Optional[int] = None
    max_slots: Optional[int] = None


class SwingInstructions(BaseModel):
    objective: str = "Make a final analytical BUY NOW or DO NOT BUY decision using the supplied evidence."
    allow_order_execution: bool = False
    do_not_invent_data: bool = True


class SwingCandidateAIRequest(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    task: str = "swing_final_buy_decision"
    module: str = "swing_trading"
    candidate: SwingCandidateIdentity
    context: SwingCandidateContext
    technical_analysis: SwingTechnicalAnalysis
    ml_analysis: SwingMLAnalysis
    trade_plan: SwingTradePlan
    historical_evidence: Optional[SwingHistoricalEvidence] = None
    portfolio_context: Optional[SwingPortfolioContext] = None
    instructions: SwingInstructions = Field(default_factory=SwingInstructions)


class SwingCandidateAIResponse(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    status: str = "success"
    candidate_id: str
    symbol: str
    final_decision: str = "DO_NOT_BUY"
    reason_code: str = SwingReasonCode.INSUFFICIENT_RISK_REWARD
    summary: str = Field(..., min_length=5, max_length=2000)
    supporting_factors: List[str] = Field(default_factory=list)
    contradicting_factors: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    conditions_for_reconsideration: List[str] = Field(default_factory=list)
    model_context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("final_decision")
    @classmethod
    def validate_final_decision(cls, v: str) -> str:
        v_upper = v.upper().strip()
        if v_upper in ("BUY_NOW", "BUY"):
            return "BUY_NOW"
        return "DO_NOT_BUY"

