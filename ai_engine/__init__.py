"""
AI Intelligence Engine Package for NSE Trading Desk.
Integrates Groq & OpenRouter free-tier LLM models with verified ML predictions.
"""
from ai_engine.model_registry import ModelRegistry, CANDIDATE_MODELS
from ai_engine.schemas import (
    PredictionInputContract,
    AIAnalysisOutput,
    AIStatusResponse,
    ModelBenchmarkReport,
)
from ai_engine.service import AIService, ai_service

__all__ = [
    "ModelRegistry",
    "CANDIDATE_MODELS",
    "PredictionInputContract",
    "AIAnalysisOutput",
    "AIStatusResponse",
    "ModelBenchmarkReport",
    "AIService",
    "ai_service",
]
