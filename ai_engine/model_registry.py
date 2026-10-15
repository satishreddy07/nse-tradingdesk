"""
AI Model Registry for Groq and OpenRouter Free-Tier Candidate Models.
Maintains exact model definitions, provider routing, context limits, and capabilities.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    provider: str  # "groq" or "openrouter"
    role: str
    free_tier_eligible: bool
    context_window: int
    max_output_tokens: int
    description: str
    recommended_temperature: float = 0.2


# The candidate models to support:
CANDIDATE_MODELS: Dict[str, ModelMetadata] = {
    # 0. Groq - Production Standard LLaMA 3.3 70B
    "llama-3.3-70b-versatile": ModelMetadata(
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        role="Primary High-Speed LPU Deep Reasoning",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=2048,
        description="Meta LLaMA 3.3 70B hosted on Groq LPUs for rapid, accurate swing trade and prediction evaluation.",
        recommended_temperature=0.2,
    ),
    "llama-3.1-8b-instant": ModelMetadata(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        role="Fast Low-Latency LPU Explanations & Fallback",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=1536,
        description="Meta LLaMA 3.1 8B hosted on Groq LPUs for rapid fallback inference.",
        recommended_temperature=0.2,
    ),
    # 1. Groq - Primary for complex interpretation & reasoning
    "openai/gpt-oss-120b": ModelMetadata(
        model_id="openai/gpt-oss-120b",
        provider="groq",
        role="Primary Deep Reasoning & Prediction Interpretation",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=2048,
        description="High-capacity open weights model hosted on Groq LPU inference for nuanced market reasoning.",
        recommended_temperature=0.2,
    ),
    # 2. Groq - Lightweight for routine explanations & low latency
    "openai/gpt-oss-20b": ModelMetadata(
        model_id="openai/gpt-oss-20b",
        provider="groq",
        role="Fast Low-Latency Routine Explanations",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=1536,
        description="Efficient smaller model on Groq for routine explanations and rapid fallback execution.",
        recommended_temperature=0.2,
    ),
    # 3. OpenRouter - Structured prediction interpretation
    "google/gemma-4-31b-it:free": ModelMetadata(
        model_id="google/gemma-4-31b-it:free",
        provider="openrouter",
        role="Structured Prediction Interpretation",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=2048,
        description="Google Gemma 4 instruction-tuned model via OpenRouter free tier for structured JSON generation.",
        recommended_temperature=0.2,
    ),
    # 4. OpenRouter - Complex reasoning & horizon conflict analysis
    "nvidia/nemotron-3-super-120b-a12b:free": ModelMetadata(
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        provider="openrouter",
        role="Complex Macro & Cross-Asset Reasoning",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=2048,
        description="NVIDIA Nemotron 3 Super high-capacity free model on OpenRouter for cross-asset conflict resolution.",
        recommended_temperature=0.2,
    ),
    # 5. OpenRouter - Routine reasoning & structured risk breakdown
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": ModelMetadata(
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        provider="openrouter",
        role="Routine Reasoning & Structured Risk Breakdown",
        free_tier_eligible=True,
        context_window=8192,
        max_output_tokens=1536,
        description="NVIDIA Nemotron 3 Nano reasoning model on OpenRouter free tier for concise risk summaries.",
        recommended_temperature=0.2,
    ),
}


class ModelRegistry:
    """Configurable registry for AI model candidates and runtime lookup."""

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelMetadata]:
        return CANDIDATE_MODELS.get(model_id)

    @classmethod
    def get_all_models(cls) -> Dict[str, ModelMetadata]:
        return dict(CANDIDATE_MODELS)

    @classmethod
    def list_model_ids(cls) -> List[str]:
        return list(CANDIDATE_MODELS.keys())

    @classmethod
    def get_models_by_provider(cls, provider: str) -> List[ModelMetadata]:
        prov = provider.lower().strip()
        return [m for m in CANDIDATE_MODELS.values() if m.provider == prov]

    @classmethod
    def is_valid_model(cls, model_id: str) -> bool:
        return model_id in CANDIDATE_MODELS

    @classmethod
    def resolve_provider_for_model(cls, model_id: str) -> str:
        meta = cls.get_model(model_id)
        if meta:
            return meta.provider
        # Heuristic fallback based on model prefix
        if model_id.startswith("openai/") or model_id.startswith("llama") or "groq" in model_id.lower():
            return "groq"
        return "openrouter"
