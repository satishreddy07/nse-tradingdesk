"""
engines/ai_engine.py
====================
Backward-compatible proxy re-exporting the modular ai_engine package.
"""
from ai_engine.service import AIService, ai_service

__all__ = ["AIService", "ai_service"]
