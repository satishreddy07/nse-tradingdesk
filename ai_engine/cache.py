"""
Thread-safe TTL Cache for AI prediction interpretations.
Conserves free-tier quotas and prevents redundant LLM calls on identical prediction inputs.
"""
import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional

from ai_engine.schemas import AIAnalysisOutput, PredictionInputContract


class AIAnalysisCache:
    """In-memory thread-safe cache with TTL and hash validation."""

    def __init__(self, default_ttl_seconds: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._latest_analysis: Optional[AIAnalysisOutput] = None
        self._lock = threading.Lock()
        self.default_ttl = default_ttl_seconds

    def compute_contract_hash(self, contract: PredictionInputContract) -> str:
        """Computes deterministic MD5 hash of the normalized prediction contract."""
        core_state = {
            "symbol": contract.symbol,
            "predictions": [p.model_dump() for p in contract.predictions],
            "spot": contract.market_context.spot_price,
            "composite": contract.market_context.composite_score,
            "regime": contract.market_context.regime,
        }
        encoded = json.dumps(core_state, sort_keys=True).encode("utf-8")
        return hashlib.md5(encoded).hexdigest()

    def get(self, contract: PredictionInputContract) -> Optional[AIAnalysisOutput]:
        """Returns cached analysis if present and not expired."""
        key = self.compute_contract_hash(contract)
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry["expires_at"] > now:
                return entry["data"]
            # Prune if expired
            if entry:
                del self._cache[key]
        return None

    def set(self, contract: PredictionInputContract, analysis: AIAnalysisOutput, ttl_seconds: Optional[int] = None) -> None:
        """Stores analysis with TTL."""
        key = self.compute_contract_hash(contract)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl
        with self._lock:
            self._cache[key] = {
                "data": analysis,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            self._latest_analysis = analysis

    def get_latest(self) -> Optional[AIAnalysisOutput]:
        """Returns the most recent valid analysis regardless of hash."""
        with self._lock:
            return self._latest_analysis

    def clear(self) -> None:
        """Wipes the cache."""
        with self._lock:
            self._cache.clear()
            self._latest_analysis = None

    def count(self) -> int:
        """Returns active cached item count."""
        now = time.time()
        with self._lock:
            return sum(1 for e in self._cache.values() if e["expires_at"] > now)


# Global cache instance
ai_cache = AIAnalysisCache(default_ttl_seconds=300)
