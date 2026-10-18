"""
Provider-independent adapter for Groq and OpenRouter APIs.
Implements secure HTTP invocation, timeout, retry with backoff, and zero API-key exposure.
"""
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple
import httpx

from core import config
from ai_engine.model_registry import ModelRegistry

logger = logging.getLogger("ai_engine.provider")


class ProviderError(Exception):
    """Base exception for AI provider invocation errors."""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None, is_transient: bool = False):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.is_transient = is_transient


class AIProviderAdapter:
    """Unified client adapter for Groq and OpenRouter OpenAI-compatible endpoints."""

    def __init__(self, timeout_seconds: Optional[int] = None, max_retries: Optional[int] = None):
        self.timeout = timeout_seconds or config.AI_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else config.AI_MAX_RETRIES

    def _get_provider_credentials(self, provider: str) -> Tuple[str, str]:
        """Resolves API key and base URL safely without exposing secrets in logs."""
        prov = provider.lower().strip()
        if prov == "groq":
            api_key = config.GROQ_API_KEY.strip()
            base_url = config.GROQ_BASE_URL.rstrip("/")
            return api_key, base_url
        elif prov == "openrouter":
            api_key = config.OPENROUTER_API_KEY.strip()
            base_url = config.OPENROUTER_BASE_URL.rstrip("/")
            return api_key, base_url
        else:
            raise ProviderError(f"Unsupported provider '{provider}'", provider=provider, is_transient=False)

    def is_provider_configured(self, provider: str) -> bool:
        """Returns True if the provider has a non-empty API key configured."""
        try:
            key, _ = self._get_provider_credentials(provider)
            return bool(key)
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        enforce_json: bool = True
    ) -> Dict[str, Any]:
        """
        Sends an inference request to the configured provider using OpenAI-compatible Chat Completions.
        Returns a dict containing:
          - 'content': raw response text string
          - 'latency_ms': response duration in ms
          - 'model': model used
          - 'provider': provider used
          - 'usage': token usage stats (if available)
        """
        provider = ModelRegistry.resolve_provider_for_model(model_id)
        api_key, base_url = self._get_provider_credentials(provider)

        if not api_key:
            raise ProviderError(
                f"Missing API key for provider '{provider}'. Configure {provider.upper()}_API_KEY in environment.",
                provider=provider,
                status_code=401,
                is_transient=False
            )

        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NSE-TradingDesk-AI/1.0",
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/satishreddy07/nse-tradingdesk"
            headers["X-Title"] = "NSE Trading Desk AI Layer"

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if enforce_json:
            payload["response_format"] = {"type": "json_object"}

        attempts = 0
        last_error = None

        while attempts <= self.max_retries:
            attempts += 1
            start_t = time.perf_counter()
            try:
                with httpx.Client(timeout=float(self.timeout)) as client:
                    resp = client.post(endpoint, json=payload, headers=headers)

                latency_ms = round((time.perf_counter() - start_t) * 1000.0, 1)

                # Check if model rejects json_object response_format
                if resp.status_code == 400 and enforce_json and "response_format" in resp.text.lower():
                    logger.warning(f"Model {model_id} rejected response_format, retrying without explicit json_object")
                    payload.pop("response_format", None)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise ProviderError("Empty choices received from provider", provider=provider, is_transient=True)

                    msg_content = choices[0].get("message", {}).get("content", "").strip()
                    usage = data.get("usage") or {}

                    return {
                        "content": msg_content,
                        "latency_ms": latency_ms,
                        "model": model_id,
                        "provider": provider,
                        "usage": usage,
                    }

                # Handle HTTP errors
                status = resp.status_code
                is_transient = status in (429, 500, 502, 503, 504)
                err_msg = f"HTTP {status} from {provider}: {resp.text[:300]}"

                if status in (401, 403):
                    # Invalid credentials / unauthorized - never retry
                    raise ProviderError(f"Authentication failed for {provider} (HTTP {status})", provider=provider, status_code=status, is_transient=False)

                if status == 429:
                    # Rate limit
                    logger.warning(f"Rate limited by {provider} (HTTP 429). Attempt {attempts}/{self.max_retries + 1}")
                    if attempts <= self.max_retries:
                        time.sleep(1.5 * attempts)
                        continue
                    raise ProviderError(f"Rate limit exceeded on {provider} free tier", provider=provider, status_code=429, is_transient=True)

                if status == 404 or (status == 400 and ("decommission" in resp.text.lower() or "not found" in resp.text.lower() or "does not exist" in resp.text.lower())):
                    raise ProviderError(f"Model '{model_id}' unavailable or decommissioned on {provider} (HTTP {status}): {resp.text[:200]}", provider=provider, status_code=status, is_transient=False)

                last_error = ProviderError(err_msg, provider=provider, status_code=status, is_transient=is_transient)

                if not is_transient:
                    raise last_error

            except httpx.TimeoutException:
                latency_ms = round((time.perf_counter() - start_t) * 1000.0, 1)
                last_error = ProviderError(f"Request timeout after {self.timeout}s to {provider}", provider=provider, status_code=408, is_transient=True)
                logger.warning(f"Timeout calling {provider} with model {model_id} ({latency_ms}ms)")
            except httpx.RequestError as exc:
                last_error = ProviderError(f"Network error calling {provider}: {str(exc)}", provider=provider, is_transient=True)
                logger.warning(f"Network error calling {provider}: {str(exc)}")
            except ProviderError:
                raise
            except Exception as exc:
                last_error = ProviderError(f"Unexpected error: {str(exc)}", provider=provider, is_transient=False)
                raise last_error

            if attempts <= self.max_retries:
                time.sleep(1.0 * attempts)

        if last_error:
            raise last_error
        raise ProviderError(f"Failed to complete request to {provider}", provider=provider, is_transient=True)
