"""
========================================================================================
Route Security & Production Provenance Isolation Tests (Phase 1 Remediation)
========================================================================================
Validates that:
1. Every mutating route (POST, PUT, PATCH, DELETE) in the application is protected by verify_admin_key.
2. Sensitive operational telemetry and broker credentials endpoints are guarded.
3. In production mode, missing ADMIN_API_KEY fails closed with HTTP 500.
4. When ADMIN_API_KEY is configured, unauthenticated or invalid requests receive HTTP 401.
5. In production mode, /api/kotak/test-tick is permanently disabled with HTTP 403.
6. Test ticks are stored exclusively in the isolated test cache and cannot be read by production live-price accessors.
7. Production live-price accessors strictly reject MOCK_INJECTION, TEST_TOKEN, and non-finite values.
"""
import math
import pytest
from fastapi.testclient import TestClient

from web.api import app, verify_admin_key, is_production_environment
from engines.kotak_neo_engine import kotak_neo_engine, KotakNeoEngine
from execution.state_engine import state_engine
from core import config


@pytest.fixture
def client():
    return TestClient(app)


def test_all_mutating_routes_are_protected_by_security_gate():
    """
    Automated security audit: Enumerates all routes mounted on FastAPI.
    Every single state-changing endpoint (POST, PUT, PATCH, DELETE) MUST have
    verify_admin_key in its dependencies. Unprotected mutating routes fail CI immediately.
    """
    mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
    unprotected = []
    total_mutating = 0

    for route in app.routes:
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        intersect = set(methods).intersection(mutating_methods)
        if intersect:
            total_mutating += 1
            dep_names = [
                d.dependency.__name__ if hasattr(d, "dependency") and hasattr(d.dependency, "__name__") else str(d)
                for d in getattr(route, "dependencies", [])
            ]
            if "verify_admin_key" not in dep_names:
                unprotected.append(f"{list(intersect)} {route.path} -> {route.name} (deps: {dep_names})")

    assert total_mutating >= 60, f"Expected at least 60 mutating routes, found {total_mutating}"
    assert len(unprotected) == 0, (
        f"Security violation! Found {len(unprotected)} unprotected mutating routes:\n"
        + "\n".join(unprotected)
    )


def test_all_sensitive_telemetry_routes_are_protected():
    """
    Verifies that sensitive portfolio, broker, execution, and operational telemetry routes
    require administrative authentication.
    """
    sensitive_routes = {
        "/api/kotak/config",
        "/api/server-ip",
        "/api/vault/status",
        "/api/execution/queue",
        "/api/system/backup/list",
        "/api/system/activity-logs",
    }
    for route in app.routes:
        if route.path in sensitive_routes:
            dep_names = [
                d.dependency.__name__ if hasattr(d, "dependency") and hasattr(d.dependency, "__name__") else str(d)
                for d in getattr(route, "dependencies", [])
            ]
            assert "verify_admin_key" in dep_names, (
                f"Sensitive telemetry route {route.path} is not protected by verify_admin_key"
            )


def test_production_environment_fails_closed_when_admin_key_missing(monkeypatch, client):
    """
    In production mode (ENVIRONMENT=production), if ADMIN_API_KEY is not set or empty,
    mutating requests MUST fail closed with HTTP 500 error (not 200 or 401).
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    # Attempt a mutating call
    res = client.post("/api/calendar/sync-holidays")
    assert res.status_code == 500
    assert "ADMIN_API_KEY must be configured in production" in res.json().get("detail", "")


def test_unauthorized_when_admin_key_configured_but_missing_or_invalid(monkeypatch, client):
    """
    When ADMIN_API_KEY is configured, requests without key or with wrong key fail with 401.
    """
    monkeypatch.setenv("ADMIN_API_KEY", "super_secret_quant_key_999")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    # 1. No auth headers provided
    res1 = client.post("/api/calendar/sync-holidays")
    assert res1.status_code == 401
    assert "Unauthorized" in res1.json().get("detail", "")

    # 2. Invalid X-API-Key
    res2 = client.post("/api/calendar/sync-holidays", headers={"X-API-Key": "wrong_key"})
    assert res2.status_code == 401

    # 3. Invalid Authorization Bearer
    res3 = client.post("/api/calendar/sync-holidays", headers={"Authorization": "Bearer wrong_token"})
    assert res3.status_code == 401


def test_authorized_access_with_valid_credentials(monkeypatch, client):
    """
    When ADMIN_API_KEY is configured and valid token is supplied, request passes gate.
    """
    monkeypatch.setenv("ADMIN_API_KEY", "super_secret_quant_key_999")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    # 1. Valid via X-API-Key
    res1 = client.post("/api/calendar/sync-holidays", headers={"X-API-Key": "super_secret_quant_key_999"})
    assert res1.status_code == 200

    # 2. Valid via Authorization: Bearer
    res2 = client.post("/api/calendar/sync-holidays", headers={"Authorization": "Bearer super_secret_quant_key_999"})
    assert res2.status_code == 200


def test_kotak_config_endpoint_masks_credentials(monkeypatch, client):
    """
    Verifies that /api/kotak/config masks sensitive credentials so raw keys are never leaked.
    """
    monkeypatch.setenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setattr(config, "KOTAK_NEO_CONSUMER_KEY", "SECRET_CONSUMER_123456789")
    monkeypatch.setattr(config, "KOTAK_NEO_UCC", "UCC987654321")
    monkeypatch.setattr(config, "KOTAK_NEO_MOBILE_NUMBER", "9876543210")
    monkeypatch.setattr(state_engine, "get_kv", lambda key: {})

    res = client.get("/api/kotak/config", headers={"X-API-Key": "test_key"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    # Secrets must be masked (showing only last 4 chars)
    assert data["consumer_key"].endswith("6789")
    assert data["consumer_key"].startswith("*")
    assert "SECRET_CONSUMER" not in data["consumer_key"]
    assert data["ucc"].endswith("4321")
    assert data["ucc"].startswith("*")
    assert "UCC987" not in data["ucc"]
    assert data["mobile_number"].endswith("3210")
    assert data["mobile_number"].startswith("*")
    assert "987654" not in data["mobile_number"]


def test_kotak_status_endpoint_masks_ucc(monkeypatch):
    """
    Verifies that KotakNeoEngine.get_status() masks saved_ucc.
    """
    monkeypatch.setattr(config, "KOTAK_NEO_UCC", "MY_CONFIDENTIAL_UCC_1234")
    monkeypatch.setattr(state_engine, "get_kv", lambda key: {})
    status = kotak_neo_engine.get_status()
    saved_ucc = status.get("saved_ucc", "")
    assert "CONFIDENTIAL" not in saved_ucc
    assert saved_ucc.endswith("1234")
    assert saved_ucc.startswith("*")


def test_test_tick_permanently_disabled_in_production(monkeypatch, client):
    """
    Verifies that /api/kotak/test-tick is blocked with HTTP 403 in production.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ADMIN_API_KEY", "prod_key")

    res = client.post(
        "/api/kotak/test-tick?symbol=RELIANCE&ltp=3000.0",
        headers={"X-API-Key": "prod_key"}
    )
    assert res.status_code == 403
    assert "permanently disabled in production" in res.json().get("detail", "")


def test_test_tick_cache_isolation_and_live_price_rejection():
    """
    Verifies that injected test ticks are placed strictly in _test_ticks
    and CANNOT be read by production live-price accessors.
    """
    kotak_neo_engine.clear_test_ticks()
    sym = "TEST_ISOLATION_SYM"
    kotak_neo_engine.inject_test_tick(symbol=sym, ltp=2550.0)

    # 1. Accessible via test-only accessors
    assert kotak_neo_engine.get_test_price(sym) == 2550.0
    test_tick = kotak_neo_engine.get_test_tick(sym)
    assert test_tick is not None
    assert test_tick["ltp"] == 2550.0
    assert test_tick["source"] == "MOCK_INJECTION"
    assert test_tick["token"] == "TEST_TOKEN"
    assert sym in kotak_neo_engine.get_all_test_ticks()

    # 2. Strictly REJECTED by production accessors
    assert kotak_neo_engine.get_live_price(sym) is None
    assert kotak_neo_engine.get_live_tick(sym) is None
    assert sym not in kotak_neo_engine.get_all_ticks()


def test_live_price_accessor_rejects_mock_sources_and_nonfinite():
    """
    Verifies KotakNeoEngine._is_valid_live_tick validation matrix.
    """
    valid_tick = {
        "symbol": "TCS",
        "token": "11536",
        "ltp": 4150.0,
        "source": "KOTAK_NEO_WS"
    }
    assert KotakNeoEngine._is_valid_live_tick(valid_tick) is True

    # Rejection of mock/simulated sources
    for bad_source in ("MOCK_INJECTION", "SIMULATED", "TEST", "TEST_ONLY"):
        t = dict(valid_tick, source=bad_source)
        assert KotakNeoEngine._is_valid_live_tick(t) is False, f"Failed to reject source={bad_source}"

    # Rejection of test token
    assert KotakNeoEngine._is_valid_live_tick(dict(valid_tick, token="TEST_TOKEN")) is False

    # Rejection of non-positive or non-finite ltp
    for bad_ltp in (None, 0, 0.0, -100.0, float("nan"), float("inf"), float("-inf"), "invalid"):
        t = dict(valid_tick, ltp=bad_ltp)
        assert KotakNeoEngine._is_valid_live_tick(t) is False, f"Failed to reject ltp={bad_ltp}"


def test_cors_configuration_not_wildcard_with_credentials():
    """
    Verifies CORS configuration does not mix allow_credentials=True with wildcard origin ['*'].
    """
    from fastapi.middleware.cors import CORSMiddleware
    cors_mw = None
    for mw in app.user_middleware:
        if mw.cls == CORSMiddleware:
            cors_mw = mw
            break

    assert cors_mw is not None, "CORSMiddleware not registered on app"
    options = cors_mw.kwargs
    allow_origins = options.get("allow_origins", [])
    allow_credentials = options.get("allow_credentials", False)

    if allow_credentials:
        assert "*" not in allow_origins, "Security error: allow_origins contains '*' while allow_credentials is True"
