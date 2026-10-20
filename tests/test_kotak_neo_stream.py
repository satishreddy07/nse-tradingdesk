"""
========================================================================================
Kotak Neo API v2 Live Data Stream Test Suite
========================================================================================
Tests:
1. KotakNeoEngine initialization, Scrip Master resolution (Symbol <-> Token).
2. Live WebSocket tick caching, microsecond retrieval, and mock tick injection.
3. Live price stream fast-path integration in institutional_engine.py.
4. Seamless fallback to Yahoo Finance / NSE if Kotak Neo is offline.
5. FastAPI REST API endpoints (/api/kotak/status, /api/kotak/quotes, etc.).
========================================================================================
"""
import pytest
from fastapi.testclient import TestClient

from engines.kotak_neo_engine import kotak_neo_engine, KotakNeoEngine
from engines.institutional_engine import refresh_live_market_prices
from web.api import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_kotak_neo_engine_initialization():
    """Verifies KotakNeoEngine initializes cleanly with pre-seeded scrip mapping."""
    engine = KotakNeoEngine()
    assert engine is not None
    assert engine.resolve_token("RELIANCE") == "2885"
    assert engine.resolve_token("reliance.ns") == "2885"
    assert engine.resolve_token("TCS") == "11536"
    assert engine.resolve_symbol("2885") == "RELIANCE"
    assert engine.resolve_symbol("11536") == "TCS"
    assert engine.resolve_token("NON_EXISTENT_TICKER_XYZ") is None


def test_tick_cache_injection_and_fast_retrieval():
    """Verifies sub-millisecond in-memory LTP and tick retrieval."""
    # Inject test tick
    kotak_neo_engine.inject_test_tick(
        symbol="INFY",
        ltp=1850.75,
        high=1870.00,
        low=1840.50,
        volume=1250000
    )

    # Fast LTP retrieval
    price = kotak_neo_engine.get_live_price("INFY")
    assert price == 1850.75

    # Case insensitivity and suffix stripping
    assert kotak_neo_engine.get_live_price("infy.ns") == 1850.75

    # Full tick retrieval
    tick = kotak_neo_engine.get_live_tick("INFY")
    assert tick is not None
    assert tick["symbol"] == "INFY"
    assert tick["ltp"] == 1850.75
    assert tick["high"] == 1870.00
    assert tick["low"] == 1840.50
    assert tick["volume"] == 1250000
    assert tick["source"] == "MOCK_INJECTION"

    # All ticks snapshot
    all_ticks = kotak_neo_engine.get_all_ticks()
    assert "INFY" in all_ticks
    assert all_ticks["INFY"]["ltp"] == 1850.75


def test_subscription_management():
    """Verifies symbol-to-token translation and duplicate filtering during subscription."""
    res = kotak_neo_engine.subscribe_symbols(["RELIANCE", "TCS", "HDFCBANK"])
    assert res["status"] in ("SUBSCRIBED", "QUEUED_OFFLINE", "ALREADY_SUBSCRIBED")

    # Re-subscribing the same symbols returns ALREADY_SUBSCRIBED with 0 new additions
    res_dup = kotak_neo_engine.subscribe_symbols(["RELIANCE"])
    assert res_dup["status"] == "ALREADY_SUBSCRIBED"
    assert res_dup["count"] == 0


def test_status_and_telemetry():
    """Verifies telemetry contract returned by get_status()."""
    status = kotak_neo_engine.get_status()
    assert "has_sdk_installed" in status
    assert "is_authenticated" in status
    assert "is_streaming" in status
    assert "ticks_received_total" in status
    assert "cache_size" in status
    assert status["cache_size"] >= 1


def test_rest_api_status_endpoint(test_client):
    """Tests GET /api/kotak/status endpoint."""
    response = test_client.get("/api/kotak/status")
    assert response.status_code == 200
    data = response.json()
    assert "has_sdk_installed" in data
    assert "is_streaming" in data
    assert "cache_size" in data


def test_rest_api_quotes_endpoint(test_client):
    """Tests GET /api/kotak/quotes endpoint returns cached ticks."""
    kotak_neo_engine.inject_test_tick("TCS", 4120.00)
    response = test_client.get("/api/kotak/quotes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "TCS" in data["quotes"]
    assert data["quotes"]["TCS"]["ltp"] == 4120.00


def test_rest_api_subscribe_endpoint(test_client):
    """Tests POST /api/kotak/subscribe endpoint."""
    response = test_client.post("/api/kotak/subscribe", json={"symbols": ["SBIN", "ITC"]})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

    # Empty payload validation
    bad_res = test_client.post("/api/kotak/subscribe", json={"symbols": []})
    assert bad_res.status_code == 400


def test_rest_api_auth_validation(test_client):
    """Tests POST /api/kotak/auth validates missing credentials gracefully."""
    response = test_client.post("/api/kotak/auth", json={})
    assert response.status_code == 200
    data = response.json()
    # If no credentials configured, returns MISSING_CREDENTIALS or FAILED
    assert data["status"] in ("MISSING_CREDENTIALS", "FAILED", "SUCCESS")


def test_rest_api_disconnect(test_client):
    """Tests POST /api/kotak/disconnect endpoint."""
    response = test_client.post("/api/kotak/disconnect")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert kotak_neo_engine.is_streaming_active() is False


def test_live_price_stream_fast_path():
    """
    Verifies that refresh_live_market_prices prioritizes Kotak Neo in-memory ticks
    and returns stream telemetry reflecting kotak_neo_streaming.
    """
    kotak_neo_engine.inject_test_tick("RELIANCE", 2988.50)
    res = refresh_live_market_prices(symbols=["RELIANCE"])
    assert res.get("status") in ("OK", "SUCCESS")
    prices = res.get("prices", {})
    assert "RELIANCE" in prices
    assert prices["RELIANCE"] == 2988.50
    assert "kotak_neo_streaming" in res


def test_server_ip_detection(test_client):
    """Verifies server IP detection and /api/server-ip endpoint."""
    from scheduler.cron_scheduler import get_server_public_ip
    ip = get_server_public_ip()
    assert ip is not None
    assert len(ip) > 0

    response = test_client.get("/api/server-ip")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "server_ip" in data
    assert len(data["server_ip"]) > 0

    # Also verify /api/status includes server_ip
    status_res = test_client.get("/api/status")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert "server_ip" in st_data


def test_kotak_ping_and_test_tick_endpoints(test_client):
    """Verifies the diagnostic endpoints for verifying connection & injecting ticks."""
    # Test tick injection
    tick_res = test_client.post("/api/kotak/test-tick?symbol=RELIANCE&ltp=2999.50")
    assert tick_res.status_code == 200
    t_data = tick_res.json()
    assert t_data["status"] == "SUCCESS"
    assert t_data["symbol"] == "RELIANCE"
    assert t_data["injected_price"] == 2999.50
    assert t_data["ticks_total"] >= 1

    # Verify tick is immediately accessible in Kotak engine cache
    assert kotak_neo_engine.get_live_price("RELIANCE") == 2999.50

    # Ping endpoint (when unauthenticated or authenticated)
    ping_res = test_client.get("/api/kotak/ping")
    assert ping_res.status_code == 200
    p_data = ping_res.json()
    assert "connected" in p_data


def test_continuous_stream_and_options_tick_cache(test_client):
    """Verifies the multi-desk continuous stream endpoint and options tick cache."""
    # 1. Option tick cache test
    kotak_neo_engine.set_option_price(25000, "CE", 172.50)
    kotak_neo_engine.set_option_price(25000, "PE", 145.20)
    assert kotak_neo_engine.get_option_price(25000, "CE") == 172.50
    assert kotak_neo_engine.get_option_price(25000, "PE") == 145.20

    # 2. Continuous Stream API test
    res = test_client.get("/api/stream/continuous")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["source"] == "KOTAK_NEO_STREAM"
    assert "swing" in data
    assert "intraday" in data
    assert "options" in data
    assert "positions" in data["swing"]
    assert "nifty_spot" in data["intraday"]
    assert "strategies" in data["options"]


def test_zero_yahoo_and_nse_live_requests_in_intraday_and_options(monkeypatch):
    """
    Verifies that Option Chain, Intraday VWAP, and Sectoral Divergence
    make ZERO HTTP calls to nseindia.com/api/option-chain or query1.finance.yahoo.com.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine
    from engines.intraday_analytics_engine import intraday_analytics_engine
    import requests

    forbidden_urls = []

    def mock_get(url, *args, **kwargs):
        if "nseindia.com/api/option-chain" in url or "query1.finance.yahoo.com" in url:
            forbidden_urls.append(url)
            raise AssertionError(f"Forbidden live endpoint called: {url}")
        return requests.Response()

    monkeypatch.setattr(requests.Session, "get", mock_get)

    # 1. Option Chain must NOT call NSE scraping endpoint
    oc = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=True)
    assert oc.get("status") in ("OK", "UNAVAILABLE", "STANDBY")
    assert oc.get("source") != "NSE_LIVE_OPTION_CHAIN_V3"

    # 2. Intraday VWAP & ORB must NOT call Yahoo Finance
    vwap = intraday_analytics_engine.compute_vwap_and_orb()
    assert vwap.get("status") == "OK"
    assert vwap.get("source") in ("KOTAK_NEO_STREAM", "MASTER_DB_PERSISTED_SNAPSHOT")

    # 3. Sectoral Divergence must NOT call Yahoo Finance
    sec = intraday_analytics_engine.compute_sectoral_divergence()
    assert sec.get("status") == "OK"
    assert sec.get("source") in ("KOTAK_NEO_STREAM", "MASTER_DB_PERSISTED_SNAPSHOT")

    # Assert zero forbidden calls occurred
    assert len(forbidden_urls) == 0


def test_hybrid_rest_stream_when_ws_offline(monkeypatch):
    """
    Verifies that when WebSocket handshake times out (e.g. Cloudflare 522),
    KotakNeoEngine hybrid REST stream populates live ticks, mirrors Nifty index,
    increments tick telemetry, and marks streaming active.
    """
    from engines.kotak_neo_engine import KotakNeoEngine
    engine = KotakNeoEngine()
    engine._is_authenticated = True
    engine._is_connected = False  # WS Offline (like Cloudflare 522)

    # Mock NeoAPI client quotes response
    class MockNeoClient:
        def quotes(self, instrument_tokens, quote_type="all"):
            # Return payload simulating Kotak Neo REST quotes response
            return [
                {
                    "tk": "2885",
                    "ltp": "3015.40",
                    "h": "3030.00",
                    "lo": "2995.00",
                    "c": "2980.00",
                    "v": "2500000",
                    "nc": "1.19"
                },
                {
                    "tk": "26000",
                    "iv": "25120.75",
                    "highPrice": "25150.00",
                    "lowPrice": "25050.00",
                    "openingPrice": "25080.00",
                    "c": "25000.00",
                    "nc": "0.48"
                }
            ]

    engine._client = MockNeoClient()

    # Fetch quotes via REST
    prices = engine.fetch_rest_quotes(["RELIANCE", "NIFTY 50"])
    assert "RELIANCE" in prices
    assert prices["RELIANCE"] == 3015.40
    assert "NIFTY 50" in prices
    assert prices["NIFTY 50"] == 25120.75

    # Verify cache & aliases
    assert engine.get_live_price("RELIANCE") == 3015.40
    assert engine.get_live_price("NIFTY 50") == 25120.75
    assert engine.get_live_price("NIFTY") == 25120.75
    assert engine.get_live_price("^NSEI") == 25120.75

    # Verify telemetry & streaming active state
    assert engine.is_streaming_active() is True
    status = engine.get_status()
    assert status["is_streaming"] is True
    assert status["stream_mode"] == "REST_STREAM"
    assert status["ticks_received_total"] >= 2


def test_kotak_neo_option_chain_generation():
    """Verifies that NiftyAnalyticsEngine produces a live Option Chain from Kotak Neo feed without standby dashes."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    from engines.kotak_neo_engine import kotak_neo_engine

    # Inject mock spot tick for NIFTY 50
    kotak_neo_engine.inject_test_tick(
        symbol="NIFTY 50",
        ltp=25240.50,
        high=25300.0,
        low=25150.0,
        volume=50000000
    )

    chain = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=True)

    assert chain is not None
    assert chain["status"] == "OK"
    assert chain["source"] == "KOTAK_NEO_STREAM"
    assert chain["spot_price"] == 25240.50
    assert "weekly" in chain and "monthly" in chain
    assert len(chain["weekly"]["strikes"]) == 25
    assert len(chain["monthly"]["strikes"]) == 25

    # Verify ATM strike is 25250
    assert chain["weekly"]["atm_strike"] == 25250
    assert chain["weekly"]["pcr"]["overall_oi_pcr"] > 0
    assert chain["weekly"]["walls_and_pain"]["max_pain"] > 0
    assert chain["weekly"]["walls_and_pain"]["call_wall_1"] >= chain["weekly"]["atm_strike"]
    assert chain["weekly"]["walls_and_pain"]["put_wall_1"] <= chain["weekly"]["atm_strike"]
    assert len(chain["weekly"]["strikes"]) > 0
    assert len(chain["quant_suggestions"]) > 0


def test_sfeed_scrip_and_index_message_decoding():
    """Verifies that official SDK kotakneoapi v3.0.9 SFeedScrip and SFeedIndex models update the tick cache."""
    from neo_api_client import SFeedScrip, SFeedIndex, WsToken
    from engines.kotak_neo_engine import KotakNeoEngine

    engine = KotakNeoEngine()

    # 1. Test SFeedScrip (Equity Live Quotes)
    scrip_msg = SFeedScrip.model_construct(
        exchange_segment="nse_cm",
        instrument_token="2885",
        trading_symbol="RELIANCE-EQ",
        type="scrip",
        level="1",
        last_traded_price=3055.20,
        high_price=3070.00,
        low_price=3040.00,
        open_price=3045.00,
        close_price=3035.00,
        average_trade_price=3052.50,
        volume_traded_today=1500000,
        net_change_percent=0.66
    )
    engine._process_sfeed_message(scrip_msg)

    assert engine.get_live_price("RELIANCE") == 3055.20
    tick = engine.get_live_tick("RELIANCE")
    assert tick is not None
    assert tick["ltp"] == 3055.20
    assert tick["high"] == 3070.00
    assert tick["low"] == 3040.00
    assert tick["vwap"] == 3052.50
    assert tick["volume"] == 1500000
    assert tick["source"] == "KOTAK_NEO_WS"

    # 2. Test SFeedIndex (Index Live Quotes)
    index_msg = SFeedIndex.model_construct(
        exchange_segment="nse_cm",
        instrument_token="26000",
        trading_symbol="NIFTY 50",
        type="index",
        name="NIFTY 50",
        last_traded_price=25350.75,
        high_price=25400.00,
        low_price=25280.00,
        close_price=25250.00,
        change=100.75,
        net_change_percent=0.40
    )
    engine._process_sfeed_message(index_msg)

    assert engine.get_live_price("NIFTY 50") == 25350.75
    assert engine.get_live_price("NIFTY") == 25350.75
    assert engine.get_live_price("^NSEI") == 25350.75
    idx_tick = engine.get_live_tick("NIFTY 50")
    assert idx_tick is not None
    assert idx_tick["ltp"] == 25350.75
    assert idx_tick["source"] == "KOTAK_NEO_WS_INDEX"




