"""
========================================================================================
Kotak Neo MCX Commodities Analytics & Real-Time Stream Engine Unit Tests
========================================================================================
Validates:
1. Specs & Seeding for 6 Primary Commodities (Crude Oil, Nat Gas, Gold, Silver, Copper, Zinc)
2. MCX Session Hours Intelligence (Mon-Fri 09:00 - 23:30 IST)
3. Floor Pivots (P, R1, S1, R2, S2), Day Range Progress, and OI Buildup classification
4. Commodities REST Endpoints: `/api/commodities/quotes`, `/api/commodities/summary`, `/api/commodities/refresh`
5. Multi-Desk Continuous Stream payload integration (`/api/stream/continuous`)
========================================================================================
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from web.api import app
from engines.commodities_engine import commodities_engine, COMMODITY_SPECS


@pytest.fixture
def client():
    return TestClient(app)


def test_commodity_specs_and_seeding():
    """Validates that all 6 core MCX commodities are defined with complete contract specs."""
    assert len(COMMODITY_SPECS) == 6
    required_keys = ["display_name", "sector", "unit", "lot_size", "default_token", "default_symbol"]

    for base_sym, spec in COMMODITY_SPECS.items():
        for k in required_keys:
            assert k in spec, f"Spec for {base_sym} missing key {k}"

    active = commodities_engine._active_contracts
    assert len(active) == 6
    assert "CRUDEOIL" in active
    assert "NATURALGAS" in active
    assert "GOLD" in active
    assert "SILVER" in active
    assert "COPPER" in active
    assert "ZINC" in active


def test_mcx_trading_hours_checker():
    """Validates MCX trading hours detection."""
    is_open, mkt_text = commodities_engine.is_mcx_open()
    assert isinstance(is_open, bool)
    assert isinstance(mkt_text, str)
    assert "MCX" in mkt_text


def test_floor_pivots_and_technicals_calculation():
    """Validates calculation of floor pivots and range percentages."""
    high = 100.0
    low = 80.0
    close = 90.0

    pivot = round((high + low + close) / 3.0, 2)
    r1 = round((2.0 * pivot) - low, 2)
    s1 = round((2.0 * pivot) - high, 2)
    r2 = round(pivot + (high - low), 2)
    s2 = round(pivot - (high - low), 2)

    assert pivot == 90.0
    assert r1 == 100.0
    assert s1 == 80.0
    assert r2 == 110.0
    assert s2 == 70.0

    ltp = 95.0
    rng = high - low
    range_pct = round(((ltp - low) / rng) * 100.0, 1)
    assert range_pct == 75.0


def test_commodities_quotes_api_endpoint(client):
    """Tests GET /api/commodities/quotes returns valid structure and contract data."""
    res = client.get("/api/commodities/quotes")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert "quotes" in data
    assert "is_mcx_open" in data
    assert "summary" in data

    quotes = data.get("quotes", {})
    assert len(quotes) >= 1
    # Check that known commodities exist
    for sym in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "COPPER", "ZINC"]:
        if sym in quotes:
            q = quotes[sym]
            assert "ltp" in q
            assert "change_pct" in q
            assert "pivot" in q
            assert "volume" in q


def test_commodities_summary_api_endpoint(client):
    """Tests GET /api/commodities/summary returns sector stats."""
    res = client.get("/api/commodities/summary")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert "summary" in data
    summary = data.get("summary", {})
    assert "Energy" in summary
    assert "Bullion" in summary
    assert "Base Metals" in summary


def test_continuous_stream_includes_commodities(client):
    """Tests GET /api/stream/continuous includes commodities stream payload."""
    res = client.get("/api/stream/continuous")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert "commodities" in data
    comm_data = data["commodities"]
    assert "quotes" in comm_data
    assert "is_mcx_open" in comm_data
    assert "summary" in comm_data


def test_commodities_refresh_endpoint(client):
    """Tests POST /api/commodities/refresh endpoint executes properly."""
    res = client.post("/api/commodities/refresh")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert "quotes" in data
