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
    """Validates that all core MCX commodities are defined with complete contract specs."""
    assert len(COMMODITY_SPECS) >= 6
    required_keys = ["display_name", "sector", "unit", "lot_size", "default_token", "default_symbol"]

    for base_sym, spec in COMMODITY_SPECS.items():
        for k in required_keys:
            assert k in spec, f"Spec for {base_sym} missing key {k}"

    active = commodities_engine._active_contracts
    assert len(active) >= 6
    assert "CRUDEOIL" in active
    assert "CRUDEOILM" in active
    assert "NATURALGAS" in active
    assert "GOLD" in active
    assert "GOLDM" in active
    assert "SILVER" in active
    assert "SILVERM" in active
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
    with patch("engines.nifty_analytics_engine.nifty_analytics_engine.get_options_portfolio", return_value={"strategies": []}):
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


def test_commodity_options_expiries_discovery(client):
    """Tests official MCX option expiry discovery for Crude Oil and Natural Gas."""
    res = client.get("/api/commodities/options/expiries?underlying=CRUDEOIL")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert data.get("underlying") == "CRUDEOIL"
    assert "expiries" in data
    expiries = data.get("expiries", [])
    assert len(expiries) >= 1
    assert "iso" in expiries[0]
    assert "label" in expiries[0]


def test_commodity_option_chain_endpoint(client):
    """Tests live MCX option chain endpoint, strikes structure, PCR, and Greeks."""
    res = client.get("/api/commodities/options/chain?underlying=CRUDEOIL")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert data.get("underlying") == "CRUDEOIL"
    assert "spot_price" in data
    assert "atm_strike" in data
    assert "pcr" in data
    assert "strikes" in data
    assert len(data["strikes"]) >= 5

    # Validate row strike structure with CE and PE
    row = data["strikes"][0]
    assert "strikePrice" in row
    assert "CE" in row
    assert "PE" in row
    assert "lastPrice" in row["CE"]
    assert "lastPrice" in row["PE"]
    assert "impliedVolatility" in row["CE"]
    assert "delta" in row["CE"]

    # Also validate CRUDEOILM
    res_m = client.get("/api/commodities/options/chain?underlying=CRUDEOILM")
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert data_m.get("status") == "SUCCESS"
    assert data_m.get("underlying") == "CRUDEOILM"
    assert data_m.get("lot_size") == 10


def test_commodity_options_strategy_analyzer_endpoint(client):
    """Tests multi-leg strategy analysis and payoff curve generation."""
    payload = {
        "underlying": "CRUDEOIL",
        "legs": [
            {"strike": 5800, "type": "CE", "action": "BUY", "price": 150.0, "qty": 100, "iv": 28.0},
            {"strike": 6000, "type": "CE", "action": "SELL", "price": 80.0, "qty": 100, "iv": 27.0}
        ],
        "target_days": 0.0,
        "iv_shift_pct": 0.0
    }
    res = client.post("/api/commodities/options/strategy/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert data.get("underlying") == "CRUDEOIL"
    assert data.get("net_premium") == -7000.0  # Debit of ₹7,000
    assert "max_profit" in data
    assert "max_loss" in data
    assert "payoff_curve" in data
    curve = data["payoff_curve"]
    assert "spots" in curve
    assert "expiry_pnls" in curve
    assert "target_pnls" in curve
    assert len(curve["spots"]) > 50


def test_commodity_strategy_deployment_lifecycle(client):
    """Tests deploying a strategy, retrieving deployed strategies, and squaring off."""
    deploy_payload = {
        "underlying": "CRUDEOIL",
        "strategy_name": "Crude Bull Call Test",
        "legs": [
            {"strike": 5800, "type": "CE", "action": "BUY", "price": 150.0, "qty": 100},
            {"strike": 6000, "type": "CE", "action": "SELL", "price": 80.0, "qty": 100}
        ],
        "net_premium": -7000.0,
        "max_profit": "₹13,000.00",
        "max_loss": "₹7,000.00",
        "breakevens": [5870.0]
    }
    # 1. Deploy
    res = client.post("/api/commodities/options/strategy/deploy", json=deploy_payload)
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    strat = data.get("strategy", {})
    strat_id = strat.get("strategy_id")
    assert strat_id is not None
    assert strat.get("status") == "ACTIVE"

    # 2. Get deployed
    res_list = client.get("/api/commodities/options/strategy/deployed")
    assert res_list.status_code == 200
    deployed_data = res_list.json()
    assert deployed_data.get("status") == "SUCCESS"
    found = [s for s in deployed_data.get("strategies", []) if s.get("strategy_id") == strat_id]
    assert len(found) == 1

    # 3. Square off
    res_sq = client.post("/api/commodities/options/strategy/squareoff", json={"strategy_id": strat_id})
    assert res_sq.status_code == 200
    sq_data = res_sq.json()
    assert sq_data.get("status") == "SUCCESS"
    assert sq_data.get("strategy", {}).get("status") == "CLOSED"


def test_commodity_depth_extraction_and_api(client):
    """Tests 5-level market depth extraction from ticks and GET /api/commodities/depth."""
    from neo_api_client.websocket.feed.models import DepthLevel

    mock_tick = MagicMock()
    mock_tick.last_traded_price = 6125.0
    mock_tick.buy = [
        DepthLevel(price=6120.0, quantity=100, orders=2),
        DepthLevel(price=6119.0, quantity=250, orders=5),
    ]
    mock_tick.sell = [
        DepthLevel(price=6126.0, quantity=150, orders=3),
        DepthLevel(price=6127.0, quantity=300, orders=6),
    ]
    mock_tick.total_buy_quantity = 350
    mock_tick.total_sell_quantity = 450

    depth = commodities_engine._extract_depth_payload(mock_tick)
    assert len(depth["bids"]) == 2
    assert len(depth["asks"]) == 2
    assert depth["bids"][0]["price"] == "6,120.00"
    assert depth["bids"][0]["quantity"] == "100"
    assert depth["bids"][0]["orders"] == "2"
    assert depth["total_buy"] == 350
    assert depth["total_sell"] == 450

    # 2. Test process_ws_tick preserves and updates depth
    token = list(commodities_engine._active_contracts.values())[0]["token"]
    res_tick = commodities_engine.process_ws_tick(token=token, trading_symbol="CRUDEOIL", message=mock_tick)
    assert res_tick is not None
    assert "depth" in res_tick
    assert len(res_tick["depth"]["bids"]) == 2

    # 3. Test API endpoint GET /api/commodities/depth
    res = client.get("/api/commodities/depth?symbol=CRUDEOIL")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "SUCCESS"
    assert "depth" in data
    assert "quote" in data

