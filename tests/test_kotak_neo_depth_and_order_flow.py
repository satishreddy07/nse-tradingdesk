import pytest
from fastapi.testclient import TestClient
from web.api import app
from engines.kotak_neo_engine import kotak_neo_engine
from engines.intraday_analytics_engine import intraday_analytics_engine

client = TestClient(app)


def test_extract_market_depth_payload():
    raw_depth = {
        "buy": [
            {"price": 100.5, "quantity": 1500, "orders": 12},
            {"price": 100.0, "quantity": 2500, "orders": 20},
            {"price": 99.5, "quantity": 1000, "orders": 8},
            {"price": 99.0, "quantity": 500, "orders": 4},
            {"price": 98.5, "quantity": 250, "orders": 2},
        ],
        "sell": [
            {"price": 101.0, "quantity": 1000, "orders": 10},
            {"price": 101.5, "quantity": 1200, "orders": 15},
            {"price": 102.0, "quantity": 800, "orders": 6},
            {"price": 102.5, "quantity": 600, "orders": 5},
            {"price": 103.0, "quantity": 400, "orders": 3},
        ]
    }
    parsed = kotak_neo_engine._extract_market_depth_payload(raw_depth)
    assert parsed["total_buy_quantity"] == 5750
    assert parsed["total_sell_quantity"] == 4000
    assert parsed["spread"] == 0.5
    assert parsed["spread_pct"] == pytest.approx(0.5 / 100.75 * 100, 0.01)
    assert parsed["obi"] == pytest.approx((5750 - 4000) / (5750 + 4000), 0.001)
    assert parsed["flow_bias"] == "MODERATE_BUYING"
    assert len(parsed["bid_levels"]) == 5
    assert len(parsed["ask_levels"]) == 5


def test_order_flow_market_depth_computation():
    mock_chain = {
        "spot_price": 22800.0,
        "expiry": "27-Oct-2026",
        "strikes": [
            {
                "strike": 22800,
                "ce_ltp": 120.0,
                "pe_ltp": 115.0,
                "has_depth": True,
                "ce_total_buy": 10000,
                "ce_total_sell": 6000,
                "ce_depth_obi": 0.25,
                "ce_micro_price": 120.3,
                "ce_spread": 0.5,
                "pe_total_buy": 5000,
                "pe_total_sell": 9000,
                "pe_depth_obi": -0.28,
                "pe_micro_price": 114.8,
                "pe_spread": 0.6,
            }
        ]
    }
    res = intraday_analytics_engine.compute_order_flow_market_depth(
        spot_price=22800.0,
        weekly_chain_data=mock_chain,
        force_refresh=True
    )
    assert res["status"] in ("OK", "DATA_NOT_AVAILABLE")
    if res["status"] == "OK":
        assert res["atm_strike"] == 22800
        assert "net_flow_index" in res
        assert "flow_regime" in res
        assert "aggregate_call_obi" in res
        assert "aggregate_put_obi" in res
        assert "call_put_depth_ratio" in res
        assert "avg_spread_pts" in res
        assert "liquidity_status" in res


def test_api_options_strike_depth_endpoint():
    res = client.get("/api/nifty/options/strike-depth?strike=22800&option_type=CE&expiry_type=weekly")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["strike"] == 22800
    assert data["opt_type"] == "CE"
    assert data["expiry_type"] == "weekly"


def test_api_intraday_order_flow_endpoint():
    res = client.get("/api/nifty/intraday/order-flow")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "near_atm_strikes" in data
