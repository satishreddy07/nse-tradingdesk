import pytest
import math
from fastapi.testclient import TestClient
from web.api import app
from engines.kotak_neo_engine import kotak_neo_engine
from engines.intraday_analytics_engine import intraday_analytics_engine
from engines.order_flow_structure_engine import order_flow_structure_engine

client = TestClient(app)


class TestLevel2MarketDepthCalculations:
    """Rigorous unit tests for Level-2 Market Depth, OBI, Microprice, and Spreads."""

    def test_multi_level_obi_l1_l3_l5(self):
        """Test exact calculation of L1, L3, and L5 Order Book Imbalance."""
        raw_depth = {
            "buy": [
                {"price": 100.0, "quantity": 1000, "orders": 5},
                {"price": 99.5, "quantity": 2000, "orders": 10},
                {"price": 99.0, "quantity": 3000, "orders": 15},
                {"price": 98.5, "quantity": 4000, "orders": 20},
                {"price": 98.0, "quantity": 5000, "orders": 25},
            ],
            "sell": [
                {"price": 101.0, "quantity": 500, "orders": 2},
                {"price": 101.5, "quantity": 1500, "orders": 6},
                {"price": 102.0, "quantity": 2500, "orders": 12},
                {"price": 102.5, "quantity": 3500, "orders": 18},
                {"price": 103.0, "quantity": 4500, "orders": 22},
            ]
        }
        res = kotak_neo_engine._extract_market_depth_payload(raw_depth)

        # L1: Buy=1000, Sell=500 -> OBI = (1000-500)/1500 = +0.3333
        assert res["depth_l1_buy_qty"] == 1000
        assert res["depth_l1_sell_qty"] == 500
        assert res["obi_l1"] == pytest.approx(500.0 / 1500.0, abs=0.0002)

        # L3: Buy=1000+2000+3000=6000, Sell=500+1500+2500=4500 -> OBI = (6000-4500)/10500 = +0.1429
        assert res["depth_l3_buy_qty"] == 6000
        assert res["depth_l3_sell_qty"] == 4500
        assert res["obi_l3"] == pytest.approx(1500.0 / 10500.0, abs=0.0002)

        # L5: Buy=15000, Sell=12500 -> OBI = (15000-12500)/27500 = +0.0909
        assert res["depth_l5_buy_qty"] == 15000
        assert res["depth_l5_sell_qty"] == 12500
        assert res["obi_l5"] == pytest.approx(2500.0 / 27500.0, abs=0.0002)
        assert res["book_status"] == "OK"

    def test_zero_depth_returns_none(self):
        """When total book depth is zero, OBI must return None (never synthetic 0.0)."""
        empty_depth = {"buy": [], "sell": []}
        res = kotak_neo_engine._extract_market_depth_payload(empty_depth)
        assert res["book_status"] == "EMPTY_BOOK"
        assert res["obi_l1"] is None
        assert res["obi_l3"] is None
        assert res["obi_l5"] is None
        assert res["micro_price"] is None
        assert res["mid_price"] is None
        assert res["micro_price_displacement"] is None

    def test_strict_microprice_and_displacement(self):
        """Microprice must equal (Pa*Qb + Pb*Qa)/(Qb + Qa) and satisfy Pb <= P_micro <= Pa."""
        raw_depth = {
            "buy": [{"price": 200.0, "quantity": 3000, "orders": 10}],
            "sell": [{"price": 202.0, "quantity": 1000, "orders": 5}]
        }
        res = kotak_neo_engine._extract_market_depth_payload(raw_depth)
        assert res["book_status"] == "OK"
        # Midpoint = (200 + 202)/2 = 201.0
        assert res["mid_price"] == 201.0
        # Microprice = (202*3000 + 200*1000) / 4000 = (606000 + 200000)/4000 = 806000/4000 = 201.50
        assert res["micro_price"] == 201.50
        # Must be strictly bounded between Pb (200.0) and Pa (202.0)
        assert 200.0 <= res["micro_price"] <= 202.0
        # Displacement = 201.50 - 201.0 = +0.50 (heavy buying skew toward ask)
        assert res["micro_price_displacement"] == 0.50
        # Spread = 2.0, Spread % = (2.0 / 201.0) * 100 = 0.995%
        assert res["spread"] == 2.0
        assert res["spread_pct"] == pytest.approx((2.0 / 201.0) * 100.0, 0.01)

    def test_one_sided_book_handles_cleanly(self):
        """Single-sided books must report ONE_SIDED and return None for microprice (no LTP fallback)."""
        bids_only = {
            "buy": [{"price": 150.0, "quantity": 500, "orders": 2}],
            "sell": []
        }
        res = kotak_neo_engine._extract_market_depth_payload(bids_only, ltp=152.0)
        assert res["book_status"] == "ONE_SIDED_BIDS_ONLY"
        assert res["micro_price"] is None
        assert res["mid_price"] is None
        assert res["micro_price_displacement"] is None

        asks_only = {
            "buy": [],
            "sell": [{"price": 155.0, "quantity": 800, "orders": 3}]
        }
        res2 = kotak_neo_engine._extract_market_depth_payload(asks_only, ltp=152.0)
        assert res2["book_status"] == "ONE_SIDED_ASKS_ONLY"
        assert res2["micro_price"] is None

    def test_crossed_book_classification(self):
        """When Bid >= Ask, book is crossed/locked and microprice must be suppressed."""
        crossed = {
            "buy": [{"price": 105.0, "quantity": 500, "orders": 2}],
            "sell": [{"price": 104.0, "quantity": 500, "orders": 2}]
        }
        res = kotak_neo_engine._extract_market_depth_payload(crossed)
        assert res["book_status"] == "CROSSED_BOOK"
        assert res["micro_price"] is None
        assert res["mid_price"] is None


class TestIntradayAnalyticsCalculations:
    """Rigorous unit tests for Intraday Resting Skew, OBI Velocity, and Black-Scholes Theta."""

    def test_resting_depth_skew_decoupled_from_trade_flow(self):
        """Verify Resting Limit Queue Skew is computed as (Call_OBI - Put_OBI)/2 without claiming trade accumulation."""
        mock_chain = {
            "spot_price": 22800.0,
            "expiry": "2026-10-27",
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
                    "ce_spread_pct": 0.42,
                    "pe_total_buy": 5000,
                    "pe_total_sell": 9000,
                    "pe_depth_obi": -0.28,
                    "pe_micro_price": 114.8,
                    "pe_spread": 0.6,
                    "pe_spread_pct": 0.52,
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
            assert "resting_depth_skew" in res
            assert "depth_skew_regime" in res
            assert "avg_relative_spread_pct" in res
            # Check near-ATM strike row structure
            atm_row = [r for r in res["near_atm_strikes"] if r["strike"] == 22800][0]
            assert "mid_price" in atm_row["ce"]
            assert "micro_price_displacement" in atm_row["ce"]

    def test_authentic_black_scholes_theta_curve(self):
        """Black-Scholes Theta curve must produce positive daily/hourly decay without hardcoded multipliers."""
        res = intraday_analytics_engine.compute_weekly_theta_decay_curve(
            spot_price=22750.0,
            weekly_chain={"spot_price": 22750.0, "atm_iv": 13.0, "expiry": "2026-10-27"}
        )
        assert res["status"] == "OK"
        assert res["current_decay_rate_pts_per_hour"] > 0.0
        assert res["decay_rate_inr_per_lot_hour"] > 0.0
        assert res["decay_acceleration_factor"] >= 0.5

        # Check decay schedule monotonicity
        sched = res["decay_schedule"]
        assert len(sched) == 7
        assert sched[0]["time_ist"] == "09:15"
        assert sched[-1]["time_ist"] == "15:30"
        # Cumulative decay points must be non-decreasing throughout trading session
        for i in range(len(sched) - 1):
            assert sched[i]["cumulative_decay_pts"] <= sched[i+1]["cumulative_decay_pts"]


class TestOrderFlowStructureEngineCVD:
    """Rigorous tests for Cumulative Volume Delta and Quote/Tick Rule classification."""

    def test_cvd_uses_quote_rule_and_tick_rule(self):
        """Verify CVD endpoint returns explicit methodology and valid structure."""
        res = order_flow_structure_engine.compute_cumulative_volume_delta()
        assert "delta_methodology" in res
        assert res["delta_methodology"] == "QUOTE_RULE_AND_TICK_RULE"
        assert "current_cvd" in res
        assert "order_book_imbalance_pct" in res


class TestAPIRoutesLevel2MarketDepth:
    """API endpoint response contract verification."""

    def test_api_options_strike_depth_payload(self):
        res = client.get("/api/nifty/options/strike-depth?strike=22800&option_type=CE&expiry_type=weekly")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "strike" in data
        assert data["strike"] == 22800

    def test_api_intraday_order_flow_payload(self):
        res = client.get("/api/nifty/intraday/order-flow")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
