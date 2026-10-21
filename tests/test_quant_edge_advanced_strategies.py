"""
tests/test_quant_edge_advanced_strategies.py
============================================
Comprehensive test suite verifying institutional quant edge engines:
1. Level-2 Order Book Imbalance (OBI) Velocity (dOBI/dt) & Acceleration
2. Iceberg / Passive Absorption Detector
3. Priority Gamma Squeeze Alert with Depth Confirmation
4. Intraday Theta Decay Acceleration Curve (dTheta/dt)
5. Dynamic Max Pain Migration Tracker (Delta K drift)
6. Level-2 Micro-Price Smart Execution Router (Slippage Eliminator)
7. Term Structure Volatility Skew & Calendar Spread Arbitrage
8. Monthly Roll-Over Flow Radar
9. Delta-Neutral Dynamic Hedging Band
10. Complete Web API Endpoints Test
"""

import pytest
from fastapi.testclient import TestClient
from web.api import app
from engines.intraday_analytics_engine import intraday_analytics_engine
from engines.weekly_monthly_options_strategy_engine import WeeklyMonthlyOptionsStrategyEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_obi_velocity_and_iceberg_detector():
    """Verifies that OBI velocity and passive iceberg detection correctly compute rates of change."""
    dummy_depth = {
        "status": "OK",
        "net_flow_index": 0.35,
        "aggregate_call_obi": 0.45,
        "aggregate_put_obi": -0.25,
        "near_atm_strikes": [
            {
                "strike": 22800,
                "is_atm": True,
                "ce": {"micro_price": 125.5, "ltp": 125.0, "total_buy": 15000, "total_sell": 4500},
                "pe": {"micro_price": 110.0, "ltp": 110.0, "total_buy": 4000, "total_sell": 12000}
            }
        ]
    }
    vel_res = intraday_analytics_engine.compute_obi_velocity_and_icebergs(dummy_depth)
    assert "velocity_1m" in vel_res
    assert "velocity_3m" in vel_res
    assert "acceleration" in vel_res
    assert "velocity_regime" in vel_res
    assert "iceberg_alerts" in vel_res
    assert "absorption_regime" in vel_res


def test_gamma_squeeze_depth_alert():
    """Verifies that gamma squeeze alerts fire when spot approaches call/put wall with depth confirmation."""
    flow_data = {
        "net_flow_index": 0.40,
        "aggregate_call_obi": 0.50,
        "velocity_1m": 0.35
    }
    # Reset last alert state to test firing
    intraday_analytics_engine._last_gamma_alert_state = {"wall": None, "ts": 0.0}
    # Spot near Call Wall 22900 (spot 22890)
    alert = intraday_analytics_engine.check_gamma_squeeze_depth_alert(
        spot=22890.0,
        call_wall=22900,
        put_wall=22700,
        flow_data=flow_data
    )
    assert alert is not None
    assert "GAMMA SQUEEZE" in alert
    assert "22900 CE" in alert


def test_weekly_theta_decay_curve():
    """Verifies intraday theta decay schedule and non-linear acceleration factors."""
    theta_res = intraday_analytics_engine.compute_weekly_theta_decay_curve(spot_price=22811.35)
    assert theta_res["status"] == "OK"
    assert "theta_phase" in theta_res
    assert "decay_schedule" in theta_res
    assert len(theta_res["decay_schedule"]) > 0
    assert "decay_rate_inr_per_lot_hour" in theta_res
    assert theta_res["base_atm_straddle_daily_pts"] > 0


def test_dynamic_max_pain_migration():
    """Verifies Max Pain drift points and pin strength classification."""
    mp_res = intraday_analytics_engine.compute_dynamic_max_pain_migration()
    assert mp_res["status"] == "OK"
    assert "current_max_pain" in mp_res
    assert "opening_max_pain" in mp_res
    assert "drift_points" in mp_res
    assert "migration_status" in mp_res
    assert "pin_strength" in mp_res


def test_smart_micro_price_execution():
    """Verifies that Level-2 Micro-Price routing eliminates spread slippage and calculates INR savings."""
    legs = [
        {"strike": 22800, "opt_type": "CE", "action": "SELL", "lots": 2, "price": 125.0},
        {"strike": 22800, "opt_type": "PE", "action": "SELL", "lots": 2, "price": 115.0}
    ]
    res = intraday_analytics_engine.calculate_smart_micro_price_execution(legs, spot_price=22811.35)
    assert res["status"] == "OK"
    assert res["total_legs"] == 2
    assert "total_slippage_savings_inr" in res
    assert "legs" in res
    for leg in res["legs"]:
        assert "smart_limit_price" in leg
        assert "market_fill_price" in leg
        assert "slippage_saving_pts" in leg
        assert "slippage_saving_inr" in leg


def test_term_structure_and_calendar_skew():
    """Verifies term structure IV skew and calendar spread arbitrage signal."""
    skew_res = intraday_analytics_engine.compute_term_structure_and_calendar_skew()
    assert skew_res["status"] == "OK"
    assert "weekly_atm_iv" in skew_res
    assert "monthly_atm_iv" in skew_res
    assert "iv_skew_pts" in skew_res
    assert "term_structure_regime" in skew_res
    assert "arbitrage_opportunity" in skew_res


def test_monthly_rollover_radar():
    """Verifies monthly rollover pacing, stage, and directional bias."""
    radar_res = intraday_analytics_engine.compute_monthly_rollover_radar()
    assert radar_res["status"] == "OK"
    assert "days_to_monthly_expiry" in radar_res
    assert "roll_stage" in radar_res
    assert "estimated_rollover_pct" in radar_res
    assert "rollover_directional_bias" in radar_res


def test_delta_neutral_hedging_band():
    """Verifies 1-sigma and 2-sigma boundaries and delta drift adjustment alerts."""
    hedging = intraday_analytics_engine.compute_delta_neutral_hedging_band(
        spot_price=22811.35,
        iv=13.5,
        dte_days=4.0,
        net_delta=0.25 # Drifted beyond threshold
    )
    assert hedging["status"] == "OK"
    assert hedging["upper_1_sigma"] > 22811.35
    assert hedging["lower_1_sigma"] < 22811.35
    assert hedging["action_required"] is True
    assert "REBALANCE_TRIGGERED" in hedging["delta_status"]


def test_weekly_monthly_engine_delegates():
    """Verifies that WeeklyMonthlyOptionsStrategyEngine delegate methods function properly."""
    engine = WeeklyMonthlyOptionsStrategyEngine()
    legs = [{"strike": 22800, "opt_type": "CE", "action": "BUY", "lots": 1, "price": 100.0}]
    smart = engine.calculate_smart_micro_price_execution(legs, spot_price=22811.35)
    assert smart["status"] == "OK"
    term = engine.compute_term_structure_and_calendar_skew()
    assert term["status"] == "OK"
    roll = engine.compute_monthly_rollover_radar()
    assert roll["status"] == "OK"


def test_api_endpoints(client):
    """Verifies all FastAPI REST endpoints return status 200 and expected schemas."""
    # 1. Order Flow Analytics
    r1 = client.get("/api/nifty/order-flow/analytics")
    assert r1.status_code == 200
    assert r1.json()["status"] == "OK"

    # 2. Theta Decay Curve
    r2 = client.get("/api/nifty/options/theta-decay-curve")
    assert r2.status_code == 200
    assert r2.json()["status"] == "OK"

    # 3. Max Pain Migration
    r3 = client.get("/api/nifty/options/max-pain-migration")
    assert r3.status_code == 200
    assert r3.json()["status"] == "OK"

    # 4. Term Structure
    r4 = client.get("/api/nifty/options/term-structure")
    assert r4.status_code == 200
    assert r4.json()["status"] == "OK"

    # 5. Delta Hedging Band
    r5 = client.get("/api/nifty/options/delta-hedging-band?net_delta=0.05")
    assert r5.status_code == 200
    assert r5.json()["status"] == "OK"

    # 6. Smart Execution
    r6 = client.post("/api/nifty/options/smart-execution", json={
        "legs": [
            {"strike": 22800, "opt_type": "CE", "action": "SELL", "lots": 1, "price": 120.0},
            {"strike": 22800, "opt_type": "PE", "action": "SELL", "lots": 1, "price": 110.0}
        ],
        "spot_price": 22811.35
    })
    assert r6.status_code == 200
    assert r6.json()["status"] == "OK"
    assert r6.json()["total_slippage_savings_inr"] >= 0
