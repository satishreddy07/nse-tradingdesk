"""
Automated tests for Auto Options Trading master toggle and ML deactivation gating.
Complies with AGENTS.md and ANTI_FABRICATION_POLICY.md.
"""
import pytest
from fastapi.testclient import TestClient
from web.api import app
from execution.state_engine import state_engine
from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine


client = TestClient(app)


def test_auto_options_toggle_persistence_and_endpoints():
    # 1. Test setting to False (toggled OFF)
    resp = client.post("/api/options/auto-trade-toggle", json={"enabled": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["enabled"] is False
    assert data["toggle_state"] == "OFF"

    # Verify state_engine reflects False
    assert state_engine.is_auto_options_trading_enabled() is False

    # GET endpoint test
    get_resp = client.get("/api/options/auto-trade-toggle")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["enabled"] is False
    assert get_data["toggle_state"] == "OFF"

    # 2. Test fail-closed behavior when OFF
    # Intraday auto-deploy
    intra_res = intraday_options_strategy_engine.evaluate_and_execute_auto_trades()
    assert intra_res["status"] == "DISABLED_BY_USER"
    assert intra_res["executed"] is False

    # Intraday ML training
    intra_ml = intraday_options_strategy_engine.train_options_ml_model()
    assert intra_ml["status"] == "PAUSED_BY_USER"
    assert intra_ml["trained"] is False

    # Positional auto-deploy
    pos_res = weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades("WEEKLY")
    assert pos_res["status"] == "DISABLED_BY_USER"
    assert pos_res["executed"] is False

    # Positional ML training
    pos_ml = weekly_monthly_options_strategy_engine.train_positional_options_ml_model("WEEKLY")
    assert pos_ml["status"] == "PAUSED_BY_USER"
    assert pos_ml["trained"] is False

    # API endpoint auto-deploy when OFF
    api_deploy = client.post("/api/options/auto-deploy", json={"modes": ["INTRADAY"]})
    assert api_deploy.status_code == 200
    assert api_deploy.json()["status"] == "DISABLED_BY_USER"

    # 3. Test setting to True (toggled ON)
    resp_on = client.post("/api/options/auto-trade-toggle", json={"enabled": True})
    assert resp_on.status_code == 200
    data_on = resp_on.json()
    assert data_on["enabled"] is True
    assert data_on["toggle_state"] == "ON"
    assert state_engine.is_auto_options_trading_enabled() is True

    # Leave toggle ON as default for subsequent strategy test suites
    client.post("/api/options/auto-trade-toggle", json={"enabled": True})
    assert state_engine.is_auto_options_trading_enabled() is True
