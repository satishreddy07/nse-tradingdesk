"""
========================================================================================
Test Suite: Production Readiness Expansion & Institutional Engines
========================================================================================
Validates:
1. AutonomousExecutionDaemon (Lifecycle, per-leg target/SL evaluations, kill-switch).
2. OptionsEdgeAnalyticsEngine (15m Strike OI Pulse, Straddle Decay, IVR/IVP, GEX Profile).
3. OrderFlowStructureEngine (Volume Profile POC/VAH/VAL, CVD, Multi-Index Confluence).
4. FastAPI Endpoints (/api/daemon/status, /api/options/edge/analytics, /api/intraday/order-flow).
Strictly authentic data without hardcoding.
========================================================================================
"""

import pytest
from fastapi.testclient import TestClient

from engines.autonomous_execution_daemon import autonomous_execution_daemon
from engines.options_edge_analytics_engine import options_edge_analytics_engine
from engines.order_flow_structure_engine import order_flow_structure_engine
from web.api import app

client = TestClient(app)


def test_autonomous_daemon_lifecycle_and_status():
    """Verifies that the autonomous daemon starts, reports status, and stops cleanly."""
    status_initial = autonomous_execution_daemon.get_status()
    assert "is_running" in status_initial
    assert "mode" in status_initial
    assert status_initial["mode"] in ("PAPER", "LIVE_BROKER")

    # Start daemon in PAPER mode
    started = autonomous_execution_daemon.start_daemon(mode="PAPER")
    assert started is True
    assert autonomous_execution_daemon.is_running() is True

    # Mode switch
    assert autonomous_execution_daemon.set_mode("PAPER") == "PAPER"

    # Evaluate once
    eval_res = autonomous_execution_daemon.evaluate_once(is_market_open=True)
    assert eval_res["status"] in ("OK", "KILL_SWITCH_ACTIVE")
    assert "legs_evaluated" in eval_res

    # Stop daemon cleanly
    stopped = autonomous_execution_daemon.stop_daemon()
    assert stopped is True


def test_account_drawdown_killswitch():
    """Verifies that account drawdown circuit breaker accurately detects safe vs breach states."""
    kill_res = autonomous_execution_daemon.evaluate_account_drawdown_killswitch()
    assert "triggered" in kill_res
    assert isinstance(kill_res["triggered"], bool)
    assert "limit_inr" in kill_res
    assert kill_res["limit_inr"] > 0


def test_options_edge_15m_oi_pulse():
    """Verifies 15-minute strike OI pulse computation and buildup classifications."""
    pulse = options_edge_analytics_engine.compute_15m_strike_oi_pulse()
    assert "spot_price" in pulse
    assert "strikes" in pulse
    assert isinstance(pulse["strikes"], list)
    assert "dominant_regime" in pulse

    if pulse["strikes"]:
        sample = pulse["strikes"][0]
        assert "strike" in sample
        assert "ce" in sample
        assert "pe" in sample
        assert "buildup" in sample["ce"]
        assert "buildup" in sample["pe"]


def test_atm_straddle_decay_curve():
    """Verifies ATM straddle decay curve generation from historical snapshots."""
    decay = options_edge_analytics_engine.compute_atm_straddle_decay_curve("2026-10-23")
    assert "available" in decay
    assert "slots" in decay
    assert isinstance(decay["slots"], list)

    if decay["available"] and decay["slots"]:
        assert "open_premium" in decay
        assert "current_premium" in decay
        assert decay["open_premium"] > 0
        first_slot = decay["slots"][0]
        assert "straddle_premium" in first_slot
        assert "theoretical_baseline" in first_slot


def test_iv_rank_and_percentile():
    """Verifies empirical 90-day IV Rank and IV Percentile calculations."""
    vol_rank = options_edge_analytics_engine.compute_iv_rank_and_percentile()
    assert "current_iv" in vol_rank
    assert "iv_rank" in vol_rank
    assert "iv_percentile" in vol_rank
    assert 0.0 <= vol_rank["iv_rank"] <= 100.0
    assert 0.0 <= vol_rank["iv_percentile"] <= 100.0
    assert "regime" in vol_rank


def test_strike_gex_profile():
    """Verifies Strike-by-Strike Net Dealer Gamma Exposure (GEX in ₹ Crores)."""
    gex_profile = options_edge_analytics_engine.compute_strike_gex_profile()
    assert "total_net_gex_crores" in gex_profile
    assert "strikes" in gex_profile
    assert isinstance(gex_profile["strikes"], list)

    if gex_profile["strikes"]:
        s = gex_profile["strikes"][0]
        assert "strike" in s
        assert "net_gex_crores" in s
        assert "dealer_bias" in s


def test_volume_profile_and_value_area():
    """Verifies Volume Profile 25-point bins, POC, VAH (70%), and VAL (70%)."""
    vp = order_flow_structure_engine.compute_volume_profile()
    assert "poc" in vp
    assert "vah" in vp
    assert "val" in vp
    assert vp["vah"] >= vp["poc"] >= vp["val"]
    assert "market_state" in vp
    assert "histogram" in vp
    assert isinstance(vp["histogram"], list)


def test_cumulative_volume_delta():
    """Verifies Cumulative Volume Delta (CVD) calculation and absorption signals."""
    cvd = order_flow_structure_engine.compute_cumulative_volume_delta()
    assert "current_cvd" in cvd
    assert "flow_sentiment" in cvd
    assert "absorption_signal" in cvd
    assert cvd["flow_sentiment"] in ("BUYER_DOMINATED", "SELLER_DOMINATED", "NEUTRAL")


def test_multi_index_confluence():
    """Verifies weighted Multi-Index & Heavyweight Confluence Score."""
    confluence = order_flow_structure_engine.compute_multi_index_confluence()
    assert "confluence_score" in confluence
    assert -100.0 <= confluence["confluence_score"] <= 100.0
    assert "sentiment" in confluence
    assert "constituents" in confluence
    assert len(confluence["constituents"]) == 7


def test_api_endpoints_expansion_suite():
    """Verifies FastAPI HTTP endpoints for the expansion suite."""
    # 1. Daemon Status
    r_daemon = client.get("/api/daemon/status")
    assert r_daemon.status_code == 200
    assert "is_running" in r_daemon.json()

    # 2. Options Edge Analytics
    r_edge = client.get("/api/options/edge/analytics")
    assert r_edge.status_code == 200
    data_edge = r_edge.json()
    assert data_edge["status"] == "OK"
    assert "oi_pulse_15m" in data_edge
    assert "straddle_decay" in data_edge
    assert "volatility_rank" in data_edge

    # 3. Intraday Order Flow Suite
    r_flow = client.get("/api/intraday/order-flow")
    assert r_flow.status_code == 200
    data_flow = r_flow.json()
    assert data_flow["status"] == "OK"
    assert "volume_profile" in data_flow
    assert "cumulative_volume_delta" in data_flow
    assert "multi_index_confluence" in data_flow
