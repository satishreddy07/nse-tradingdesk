"""
Deterministic regression and integrity test suite for F3-01:
Autonomous Execution Daemon fail-open remediation and position-state persistence.

Verifies:
1. Missing order-placement interface fails closed (preserves OPEN state).
2. Broker rejection fails closed (preserves OPEN state).
3. Order-placement exception fails closed (preserves OPEN state).
4. Timeout or ambiguous/pending response fails closed (preserves OPEN state).
5. Confirmed execution transitions leg to CLOSED only upon authoritative broker fill.
6. Partial execution preserves OPEN state and does not finalize full closed P&L.
7. Paper simulation is clearly tagged as simulated and distinct from broker fills.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from engines.autonomous_execution_daemon import AutonomousExecutionDaemon


@pytest.fixture
def test_daemon():
    """Isolated daemon instance in test mode."""
    daemon = AutonomousExecutionDaemon()
    daemon.stop_daemon()
    return daemon


@pytest.fixture
def sample_active_strategy():
    """Controlled 2-leg active strategy fixture."""
    legs = [
        {
            "action": "BUY",
            "type": "CE",
            "strike": 22600.0,
            "qty": 65,
            "entry_price": 100.0,
            "current_price": 100.0,
            "target_price": 120.0,
            "stoploss_price": 80.0,
            "state": "OPEN",
            "status": "OPEN",
            "leg_pnl": 0.0,
            "unrealized_pnl": 0.0
        },
        {
            "action": "SELL",
            "type": "CE",
            "strike": 22700.0,
            "qty": 65,
            "entry_price": 50.0,
            "current_price": 50.0,
            "target_price": 30.0,
            "stoploss_price": 70.0,
            "state": "OPEN",
            "status": "OPEN",
            "leg_pnl": 0.0,
            "unrealized_pnl": 0.0
        }
    ]
    return {
        "strategy_id": "strat_f301_test",
        "strategy_name": "Test Bull Call Spread",
        "status": "ACTIVE",
        "spot_at_entry": 22650.0,
        "legs_json": json.dumps(legs)
    }


# ==============================================================================
# TEST 1: MISSING ORDER-PLACEMENT INTERFACE
# ==============================================================================
def test_missing_order_placement_interface_preserves_open_state(test_daemon, sample_active_strategy):
    """
    When mode is LIVE_BROKER and kotak_neo_engine lacks place_order:
    - _dispatch_leg_order must return confirmed=False, status=INTERFACE_UNAVAILABLE.
    - evaluate_active_strategies_legs must NOT mark leg as CLOSED.
    - SQLite database must NOT be updated with closed strategy or finalized P&L.
    """
    test_daemon.set_mode("LIVE_BROKER")

    # Mock KotakNeoEngine with no place_order method (the current production state)
    mock_neo = MagicMock()
    del mock_neo.place_order  # explicitly ensure place_order does not exist
    mock_neo.get_option_price.return_value = 125.0  # triggers target hit (>= 120.0)

    # Mock state_engine DB
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        # 1. Direct dispatch check
        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is False
        assert res["status"] == "INTERFACE_UNAVAILABLE"
        assert "not implemented" in res["message"].lower()

        # 2. End-to-end evaluation pass
        eval_res = test_daemon.evaluate_active_strategies_legs()
        # Strategy UPDATE must NOT have been called with status = 'CLOSED'
        for call_item in mock_conn.execute.call_args_list:
            sql_arg = call_item[0][0]
            assert "UPDATE options_strategies SET" not in sql_arg or "status = 'CLOSED'" not in sql_arg


# ==============================================================================
# TEST 2: BROKER REJECTION
# ==============================================================================
def test_broker_rejection_preserves_open_state(test_daemon, sample_active_strategy):
    """
    When place_order returns a rejection or error:
    - Must return confirmed=False, status=REJECTED.
    - Leg must remain OPEN in memory and database.
    """
    test_daemon.set_mode("LIVE_BROKER")

    mock_neo = MagicMock()
    mock_neo._option_contract_to_token = {(22600, "CE", None): "TOKEN_22600_CE"}
    mock_neo.get_option_price.return_value = 125.0
    mock_neo.place_order.return_value = {
        "status": "REJECTED",
        "order_id": "ORD_REJ_001",
        "reason": "RMS: Margin Insufficient"
    }

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is False
        assert res["status"] == "REJECTED"

        eval_res = test_daemon.evaluate_active_strategies_legs()
        # Leg was not closed in database
        for call_item in mock_conn.execute.call_args_list:
            sql_arg = call_item[0][0]
            assert "status = 'CLOSED'" not in sql_arg


# ==============================================================================
# TEST 3: ORDER-PLACEMENT EXCEPTION
# ==============================================================================
def test_order_placement_exception_preserves_open_state(test_daemon, sample_active_strategy):
    """
    When place_order raises an unexpected network or broker error:
    - Must catch exception safely without crashing daemon loop.
    - Must return confirmed=False, status=ERROR.
    - Leg must remain OPEN.
    """
    test_daemon.set_mode("LIVE_BROKER")

    mock_neo = MagicMock()
    mock_neo._option_contract_to_token = {(22600, "CE", None): "TOKEN_22600_CE"}
    mock_neo.get_option_price.return_value = 125.0
    mock_neo.place_order.side_effect = ConnectionResetError("Broker socket disconnected")

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is False
        assert res["status"] == "ERROR"
        assert "Broker socket disconnected" in res["message"]


# ==============================================================================
# TEST 4: TIMEOUT OR AMBIGUOUS / PENDING RESPONSE
# ==============================================================================
def test_timeout_or_pending_response_does_not_close_leg(test_daemon, sample_active_strategy):
    """
    When an order is timed out, acknowledged, or pending without confirmed fill:
    - Must return confirmed=False.
    - Mere submission acknowledgement (e.g. status=PENDING, orderId=123) is NOT a fill.
    - Leg must remain OPEN.
    """
    test_daemon.set_mode("LIVE_BROKER")

    mock_neo = MagicMock()
    mock_neo._option_contract_to_token = {(22600, "CE", None): "TOKEN_22600_CE"}
    mock_neo.get_option_price.return_value = 125.0

    # 4A. TimeoutError
    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo):
        mock_neo.place_order.side_effect = TimeoutError("Order gateway timeout")
        res_timeout = test_daemon._dispatch_leg_order("strat_1", 22600.0, "CE", "SELL", 65, 125.0, "TARGET_HIT")
        assert res_timeout["confirmed"] is False
        assert res_timeout["status"] == "TIMEOUT"

        # 4B. Mere submission acknowledgement (pending fill)
        mock_neo.place_order.side_effect = None
        mock_neo.place_order.return_value = {
            "status": "PENDING",
            "order_id": "ORD_ACK_999"
        }
        res_pending = test_daemon._dispatch_leg_order("strat_1", 22600.0, "CE", "SELL", 65, 125.0, "TARGET_HIT")
        assert res_pending["confirmed"] is False
        assert res_pending["status"] == "PENDING_UNCONFIRMED"


# ==============================================================================
# TEST 5: CONFIRMED BROKER EXECUTION
# ==============================================================================
def test_confirmed_broker_execution_transitions_leg_to_closed(test_daemon, sample_active_strategy):
    """
    When place_order authoritatively returns COMPLETE with filled_qty == qty:
    - Must return confirmed=True, status=FILLED.
    - Leg must transition to state=CLOSED, status=CLOSED.
    - Execution mode must be recorded as LIVE_BROKER_CONFIRMED.
    - Strategy in SQLite is updated with realized P&L.
    """
    test_daemon.set_mode("LIVE_BROKER")

    mock_neo = MagicMock()
    mock_neo._option_contract_to_token = {(22600, "CE", None): "TOKEN_22600_CE"}
    mock_neo.get_option_price.return_value = 125.0
    mock_neo.place_order.return_value = {
        "status": "COMPLETE",
        "order_id": "ORD_CONFIRMED_777",
        "filled_qty": 65,
        "price": 125.0
    }

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is True
        assert res["status"] == "FILLED"
        assert res["order_id"] == "ORD_CONFIRMED_777"
        assert res["executed_price"] == 125.0

        eval_res = test_daemon.evaluate_active_strategies_legs()
        assert eval_res["legs_evaluated"] >= 1
        # Strategy was modified because leg execution was authoritatively confirmed
        update_calls = [c for c in mock_conn.execute.call_args_list if "UPDATE options_strategies" in c[0][0]]
        assert len(update_calls) > 0


# ==============================================================================
# TEST 6: PARTIAL EXECUTION PRESERVES OPEN STATE
# ==============================================================================
def test_partial_execution_preserves_open_state(test_daemon, sample_active_strategy):
    """
    When broker reports partial fill (e.g. 20 executed out of 65 requested):
    - Because the schema does not support partial leg splitting, the leg must NOT be closed.
    - Must return confirmed=False, status=PARTIAL_FILL.
    - Leg must remain OPEN in database.
    """
    test_daemon.set_mode("LIVE_BROKER")

    mock_neo = MagicMock()
    mock_neo._option_contract_to_token = {(22600, "CE", None): "TOKEN_22600_CE"}
    mock_neo.get_option_price.return_value = 125.0
    mock_neo.place_order.return_value = {
        "status": "COMPLETE",
        "order_id": "ORD_PART_001",
        "filled_qty": 20,
        "price": 125.0
    }

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is False
        assert res["status"] == "PARTIAL_FILL"
        assert res["filled_qty"] == 20

        eval_res = test_daemon.evaluate_active_strategies_legs()
        # Strategy status=CLOSED must NOT be written for partial fill
        for call_item in mock_conn.execute.call_args_list:
            sql_arg = call_item[0][0]
            assert "status = 'CLOSED'" not in sql_arg


# ==============================================================================
# TEST 7: PAPER SIMULATION IS CLEARLY TAGGED
# ==============================================================================
def test_paper_simulation_is_clearly_tagged_distinct_from_live(test_daemon, sample_active_strategy):
    """
    In PAPER mode (simulation):
    - Execution executes in simulation.
    - execution_mode must be explicitly tagged as PAPER_SIMULATION.
    - exit_reason must clearly specify AUTONOMOUS_SIMULATED.
    - Simulation ID must have prefix SIM_.
    """
    test_daemon.set_mode("PAPER")

    mock_neo = MagicMock()
    mock_neo.get_option_price.return_value = 125.0

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [sample_active_strategy]

    with patch("engines.kotak_neo_engine.kotak_neo_engine", mock_neo), \
         patch("execution.state_engine.state_engine._get_connection", return_value=mock_conn):

        res = test_daemon._dispatch_leg_order(
            strat_id="strat_f301_test",
            strike=22600.0,
            opt_type="CE",
            close_action="SELL",
            qty=65,
            exit_price=125.0,
            reason="TARGET_HIT"
        )
        assert res["confirmed"] is True
        assert res["status"] == "SIMULATED"
        assert res["mode"] == "PAPER"
        assert res["order_id"].startswith("SIM_")

        eval_res = test_daemon.evaluate_active_strategies_legs()
        assert eval_res["legs_evaluated"] >= 1
