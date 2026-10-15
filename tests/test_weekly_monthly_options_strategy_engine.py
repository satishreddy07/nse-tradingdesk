"""
tests/test_weekly_monthly_options_strategy_engine.py
====================================================
Unit tests for Weekly and Monthly Options Strategy Engine:
1. Dynamic 1-sigma strike calculation with expected move formula for Weekly and Monthly.
2. 5-Regime strategy rules verification (Weekly 40% SL, Monthly 50% SL).
3. State machine verification for Weekly (40% SL, price-triggered re-entry at <= P0).
4. State machine verification for Monthly (50% SL, price-triggered re-entry at <= P0).
5. Strict 1-time re-entry enforcement and permanent final SL exit.
6. Independent backtest execution for Weekly and Monthly.
7. Isolated machine learning model training without data mixing.
"""

import pytest
import math
from engines.weekly_monthly_options_strategy_engine import (
    PositionalOptionLegTracker,
    WeeklyMonthlyOptionsStrategyEngine,
    weekly_monthly_options_strategy_engine
)


def test_weekly_monthly_dynamic_sigma_strike_selection():
    """Verify dynamic 1-sigma strike pricing for Weekly and Monthly horizons."""
    spot = 22600.0
    iv_pct = 13.2

    # 1. Weekly Strike Calculation (DTE = 5 days)
    w_res = WeeklyMonthlyOptionsStrategyEngine.calculate_sigma_strikes(
        spot=spot,
        iv_pct=iv_pct,
        horizon="WEEKLY"
    )
    assert w_res["horizon"] == "WEEKLY"
    assert w_res["spot"] == 22600.0
    assert w_res["iv_pct"] == 13.2
    assert w_res["dte"] > 0
    assert w_res["expected_move"] > 0
    # Strike grid multiple of 50
    assert w_res["strikes"]["strike_1s_pe"] % 50 == 0
    assert w_res["strikes"]["strike_1s_ce"] % 50 == 0
    assert w_res["strikes"]["strike_1s_pe"] < spot
    assert w_res["strikes"]["strike_1s_ce"] > spot

    # 2. Monthly Strike Calculation (DTE = 25 days)
    m_res = WeeklyMonthlyOptionsStrategyEngine.calculate_sigma_strikes(
        spot=spot,
        iv_pct=iv_pct,
        horizon="MONTHLY"
    )
    assert m_res["horizon"] == "MONTHLY"
    assert m_res["dte"] > w_res["dte"]
    # Monthly expected move must be strictly larger than Weekly expected move
    assert m_res["expected_move"] > w_res["expected_move"]
    assert m_res["strikes"]["strike_1s_pe"] % 50 == 0
    assert m_res["strikes"]["strike_1s_ce"] % 50 == 0
    # Monthly strikes must be further OTM than Weekly strikes
    assert m_res["strikes"]["strike_1s_pe"] < w_res["strikes"]["strike_1s_pe"]
    assert m_res["strikes"]["strike_1s_ce"] > w_res["strikes"]["strike_1s_ce"]


def test_weekly_five_regimes_rules():
    """Verify Weekly strategies: all regimes use 40% Stop-Loss."""
    # 1. Strong Bullish: Sell PE @ 1-sigma, 40% SL
    r_sbull = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("STRONG BULLISH", horizon="WEEKLY")
    assert r_sbull["sl_pct"] == 0.40
    assert len(r_sbull["strategy"]["legs"]) == 1
    assert r_sbull["strategy"]["legs"][0]["type"] == "PE"

    # 2. Strong Bearish: Sell CE @ 1-sigma, 40% SL
    r_sbear = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("STRONG BEARISH", horizon="WEEKLY")
    assert r_sbear["sl_pct"] == 0.40
    assert len(r_sbear["strategy"]["legs"]) == 1
    assert r_sbear["strategy"]["legs"][0]["type"] == "CE"

    # 3. Mild Bullish: Sell PE @ 1-sigma, 40% SL
    r_mbull = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("MILD BULLISH", horizon="WEEKLY")
    assert r_mbull["sl_pct"] == 0.40
    assert len(r_mbull["strategy"]["legs"]) == 1
    assert r_mbull["strategy"]["legs"][0]["type"] == "PE"

    # 4. Mild Bearish: Sell CE @ 1-sigma, 40% SL
    r_mbear = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("MILD BEARISH", horizon="WEEKLY")
    assert r_mbear["sl_pct"] == 0.40
    assert len(r_mbear["strategy"]["legs"]) == 1
    assert r_mbear["strategy"]["legs"][0]["type"] == "CE"

    # 5. Range-Bound: Sell CE @ 1-sigma + Sell PE @ 1-sigma, 40% SL independently
    r_range = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("RANGE BOUND", horizon="WEEKLY")
    assert r_range["sl_pct"] == 0.40
    assert len(r_range["strategy"]["legs"]) == 2
    types = [l["type"] for l in r_range["strategy"]["legs"]]
    assert "CE" in types and "PE" in types


def test_monthly_five_regimes_rules():
    """Verify Monthly strategies: all regimes use 50% Stop-Loss."""
    # 1. Strong Bullish: Sell PE @ 1-sigma, 50% SL
    r_sbull = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("STRONG BULLISH", horizon="MONTHLY")
    assert r_sbull["sl_pct"] == 0.50
    assert len(r_sbull["strategy"]["legs"]) == 1
    assert r_sbull["strategy"]["legs"][0]["type"] == "PE"

    # 2. Strong Bearish: Sell CE @ 1-sigma, 50% SL
    r_sbear = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("STRONG BEARISH", horizon="MONTHLY")
    assert r_sbear["sl_pct"] == 0.50
    assert len(r_sbear["strategy"]["legs"]) == 1
    assert r_sbear["strategy"]["legs"][0]["type"] == "CE"

    # 3. Mild Bullish: Sell PE @ 1-sigma, 50% SL
    r_mbull = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("MILD BULLISH", horizon="MONTHLY")
    assert r_mbull["sl_pct"] == 0.50
    assert len(r_mbull["strategy"]["legs"]) == 1
    assert r_mbull["strategy"]["legs"][0]["type"] == "PE"

    # 4. Mild Bearish: Sell CE @ 1-sigma, 50% SL
    r_mbear = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("MILD BEARISH", horizon="MONTHLY")
    assert r_mbear["sl_pct"] == 0.50
    assert len(r_mbear["strategy"]["legs"]) == 1
    assert r_mbear["strategy"]["legs"][0]["type"] == "CE"

    # 5. Range-Bound: Sell CE @ 1-sigma + Sell PE @ 1-sigma, 50% SL independently
    r_range = WeeklyMonthlyOptionsStrategyEngine.get_strategy_rules("RANGE BOUND", horizon="MONTHLY")
    assert r_range["sl_pct"] == 0.50
    assert len(r_range["strategy"]["legs"]) == 2
    types = [l["type"] for l in r_range["strategy"]["legs"]]
    assert "CE" in types and "PE" in types


def test_weekly_reentry_state_machine_40pct():
    """
    Weekly test rule: CE sold at Rs.100 with 40% stop-loss.
    1. Initial entry: Sell CE @ 100.
    2. Stop-loss: If premium reaches 140, trigger exit and record actual fill.
    3. After exit: Monitor same CE contract and strike.
    4. Re-entry: If premium subsequently falls back to 100 or below, submit new sell order.
    5. Permit only ONE re-entry per leg during original trade cycle.
    6. Calculate new stop-loss from actual re-entry fill: fill * 1.40.
    7. If re-entered position hits stop-loss, exit and prohibit further re-entry.
    """
    leg = PositionalOptionLegTracker(
        leg_id="W-LEG-CE-100",
        horizon="WEEKLY",
        strike=22800,
        option_type="CE",
        action="SELL",
        qty=75,
        entry_premium=100.0,
        sl_pct=0.40,
        entry_time="2026-10-15 09:16:00 IST",
        expiry_date="22-Oct-2026",
        dte_at_entry=7
    )
    assert leg.state == "ACTIVE"
    assert leg.original_entry_premium == 100.0
    # 40% SL = 100 * 1.40 = 140.0
    assert leg.initial_sl_trigger == 140.0

    # Tick 1: Normal variation at 125 (below SL)
    leg.tick(current_bid=124.0, current_ask=125.0, current_ltp=124.5, timestamp_ist="2026-10-16")
    assert leg.state == "ACTIVE"

    # Tick 2: Premium spikes to 142 (triggers 40% SL)
    leg.tick(current_bid=141.0, current_ask=142.0, current_ltp=141.5, timestamp_ist="2026-10-17")
    assert leg.state == "SL_EXITED"
    assert leg.first_sl_hit is True
    assert leg.first_exit_price == 142.0
    assert leg.re_entry_count == 0

    # Tick 3: Premium stays at 115 (above P0 = 100) -> NO re-entry
    leg.tick(current_bid=114.0, current_ask=115.0, current_ltp=114.5, timestamp_ist="2026-10-18")
    assert leg.state == "SL_EXITED"
    assert leg.re_entry_count == 0

    # Tick 4: Premium decays back to <= P0 (e.g. bid = 98.0 <= 100.0) -> Re-entry triggers!
    leg.tick(current_bid=98.0, current_ask=99.0, current_ltp=98.5, timestamp_ist="2026-10-19")
    assert leg.state == "RE_ENTERED"
    assert leg.re_entry_count == 1
    assert leg.re_entry_price == 98.0
    # New SL = 98.0 * 1.40 = 137.20
    assert leg.re_entry_sl_trigger == round(98.0 * 1.40, 2)

    # Tick 5: Expiry settlement at 10.0
    leg.tick(current_bid=9.5, current_ask=10.0, current_ltp=9.8, timestamp_ist="2026-10-22", is_expiry_settlement=True)
    assert leg.state == "CLOSED"
    assert leg.settlement_price == 10.0
    # Total Gross PnL = (100 - 142) + (98 - 10) = -42 + 88 = +46 pts
    assert round(leg.total_gross_pnl_pts, 1) == 46.0
    assert leg.net_pnl_inr > 0


def test_monthly_reentry_state_machine_50pct():
    """
    Monthly test rule: Entry at Rs.100 has initial stop trigger of Rs.150 (50% SL).
    Re-entry at <= 100.
    New stop trigger = re_entry_fill * 1.50.
    """
    leg = PositionalOptionLegTracker(
        leg_id="M-LEG-PE-100",
        horizon="MONTHLY",
        strike=22000,
        option_type="PE",
        action="SELL",
        qty=75,
        entry_premium=100.0,
        sl_pct=0.50,
        entry_time="2026-10-15 09:16:00 IST",
        expiry_date="26-Nov-2026",
        dte_at_entry=35
    )
    assert leg.state == "ACTIVE"
    # 50% SL = 100 * 1.50 = 150.0
    assert leg.initial_sl_trigger == 150.0

    # 1. Trigger Initial SL at 152
    leg.tick(current_bid=151.0, current_ask=152.0, current_ltp=151.5, timestamp_ist="2026-10-20")
    assert leg.state == "SL_EXITED"

    # 2. Premium falls back to <= 100 (re-enter at 95.0)
    leg.tick(current_bid=95.0, current_ask=96.0, current_ltp=95.5, timestamp_ist="2026-10-25")
    assert leg.state == "RE_ENTERED"
    # New SL = 95.0 * 1.50 = 142.5
    assert leg.re_entry_sl_trigger == round(95.0 * 1.50, 2)

    # 3. Premium surges again and hits the new SL trigger (145 >= 142.5)
    leg.tick(current_bid=144.0, current_ask=145.0, current_ltp=144.5, timestamp_ist="2026-11-05")
    assert leg.state == "FINAL_SL_EXITED"
    assert leg.final_sl_hit is True
    assert leg.re_entry_count == 1

    # 4. No second re-entry even if premium drops below 100 again
    leg.tick(current_bid=80.0, current_ask=81.0, current_ltp=80.5, timestamp_ist="2026-11-10")
    assert leg.state == "FINAL_SL_EXITED"
    assert leg.re_entry_count == 1


def test_independent_backtest_weekly_and_monthly():
    """Verify that Weekly and Monthly backtests run independently."""
    # Weekly backtest
    w_bt = weekly_monthly_options_strategy_engine.run_positional_backtest(horizon="WEEKLY", lookback_sessions=15)
    assert w_bt["status"] == "OK"
    assert w_bt["horizon"] == "WEEKLY"
    assert w_bt["specified_sl_pct"] == 0.40
    assert "SPECIFIED_WITH_REENTRY" in w_bt["comparison_matrix"]
    assert "SPECIFIED_WITHOUT_REENTRY" in w_bt["comparison_matrix"]

    # Monthly backtest
    m_bt = weekly_monthly_options_strategy_engine.run_positional_backtest(horizon="MONTHLY", lookback_sessions=15)
    assert m_bt["status"] == "OK"
    assert m_bt["horizon"] == "MONTHLY"
    assert m_bt["specified_sl_pct"] == 0.50
    assert "SPECIFIED_WITH_REENTRY" in m_bt["comparison_matrix"]
    assert "SPECIFIED_WITHOUT_REENTRY" in m_bt["comparison_matrix"]


def test_ml_training_strict_isolation():
    """Verify that Weekly and Monthly ML models train independently and do not mix data."""
    w_ml = weekly_monthly_options_strategy_engine.train_positional_options_ml_model(horizon="WEEKLY")
    assert w_ml["horizon"] == "WEEKLY"
    assert w_ml["status"] == "TRAINED_ONLINE"
    assert "days_to_expiry" in w_ml["feature_importance"]

    m_ml = weekly_monthly_options_strategy_engine.train_positional_options_ml_model(horizon="MONTHLY")
    assert m_ml["horizon"] == "MONTHLY"
    assert m_ml["status"] == "TRAINED_ONLINE"
    assert "days_to_expiry" in m_ml["feature_importance"]


def test_weekly_monthly_trades_based_on_weekly_prediction_model():
    """Verify that Weekly & Monthly trades are derived strictly from the Weekly Prediction Model (T+5 Structural)."""
    weekly_info = weekly_monthly_options_strategy_engine.get_weekly_prediction_model_regime(force_refresh=True)
    assert weekly_info["decision_source"] == "WEEKLY_PREDICTION_MODEL_T5"
    assert weekly_info["regime"] in ["STRONG BULLISH", "MILD BULLISH", "STRONG BEARISH", "MILD BEARISH", "RANGE-BOUND"]
    assert "weekly_composite_score" in weekly_info
    assert "probabilities" in weekly_info
    assert "t5_verdict" in weekly_info

    # Test auto deployment returns WEEKLY_PREDICTION_MODEL_T5 for Weekly
    w_deploy = weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades(horizon="WEEKLY", force_refresh=True)
    assert w_deploy["decision_source"] == "WEEKLY_PREDICTION_MODEL_T5"
    assert "weekly_prediction_model" in w_deploy

    # Test auto deployment returns WEEKLY_PREDICTION_MODEL_T5 for Monthly
    m_deploy = weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades(horizon="MONTHLY", force_refresh=True)
    assert m_deploy["decision_source"] == "WEEKLY_PREDICTION_MODEL_T5"
    assert "weekly_prediction_model" in m_deploy
