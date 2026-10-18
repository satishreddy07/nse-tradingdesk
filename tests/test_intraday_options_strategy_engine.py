"""
tests/test_intraday_options_strategy_engine.py
==============================================
Unit tests for the 5-Regime Intraday Options Strategy Engine:
1. Dynamic Sigma Strike Selection (1-sigma & 2-sigma using expected move formula).
2. Strategy formulation across all 5 regimes (Primary & Experimental Asymmetric).
3. Independent per-leg state machine (ACTIVE -> SL_EXITED -> RE_ENTERED -> FINAL_SL_EXITED -> CLOSED).
4. Strict price-triggered 1-time re-entry when premium drops to <= P0 (original entry fill).
5. Stop-loss calculation from actual re-entry fill.
6. Backtesting matrix (30% vs 40% SL, with vs without re-entry, net P&L after regulatory costs).
7. Machine learning model online training and telemetry.
"""

import pytest
import math
from engines.intraday_options_strategy_engine import (
    OptionLegTracker,
    IntradayOptionsStrategyEngine,
    intraday_options_strategy_engine,
    _black_scholes_price
)

def test_dynamic_sigma_strike_selection():
    """Verify expected move formula and strike rounding to 50-strike grid."""
    spot = 22600.0
    iv_pct = 13.2
    # 1-day intraday time to expiry
    t_years = 1.0 / 365.0
    
    strikes_res = IntradayOptionsStrategyEngine.calculate_sigma_strikes(
        spot=spot,
        iv_pct=iv_pct,
        time_to_expiry_years=t_years,
        strike_step=50
    )
    
    inputs = strikes_res["inputs"]
    assert inputs["spot_price"] == 22600.0
    assert inputs["iv_pct"] == 13.2
    assert inputs["expected_move_1_sigma"] > 0
    
    strikes = strikes_res["strikes"]
    assert strikes["strike_1s_pe"] % 50 == 0
    assert strikes["strike_1s_ce"] % 50 == 0
    assert strikes["strike_2s_pe"] % 50 == 0
    assert strikes["strike_2s_ce"] % 50 == 0
    
    # 1-sigma PE strike must be below spot; 1-sigma CE strike must be above spot
    assert strikes["strike_1s_pe"] < spot
    assert strikes["strike_1s_ce"] > spot
    # 2-sigma PE strike must be below 1-sigma PE; 2-sigma CE must be above 1-sigma CE
    assert strikes["strike_2s_pe"] < strikes["strike_1s_pe"]
    assert strikes["strike_2s_ce"] > strikes["strike_1s_ce"]

def test_five_regimes_formulation():
    """Verify strategy legs and stop-loss percentages for all 5 regimes."""
    # 1. Strong Bullish
    r_sbull = IntradayOptionsStrategyEngine.get_strategy_rules("STRONG BULLISH")
    assert r_sbull["primary"]["sl_pct"] == 0.30
    assert len(r_sbull["primary"]["legs"]) == 1
    assert r_sbull["primary"]["legs"][0]["type"] == "PE"
    assert r_sbull["primary"]["legs"][0]["sigma"] == 1.0
    
    assert r_sbull["experimental_asymmetric"] is not None
    assert len(r_sbull["experimental_asymmetric"]["legs"]) == 2
    assert r_sbull["experimental_asymmetric"]["legs"][0]["type"] == "PE" # 1-sigma PE
    assert r_sbull["experimental_asymmetric"]["legs"][1]["type"] == "CE" # 2-sigma CE
    assert r_sbull["experimental_asymmetric"]["sl_pct"] == 0.30

    # 2. Strong Bearish
    r_sbear = IntradayOptionsStrategyEngine.get_strategy_rules("STRONG BEARISH")
    assert r_sbear["primary"]["sl_pct"] == 0.30
    assert len(r_sbear["primary"]["legs"]) == 1
    assert r_sbear["primary"]["legs"][0]["type"] == "CE"
    assert r_sbear["primary"]["legs"][0]["sigma"] == 1.0
    
    assert r_sbear["experimental_asymmetric"] is not None
    assert len(r_sbear["experimental_asymmetric"]["legs"]) == 2
    assert r_sbear["experimental_asymmetric"]["legs"][0]["type"] == "CE" # 1-sigma CE
    assert r_sbear["experimental_asymmetric"]["legs"][1]["type"] == "PE" # 2-sigma PE
    assert r_sbear["experimental_asymmetric"]["sl_pct"] == 0.30

    # 3. Mild Bullish
    r_mbull = IntradayOptionsStrategyEngine.get_strategy_rules("MILD BULLISH")
    assert r_mbull["primary"]["sl_pct"] == 0.40
    assert len(r_mbull["primary"]["legs"]) == 1
    assert r_mbull["primary"]["legs"][0]["type"] == "PE"
    assert r_mbull["experimental_asymmetric"] is None

    # 4. Mild Bearish
    r_mbear = IntradayOptionsStrategyEngine.get_strategy_rules("MILD BEARISH")
    assert r_mbear["primary"]["sl_pct"] == 0.40
    assert len(r_mbear["primary"]["legs"]) == 1
    assert r_mbear["primary"]["legs"][0]["type"] == "CE"
    assert r_mbear["experimental_asymmetric"] is None

    # 5. Range-Bound
    r_range = IntradayOptionsStrategyEngine.get_strategy_rules("RANGE BOUND")
    assert r_range["primary"]["sl_pct"] == 0.40
    assert len(r_range["primary"]["legs"]) == 2
    types = [l["type"] for l in r_range["primary"]["legs"]]
    assert "CE" in types and "PE" in types
    assert r_range["primary"]["legs"][0]["sigma"] == 1.0
    assert r_range["primary"]["legs"][1]["sigma"] == 1.0

def test_option_leg_state_machine_and_reentry():
    """
    Test exact rule:
    CE sold at Rs.100 with 30% SL.
    1. Initial entry: Sell CE @ 100.
    2. Stop-loss: If premium reaches 130, trigger exit and record actual fill.
    3. After exit: Monitor same CE contract and strike.
    4. Re-entry: If premium subsequently falls back to 100 or below, submit new sell order.
    5. Allow only ONE re-entry per leg during trade cycle.
    6. Calculate new stop-loss from actual re-entry fill (e.g. 100 * 1.30 = 130).
    7. If re-entered position hits stop-loss, exit and prohibit further re-entry.
    """
    leg = OptionLegTracker(
        leg_id="LEG-CE-100",
        strike=22750,
        option_type="CE",
        action="SELL",
        qty=75,
        entry_premium=100.0,
        sl_pct=0.30,
        entry_time="2026-10-15 09:16:00 IST"
    )

    assert leg.state == "ACTIVE"
    assert leg.original_entry_premium == 100.0
    assert leg.initial_sl_trigger == 130.0
    assert leg.re_entry_count == 0

    # Tick 1: Normal premium fluctuation below SL (Rs. 115)
    leg.tick(current_bid=114.5, current_ask=115.5, current_ltp=115.0, timestamp_ist="2026-10-15 09:30:00 IST")
    assert leg.state == "ACTIVE"
    assert not leg.first_sl_hit

    # Tick 2: Premium spikes to Rs. 132 (triggers Stop Loss)
    # On executable ask of 132.5, exit fills
    leg.tick(current_bid=131.5, current_ask=132.5, current_ltp=132.0, timestamp_ist="2026-10-15 10:15:00 IST")
    assert leg.state == "SL_EXITED"
    assert leg.first_sl_hit is True
    assert leg.first_exit_price == 132.5
    assert leg.first_exit_time == "2026-10-15 10:15:00 IST"
    # Phase 1 PnL: Sold @ 100, Bought back @ 132.5 = -32.5 pts
    assert leg.phase1_pnl_pts == -32.5

    # Tick 3: Premium stays elevated at Rs. 110 (above P0 = 100) -> NO re-entry yet
    leg.tick(current_bid=109.5, current_ask=110.5, current_ltp=110.0, timestamp_ist="2026-10-15 11:00:00 IST")
    assert leg.state == "SL_EXITED"
    assert leg.re_entered is False
    assert leg.re_entry_count == 0

    # Tick 4: Premium decays back to <= P0 (e.g. executable bid is Rs. 99.0 <= 100.0) -> Re-entry fires!
    leg.tick(current_bid=99.0, current_ask=100.0, current_ltp=99.5, timestamp_ist="2026-10-15 12:00:00 IST")
    assert leg.state == "RE_ENTERED"
    assert leg.re_entered is True
    assert leg.re_entry_count == 1
    assert leg.re_entry_price == 99.0
    # New SL trigger = 99.0 * 1.30 = 128.70
    assert leg.re_entry_sl_trigger == 128.7

    # Tick 5: Re-entered position expires at market-on-close (MOC) at Rs. 15.0
    leg.tick(current_bid=14.8, current_ask=15.2, current_ltp=15.0, timestamp_ist="2026-10-15 15:25:00 IST", is_moc_settlement=True)
    assert leg.state == "CLOSED"
    assert leg.settlement_price == 15.2
    # Phase 2 PnL: Re-sold @ 99.0, Bought back at MOC @ 15.2 = +83.8 pts
    assert leg.phase2_pnl_pts == (99.0 - 15.2)
    # Total Gross PnL = -32.5 + 83.8 = +51.3 pts
    assert round(leg.total_gross_pnl_pts, 1) == 51.3
    # Net PnL in INR after deducting transaction costs should be positive
    assert leg.net_pnl_inr > 0

def test_reentry_final_sl_exit_rule():
    """Verify that if re-entered position hits its stop-loss, it exits permanently with zero further re-entries."""
    leg = OptionLegTracker(
        leg_id="LEG-PE-80",
        strike=22450,
        option_type="PE",
        action="SELL",
        qty=75,
        entry_premium=80.0,
        sl_pct=0.40,
        entry_time="2026-10-15 09:16:00 IST"
    )
    # Initial SL = 80 * 1.40 = 112.0
    assert leg.initial_sl_trigger == 112.0

    # 1. Trigger Initial SL at 115
    leg.tick(current_bid=114.0, current_ask=115.0, current_ltp=114.5, timestamp_ist="2026-10-15 10:00:00 IST")
    assert leg.state == "SL_EXITED"

    # 2. Price falls back to <= 80 (re-enter at 78)
    leg.tick(current_bid=78.0, current_ask=79.0, current_ltp=78.5, timestamp_ist="2026-10-15 11:30:00 IST")
    assert leg.state == "RE_ENTERED"
    # New SL = 78 * 1.40 = 109.2
    assert leg.re_entry_sl_trigger == round(78.0 * 1.40, 2)

    # 3. Premium surges again and hits the new SL trigger (110 >= 109.2)
    leg.tick(current_bid=109.5, current_ask=110.5, current_ltp=110.0, timestamp_ist="2026-10-15 13:00:00 IST")
    assert leg.state == "FINAL_SL_EXITED"
    assert leg.final_sl_hit is True
    assert leg.re_entry_count == 1

    # 4. Even if premium drops back below 80 again, NO second re-entry is permitted
    leg.tick(current_bid=70.0, current_ask=71.0, current_ltp=70.5, timestamp_ist="2026-10-15 14:00:00 IST")
    assert leg.state == "FINAL_SL_EXITED"
    assert leg.re_entry_count == 1

def test_backtest_matrix_execution():
    """Verify that comprehensive backtest runs and outputs the required comparison metrics."""
    backtest_res = intraday_options_strategy_engine.run_comprehensive_backtest(
        lookback_sessions=15,
        lot_size=75
    )
    if backtest_res["status"] == "DATA_UNAVAILABLE":
        assert backtest_res.get("message")
        assert backtest_res.get("sessions_evaluated", 0) < 15
        return
    assert backtest_res["status"] == "OK"
    assert backtest_res["sessions_evaluated"] >= 15

    matrix = backtest_res["comparison_matrix"]
    assert "PRIMARY_WITH_REENTRY" in matrix
    assert "PRIMARY_WITHOUT_REENTRY" in matrix
    assert "ASYMMETRIC_WITH_REENTRY" in matrix
    assert "SL_30_SENSITIVITY" in matrix
    assert "SL_40_SENSITIVITY" in matrix
    
    prim_re = matrix["PRIMARY_WITH_REENTRY"]
    assert "win_rate_pct" in prim_re
    assert "total_net_pnl_inr" in prim_re
    assert "profit_factor" in prim_re
    assert "expectancy_inr" in prim_re
    assert "max_drawdown_inr" in prim_re
    assert "tail_loss_worst_5pct_inr" in prim_re
    
    # Regime breakdown must have all 5 regimes
    regimes = backtest_res["regime_breakdown"]
    for reg in ["STRONG_BULLISH", "STRONG_BEARISH", "MILD_BULLISH", "MILD_BEARISH", "RANGE_BOUND"]:
        assert reg in regimes

def test_train_options_ml_model():
    """Verify that the ML meta-model trains online and returns valid metrics."""
    ml_meta = intraday_options_strategy_engine.train_options_ml_model()
    assert ml_meta["status"] == "TRAINED_ONLINE"
    assert ml_meta["total_samples"] > 0
    assert "win_rate_accuracy_pct" in ml_meta
    assert "expectancy_r2" in ml_meta
    assert "feature_importance" in ml_meta
    assert len(ml_meta["feature_importance"]) == 5

def test_intraday_trades_based_on_intraday_analysis():
    """Verify that intraday trades are strictly derived from pure intraday analysis (VWAP, ORB, GEX, Intraday ML)."""
    regime_info = intraday_options_strategy_engine.get_intraday_market_regime(force_refresh=True)
    assert regime_info["decision_source"] == "INTRADAY_ANALYSIS"
    assert regime_info["regime"] in ["STRONG BULLISH", "MILD BULLISH", "STRONG BEARISH", "MILD BEARISH", "RANGE-BOUND"]
    assert "intraday_composite_score" in regime_info
    assert "vwap" in regime_info
    assert "orb_status" in regime_info
    assert "choppiness_index" in regime_info
    assert "zero_gamma_flip" in regime_info
    assert "writer_dominance_score" in regime_info
    assert "probabilities" in regime_info
    assert "near_atm_pcr" in regime_info
    
    # Auto deploy evaluation should also flag decision_source as INTRADAY_ANALYSIS
    deploy_res = intraday_options_strategy_engine.evaluate_and_execute_auto_trades(force_refresh=True)
    assert deploy_res["decision_source"] == "INTRADAY_ANALYSIS"
    assert "intraday_analysis" in deploy_res


def test_intraday_sigma_rejects_invalid_expiry_instead_of_using_calendar_fallback():
    """A malformed explicit expiry must not silently become the calendar's next expiry."""
    with pytest.raises(ValueError, match="Explicit option expiry could not be parsed"):
        IntradayOptionsStrategyEngine.calculate_sigma_strikes(
            spot=22600.0,
            iv_pct=13.5,
            expiry_date="not-an-expiry",
        )


def test_intraday_everyday_0dte_and_non_0dte_confidence():
    """Verify that intraday trades support 0DTE and non-0DTE cycles and execute based on daytime confidence."""
    # 1. Test 0DTE calculation
    res_0dte = IntradayOptionsStrategyEngine.calculate_sigma_strikes(
        spot=22600.0,
        iv_pct=13.5,
        dte=0
    )
    assert res_0dte["is_zero_dte"] is True
    assert res_0dte["expiry_cycle"] == "0DTE"
    assert res_0dte["dist_1sigma"] > 0

    # 2. Test Non-0DTE calculation (e.g. 2DTE Wednesday before Thursday weekly expiry)
    res_2dte = IntradayOptionsStrategyEngine.calculate_sigma_strikes(
        spot=22600.0,
        iv_pct=13.5,
        dte=2
    )
    assert res_2dte["is_zero_dte"] is False
    assert res_2dte["expiry_cycle"] == "2DTE_INTRADAY"
    # Expected move for 2DTE must be strictly larger than 0DTE
    assert res_2dte["dist_1sigma"] > res_0dte["dist_1sigma"]

    # 3. Test Daytime Confidence calculation
    mock_regime_info = {
        "regime": "STRONG BULLISH",
        "intraday_composite_score": 28.5,
        "dist_vwap_pts": 32.0,
        "orb_status": "BREAKOUT_UP",
        "choppiness_index": 38.0,
        "writer_dominance_score": 14.0,
        "near_atm_pcr": 1.25
    }
    conf = IntradayOptionsStrategyEngine.calculate_daytime_confidence(mock_regime_info)
    assert conf["intraday_confidence_pct"] >= 65.0
    assert conf["is_confident_for_0dte"] is True
    assert conf["is_confident_for_intraday"] is True
    assert "breakdown" in conf

    # 4. Auto deploy evaluation passes cycle & confidence metrics
    deploy_res = intraday_options_strategy_engine.evaluate_and_execute_auto_trades(force_refresh=True)
    assert "intraday_confidence_pct" in deploy_res
    assert "expiry_cycle" in deploy_res
    if deploy_res.get("status") == "DATA_UNAVAILABLE":
        assert deploy_res.get("reason")
        assert deploy_res["deployed_strategies"] == []
    else:
        assert deploy_res["status"] in ("DEPLOYED", "SKIPPED")

