"""
========================================================================================
Test Suite: Portfolio 3 Pre-Breakout Coil & Split Exit Strategy Engine
File: tests/test_strategy_engine.py
========================================================================================
Verifies:
1. Pre-breakout coil evaluation criteria (VDU, 20 EMA proximity, range contraction, trend filter).
2. Intraday breakout trigger & volume pacing confirmation filter (>= 3.12x threshold).
3. Fixed 1% risk capital position sizing & 50/50 quantity splitting.
4. Order management via OMS backend (StateEngine + p3_position_legs table).
5. Automated Split Exit:
   - Scenario A: Leg 1 target (+2.0R) hit, 50% booked, Leg 2 stop ratcheted to Breakeven.
   - Scenario B: Leg 2 runner trailing with 10 EMA.
   - Scenario C: Initial SL hit before target (-1.0R full exit).
6. State tracking dictionary & SQLite schema synchronization.
========================================================================================
"""

import datetime
import math
import os
import sqlite3
import pytest
import numpy as np
import pandas as pd

from core.config import P3_BASE_CAPITAL, P3_RISK_PER_TRADE, P3_VOLUME_PACING_THRESHOLD
from core.data_quality import now_ist, IST
from engines.strategy_engine import (
    Portfolio3StrategyEngine,
    portfolio3_strategy_engine,
    evaluate_prebreakout_coil,
    calculate_volume_pacing,
    PORTFOLIO_TYPE
)
from execution.state_engine import state_engine


def generate_synthetic_coil_df(
    n_bars: int = 70,
    base_price: float = 100.0,
    vdu_ratio: float = 0.65,
    range_contraction_pct: float = 2.5,
    ema20_dist_pct: float = 1.0,
    uptrend: bool = True,
    vol_fade: bool = True,
    atr_squeeze: bool = True
) -> pd.DataFrame:
    """Generates synthetic daily OHLCV dataframe configured for hybrid squeeze coil testing."""
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="B")
    opens, highs, lows, closes, vols = [], [], [], [], []

    current_p = base_price
    for i in range(n_bars):
        drift = 0.3 if uptrend else -0.3
        current_p = max(20.0, current_p + drift)
        c = current_p

        if atr_squeeze and i < n_bars - 5:
            half_range = 2.5
        elif not atr_squeeze:
            half_range = 1.5
        else:
            half_range = 0.8

        if i == n_bars - 1:
            # T-1 Bar (Coil Candidate Bar)
            half_range = (c * (range_contraction_pct / 100.0)) / 2.0
            h = c + half_range
            l = c - half_range
            o = c - (half_range * 0.2)
            v = 1000000.0 * vdu_ratio
        elif i == n_bars - 2 and vol_fade:
            h = c + half_range
            l = c - half_range
            o = c - (half_range * 0.2)
            v = 1200000.0  # T-2
        elif i == n_bars - 3 and vol_fade:
            h = c + half_range
            l = c - half_range
            o = c - (half_range * 0.2)
            v = 1500000.0  # T-3 (1.5M > 1.2M > T-1 Volume)
        else:
            h = c + half_range
            l = c - half_range
            o = c - (half_range * 0.2)
            v = 1000000.0

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(v)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols
    }, index=dates)
    return df


# ------------------------------------------------------------------------------
# 1. WATCHLIST EVALUATION TESTS
# ------------------------------------------------------------------------------

def test_evaluate_prebreakout_coil_valid():
    """Test that a stock meeting all 6 rules is identified as a valid coil candidate."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=0.65, range_contraction_pct=2.5, uptrend=True)
    res = evaluate_prebreakout_coil(df, "TESTCOIL")

    assert res is not None
    assert res["symbol"] == "TESTCOIL"
    assert res["vdu_ratio"] <= 0.80
    assert res["range_contraction_pct"] <= 3.5
    assert res["ema20_dist_pct"] <= 4.0
    assert res["t_minus_1_close"] >= res["ema20"] * 0.98
    assert res["ema20"] > res["ema50"]
    assert res["vol_fade_passed"] == 1
    assert res["atr_squeeze_ratio"] <= 0.80
    # Check stop loss anchored at Base Low * 0.998
    expected_sl = round(res["base_low"] * 0.998, 2)
    assert res["initial_sl"] == expected_sl
    assert res["risk_per_share"] > 0.0


def test_evaluate_prebreakout_coil_fails_vdu():
    """Test that high volume (> 0.8x SMA) fails VDU filter."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=1.45, range_contraction_pct=2.0, uptrend=True)
    res = evaluate_prebreakout_coil(df, "FAIL_VDU")
    assert res is None


def test_evaluate_prebreakout_coil_fails_range_contraction():
    """Test that wide range (> 3.5%) fails range contraction filter."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=0.60, range_contraction_pct=5.5, uptrend=True)
    res = evaluate_prebreakout_coil(df, "FAIL_RANGE")
    assert res is None


def test_evaluate_prebreakout_coil_fails_downtrend():
    """Test that downtrend (20 EMA <= 50 SMA) is blocked by trend filter."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=0.60, range_contraction_pct=2.0, uptrend=False)
    res = evaluate_prebreakout_coil(df, "FAIL_TREND")
    assert res is None


def test_evaluate_prebreakout_coil_fails_vol_fade():
    """Test that failure of 3-day progressive volume fade disqualifies candidate."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=0.65, range_contraction_pct=2.0, uptrend=True, vol_fade=False)
    res = evaluate_prebreakout_coil(df, "FAIL_VOL_FADE")
    assert res is None


def test_evaluate_prebreakout_coil_fails_atr_squeeze():
    """Test that absence of dynamic volatility squeeze (ATR5/ATR20 > 0.80) fails setup."""
    df = generate_synthetic_coil_df(n_bars=70, vdu_ratio=0.65, range_contraction_pct=2.0, uptrend=True, atr_squeeze=False)
    res = evaluate_prebreakout_coil(df, "FAIL_ATR_SQUEEZE")
    assert res is None


# ------------------------------------------------------------------------------
# 2. INTRADAY TRIGGER & VOLUME PACING TESTS
# ------------------------------------------------------------------------------

def test_volume_pacing_calculation():
    """Test time-adjusted volume pacing relative to the intraday volume factor."""
    now_test = datetime.datetime(2026, 4, 15, 9, 45, 0, tzinfo=IST) # 30 mins elapsed
    avg_vol_20 = 1000000.0

    # Strong morning volume: 600,000 shares in first 30 mins
    pacing_high = calculate_volume_pacing(live_volume=600000.0, avg_volume_20=avg_vol_20, current_time=now_test)
    assert pacing_high >= 3.12

    # Weak morning volume: 50,000 shares in first 30 mins
    pacing_low = calculate_volume_pacing(live_volume=50000.0, avg_volume_20=avg_vol_20, current_time=now_test)
    assert pacing_low < 3.12


def test_evaluate_breakout_candidate_confirmed():
    """Test confirmed breakout when price crosses T-1 High AND pacing >= 3.12x."""
    watch_item = {
        "symbol": "CONFIRMED_STK",
        "t_minus_1_high": 105.0,
        "initial_sl": 98.0,
        "avg_vol_20": 1000000.0
    }
    quote_data = {
        "ltp": 105.5, # Crossed above 105.0
        "high": 106.0,
        "volume": 600000.0 # High pacing
    }
    now_test = datetime.datetime(2026, 4, 15, 9, 45, 0, tzinfo=IST)

    res = portfolio3_strategy_engine.evaluate_breakout_candidate(watch_item, quote_data, now_dt=now_test)
    assert res["price_crossed"] is True
    assert res["volume_confirmed"] is True
    assert res["should_enter"] is True
    assert res["reason"] == "CONFIRMED_BREAKOUT"


def test_evaluate_breakout_candidate_unconfirmed_volume():
    """Test unconfirmed breakout when price crosses T-1 High but volume pacing is inadequate."""
    watch_item = {
        "symbol": "UNCONFIRMED_STK",
        "t_minus_1_high": 105.0,
        "initial_sl": 98.0,
        "avg_vol_20": 1000000.0
    }
    quote_data = {
        "ltp": 105.5, # Crossed above 105.0
        "high": 105.5,
        "volume": 50000.0 # Low pacing (< 3.12x)
    }
    now_test = datetime.datetime(2026, 4, 15, 9, 45, 0, tzinfo=IST)

    res = portfolio3_strategy_engine.evaluate_breakout_candidate(watch_item, quote_data, now_dt=now_test)
    assert res["price_crossed"] is True
    assert res["volume_confirmed"] is False
    assert res["should_enter"] is False
    assert res["reason"] == "UNCONFIRMED_VOLUME_PACE"


# ------------------------------------------------------------------------------
# 3. POSITION SIZING & SPLIT EXIT ORDER MANAGEMENT TESTS
# ------------------------------------------------------------------------------

def test_position_sizing_fixed_1pct_risk():
    """Test position sizing based on fixed 1% risk capital model."""
    capital = 200000.0 # ₹2,00,000 capital -> ₹2,000 risk budget
    entry_price = 100.0
    initial_sl = 95.0 # Risk per share = ₹5.00

    total_qty, risk_per_share, risk_capital = portfolio3_strategy_engine.calculate_position_size(
        entry_price=entry_price,
        initial_sl=initial_sl,
        account_capital=capital
    )

    assert risk_per_share == 5.0
    # Expected quantity: 2000 / 5 = 400 shares
    assert total_qty == 400
    assert risk_capital == 2000.0


def test_split_exit_order_execution_and_lifecycle():
    """
    Comprehensive test of the 50/50 Split Exit lifecycle:
    1. Order execution: Leg 1 (50% target @ +2.0R) and Leg 2 (50% runner).
    2. Leg 1 Target Hit: Books 50%, moves Leg 2 SL to Breakeven.
    3. Leg 2 Runner Exits on 10 EMA trailing stop.
    """
    sym = "TEST_SPLIT_RUNNER"
    # Clean previous records if any
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM active_positions WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM trade_history WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM p3_position_legs WHERE symbol = ?", (sym,))

    eval_result = {
        "symbol": sym,
        "ltp": 100.0,
        "initial_sl": 95.0, # 1R = ₹5.00
        "volume_pacing": 3.85
    }

    # 1. Execute Breakout Order
    order_res = portfolio3_strategy_engine.execute_breakout_order(eval_result, bypass_gate=True)
    assert order_res is not None
    assert order_res["symbol"] == sym
    assert order_res["entry_price"] == 100.0
    assert order_res["initial_sl"] == 95.0
    # Target 1 = Entry + 2.0 * 5 = 110.0
    assert order_res["leg1_target"] == 110.0
    total_qty = order_res["total_quantity"]
    assert order_res["leg1_qty"] == total_qty // 2
    assert order_res["leg2_qty"] == total_qty - (total_qty // 2)

    # Verify active position exists in StateEngine
    active = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    pos = next((p for p in active if p["symbol"] == sym), None)
    assert pos is not None
    assert pos["quantity"] == total_qty
    assert pos["stop_loss"] == 95.0
    assert pos["target"] == 110.0

    # Verify p3_position_legs record
    with portfolio3_strategy_engine._get_connection() as conn:
        leg_row = conn.execute("SELECT * FROM p3_position_legs WHERE symbol = ?", (sym,)).fetchone()
        assert leg_row is not None
        assert leg_row["leg1_status"] == "OPEN"
        assert leg_row["leg2_status"] == "OPEN"
        assert leg_row["leg1_target_price"] == 110.0

    # 2. Simulate Target Hit for Leg 1 (High reaches 112.0)
    # Mock daily OHLCV
    mock_df = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [105.0, 112.0], # Target hit!
        "Low": [98.0, 101.0],
        "Close": [104.0, 111.0],
        "Volume": [1000000.0, 1500000.0]
    }, index=pd.date_range("2026-04-14", periods=2, freq="B"))

    # Mock daily OHLCV provider
    orig_fetch = portfolio3_strategy_engine.get_stock_daily_df
    portfolio3_strategy_engine.get_stock_daily_df = lambda s: mock_df

    try:
        portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=True)

        # Check that Leg 1 was booked
        active_after_t1 = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
        pos_t1 = next((p for p in active_after_t1 if p["symbol"] == sym), None)
        assert pos_t1 is not None
        assert pos_t1["t1_hit"] == 1
        assert pos_t1["quantity"] == order_res["leg2_qty"]
        # Stop loss should be at least Breakeven (100.0)
        assert pos_t1["stop_loss"] >= 100.0

        with portfolio3_strategy_engine._get_connection() as conn:
            leg_row_t1 = conn.execute("SELECT * FROM p3_position_legs WHERE symbol = ?", (sym,)).fetchone()
            assert leg_row_t1["leg1_status"] == "TARGET_HIT"
            assert leg_row_t1["overall_status"] == "PARTIAL_CLOSED"
            assert leg_row_t1["leg1_pnl"] > 0

        # 3. Simulate Leg 2 Runner Exit on 10 EMA Trailing Stop
        # Next candle drops below the trailing stop
        mock_df_exit = pd.DataFrame({
            "Open": [100.0, 102.0, 105.0],
            "High": [105.0, 112.0, 106.0],
            "Low": [98.0, 101.0, 97.0],
            "Close": [104.0, 111.0, 98.0], # Drops below 100.0 breakeven stop!
            "Volume": [1000000.0, 1500000.0, 1200000.0]
        }, index=pd.date_range("2026-04-14", periods=3, freq="B"))

        portfolio3_strategy_engine.get_stock_daily_df = lambda s: mock_df_exit
        closed_trades = portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=True)

        # Verify position is closed in active_positions
        active_final = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
        pos_final = next((p for p in active_final if p["symbol"] == sym), None)
        assert pos_final is None # Closed!

        # Verify p3_position_legs marked CLOSED
        with portfolio3_strategy_engine._get_connection() as conn:
            leg_row_final = conn.execute("SELECT * FROM p3_position_legs WHERE symbol = ?", (sym,)).fetchone()
            assert leg_row_final["overall_status"] == "CLOSED"
            assert leg_row_final["leg2_status"] == "TRAILED_OUT"
    finally:
        portfolio3_strategy_engine.get_stock_daily_df = orig_fetch


def test_initial_stop_hit_before_target():
    """Test scenario where initial stop is hit before Leg 1 target (+2.0R)."""
    sym = "TEST_INITIAL_STOP"
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM active_positions WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM trade_history WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM p3_position_legs WHERE symbol = ?", (sym,))

    eval_result = {
        "symbol": sym,
        "ltp": 100.0,
        "initial_sl": 95.0,
        "volume_pacing": 3.50
    }
    order_res = portfolio3_strategy_engine.execute_breakout_order(eval_result, bypass_gate=True)
    assert order_res is not None

    # Simulate immediate stop-out: Close falls to 94.0
    mock_df_stop = pd.DataFrame({
        "Open": [100.0, 96.0],
        "High": [101.0, 97.0],
        "Low": [98.0, 93.0],
        "Close": [99.0, 94.0], # Breaches 95.0 stop loss
        "Volume": [1000000.0, 1000000.0]
    }, index=pd.date_range("2026-04-14", periods=2, freq="B"))

    orig_fetch = portfolio3_strategy_engine.get_stock_daily_df
    portfolio3_strategy_engine.get_stock_daily_df = lambda s: mock_df_stop

    try:
        portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=True)

        # Check that position is closed
        active = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
        assert next((p for p in active if p["symbol"] == sym), None) is None

        # Verify dual legs both stopped out
        with portfolio3_strategy_engine._get_connection() as conn:
            leg_row = conn.execute("SELECT * FROM p3_position_legs WHERE symbol = ?", (sym,)).fetchone()
            assert leg_row["overall_status"] == "CLOSED"
            assert leg_row["leg1_status"] == "STOPPED_OUT"
            assert leg_row["leg2_status"] == "STOPPED_OUT"
    finally:
        portfolio3_strategy_engine.get_stock_daily_df = orig_fetch


def test_evaluate_breakout_inside_bar_rolling():
    """Test rolling inside bar when volume remains <= 0.80x and price holds 20 EMA."""
    watch_item = {
        "symbol": "ROLLING_STK",
        "trigger_high": 100.0,
        "base_low": 96.0,
        "avg_vol_20": 1000000.0,
        "ema20": 95.0,
        "days_tracked": 0
    }
    quote_data = {
        "ltp": 99.0, # Does not trigger (trigger is 100 * 1.001 = 100.1)
        "high": 99.5,
        "low": 96.5,
        "volume": 600000.0 # 0.60x <= 0.80x
    }
    res = portfolio3_strategy_engine.evaluate_breakout_candidate(watch_item, quote_data)
    assert res["should_enter"] is False
    assert res["is_inside_bar_roll"] is True
    assert res["next_days_tracked"] == 1
    assert res["new_trigger_high"] == 100.0
    assert res["new_base_low"] == 96.0


def test_evaluate_breakout_invalidation_below_20ema():
    """Test invalidation when price closes below 20 EMA."""
    watch_item = {
        "symbol": "INVALID_STK",
        "trigger_high": 100.0,
        "base_low": 96.0,
        "avg_vol_20": 1000000.0,
        "ema20": 95.0,
        "days_tracked": 0
    }
    quote_data = {
        "ltp": 94.0, # Below 20 EMA
        "high": 97.0,
        "low": 93.5,
        "volume": 500000.0
    }
    res = portfolio3_strategy_engine.evaluate_breakout_candidate(watch_item, quote_data)
    assert res["should_enter"] is False
    assert res["is_invalidated"] is True
    assert res["invalidation_reason"] == "CLOSED_BELOW_20EMA"


def test_friction_deduction_and_split_exit_r_multiple():
    """Verify that 0.30% round-trip friction is subtracted and R-multiple is accurately computed."""
    sym = "TEST_FRICTION"
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM active_positions WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM trade_history WHERE symbol = ? AND portfolio_type = ?", (sym, PORTFOLIO_TYPE))
            conn.execute("DELETE FROM p3_position_legs WHERE symbol = ?", (sym,))

    eval_result = {
        "symbol": sym,
        "ltp": 100.0,
        "initial_sl": 95.0, # 1R = 5.0
        "volume_pacing": 3.50
    }
    order_res = portfolio3_strategy_engine.execute_breakout_order(eval_result, bypass_gate=True)
    assert order_res is not None

    # Part 1 target reaches 110.0 (+2R)
    mock_df_t1 = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [105.0, 110.5],
        "Low": [98.0, 101.0],
        "Close": [104.0, 110.0],
        "Volume": [1000000.0, 1500000.0]
    }, index=pd.date_range("2026-04-14", periods=2, freq="B"))

    orig_fetch = portfolio3_strategy_engine.get_stock_daily_df
    portfolio3_strategy_engine.get_stock_daily_df = lambda s: mock_df_t1

    try:
        portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=True)

        # Part 2 runner trailed out at 105.0 (+1R)
        mock_df_exit = pd.DataFrame({
            "Open": [100.0, 102.0, 107.0],
            "High": [105.0, 110.5, 108.0],
            "Low": [98.0, 101.0, 104.0],
            "Close": [104.0, 110.0, 105.0],
            "Volume": [1000000.0, 1500000.0, 1200000.0]
        }, index=pd.date_range("2026-04-14", periods=3, freq="B"))

        portfolio3_strategy_engine.get_stock_daily_df = lambda s: mock_df_exit
        portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=True)

        with portfolio3_strategy_engine._get_connection() as conn:
            leg_row = conn.execute("SELECT * FROM p3_position_legs WHERE symbol = ?", (sym,)).fetchone()
            assert leg_row["overall_status"] == "CLOSED"
            # Part 1 return: (110 - 100) / 100 = 10%
            # Part 2 return: (105 - 100) / 100 = 5%
            # Blended gross return: 0.5 * 10 + 0.5 * 5 = 7.5%
            # Net return after 0.30% friction: 7.5% - 0.30% = 7.20%
            assert abs(leg_row["net_ret_pct"] - 7.20) < 0.05
            # Part 1 R: +2.0R, Part 2 R: +1.0R -> blended R: 0.5 * 2.0 + 0.5 * 1.0 = +1.5R
            assert abs(leg_row["r_multiple"] - 1.50) < 0.05
    finally:
        portfolio3_strategy_engine.get_stock_daily_df = orig_fetch

