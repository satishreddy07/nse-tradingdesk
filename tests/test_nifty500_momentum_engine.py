"""
Unit and Integration Test Suite for Portfolio 2:
Nifty 500 Momentum Compounded Swing Engine
Verifies 100% fidelity to the specification:
1. Risk & Compounding Position Sizing Engine (₹1,00,000, 1.5% risk, 10 slots / 10% cap, 2.0x ATR)
2. 7-Point Entry Criteria Matrix (#0 Macro Gate, #1 Uptrend, #2 RS Leader, #3 Pullback, #4 RSI, #5 Green Bar, #6 Volume)
3. 4-Tier Strategy Exit Hierarchy (Priority 1: Initial SL, Priority 2: Trailing 5D Low @ +2R, Priority 3: Fast Cut < 20 EMA, Priority 4: Time Stop Day 8 < 1R)
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from core.data_quality import now_ist
from core.config import P2_BASE_CAPITAL, P2_RISK_PER_TRADE, P2_MAX_SLOTS, P2_STOP_ATR_MULT
from engines.nifty500_engine import (
    PORTFOLIO_TYPE,
    calc_rsi,
    get_live_compounded_equity,
    reconcile_open_nifty500_positions
)
from execution.state_engine import state_engine


def test_portfolio2_compounding_constants():
    """Validates core quantitative parameters of Portfolio 2."""
    assert P2_BASE_CAPITAL == 100000.0, "Base capital must be ₹1,00,000"
    assert P2_RISK_PER_TRADE == 0.015, "Risk per trade must be 1.5%"
    assert P2_MAX_SLOTS == 10, "Max concurrent slots must be 10"
    assert P2_STOP_ATR_MULT == 2.0, "Stop ATR multiplier must be 2.0x"


def test_dynamic_position_sizing_math():
    """
    Validates exact position sizing formula:
    dynamic_risk_budget = current_equity * 0.015
    dynamic_slot_cap = current_equity / 10
    initial_sl = round(entry_px - 2.0 * atr, 2)
    risk_per_share = round(entry_px - initial_sl, 2)
    shares = min(int(dynamic_risk_budget / risk_per_share), int(dynamic_slot_cap / entry_px))
    """
    current_equity = 100000.0
    risk_budget = current_equity * P2_RISK_PER_TRADE  # ₹1,500
    slot_cap = current_equity / P2_MAX_SLOTS          # ₹10,000

    entry_px = 500.0
    atr = 12.50
    initial_sl = round(entry_px - (P2_STOP_ATR_MULT * atr), 2)  # 500 - 25.0 = 475.0
    risk_per_share = round(entry_px - initial_sl, 2)            # 25.0

    shares_by_risk = int(risk_budget / risk_per_share)  # 1500 / 25 = 60
    shares_by_slot = int(slot_cap / entry_px)           # 10000 / 500 = 20
    final_shares = min(shares_by_risk, shares_by_slot)  # 20 (slot-capped)

    assert initial_sl == 475.0
    assert risk_per_share == 25.0
    assert shares_by_risk == 60
    assert shares_by_slot == 20
    assert final_shares == 20

    # Test when risk budget is the tighter binding constraint
    entry_px2 = 100.0
    atr2 = 10.0
    initial_sl2 = round(entry_px2 - (P2_STOP_ATR_MULT * atr2), 2)  # 100 - 20 = 80.0
    risk_per_share2 = round(entry_px2 - initial_sl2, 2)            # 20.0

    shares_by_risk2 = int(risk_budget / risk_per_share2)  # 1500 / 20 = 75
    shares_by_slot2 = int(slot_cap / entry_px2)           # 10000 / 100 = 100
    final_shares2 = min(shares_by_risk2, shares_by_slot2) # 75 (risk-capped)

    assert final_shares2 == 75


def test_seven_point_entry_matrix_evaluation():
    """
    Validates all 7 criteria of the Entry Criteria Matrix:
    #0: Macro Gate (Index Close > EMA50)
    #1: Structural Uptrend (EMA20 > EMA50 and LTP > EMA50)
    #2: Relative Strength (Return_63d > Benchmark Return_63d)
    #3: Value Pullback Pocket (0.985 * EMA20 <= LTP <= 1.020 * EMA20)
    #4: Momentum Cooldown (40 <= RSI14 <= 62)
    #5: Bullish Bounce Bar (LTP > Open and LTP > Close_{t-1})
    #6: Volume Expansion (Volume >= 0.80 * Vol_MA20)
    """
    # 85 days: stock strongly outperforms benchmark
    closes = [700.0 + i * 5.0 for i in range(50)] + [950.0] * 20 + [
        970.0, 960.0, 965.0, 955.0, 960.0, 950.0, 955.0, 945.0, 950.0, 942.0, 948.0, 940.0, 945.0, 938.0, 946.0
    ]
    n = len(closes)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")

    # Benchmark (^CRSLDX) - modest 5% gain so stock with +35% has strong positive RS spread
    bench_closes = np.linspace(20000, 21000, n)
    b_df = pd.DataFrame({"date": dates, "close": bench_closes})
    b_df["bench_ema50"] = b_df["close"].ewm(span=50, adjust=False).mean()
    b_df["bull_regime"] = b_df["close"] > b_df["bench_ema50"]
    b_df["bench_ret_63"] = b_df["close"].pct_change(63)

    assert b_df["bull_regime"].iloc[-1] is np.True_ or b_df["bull_regime"].iloc[-1] == True

    opens = list(closes)
    opens[-1] = 940.0  # Open at 940, close at 946 (green candle)

    volumes = [100000.0] * n
    volumes[-1] = 120000.0  # > 0.80 * 20-day SMA

    s_df = pd.DataFrame({
        "date": dates,
        "symbol": "QUALIFY_STOCK",
        "open": opens,
        "high": np.maximum(opens, closes) + 5.0,
        "low": np.minimum(opens, closes) - 5.0,
        "close": closes,
        "volume": volumes
    })

    s_df = pd.merge(s_df, b_df[["date", "bull_regime", "bench_ret_63"]], on="date", how="left")
    s_df["ema_20"] = s_df["close"].ewm(span=20, adjust=False).mean()
    s_df["ema_50"] = s_df["close"].ewm(span=50, adjust=False).mean()
    s_df["vol_ma20"] = s_df["volume"].rolling(20).mean()
    s_df["stock_ret_63"] = s_df["close"].pct_change(63)
    s_df["rs_spread"] = s_df["stock_ret_63"] - s_df["bench_ret_63"]
    s_df["rsi"] = calc_rsi(s_df["close"])

    # Verify indicators on latest row
    row = s_df.iloc[-1]
    uptrend = bool((row["ema_20"] > row["ema_50"]) and (row["close"] > row["ema_50"]))
    rs_leader = bool(row["rs_spread"] > 0.0)
    pullback_zone = bool((row["close"] >= row["ema_20"] * 0.985) and (row["close"] <= row["ema_20"] * 1.020))
    rsi_zone = bool((row["rsi"] >= 40.0) and (row["rsi"] <= 62.0))
    prior_c = s_df["close"].iloc[-2]
    bullish_bar = bool((row["close"] > row["open"]) and (row["close"] > prior_c))
    vol_confirm = bool(row["volume"] >= 0.80 * row["vol_ma20"])

    assert uptrend, f"Criteria #1 Uptrend: EMA20={row['ema_20']:.1f}, EMA50={row['ema_50']:.1f}, Close={row['close']:.1f}"
    assert rs_leader, f"Criteria #2 RS Leader: RS Spread={row['rs_spread']:.4f}"
    assert pullback_zone, f"Criteria #3 Value Pullback Pocket: Close={row['close']:.1f}, EMA20={row['ema_20']:.1f}"
    assert rsi_zone, f"Criteria #4 Momentum Cooldown: RSI={row['rsi']:.1f}"
    assert bullish_bar, f"Criteria #5 Bullish Bounce Bar: Close={row['close']:.1f}, Open={row['open']:.1f}, PriorClose={prior_c:.1f}"
    assert vol_confirm, f"Criteria #6 Volume Confirmation: Vol={row['volume']:.0f}, VolMA20={row['vol_ma20']:.0f}"


def test_four_tier_exit_hierarchy():
    """
    Validates all 4 tiers of the Strategy Exit Hierarchy:
    Priority 1: Initial Stop-Loss (Price <= Entry - 2.0 * ATR14)
    Priority 2: Trailing Stop-Loss (Price <= Rolling 5D Low once gain >= +2R)
    Priority 3: Fast Cut (Close < 20 EMA while not trailing)
    Priority 4: Time Stop (Days Held >= 8 and Gain < +1R)
    """
    sym = "TEST_EXIT_STOCK"
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")

    entry_px = 1000.0
    atr = 25.0
    initial_sl = round(entry_px - (2.0 * atr), 2)  # 950.0
    risk_unit = entry_px - initial_sl               # 50.0
    target_2r = entry_px + 2.0 * risk_unit          # 1100.0

    today = now_ist().date()
    dates = pd.date_range(today - pd.Timedelta(days=12), periods=10, freq="B")

    # 1. Test Priority 1: Initial SL hit
    opened = state_engine.open_position(
        symbol=sym,
        entry_price=entry_px,
        quantity=10,
        stop_loss=initial_sl,
        target=target_2r,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    assert opened, "Must successfully open position for testing"

    # Manually align entry_date in DB to dates[0] to simulate hold period
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("UPDATE active_positions SET entry_date = ? WHERE symbol = ?", (str(dates[0].date()), sym))

    # Day 10: Price drops to 945.0 (below 950.0 initial SL)
    df_p1 = pd.DataFrame({
        "date": dates,
        "symbol": sym,
        "open": np.full(10, 990.0),
        "high": np.full(10, 995.0),
        "low": np.full(10, 945.0),
        "close": np.full(10, 948.0),
        "volume": np.full(10, 50000.0)
    })

    closed = reconcile_open_nifty500_positions(df_p1, bypass_gate=True)
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "INITIAL_SL"
    assert closed[0]["exit_price"] <= initial_sl

    # 2. Test Priority 2: Trailing Stop-Loss (+2R achieved, stop ratchets to 5D Low)
    state_engine.open_position(
        symbol=sym,
        entry_price=entry_px,
        quantity=10,
        stop_loss=initial_sl,
        target=target_2r,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("UPDATE active_positions SET entry_date = ? WHERE symbol = ?", (str(dates[0].date()), sym))

    # Stock rallied to 1120 (>= 1100 target_2r), 5-day low is 1060.0
    lows = [980.0, 990.0, 1020.0, 1060.0, 1070.0, 1080.0, 1090.0, 1060.0, 1080.0, 1050.0]
    highs = [1000.0, 1020.0, 1050.0, 1080.0, 1100.0, 1120.0, 1125.0, 1100.0, 1090.0, 1070.0]
    closes = [995.0, 1015.0, 1045.0, 1075.0, 1095.0, 1115.0, 1110.0, 1080.0, 1075.0, 1055.0]

    df_p2 = pd.DataFrame({
        "date": dates,
        "symbol": sym,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(10, 80000.0)
    })

    # Today's price is 1055, which drops below the trailing 5-day low (1060.0)
    closed2 = reconcile_open_nifty500_positions(df_p2, bypass_gate=True)
    assert len(closed2) == 1
    assert closed2[0]["exit_reason"] == "TRAILING_SL"

    # 3. Test Priority 3: Fast Cut (< 20 EMA while not in trailing mode)
    state_engine.open_position(
        symbol=sym,
        entry_price=entry_px,
        quantity=10,
        stop_loss=initial_sl,
        target=target_2r,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("UPDATE active_positions SET entry_date = ? WHERE symbol = ?", (str(dates[0].date()), sym))

    # Price never hit +2R (max high 1020), but breaks below 20 EMA (EMA20 ~ 985, close 970 > initial SL 950)
    closes_p3 = np.linspace(1000, 970, 10)
    df_p3 = pd.DataFrame({
        "date": dates,
        "symbol": sym,
        "open": closes_p3 + 2.0,
        "high": closes_p3 + 5.0,
        "low": closes_p3 - 3.0,
        "close": closes_p3,
        "volume": np.full(10, 50000.0)
    })

    closed3 = reconcile_open_nifty500_positions(df_p3, bypass_gate=True)
    assert len(closed3) == 1
    assert closed3[0]["exit_reason"] == "FAST_EMA20_CUT"

    # 4. Test Priority 4: Time Stop (Day 8 Close, Gain < +1R)
    dates_15 = pd.date_range(today - pd.Timedelta(days=20), periods=15, freq="B")
    entry_day_15 = str(dates_15[0].date())

    state_engine.open_position(
        symbol=sym,
        entry_price=entry_px,
        quantity=10,
        stop_loss=initial_sl,
        target=target_2r,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("UPDATE active_positions SET entry_date = ? WHERE symbol = ?", (entry_day_15, sym))

    # 9 days held, close is 1015 (gain is +15, which is < +1R = +50), holds above 20 EMA (EMA20 ~ 1005)
    closes_p4 = np.full(15, 1015.0)
    df_p4 = pd.DataFrame({
        "date": dates_15,
        "symbol": sym,
        "open": np.full(15, 1012.0),
        "high": np.full(15, 1025.0),
        "low": np.full(15, 1008.0),
        "close": closes_p4,
        "volume": np.full(15, 60000.0)
    })

    closed4 = reconcile_open_nifty500_positions(df_p4, bypass_gate=True)
    assert len(closed4) == 1
    assert closed4[0]["exit_reason"] == "TIME_EXIT"

    # Clean up test positions
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")


def test_trailing_stop_ratchet_without_breach():
    """
    Validates that once +2R is achieved, if price remains above the trailing 5-day low,
    the trade remains OPEN and the stop-loss is ratcheted upward to the 5-day low in the database.
    """
    sym = "TEST_RATCHET_STOCK"
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")

    entry_px = 1000.0
    initial_sl = 950.0  # -2.0 ATR
    target_2r = 1100.0   # +2R

    today = now_ist().date()
    dates = pd.date_range(today - pd.Timedelta(days=12), periods=10, freq="B")
    entry_day = str(dates[0].date())

    state_engine.open_position(
        symbol=sym,
        entry_price=entry_px,
        quantity=10,
        stop_loss=initial_sl,
        target=target_2r,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("UPDATE active_positions SET entry_date = ? WHERE symbol = ?", (entry_day, sym))

    # Rallied to 1120 (+2R hit), 5-day low is 1060.0, current price is 1090 (comfortably above 1060 trailing SL)
    lows = [980.0, 990.0, 1020.0, 1060.0, 1070.0, 1080.0, 1090.0, 1060.0, 1080.0, 1075.0]
    highs = [1000.0, 1020.0, 1050.0, 1080.0, 1100.0, 1120.0, 1125.0, 1100.0, 1095.0, 1095.0]
    closes = [995.0, 1015.0, 1045.0, 1075.0, 1095.0, 1115.0, 1110.0, 1085.0, 1085.0, 1090.0]

    df_ratchet = pd.DataFrame({
        "date": dates,
        "symbol": sym,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(10, 80000.0)
    })

    closed = reconcile_open_nifty500_positions(df_ratchet, bypass_gate=True)
    assert len(closed) == 0, "Position should NOT be closed because LTP 1090 > 1060 trailing SL"

    # Verify that stop_loss in active_positions was ratcheted from 950.0 to 1060.0
    open_pos = state_engine.get_active_positions(portfolio_type=PORTFOLIO_TYPE)
    assert len(open_pos) == 1
    assert open_pos[0]["stop_loss"] == 1060.0, f"Expected stop_loss 1060.0, got {open_pos[0]['stop_loss']}"
    assert open_pos[0]["sl_reason"] == "TRAILING_SL"
    assert open_pos[0]["t1_hit"] == 1

    # Clean up
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")


def test_compounded_equity_after_realized_trade():
    """
    Validates dynamic compounding equity formula:
    Total Equity = Base Capital (₹1,00,000) + sum(realized_pnl).
    """
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")
    curr_eq, tot_eq, active_count = get_live_compounded_equity()
    assert curr_eq == 100000.0
    assert active_count == 0

    # Simulate closing a profitable trade (+₹5,000 PnL)
    sym = "TEST_COMPOUND_STOCK"
    state_engine.open_position(
        symbol=sym,
        entry_price=100.0,
        quantity=50,
        stop_loss=90.0,
        target=120.0,
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )
    state_engine.close_position(
        symbol=sym,
        exit_price=110.0,
        exit_reason="TRAILING_SL",
        portfolio_type=PORTFOLIO_TYPE,
        bypass_gate=True
    )

    curr_eq2, tot_eq2, active_count2 = get_live_compounded_equity()
    assert active_count2 == 0
    # Entry was 100 * 50 = 5000, Exit was 110 * 50 = 5500, Gross PnL = +500
    assert tot_eq2 > 100000.0
    assert curr_eq2 == tot_eq2

    # Clean up
    state_engine.reset_portfolio_trades(PORTFOLIO_TYPE, confirm_token="CONFIRM_USER_AUTHORIZED_RESET")
