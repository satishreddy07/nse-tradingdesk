"""
========================================================================================
Test Suite: Deterministic Trade Simulator Execution Fixtures
========================================================================================
Verifies 10 exact execution scenarios defined in Phase 2F:
1. Target reached before stop
2. Stop reached before target
3. Gap-down through stop (fills at open)
4. Gap-up through target (fills at open)
5. Same-candle conflict (conservative stop first)
6. Time exit at Day 10 close
7. Corrupted candle data (NaN or invalid prices)
8. Corporate action discontinuity (>40% unadjusted jump excluded)
9. Invalid entry price (<= 0 or NaN excluded)
10. MFE above/near target but realized loss (strictly distinguishes MFE from realized return)
Plus:
- NEXT_OPEN vs SAME_CLOSE entry models
- Statutory Indian equities cost schedule vs flat percentage model
========================================================================================
"""

import pytest
import numpy as np
import pandas as pd

from engines.trade_simulator import (
    TradeSimulator,
    TransactionCostModel,
    SIMULATOR_VERSION
)


def create_base_df(n_bars=15, base_price=100.0):
    """Creates a basic neutral OHLCV DataFrame for testing."""
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="B")
    return pd.DataFrame({
        "Open": [base_price] * n_bars,
        "High": [base_price + 1.0] * n_bars,
        "Low": [base_price - 1.0] * n_bars,
        "Close": [base_price] * n_bars,
        "Volume": [1000000.0] * n_bars
    }, index=dates)


def test_fixture_1_target_reached_before_stop():
    """Fixture 1: Target reached before stop loss."""
    df = create_base_df(15, 100.0)
    # Bar 1 hits target 105.0 cleanly without touching stop 96.5
    df.iloc[1, df.columns.get_loc("Open")] = 101.0
    df.iloc[1, df.columns.get_loc("High")] = 106.0
    df.iloc[1, df.columns.get_loc("Low")] = 99.0
    df.iloc[1, df.columns.get_loc("Close")] = 105.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_TARGET")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "PROFIT_TARGET"
    assert res["exit_price"] == 105.0
    assert res["gross_return_pct"] == 5.0
    assert res["net_return_pct"] > 0
    assert res["is_win"] == 1
    assert res["hold_days"] == 1


def test_fixture_2_stop_reached_before_target():
    """Fixture 2: Stop loss reached before profit target."""
    df = create_base_df(15, 100.0)
    # Bar 1 drops to 95.0, breaching stop 96.5 without touching target 105.0
    df.iloc[1, df.columns.get_loc("Open")] = 99.0
    df.iloc[1, df.columns.get_loc("High")] = 101.0
    df.iloc[1, df.columns.get_loc("Low")] = 95.0
    df.iloc[1, df.columns.get_loc("Close")] = 96.0

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_STOP")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "STOP_LOSS"
    assert res["exit_price"] == 96.5
    assert res["gross_return_pct"] == -3.5
    assert res["net_return_pct"] < -3.5
    assert res["is_win"] == 0
    assert res["hold_days"] == 1


def test_fixture_3_gap_down_through_stop():
    """Fixture 3: Gap-down opens below stop price (fills at Open, expanding loss)."""
    df = create_base_df(15, 100.0)
    # Stop is 96.5; next bar opens at 94.0
    df.iloc[1, df.columns.get_loc("Open")] = 94.0
    df.iloc[1, df.columns.get_loc("High")] = 95.0
    df.iloc[1, df.columns.get_loc("Low")] = 93.0
    df.iloc[1, df.columns.get_loc("Close")] = 94.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_GAP_DOWN")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "STOP_GAP"
    assert res["exit_price"] == 94.0
    assert res["gross_return_pct"] == -6.0
    assert res["is_win"] == 0


def test_fixture_4_gap_up_through_target():
    """Fixture 4: Gap-up opens above target price (fills at Open, expanding gain)."""
    df = create_base_df(15, 100.0)
    # Target is 105.0; next bar opens at 108.0
    df.iloc[1, df.columns.get_loc("Open")] = 108.0
    df.iloc[1, df.columns.get_loc("High")] = 109.0
    df.iloc[1, df.columns.get_loc("Low")] = 107.0
    df.iloc[1, df.columns.get_loc("Close")] = 108.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_GAP_UP")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "TARGET_GAP"
    assert res["exit_price"] == 108.0
    assert res["gross_return_pct"] == 8.0
    assert res["is_win"] == 1


def test_fixture_5_same_candle_conflict():
    """Fixture 5: Ambiguous bar breaching both target and stop (conservative stop first)."""
    df = create_base_df(15, 100.0)
    # Bar 1 has high 107.0 (>= 105.0) and low 95.0 (<= 96.5)
    df.iloc[1, df.columns.get_loc("Open")] = 100.0
    df.iloc[1, df.columns.get_loc("High")] = 107.0
    df.iloc[1, df.columns.get_loc("Low")] = 95.0
    df.iloc[1, df.columns.get_loc("Close")] = 103.0

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_CONFLICT")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "STOP_CONFLICT"
    assert res["exit_price"] == 96.5
    assert res["gross_return_pct"] == -3.5
    assert res["is_win"] == 0


def test_fixture_6_time_exit_at_day_10_close():
    """Fixture 6: Neither target nor stop reached; trade exits at Day 10 close."""
    df = create_base_df(15, 100.0)
    # Keep all bars within 98.0 to 103.0; Bar 10 closes at 101.5
    for k in range(1, 11):
        df.iloc[k, df.columns.get_loc("Open")] = 100.0
        df.iloc[k, df.columns.get_loc("High")] = 102.0
        df.iloc[k, df.columns.get_loc("Low")] = 98.5
        df.iloc[k, df.columns.get_loc("Close")] = 100.5
    df.iloc[10, df.columns.get_loc("Close")] = 101.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_TIME_EXPIRY")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "TIME_EXPIRY"
    assert res["exit_price"] == 101.5
    assert res["hold_days"] == 10
    assert res["gross_return_pct"] == 1.5


def test_fixture_7_corrupt_candle_data():
    """Fixture 7: Corrupted candle (NaN price or inverted high/low) produces ERROR."""
    df = create_base_df(15, 100.0)
    df.iloc[1, df.columns.get_loc("Close")] = np.nan

    sim = TradeSimulator()
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_CORRUPT")
    assert res["status"] == "ERROR"
    assert res["error_reason"] == "CORRUPT_CANDLE_DATA"

    # Inverted high/low
    df2 = create_base_df(15, 100.0)
    df2.iloc[1, df2.columns.get_loc("High")] = 90.0
    df2.iloc[1, df2.columns.get_loc("Low")] = 110.0
    res2 = sim.simulate_trade(df2, signal_idx=0, symbol="TEST_INVERTED")
    assert res2["status"] == "ERROR"
    assert res2["error_reason"] == "CORRUPT_CANDLE_DATA"


def test_fixture_8_corporate_action_discontinuity():
    """Fixture 8: Unadjusted price jump >= 40% is safely EXCLUDED."""
    # Discontinuity at entry
    df = create_base_df(15, 100.0)
    df.iloc[0, df.columns.get_loc("Close")] = 50.0  # 50% drop from prev bar
    sim = TradeSimulator()
    res = sim.simulate_trade(df, signal_idx=1, symbol="TEST_CORP_ENTRY")
    # If bar 0 was 50 and bar 1 entry is 100, jump is 100% -> EXCLUDED
    assert res["status"] == "EXCLUDED"
    assert res["error_reason"] == "CORP_ACTION_DISCONTINUITY"

    # Discontinuity during forward holding
    df2 = create_base_df(15, 100.0)
    df2.iloc[2, df2.columns.get_loc("Open")] = 50.0  # 50% overnight drop
    res2 = sim.simulate_trade(df2, signal_idx=0, symbol="TEST_CORP_HOLD")
    assert res2["status"] == "EXCLUDED"
    assert res2["error_reason"] == "CORP_ACTION_DISCONTINUITY"


def test_fixture_9_invalid_entry_price():
    """Fixture 9: Zero, negative, or NaN entry price produces ERROR."""
    df = create_base_df(15, 0.0)
    sim = TradeSimulator()
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_ZERO")
    assert res["status"] == "ERROR"
    assert res["error_reason"] == "INVALID_ENTRY_PRICE"


def test_fixture_10_mfe_above_near_target_but_realized_loss():
    """
    Fixture 10: Peak excursion reaches +4.9% (MFE) but fails to reach +5.0% target,
    subsequently hitting stop loss. Proves that MFE is strictly isolated from realized P&L.
    """
    df = create_base_df(15, 100.0)
    # Bar 1 rallies to 104.9 (+4.9%), close 104.0
    df.iloc[1, df.columns.get_loc("Open")] = 100.5
    df.iloc[1, df.columns.get_loc("High")] = 104.9
    df.iloc[1, df.columns.get_loc("Low")] = 99.5
    df.iloc[1, df.columns.get_loc("Close")] = 104.0

    # Bar 2 crashes to 95.0, hitting stop at 96.5
    df.iloc[2, df.columns.get_loc("Open")] = 103.0
    df.iloc[2, df.columns.get_loc("High")] = 103.5
    df.iloc[2, df.columns.get_loc("Low")] = 95.0
    df.iloc[2, df.columns.get_loc("Close")] = 95.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="SAME_CLOSE")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_MFE_NOT_REALIZED")

    assert res["status"] == "COMPLETED"
    assert res["exit_reason"] == "STOP_LOSS"
    assert res["gross_return_pct"] == -3.5
    assert res["net_return_pct"] < -3.5
    assert res["is_win"] == 0
    # MFE correctly recorded as diagnostic excursion (+4.9%) without contaminating realized return
    assert res["mfe_pct"] == 4.9


def test_next_open_entry_model():
    """Verifies NEXT_OPEN enters at bar T+1 Open rather than bar T Close."""
    df = create_base_df(15, 100.0)
    # Signal at bar 0 (Close = 100.0); Bar 1 opens at 102.0
    df.iloc[1, df.columns.get_loc("Open")] = 102.0
    # Bar 2 hits target (102.0 * 1.05 = 107.1)
    df.iloc[2, df.columns.get_loc("Open")] = 103.0
    df.iloc[2, df.columns.get_loc("High")] = 108.0
    df.iloc[2, df.columns.get_loc("Low")] = 101.0
    df.iloc[2, df.columns.get_loc("Close")] = 107.5

    sim = TradeSimulator(target_pct=5.0, stop_pct=-3.5, horizon_days=10, entry_model="NEXT_OPEN")
    res = sim.simulate_trade(df, signal_idx=0, symbol="TEST_NEXT_OPEN")

    assert res["status"] == "COMPLETED"
    assert res["entry_price"] == 102.0
    assert res["target_price"] == 107.10
    assert res["exit_reason"] == "PROFIT_TARGET"
    assert res["exit_price"] == 107.10
    assert res["gross_return_pct"] == 5.0


def test_cost_models_statutory_vs_flat():
    """Verifies exact itemized statutory costs schedule and flat percent fallback."""
    stat_model = TransactionCostModel(cost_type="STATUTORY_NSE_CASH")
    flat_model = TransactionCostModel(cost_type="FLAT_PERCENT", flat_roundtrip_pct=0.35)

    c_stat = stat_model.calculate_costs(entry_price=1000.0, exit_price=1050.0, shares=100)
    assert c_stat["total_charges_pct"] > 0.20
    assert "brokerage_inr" in c_stat["breakdown"]
    assert "stt_inr" in c_stat["breakdown"]
    assert "exchange_charges_inr" in c_stat["breakdown"]

    c_flat = flat_model.calculate_costs(entry_price=1000.0, exit_price=1050.0, shares=100)
    assert c_flat["total_charges_pct"] == 0.35
    assert c_flat["total_charges_inr"] == 350.0
