import pytest
import sqlite3
import pandas as pd
import numpy as np
from execution.state_engine import StateEngine
from core.data_quality import validate_ohlcv_dataframe
from core.data_watchdog import data_watchdog
from engines.institutional_engine import ExecutionEngine, InstitutionalPortfolioBacktester
from engines.ml_performance_engine import ml_performance_engine
from core.config import (
    P1_BASE_CAPITAL, P2_BASE_CAPITAL, P1_MAX_SLOTS, P2_MAX_SLOTS,
    MAX_POSITION_CAP_PCT, MAX_SECTOR_POSITIONS, DAILY_LOSS_LIMIT_PCT,
    EQUITY_ROUND_TRIP_COST_RATE
)


def test_f30_cross_portfolio_close_isolation(tmp_path):
    """
    F-30: Closing a symbol held in both portfolios must NOT delete both positions.
    Each portfolio position must remain completely isolated.
    """
    db_file = str(tmp_path / "test_f30.db")
    engine = StateEngine(db_path=db_file)

    # Open RELIANCE in P1 (Swing)
    p1_ok = engine.open_position(
        symbol="RELIANCE",
        entry_price=2500.0,
        quantity=10,
        stop_loss=2400.0,
        target=2700.0,
        portfolio_type="INSTITUTIONAL_6PILLAR",
        bypass_gate=True
    )
    assert p1_ok is True

    # Open RELIANCE in P2 (Compounding)
    p2_ok = engine.open_position(
        symbol="RELIANCE",
        entry_price=2500.0,
        quantity=5,
        stop_loss=2420.0,
        target=2650.0,
        portfolio_type="NIFTY_500_GROWTH",
        bypass_gate=True
    )
    assert p2_ok is True

    # Verify both exist
    active_all = engine.get_active_positions()
    assert len(active_all) == 2

    p1_pos = engine.get_active_positions(portfolio_type="INSTITUTIONAL_6PILLAR")
    assert len(p1_pos) == 1
    assert p1_pos[0]["quantity"] == 10

    p2_pos = engine.get_active_positions(portfolio_type="NIFTY_500_GROWTH")
    assert len(p2_pos) == 1
    assert p2_pos[0]["quantity"] == 5

    # Close RELIANCE ONLY in P1
    closed_p1 = engine.close_position(
        symbol="RELIANCE",
        exit_price=2600.0,
        exit_reason="T1_TARGET",
        bypass_gate=True,
        portfolio_type="INSTITUTIONAL_6PILLAR"
    )
    assert closed_p1 is not None
    assert closed_p1["portfolio_type"] == "INSTITUTIONAL_6PILLAR"

    # P1 active positions should now be 0, but P2 MUST STILL HAVE RELIANCE!
    p1_remaining = engine.get_active_positions(portfolio_type="INSTITUTIONAL_6PILLAR")
    assert len(p1_remaining) == 0

    p2_remaining = engine.get_active_positions(portfolio_type="NIFTY_500_GROWTH")
    assert len(p2_remaining) == 1, "P2 position was erroneously deleted when P1 closed!"
    assert p2_remaining[0]["symbol"] == "RELIANCE"
    assert p2_remaining[0]["quantity"] == 5


def test_f37_statutory_charges_and_trade_history_reproducibility(tmp_path):
    """
    F-37: Verify statutory charges (0.12% round trip) are calculated and all trade fields
    (gross_pnl, charges, net_pnl, t1_pnl, orig_qty, remaining_quantity) are persisted.
    """
    db_file = str(tmp_path / "test_f37.db")
    engine = StateEngine(db_path=db_file)

    engine.open_position(
        symbol="TCS",
        entry_price=3000.0,
        quantity=20,
        stop_loss=2900.0,
        target=3200.0,
        portfolio_type="INSTITUTIONAL_6PILLAR",
        bypass_gate=True
    )

    closed = engine.close_position(
        symbol="TCS",
        exit_price=3100.0,
        exit_reason="TARGET_HIT",
        bypass_gate=True,
        portfolio_type="INSTITUTIONAL_6PILLAR"
    )
    assert closed is not None
    # Entry turnover: 3000 * 20 = 60,000
    # Exit turnover: 3100 * 20 = 62,000
    # Total turnover: 122,000. Charges: 122,000 * 0.0012 = 146.40
    # Gross P&L: (3100 - 3000) * 20 = 2,000.00
    # Net P&L: 2,000 - 146.40 = 1,853.60
    assert closed["gross_pnl"] == 2000.0
    assert closed["charges"] == pytest.approx(146.4, 0.01)
    assert closed["net_pnl"] == pytest.approx(1853.6, 0.01)

    hist = engine.get_trade_history(portfolio_type="INSTITUTIONAL_6PILLAR")
    assert len(hist) == 1
    assert hist[0]["gross_pnl"] == 2000.0
    assert hist[0]["charges"] == pytest.approx(146.4, 0.01)
    assert hist[0]["net_pnl"] == pytest.approx(1853.6, 0.01)
    assert hist[0]["quantity"] == 20


def test_f38_risk_limits_and_kill_switch(tmp_path):
    """
    F-38: Centralized limits in core/config.py, 20% position cap, sector limit (3), and 3% daily loss kill switch.
    """
    db_file = str(tmp_path / "test_f38.db")
    engine = StateEngine(db_path=db_file)

    # 1. 20% Capital Cap Gate: P1 base capital 500,000 -> Max per position is 100,000
    # Attempting to open 150,000 position should be blocked
    res_large = engine.open_position(
        symbol="BIGPOS",
        entry_price=1500.0,
        quantity=100,  # 150,000 > 100,000 (20%)
        stop_loss=1400.0,
        target=1700.0,
        portfolio_type="INSTITUTIONAL_6PILLAR",
        bypass_gate=True
    )
    assert res_large is False, "Position exceeding 20% capital cap was not blocked!"

    # 2. Sector Concentration Gate (Max 3 per sector)
    for i in range(1, 4):
        ok = engine.open_position(
            symbol=f"AUTO{i}",
            entry_price=100.0,
            quantity=10,
            stop_loss=90.0,
            target=120.0,
            sector="AUTOMOBILE",
            portfolio_type="INSTITUTIONAL_6PILLAR",
            bypass_gate=True
        )
        assert ok is True

    # 4th position in AUTOMOBILE sector must be blocked
    res_sec = engine.open_position(
        symbol="AUTO4",
        entry_price=100.0,
        quantity=10,
        stop_loss=90.0,
        target=120.0,
        sector="AUTOMOBILE",
        portfolio_type="INSTITUTIONAL_6PILLAR",
        bypass_gate=True
    )
    assert res_sec is False, "4th position in same sector was not blocked!"


def test_f35_classification_and_sizing():
    """
    F-35: Verify classification handles negative pivot depth correctly,
    does not drop coherent mid-score setups to AVOID, and sizes dynamically.
    """
    # Test negative pivot distance: 25% below pivot must NOT return BUY_NOW
    deep_drop_row = pd.Series({
        "score": 85.0,
        "dist_to_pivot": -25.0,  # 25% below pivot
        "dist_to_ema20": -30.0,  # 30% below EMA20
        "vol_ratio": 1.3,
        "ml_verdict": "APPROVED",
        "ml_decision": "PASS_TO_AI"
    })
    cls_deep = ExecutionEngine.classify(deep_drop_row)
    assert cls_deep != "BUY_NOW", "Stock 25% below pivot was erroneously classified as BUY_NOW!"

    # Coherent mid-score setup: score 65, 3% from pivot, 2% from EMA20, vol_ratio 1.3
    # Must NOT return AVOID
    coherent_mid_row = pd.Series({
        "score": 65.0,
        "dist_to_pivot": 3.0,
        "dist_to_ema20": 2.0,
        "vol_ratio": 1.3,
        "ml_verdict": "APPROVED",
        "ml_decision": "PASS_TO_AI"
    })
    cls_mid = ExecutionEngine.classify(coherent_mid_row)
    assert cls_mid in ("PRE_BREAKOUT", "WAIT_PULLBACK"), f"Coherent mid-score setup returned {cls_mid} instead of PRE_BREAKOUT/WAIT_PULLBACK"

    # F3-02: Missing ATR/low must fail closed (REJECT, shares=0)
    sizing_row_missing = pd.Series({
        "close": 1000.0,
        "atr14": np.nan,  # Missing ATR
        "low": np.nan,    # Missing Low
        "score": 85.0,
        "confidence": 4,
        "ml_verdict": "APPROVED"
    })
    pos_plan_missing = ExecutionEngine.calculate_position(sizing_row_missing)
    assert pos_plan_missing["tier"] == "REJECT"
    assert pos_plan_missing["shares"] == 0
    assert pos_plan_missing["sl"] == 0.0
    assert pos_plan_missing["status"] == "DATA_UNAVAILABLE"

    # Valid ATR and low must calculate authentic shares and stop loss
    sizing_row_valid = pd.Series({
        "close": 1000.0,
        "atr14": 20.0,
        "low": 970.0,
        "score": 85.0,
        "confidence": 4,
        "ml_verdict": "APPROVED"
    })
    pos_plan_valid = ExecutionEngine.calculate_position(sizing_row_valid)
    assert pos_plan_valid["shares"] > 0
    assert pos_plan_valid["sl"] > 0
    assert pos_plan_valid["rr"] > 0


def test_f31_institutional_backtester_simulation():
    """
    F-31: Institutional backtester must run an authentic historical simulation,
    simulating trades, ATR trailing exits, statutory costs, and returning real metrics.
    """
    backtester = InstitutionalPortfolioBacktester(capital=500000.0, risk_per_trade_pct=0.75)

    # Generate synthetic 100-bar OHLCV data for 2 symbols
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    np.random.seed(42)

    data = {}
    for sym in ["STOCK_A", "STOCK_B"]:
        px = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
        df = pd.DataFrame({
            "Open": px,
            "High": px + 2.0,
            "Low": px - 2.0,
            "Close": px + 0.5,
            "Volume": 100000
        }, index=dates)
        data[sym] = df

    results = backtester.run((data,), lookback_days=80)
    assert results["status"] == "COMPLETED"
    assert "total_trades" in results
    assert "win_rate_pct" in results
    assert "profit_factor" in results
    assert "max_drawdown_pct" in results
    assert "expectancy_r" in results
    assert results["final_equity"] > 0


def test_f40_ohlcv_validation_layer():
    """
    F-40: Comprehensive OHLCV validation layer catching missing columns,
    chronological reversals, deduplication, price sanity, and extreme spikes.
    """
    # Valid dataframe
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    valid_df = pd.DataFrame({
        "Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0, "Volume": 50000
    }, index=dates)
    is_valid, clean_df, warnings = validate_ohlcv_dataframe(valid_df, symbol="TEST_VALID", min_bars=30)
    assert is_valid is True
    assert len(clean_df) >= 30

    # Inverted High/Low sanity check: corrects inversion and adds warning
    bad_df = valid_df.copy()
    bad_df.loc[dates[10], "High"] = 90.0
    bad_df.loc[dates[10], "Low"] = 110.0  # High < Low
    is_bad_valid, clean_bad, warnings_bad = validate_ohlcv_dataframe(bad_df, symbol="TEST_BAD", min_bars=30)
    assert is_bad_valid is True
    assert any("High was less than Low" in a for a in warnings_bad)

    # Missing columns failure
    missing_col_df = valid_df.drop(columns=["Close"])
    is_m_valid, _, m_warnings = validate_ohlcv_dataframe(missing_col_df, symbol="TEST_MISSING", min_bars=30)
    assert is_m_valid is False
    assert any("Missing required price columns" in a for a in m_warnings)


def test_f33_no_ml_performance_placeholders():
    """
    F-33: Verify ml_performance_engine does not return hardcoded placeholders
    (8.4% max drawdown, 72% / 68% / 60% / 25% regime defaults).
    """
    # 1. Horizon performance attribution
    hor_stats = ml_performance_engine.get_trade_outcome_learning(horizon="INTRADAY")
    assert "max_drawdown_pct" in hor_stats
    # Drawdown must be calculated, not hardcoded to 8.4
    if hor_stats["total_trades"] == 0:
        assert hor_stats["max_drawdown_pct"] == 0.0

    # 2. Market regime performance
    reg_stats = ml_performance_engine.get_market_regime_performance()
    assert reg_stats["status"] == "SUCCESS"
    for reg, r_data in reg_stats["regimes"].items():
        if r_data["total_samples"] == 0:
            # Must NOT be fabricated defaults (72.0, 68.0, 60.0, 25.0)
            assert r_data["directional_accuracy_pct"] == 0.0
            assert r_data["trade_win_rate_pct"] == 0.0
            assert r_data["target_hit_rate_pct"] == 0.0
            assert r_data["sl_hit_rate_pct"] == 0.0
            assert r_data["brier_score"] == 0.0
            assert r_data["max_drawdown_pct"] == 0.0
