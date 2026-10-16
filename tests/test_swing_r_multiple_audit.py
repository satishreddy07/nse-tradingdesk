import pytest
import sqlite3
from execution.state_engine import StateEngine

def test_swing_r_multiple_and_quantity_audit(tmp_path):
    db_file = str(tmp_path / "test_swing.db")
    engine = StateEngine(db_path=db_file)

    # 1. Open position: 100 shares @ 100, SL=90, T1=120, T2=140
    # Risk per share = 100 - 90 = 10 Rs. Total Initial Risk = 10 * 100 = 1000 Rs (1R).
    # Bypass trading gate using direct connection or mock
    with engine._lock:
        with engine._get_connection() as conn:
            conn.execute("""
            INSERT INTO active_positions (
                symbol, portfolio_type, entry_price, quantity, original_quantity, stop_loss, original_stop_loss,
                target, target_2, current_price, highest_price, entry_date, hold_days,
                t1_hit, t1_pnl, sl_reason, sector, notes, entry_reason, entry_regime,
                metrics_json, features_json, updated_at
            ) VALUES ('TESTSYM', 'INSTITUTIONAL_6PILLAR', 100.0, 100, 100, 90.0, 90.0, 120.0, 140.0, 100.0, 100.0, '2026-10-16 09:30:00 IST', 0, 0, 0.0, 'INITIAL_SL', 'TECH', '', '', '', '{}', '{}', '2026-10-16 09:30:00 IST')
            """)
            conn.commit()

    # Verify active position has original_quantity = 100
    pos = engine.get_active_positions(portfolio_type='INSTITUTIONAL_6PILLAR')[0]
    assert pos["quantity"] == 100
    assert pos["original_quantity"] == 100

    # 2. Simulate T1 hit: Book 50 shares @ 120 (+1000 Rs profit = +1.0R on whole trade risk)
    # Remaining quantity = 50 shares, SL moved to breakeven 100
    with engine._lock:
        with engine._get_connection() as conn:
            conn.execute("""
            UPDATE active_positions
            SET quantity = 50, stop_loss = 100.0, t1_hit = 1, t1_pnl = 1000.0, sl_reason = 'BREAKEVEN_TRAILING'
            WHERE symbol = 'TESTSYM'
            """)
            conn.commit()

    # Verify portfolio stats computes exact R-multiple dynamically (not hardcoded 1.5)
    stats = engine.get_portfolio_stats(portfolio_type='INSTITUTIONAL_6PILLAR')
    assert stats["t1_booked_count"] == 1
    assert stats["t1_realized_pnl"] == 1000.0
    # Expected R for partial T1 booking:
    # 1000 PnL / (10 risk_per_share * 50 booked_shares) = 2.0 R on the half, or overall positive R
    assert stats["avg_win_r"] > 0
    assert stats["expectancy_r"] > 0

    # 3. Simulate Remaining 50 shares exit at breakeven @ 100 Rs
    # Rem PnL = 0, Total PnL = 1000 Rs.
    # Total initial risk was 1000 Rs. So overall R-multiple MUST be 1000 / 1000 = +1.0 R!
    closed = engine.close_position('TESTSYM', exit_price=100.0, exit_reason="TRAILING_BE_STOP", bypass_gate=True)
    assert closed is not None
    assert closed["pnl"] == 1000.0
    # Under old bug, this was 1000 / (10 * 50) = 2.0 R (inflated by 100%)
    # Under fix, this MUST be 1000 / (10 * 100) = 1.0 R!
    assert closed["r_multiple"] == 1.0
    assert closed["quantity"] == 100  # True total quantity traded

    # 4. Verify trade_history has correct records
    hist = engine.get_trade_history(portfolio_type='INSTITUTIONAL_6PILLAR')
    assert len(hist) == 1
    assert hist[0]["symbol"] == "TESTSYM"
    assert hist[0]["quantity"] == 100
    assert hist[0]["pnl"] == 1000.0
    assert hist[0]["r_multiple"] == 1.0
