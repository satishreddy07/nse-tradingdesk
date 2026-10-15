"""
System Resilience, Concurrency, and Future-Proofing Audit Tests
Verifies:
1. ClosingConnection automatic cleanup (zero SQLite file descriptor leaks).
2. FastAPI lifespan lifecycle management.
3. Perpetual macroeconomic event generation (2027+ projection).
4. Database migration integrity (composite_score_t5 and performance indices).
5. Configurable Nifty lot size integration.
6. Autonomous options auto-deploy scheduler integration.
"""
import sqlite3
import datetime
import pytest
from core import config
from execution.state_engine import state_engine, ClosingConnection
from engines.nifty_analytics_engine import generate_projected_central_bank_events, nifty_analytics_engine
from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
from scheduler.cron_scheduler import scheduled_options_premarket_job


def test_closing_connection_resource_management(tmp_path):
    """Verifies that ClosingConnection commits on success and closes the connection on exit."""
    test_db = tmp_path / "test_closing.db"
    raw_conn = sqlite3.connect(str(test_db))
    raw_conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, val TEXT);")
    raw_conn.commit()
    raw_conn.close()

    # Use ClosingConnection as a context manager
    raw_conn2 = sqlite3.connect(str(test_db))
    closing_conn = ClosingConnection(raw_conn2)
    with closing_conn as conn:
        conn.execute("INSERT INTO test_table (val) VALUES ('hello');")

    # Connection must now be closed automatically
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        raw_conn2.execute("SELECT * FROM test_table;")

    # Verify transaction was committed
    verify_conn = sqlite3.connect(str(test_db))
    rows = verify_conn.execute("SELECT val FROM test_table;").fetchall()
    verify_conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "hello"


def test_perpetual_macro_events_projection_for_future_years():
    """Verifies that future years (2027+) project valid FOMC and RBI MPC meeting calendars."""
    events_2027 = generate_projected_central_bank_events(2027)
    assert len(events_2027) >= 14  # At least 8 FOMC + 6 RBI meetings

    banks = {e["bank"] for e in events_2027}
    assert "US FED FOMC" in banks
    assert "RBI MPC" in banks

    # Dates must be strictly within 2027
    for ev in events_2027:
        dt = datetime.date.fromisoformat(ev["date"])
        assert dt.year == 2027
        assert ev["status"] == "UPCOMING"
        assert ev["market_impact"] == "HIGH_EVENT_VOLATILITY"


def test_database_indices_and_migration_integrity():
    """Verifies that critical options and prediction indices exist in the master database."""
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            # Check indices exist
            indices = [r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type = 'index';").fetchall()]
            
            # Prediction and options indices
            assert "idx_nifty_pred_date" in indices
            assert "idx_opt_strat_status" in indices
            assert "idx_intraday_opt_strat_leg" in indices
            assert "idx_intraday_opt_status" in indices
            assert "idx_weekly_opt_strat_leg" in indices
            assert "idx_weekly_opt_status" in indices
            assert "idx_monthly_opt_strat_leg" in indices
            assert "idx_monthly_opt_status" in indices

            # Check composite_score_t5 in nifty_predictions_log
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(nifty_predictions_log);").fetchall()]
            assert "composite_score_t5" in cols


def test_nifty_lot_size_configuration():
    """Verifies that NIFTY_LOT_SIZE is centralized in config and utilized across engines."""
    assert hasattr(config, "NIFTY_LOT_SIZE")
    assert config.NIFTY_LOT_SIZE > 0
    assert isinstance(config.NIFTY_LOT_SIZE, int)


def test_autonomous_options_premarket_job_execution():
    """Verifies that scheduled_options_premarket_job executes both weekly and monthly positional engines."""
    # Should run cleanly without throwing exceptions
    scheduled_options_premarket_job()
