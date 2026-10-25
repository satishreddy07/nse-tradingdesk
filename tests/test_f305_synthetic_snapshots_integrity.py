"""
Regression Tests for F3-05: Prevention of Synthetic Snapshot Contamination
Verifies:
1. Schema integrity & isolation: `simulated_option_snapshots` is isolated from `intraday_option_snapshots`.
2. `auto_gapfill_missed_intraday_slots` never writes to `intraday_option_snapshots`.
3. Option chain unavailability causes `capture_intraday_50min_snapshot` to fail closed (DATA_UNAVAILABLE).
4. Genuine empirical observations are explicitly recorded with data_source='LIVE'.
5. Historical synthetic records are labeled 'SYNTHETIC_GAPFILL' and excluded from live queries.
6. Downstream consumers (ML training, spot price fallback, straddle decay, watchdog) ignore synthetic snapshots.
"""

import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from engines.nifty_analytics_engine import nifty_analytics_engine
from execution.state_engine import state_engine


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh isolated SQLite database with engine tables initialized."""
    db_file = tmp_path / "test_trades.db"
    
    # Patch state_engine connection to use temp_db
    def get_conn():
        conn = sqlite3.connect(str(db_file), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    with patch.object(state_engine, "_get_connection", side_effect=get_conn):
        state_engine._init_db()
        nifty_analytics_engine._ensure_tables()
        yield db_file, get_conn


def test_schema_isolation_and_migration(temp_db):
    """Test that intraday_option_snapshots and simulated_option_snapshots are created with proper columns."""
    db_file, get_conn = temp_db
    with get_conn() as conn:
        cols_intra = [c[1] for c in conn.execute("PRAGMA table_info(intraday_option_snapshots)").fetchall()]
        cols_sim = [c[1] for c in conn.execute("PRAGMA table_info(simulated_option_snapshots)").fetchall()]
        
        assert "data_source" in cols_intra, "intraday_option_snapshots must have data_source column"
        assert "data_source" in cols_sim, "simulated_option_snapshots must have data_source column"
        assert "simulation_method" in cols_sim, "simulated_option_snapshots must have simulation_method column"


def test_historical_gapfill_tagging_without_overwriting_empirical(temp_db):
    """Test that legacy records with gapfilled JSON metadata are tagged SYNTHETIC_GAPFILL."""
    db_file, get_conn = temp_db
    with get_conn() as conn:
        # Insert a simulated record lacking data_source
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "2026-10-10", "10:00", "2026-10-10 10:00:00 IST", "2026-10-15", 22500.0, 22500,
            json.dumps({"reconciled_by": "AUTO_GAPFILL_WATCHDOG", "gapfilled": True}),
            "UNVERIFIED_HISTORICAL"
        ))
        # Insert a genuine unverified historical record
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "2026-10-10", "10:15", "2026-10-10 10:15:00 IST", "2026-10-15", 22510.0, 22500,
            json.dumps({"is_empirical": True}),
            "UNVERIFIED_HISTORICAL"
        ))
        conn.commit()

    # Re-run _ensure_tables migration
    nifty_analytics_engine._ensure_tables()

    with get_conn() as conn:
        row_gapfill = conn.execute("SELECT data_source FROM intraday_option_snapshots WHERE slot_time = '10:00'").fetchone()
        row_hist = conn.execute("SELECT data_source FROM intraday_option_snapshots WHERE slot_time = '10:15'").fetchone()

        assert row_gapfill["data_source"] == "SYNTHETIC_GAPFILL"
        assert row_hist["data_source"] == "UNVERIFIED_HISTORICAL"


def test_auto_gapfill_never_persists_to_empirical_table(temp_db):
    """Test that auto_gapfill_missed_intraday_slots persists only to simulated_option_snapshots."""
    db_file, get_conn = temp_db
    target_date = "2026-10-20"

    # Pre-seed one empirical snapshot
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                straddle_premium, near_atm_pcr, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (target_date, "09:30", f"{target_date} 09:30:00 IST", "2026-10-22", 22500.0, 22500, 100.0, 1.0, "LIVE"))
        conn.commit()

    # Run auto_gapfill for elapsed slots
    with patch.object(nifty_analytics_engine, "get_live_nifty_spot", return_value={"price": 22500.0}):
        gapfilled = nifty_analytics_engine.auto_gapfill_missed_intraday_slots(session_date=target_date, interval_mode="50m")

    assert gapfilled > 0, "Expected some missing slots to be generated for 50m schedule"

    with get_conn() as conn:
        # Check empirical table count: MUST REMAIN EXACTLY 1!
        intra_count = conn.execute("SELECT count(*) FROM intraday_option_snapshots WHERE session_date = ?", (target_date,)).fetchone()[0]
        assert intra_count == 1, "CRITICAL: auto_gapfill must NEVER insert into intraday_option_snapshots!"

        # Check simulation table: must contain the gapfilled records
        sim_count = conn.execute("SELECT count(*) FROM simulated_option_snapshots WHERE session_date = ?", (target_date,)).fetchone()[0]
        assert sim_count == gapfilled

        sim_rows = conn.execute("SELECT data_source, simulation_method FROM simulated_option_snapshots WHERE session_date = ?", (target_date,)).fetchall()
        for r in sim_rows:
            assert r["data_source"] == "SIMULATED"
            assert r["simulation_method"] == "THETA_PCR_INTERPOLATION"


def test_capture_snapshot_fails_closed_on_option_chain_unavailability(temp_db):
    """Test that capture_intraday_50min_snapshot fails closed with DATA_UNAVAILABLE when option chain fails."""
    db_file, get_conn = temp_db

    with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value={"status": "DATA_NOT_AVAILABLE", "message": "Kotak Neo Stream offline"}):
        res = nifty_analytics_engine.capture_intraday_50min_snapshot(slot_override="10:00")

    assert res.get("status") == "DATA_UNAVAILABLE"
    assert "Option chain unavailable" in res.get("message", "")

    with get_conn() as conn:
        count = conn.execute("SELECT count(*) FROM intraday_option_snapshots").fetchone()[0]
        assert count == 0, "No snapshots should be written when option chain is unavailable"


def test_capture_snapshot_records_live_provenance(temp_db):
    """Test that capture_intraday_50min_snapshot records genuine data with data_source='LIVE'."""
    db_file, get_conn = temp_db

    mock_chain = {
        "status": "OK",
        "weekly": {
            "spot_price": 22600.0,
            "atm_strike": 22600,
            "expiry": "2026-10-22",
            "nse_timestamp": "2026-10-20 10:00:00",
            "straddle": {
                "ce_ltp": 60.0,
                "pe_ltp": 55.0,
                "straddle_premium": 115.0,
            },
            "pcr": {
                "overall_oi_pcr": 1.1,
                "near_atm_pcr": 1.05,
                "exact_atm_pcr": 1.02,
                "chg_oi_pcr": 1.15
            },
            "walls_and_pain": {
                "max_pain": 22600,
                "call_wall_1": 22800,
                "put_wall_1": 22400
            },
            "volatility_and_gex": {
                "near_ce_chg_total": 50000,
                "near_pe_chg_total": 60000,
                "iv_skew": 0.5,
                "atm_iv": 14.2
            }
        }
    }

    with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value=mock_chain):
        res = nifty_analytics_engine.capture_intraday_50min_snapshot(slot_override="10:00")

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM intraday_option_snapshots WHERE slot_time = '10:00'").fetchone()
        assert row is not None
        assert row["spot_price"] == 22600.0
        assert row["data_source"] == "LIVE"

        meta = json.loads(row["snapshot_json"])
        assert meta.get("data_source") == "LIVE"
        assert meta.get("is_simulated") is False


def test_get_snapshots_excludes_synthetic_by_default(temp_db):
    """Test that _get_snapshots_for_date filters out synthetic gapfill records."""
    db_file, get_conn = temp_db
    session_date = "2026-10-21"

    with get_conn() as conn:
        # Live snapshot
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_date, "09:30", f"{session_date} 09:30:00 IST", "2026-10-22", 22500.0, 22500, json.dumps({"is_empirical": True}), "LIVE"))

        # Synthetic gapfill snapshot in intraday_option_snapshots
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_date, "09:45", f"{session_date} 09:45:00 IST", "2026-10-22", 22505.0, 22500, json.dumps({"reconciled_by": "AUTO_GAPFILL_WATCHDOG", "gapfilled": True}), "SYNTHETIC_GAPFILL"))

        # Simulated snapshot in simulated_option_snapshots
        conn.execute("""
            INSERT INTO simulated_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                data_source, simulation_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_date, "10:00", f"{session_date} 10:00:00 IST", "2026-10-22", 22510.0, 22500, "SIMULATED", "THETA_PCR_INTERPOLATION"))
        conn.commit()

    # Default query excludes synthetic gapfill and excludes simulated table
    snaps = nifty_analytics_engine._get_snapshots_for_date(session_date, include_simulated=False)
    assert len(snaps) == 1
    assert snaps[0]["slot_time"] == "09:30"
    assert snaps[0]["data_source"] == "LIVE"

    # Query with include_simulated=True merges the simulated table, but NEVER returns SYNTHETIC_GAPFILL
    all_snaps = nifty_analytics_engine._get_snapshots_for_date(session_date, include_simulated=True)
    assert len(all_snaps) == 2
    assert [s["slot_time"] for s in all_snaps] == ["09:30", "10:00"]
    assert any(s["data_source"] == "LIVE" for s in all_snaps)
    assert any(s["data_source"] == "SIMULATED" for s in all_snaps)
    assert not any(s["data_source"] == "SYNTHETIC_GAPFILL" for s in all_snaps)


def test_spot_fallback_ignores_synthetic_snapshot(temp_db):
    """Test that get_live_nifty_spot fallback step 5 never returns synthetic gapfill records."""
    db_file, get_conn = temp_db

    with get_conn() as conn:
        # Earlier empirical row
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                id, session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (1, '2026-10-20', '15:20', '2026-10-20 15:20:00 IST', '2026-10-22', 22550.0, 22550, '{}', 'LIVE')
        """)
        # Later synthetic row (higher id)
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                id, session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                snapshot_json, data_source
            ) VALUES (2, '2026-10-20', '15:30', '2026-10-20 15:30:00 IST', '2026-10-22', 99999.0, 99999, '{"reconciled_by": "AUTO_GAPFILL_WATCHDOG", "gapfilled": true}', 'SYNTHETIC_GAPFILL')
        """)
        conn.commit()

    # Simulate Kotak Neo unavailable
    with patch("engines.kotak_neo_engine.kotak_neo_engine") as mock_kotak:
        mock_kotak.get_live_tick.return_value = None
        mock_kotak.get_live_price.return_value = None
        mock_kotak._is_authenticated = False
        with patch.object(nifty_analytics_engine, "_live_spot_cache", {}):
            with patch.object(nifty_analytics_engine, "_option_chain_mem_cache", {}):
                with patch("execution.state_engine.state_engine.get_kv", return_value=None):
                    spot_res = nifty_analytics_engine.get_live_nifty_spot(force_refresh=True)

    assert spot_res.get("price") == 22550.0, "Expected spot price from empirical row (22550.0), not synthetic (99999.0)"
    assert spot_res.get("source") == "DB_SNAPSHOT_CACHE"


def test_intraday_dashboard_does_not_manufacture_missing_slots(temp_db):
    """Test that get_intraday_expiry_dashboard leaves missing slots as MISSED_OR_PRE_CAPTURE."""
    db_file, get_conn = temp_db
    session_date = "2026-10-22"

    with patch.object(nifty_analytics_engine, "fetch_and_analyze_option_chain", return_value={"status": "DATA_NOT_AVAILABLE"}):
        with patch.object(nifty_analytics_engine, "get_live_nifty_spot", return_value={"price": 22500.0}):
            dash = nifty_analytics_engine.get_intraday_expiry_dashboard(session_date=session_date, interval_mode="30m", force_refresh=True)

    with get_conn() as conn:
        count = conn.execute("SELECT count(*) FROM intraday_option_snapshots").fetchone()[0]
        assert count == 0, "Dashboard must not manufacture any snapshots in intraday_option_snapshots"

    # Verify schedule slots are not populated with fake prices
    schedule = dash.get("scheduled_slots", [])
    for slot in schedule:
        assert slot["spot_price"] is None, f"Slot {slot['slot_time']} should have None spot_price when uncaptured"
        assert slot["straddle_premium"] is None


def test_downstream_consumers_ignore_synthetic_records(temp_db):
    """Test that ML models, edge analytics, and watchdogs ignore synthetic gapfill records."""
    db_file, get_conn = temp_db
    session_date = "2026-10-23"

    with get_conn() as conn:
        # Seed 1 live record
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                straddle_premium, atm_iv, near_atm_pcr, chg_oi_pcr, snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_date, "09:30", f"{session_date} 09:30:00 IST", "2026-10-29", 22500.0, 22500, 120.0, 14.0, 1.1, 1.2, '{}', 'LIVE'))

        # Seed 1 synthetic record
        conn.execute("""
            INSERT INTO intraday_option_snapshots (
                session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                straddle_premium, atm_iv, near_atm_pcr, chg_oi_pcr, snapshot_json, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_date, "09:45", f"{session_date} 09:45:00 IST", "2026-10-29", 22520.0, 22500, 115.0, 13.5, 9.9, 9.9, '{"reconciled_by": "AUTO_GAPFILL_WATCHDOG", "gapfilled": true}', 'SYNTHETIC_GAPFILL'))
        conn.commit()

    # 1. Options Edge Analytics: compute_atm_straddle_decay_curve
    from engines.options_edge_analytics_engine import options_edge_analytics_engine
    curve = options_edge_analytics_engine.compute_atm_straddle_decay_curve(session_date)
    assert len(curve.get("slots", [])) == 1, "Decay curve must include only empirical slots, ignoring synthetic"
    assert curve["slots"][0]["slot_time"] == "09:30"

    # 2. Options Edge Analytics: get_iv_rank_and_percentile fallback
    # The synthetic IV (13.5) must not be retrieved if we query with no hist_rows
    with get_conn() as conn:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(intraday_option_snapshots)").fetchall()]
        assert "data_source" in cols
        snap_ivs = conn.execute(
            "SELECT DISTINCT atm_iv FROM intraday_option_snapshots WHERE atm_iv > 0 AND data_source = 'LIVE' "
            "AND (snapshot_json IS NULL OR (snapshot_json NOT LIKE '%AUTO_GAPFILL%' AND snapshot_json NOT LIKE '%\"gapfilled\": true%')) ORDER BY id DESC LIMIT 90"
        ).fetchall()
        assert len(snap_ivs) == 1
        assert float(snap_ivs[0][0]) == 14.0

    # 3. Nifty Forecast Learning Engine: pcr_rows query
    with get_conn() as conn:
        pcr_rows = conn.execute(
            "SELECT session_date, near_atm_pcr, chg_oi_pcr FROM intraday_option_snapshots "
            "WHERE near_atm_pcr > 0 AND data_source = 'LIVE' "
            "  AND (snapshot_json IS NULL OR (snapshot_json NOT LIKE '%AUTO_GAPFILL%' AND snapshot_json NOT LIKE '%\"gapfilled\": true%'))"
        ).fetchall()
        assert len(pcr_rows) == 1
        assert float(pcr_rows[0]["near_atm_pcr"]) == 1.1, "Synthetic PCR 9.9 must not be queried"

    # 4. Data Watchdog: latest snapshot query
    from core.data_watchdog import data_watchdog
    with get_conn() as conn:
        latest_snap = conn.execute(
            "SELECT session_date, slot_time, timestamp_ist, spot_price, atm_strike, market_state "
            "FROM intraday_option_snapshots WHERE data_source = 'LIVE' "
            "  AND (snapshot_json IS NULL OR (snapshot_json NOT LIKE '%AUTO_GAPFILL%' AND snapshot_json NOT LIKE '%\"gapfilled\": true%')) "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert latest_snap["slot_time"] == "09:30", "Data watchdog must see the empirical slot 09:30, not synthetic 09:45"
        assert latest_snap["spot_price"] == 22500.0


def test_provenance_handling_matrix_across_consumers(temp_db):
    """
    Comprehensive verification of the provenance matrix across all consumers:
    LIVE: Accepted by empirical training, live analytics, and display.
    UNVERIFIED_HISTORICAL: Excluded from empirical training & live analytics; accepted ONLY in display when explicitly requested.
    SYNTHETIC_GAPFILL: Excluded everywhere.
    SIMULATED: Excluded everywhere except explicit simulation-aware paths.
    NULL / UNKNOWN: Excluded everywhere (fail-closed).
    """
    db_file, get_conn = temp_db
    session_date = "2026-10-24"

    provenance_rows = [
        ("09:30", 22500.0, 1.05, 14.0, "LIVE"),
        ("09:45", 22510.0, 1.06, 14.1, "LIVE"),
        ("10:00", 22520.0, 1.20, 15.0, "UNVERIFIED_HISTORICAL"),
        ("10:15", 22530.0, 1.30, 16.0, "SYNTHETIC_GAPFILL"),
        ("10:30", 22540.0, 1.40, 17.0, "SIMULATED"),
        ("10:45", 22550.0, 1.50, 18.0, None),
        ("11:00", 22560.0, 1.60, 19.0, "UNKNOWN_EXPERIMENTAL"),
    ]

    with get_conn() as conn:
        for slot, spot, pcr, iv, src in provenance_rows:
            conn.execute("""
                INSERT INTO intraday_option_snapshots (
                    session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
                    atm_ce_ltp, atm_pe_ltp, straddle_premium, overall_pcr, near_atm_pcr, atm_pcr,
                    chg_oi_pcr, max_pain, call_wall, put_wall, net_ce_oi_chg, net_pe_oi_chg,
                    iv_skew, atm_iv, market_state, snapshot_json, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_date, slot, f"{session_date} {slot}:00 IST", "2026-10-29", spot, int(spot),
                50.0, 50.0, 100.0, pcr, pcr, pcr,
                pcr, int(spot), int(spot + 200), int(spot - 200), 1000, 1000,
                0.0, iv, "TEST_STATE", "{}", src
            ))
        conn.commit()

    # 1. Intraday ML Engine: training must load only LIVE rows
    from engines.intraday_ml_engine import intraday_ml_engine
    with get_conn() as conn:
        live_rows_count = conn.execute("SELECT count(*) FROM intraday_option_snapshots WHERE data_source = 'LIVE'").fetchone()[0]
        total_rows_count = conn.execute("SELECT count(*) FROM intraday_option_snapshots").fetchone()[0]
        assert live_rows_count == 2
        assert total_rows_count == 7

    ml_res = intraday_ml_engine.train_and_predict(force_refresh=True)
    assert ml_res.get("status") in ("OK", "DATA_UNAVAILABLE")

    # 2. Options Edge Analytics: Straddle decay curve must contain ONLY LIVE slots
    from engines.options_edge_analytics_engine import options_edge_analytics_engine
    curve = options_edge_analytics_engine.compute_atm_straddle_decay_curve(session_date)
    decay_slots = [s["slot_time"] for s in curve.get("slots", [])]
    assert decay_slots == ["09:30", "09:45"], (
        f"Straddle decay must include ONLY LIVE slots, got {decay_slots}"
    )

    # 3. Live Spot Fallback: Must choose the latest LIVE row (09:45 at 22510.0), NOT later unverified/synthetic rows
    with patch("engines.kotak_neo_engine.kotak_neo_engine") as mock_kotak:
        mock_kotak.get_live_tick.return_value = None
        mock_kotak.get_live_price.return_value = None
        mock_kotak._is_authenticated = False
        with patch.object(nifty_analytics_engine, "_live_spot_cache", {}):
            with patch.object(nifty_analytics_engine, "_option_chain_mem_cache", {}):
                with patch("execution.state_engine.state_engine.get_kv", return_value=None):
                    spot_res = nifty_analytics_engine.get_live_nifty_spot(force_refresh=True)
    assert spot_res.get("price") == 22510.0, (
        f"Spot fallback must select latest LIVE row (22510.0), got {spot_res.get('price')}"
    )

    # 4. Watchdog: Intraday desk must see the latest LIVE snapshot (09:45), not 11:00
    with get_conn() as conn:
        latest_live = conn.execute(
            "SELECT slot_time FROM intraday_option_snapshots WHERE data_source = 'LIVE' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert latest_live["slot_time"] == "09:45"

    # 5. Historical Inspection:
    # A. Default (allow_unverified_historical=False): returns ONLY LIVE
    live_snaps = nifty_analytics_engine._get_snapshots_for_date(session_date, allow_unverified_historical=False)
    live_slots = [s["slot_time"] for s in live_snaps]
    assert live_slots == ["09:30", "09:45"]
    assert all(s["data_source"] == "LIVE" for s in live_snaps)

    # B. Explicit historical inspection (allow_unverified_historical=True): returns LIVE + UNVERIFIED_HISTORICAL
    hist_snaps = nifty_analytics_engine._get_snapshots_for_date(session_date, allow_unverified_historical=True)
    hist_slots = [s["slot_time"] for s in hist_snaps]
    assert hist_slots == ["09:30", "09:45", "10:00"]
    # Verify SYNTHETIC_GAPFILL, SIMULATED, NULL, and UNKNOWN are excluded
    assert not any(s["data_source"] in ("SYNTHETIC_GAPFILL", "SIMULATED", None, "UNKNOWN_EXPERIMENTAL") for s in hist_snaps)


def test_unmigrated_database_fails_closed_safely(tmp_path):
    """
    Test that an unmigrated database lacking the data_source column fails closed safely:
    - ML engine reports schema lack and returns fallback without crashing.
    - _get_snapshots_for_date returns empty list when allow_unverified_historical=False.
    - Options edge analytics returns available=False.
    - Live spot fallback refuses to trust unmigrated rows.
    """
    db_file = tmp_path / "unmigrated_trades.db"
    conn = sqlite3.connect(str(db_file), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Create legacy table without data_source column
    conn.execute("""
        CREATE TABLE intraday_option_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            slot_time TEXT NOT NULL,
            timestamp_ist TEXT,
            expiry_date TEXT,
            spot_price REAL,
            atm_strike INTEGER,
            atm_ce_ltp REAL,
            atm_pe_ltp REAL,
            straddle_premium REAL,
            overall_pcr REAL,
            near_atm_pcr REAL,
            atm_pcr REAL,
            chg_oi_pcr REAL,
            max_pain INTEGER,
            call_wall INTEGER,
            put_wall INTEGER,
            net_ce_oi_chg INTEGER,
            net_pe_oi_chg INTEGER,
            iv_skew REAL,
            atm_iv REAL,
            market_state TEXT,
            what_is_happening TEXT,
            what_could_happen TEXT,
            quant_suggestion TEXT,
            snapshot_json TEXT,
            UNIQUE(session_date, slot_time)
        );
    """)
    conn.execute("""
        INSERT INTO intraday_option_snapshots (
            session_date, slot_time, timestamp_ist, expiry_date, spot_price, atm_strike,
            atm_ce_ltp, atm_pe_ltp, straddle_premium, overall_pcr, near_atm_pcr, atm_pcr,
            chg_oi_pcr, max_pain, call_wall, put_wall, net_ce_oi_chg, net_pe_oi_chg,
            iv_skew, atm_iv, market_state, snapshot_json
        ) VALUES (
            '2026-10-24', '09:30', '2026-10-24 09:30:00 IST', '2026-10-29', 22500.0, 22500,
            50.0, 50.0, 100.0, 1.0, 1.0, 1.0, 1.0, 22500, 22700, 22300, 100, 100, 0.0, 14.0, 'STATE', '{}'
        )
    """)
    conn.commit()

    def get_unmigrated_conn():
        c = sqlite3.connect(str(db_file), timeout=10.0)
        c.row_factory = sqlite3.Row
        return c

    with patch.object(state_engine, "_get_connection", side_effect=get_unmigrated_conn):
        # 1. _get_snapshots_for_date must fail closed with empty list
        snaps = nifty_analytics_engine._get_snapshots_for_date("2026-10-24", allow_unverified_historical=False)
        assert snaps == [], "Unmigrated DB must fail closed and return empty list by default"

        # 2. Intraday ML Engine must fail closed
        from engines.intraday_ml_engine import intraday_ml_engine
        ml_res = intraday_ml_engine.train_and_predict(force_refresh=True)
        assert ml_res.get("status") == "DATA_UNAVAILABLE"
        assert "data_source" in ml_res.get("reason", "")

        # 3. Options Edge Straddle Decay must return available=False
        from engines.options_edge_analytics_engine import options_edge_analytics_engine
        decay = options_edge_analytics_engine.compute_atm_straddle_decay_curve("2026-10-24")
        assert decay.get("available") is False
        assert decay.get("slots") == []

        # 4. Live spot fallback must NOT pick unmigrated row
        with patch("engines.kotak_neo_engine.kotak_neo_engine") as mock_kotak:
            mock_kotak.get_live_tick.return_value = None
            mock_kotak.get_live_price.return_value = None
            mock_kotak._is_authenticated = False
            with patch.object(nifty_analytics_engine, "_live_spot_cache", {}):
                with patch.object(nifty_analytics_engine, "_option_chain_mem_cache", {}):
                    with patch("execution.state_engine.state_engine.get_kv", return_value=None):
                        spot_res = nifty_analytics_engine.get_live_nifty_spot(force_refresh=True)
        assert spot_res.get("source") != "DB_SNAPSHOT_CACHE", "Live spot must refuse to trust unmigrated snapshot table"

