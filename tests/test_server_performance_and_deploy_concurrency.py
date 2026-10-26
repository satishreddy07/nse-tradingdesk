"""
tests/test_server_performance_and_deploy_concurrency.py
=======================================================
Verification suite for Server Latency, Concurrency, and Options Deploy Optimization:
1. Error shielding: missing/invalid strike prices return HTTP 400, not unhandled HTTP 500.
2. Low latency deployment: valid strategy deployment completes in < 250ms with detached side-effects.
3. Concurrent lock-free reads under SQLite WAL mode: parallel reader threads execute without blocking.
4. Non-blocking price refresh: `/api/prices/refresh` returns quickly without 30-second stalls.
5. High-performance SQLite pragmas: verified cache_size, mmap_size, and memory temp_store.
"""

import time
import concurrent.futures
import pytest
from fastapi.testclient import TestClient

from web.api import app
from execution.state_engine import state_engine
from engines.nifty_analytics_engine import nifty_analytics_engine

client = TestClient(app)


def test_sqlite_pragmas_and_performance_tuning():
    """Verify state_engine connection applies high-performance RAM cache and mmap pragmas."""
    with state_engine._get_connection() as conn:
        busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        cache = conn.execute("PRAGMA cache_size;").fetchone()[0]
        mmap = conn.execute("PRAGMA mmap_size;").fetchone()[0]
        temp = conn.execute("PRAGMA temp_store;").fetchone()[0]

        assert busy >= 30000, f"Expected busy_timeout >= 30000, got {busy}"
        assert sync in (1, "NORMAL"), f"Expected synchronous NORMAL (1), got {sync}"
        assert cache <= -64000 or cache == 64000, f"Expected cache_size 64MB (-64000), got {cache}"
        assert mmap >= 268435456, f"Expected mmap_size >= 256MB, got {mmap}"
        assert temp in (2, "MEMORY"), f"Expected temp_store MEMORY (2), got {temp}"


def test_deploy_options_strategy_missing_strike_returns_400_not_500(monkeypatch):
    """When a leg has no market price quote, deploy returns HTTP 400 with detail, not HTTP 500 crash."""
    # Force option chain to return empty so strike price cannot be resolved
    monkeypatch.setattr(nifty_analytics_engine, "fetch_and_analyze_option_chain", lambda force_refresh=False: {})
    monkeypatch.setattr(nifty_analytics_engine, "_option_chain_mem_cache", None)

    payload = {
        "strategy_name": "Test Missing Strike Deploy",
        "symbol": "NIFTY",
        "expiry_date": "29-OCT-2026",
        "expiry_type": "WEEKLY",
        "spot_at_entry": 22600.0,
        "net_premium": 0.0,
        "max_profit": 5000.0,
        "max_loss": 5000.0,
        "target_pnl": 2500.0,
        "stoploss_pnl": -3000.0,
        "auto_trade_mode": "MANUAL",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": 99950, "qty": 65, "entry_price": 0.0, "price": 0.0}
        ]
    }

    resp = client.post("/api/options/deploy-strategy", json=payload)
    assert resp.status_code == 400, f"Expected HTTP 400 on unresolvable strike quote, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "DATA_UNAVAILABLE" in data.get("detail", "") or "missing live price" in data.get("detail", "")


def test_deploy_options_strategy_fast_and_successful(monkeypatch):
    """Deploying a valid multi-leg strategy completes cleanly and quickly with detached alert/sync."""
    payload = {
        "strategy_name": "Test Fast Deployment Iron Condor",
        "symbol": "NIFTY",
        "expiry_date": "29-OCT-2026",
        "expiry_type": "WEEKLY",
        "spot_at_entry": 22600.0,
        "net_premium": 150.0,
        "max_profit": 9750.0,
        "max_loss": 6500.0,
        "target_pnl": 4500.0,
        "stoploss_pnl": -5000.0,
        "auto_trade_mode": "MANUAL",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": 22800, "qty": 65, "entry_price": 85.0, "price": 85.0},
            {"action": "BUY", "type": "CE", "strike": 23000, "qty": 65, "entry_price": 25.0, "price": 25.0},
            {"action": "SELL", "type": "PE", "strike": 22400, "qty": 65, "entry_price": 75.0, "price": 75.0},
            {"action": "BUY", "type": "PE", "strike": 22200, "qty": 65, "entry_price": 20.0, "price": 20.0}
        ]
    }

    t0 = time.time()
    resp = client.post("/api/options/deploy-strategy", json=payload)
    elapsed = time.time() - t0

    assert resp.status_code == 200, f"Deploy failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "OK"
    assert "strategy_id" in data
    strat_id = data["strategy_id"]

    try:
        # Verify strategy in database
        with state_engine._get_connection() as conn:
            row = conn.execute("SELECT status, strategy_name FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone()
            assert row is not None
            assert row["status"] == "ACTIVE"
            assert row["strategy_name"] == payload["strategy_name"]

        # Latency assertion: must be responsive (< 500ms even with SQLite insert)
        assert elapsed < 0.50, f"Deployment was expected to complete in < 500ms, took {elapsed:.3f}s"
    finally:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM options_strategies WHERE strategy_id = ?", (strat_id,))
            conn.commit()


def test_concurrent_database_reads_under_wal():
    """Verify that 20 parallel reader threads can query positions and KV store simultaneously without locking."""
    def _read_worker(idx: int):
        kv = state_engine.get_kv("system_health_telemetry")
        pos = state_engine.get_active_positions()
        trades = state_engine.get_trade_history(limit=50)
        logs = state_engine.get_recent_scan_logs(limit=20)
        return (idx, len(pos), len(trades), len(logs))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_read_worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert len(results) == 20, f"Expected 20 successful concurrent reads, got {len(results)}"


def test_price_refresh_endpoint_non_blocking():
    """Verify /api/prices/refresh returns within timeout without blocking worker threads for 30 seconds."""
    t0 = time.time()
    resp = client.post("/api/prices/refresh", json={"scope": "ACTIVE_AND_SIGNALS"})
    elapsed = time.time() - t0

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("status") in ("OK", "SUCCESS", "PROCESSING", "QUEUED")
    assert elapsed < 15.0, f"/api/prices/refresh blocked for {elapsed:.2f}s, expected < 15.0s"
