"""
Unit and Integration Tests for:
1. Daily MOC SQLite Backup Manager (atomic hot backup, PRAGMA integrity check, 14-day retention pruning).
2. UI Mobile Optimizations:
   - Verification that redundant top mobile-subdesk-select dropdown is removed.
   - Verification of watchdog-modal responsive fit.
   - Verification of mobile drawer z-index hierarchy and collision isolation from bottom nav.
3. AI model registration (LLaMA 3.3 on Groq).
"""
import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from core.backup_manager import DailyBackupManager
from ai_engine.model_registry import ModelRegistry
from core import config
from web.api import app


import gc

@pytest.fixture
def temp_db_and_backup_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_db = tmp_path / "test_source.db"
        backup_dir = tmp_path / "backups" / "daily"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create a valid SQLite test database with some data
        with sqlite3.connect(str(source_db)) as conn:
            conn.execute("CREATE TABLE test_trades (id INTEGER PRIMARY KEY, symbol TEXT, pnl REAL)")
            conn.execute("INSERT INTO test_trades (symbol, pnl) VALUES ('RELIANCE', 12500.5)")
            conn.execute("INSERT INTO test_trades (symbol, pnl) VALUES ('TCS', 8900.0)")
            conn.commit()

        yield source_db, backup_dir
        gc.collect()


def test_daily_moc_backup_execution_and_integrity(temp_db_and_backup_dir, monkeypatch):
    source_db, backup_dir = temp_db_and_backup_dir
    monkeypatch.setattr(config, "DB_PATH", source_db)

    mgr = DailyBackupManager(backup_dir=backup_dir, retention_days=14)
    res = mgr.run_daily_moc_backup(retention_days=14, sync_cloud=False)

    assert res["status"] == "SUCCESS"
    assert res["integrity_check"] == "ok"
    assert Path(res["backup_path"]).exists()

    # Verify backed-up data matches source
    chk = sqlite3.connect(res["backup_path"])
    try:
        rows = chk.execute("SELECT symbol, pnl FROM test_trades ORDER BY id").fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "RELIANCE"
        assert rows[1][0] == "TCS"
    finally:
        chk.close()


def test_daily_backup_retention_pruning(temp_db_and_backup_dir, monkeypatch):
    source_db, backup_dir = temp_db_and_backup_dir
    monkeypatch.setattr(config, "DB_PATH", source_db)

    mgr = DailyBackupManager(backup_dir=backup_dir, retention_days=14)

    # Create dummy older backups (one 20 days old, one 5 days old)
    old_backup = backup_dir / "swing_trades_moc_20260901_154500.db"
    old_backup.touch()
    twenty_days_ago = (datetime.now() - timedelta(days=20)).timestamp()
    os.utime(str(old_backup), (twenty_days_ago, twenty_days_ago))

    recent_backup = backup_dir / "swing_trades_moc_20261010_154500.db"
    recent_backup.touch()
    five_days_ago = (datetime.now() - timedelta(days=5)).timestamp()
    os.utime(str(recent_backup), (five_days_ago, five_days_ago))

    # Run backup
    res = mgr.run_daily_moc_backup(retention_days=14, sync_cloud=False)
    assert res["status"] == "SUCCESS"
    assert res["pruned_count"] >= 1
    assert "swing_trades_moc_20260901_154500.db" in res["pruned_files"]

    # Verify old was deleted, recent was kept
    assert not old_backup.exists()
    assert recent_backup.exists()


def test_backup_api_endpoints(temp_db_and_backup_dir, monkeypatch):
    source_db, backup_dir = temp_db_and_backup_dir
    monkeypatch.setattr(config, "DB_PATH", source_db)

    mgr = DailyBackupManager(backup_dir=backup_dir, retention_days=14)
    from core import backup_manager
    monkeypatch.setattr(backup_manager, "daily_backup_manager", mgr)
    from core.telegram_vault import telegram_vault
    monkeypatch.setattr(telegram_vault, "backup_master_file_to_cloud", lambda *a, **kw: {"status": "SUCCESS"})
    monkeypatch.setattr(telegram_vault, "send_message", lambda *a, **kw: None)

    from core.telegram_vault import telegram_vault
    monkeypatch.setattr(telegram_vault, "backup_master_file_to_cloud", lambda *a, **kw: {"status": "SUCCESS"})
    monkeypatch.setattr(telegram_vault, "send_message", lambda *a, **kw: None)

    client = TestClient(app)
    post_res = client.post("/api/system/backup/daily")
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "SUCCESS"

    list_res = client.get("/api/system/backup/list")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["status"] == "OK"
    assert len(data["backups"]) >= 1


def test_ui_templates_optimizations():
    """Verifies UI template fixes in index.html."""
    with open("web/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Duplicated top dropdown must NOT be present
    assert 'id="mobile-subdesk-select"' not in html

    # 2. Watchdog modal responsive layout classes must be present
    assert 'id="watchdog-modal"' in html
    assert 'Pipeline Watchdog' in html
    assert 'watchdog-modal-summary-badge' in html
    assert 'btn-probe-feeds' in html

    # 3. Mobile drawer and backdrop z-index hierarchy
    assert 'id="mobile-nav-backdrop"' in html
    assert 'z-[90]' in html
    assert 'id="mobile-nav-drawer"' in html
    assert 'z-[100]' in html
    assert 'translate3d(100%, 0, 0)' in html
    assert 'max-width: 28px; max-height: 28px;' in html
    assert 'max-width: 32px; max-height: 32px;' in html
    assert 'w-6.5' not in html, "Invalid Tailwind class w-6.5 must not be used on mobile elements"
    assert 'h-6.5' not in html, "Invalid Tailwind class h-6.5 must not be used on mobile elements"

    # 4. Portfolio 3 Watchlist filter counts & LTP column
    assert 'id="btn-p3-filter-active"' in html
    assert 'id="btn-p3-filter-all"' in html
    assert 'id="p3-filter-active-count"' in html
    assert 'id="p3-filter-all-count"' in html
    assert 'Live LTP & Spread' in html
    assert '🚫 INVALIDATED' in html
    assert 'Below 20 EMA' in html
    assert '⚡ Auto Execute' in html


def test_groq_llama_model_registration():
    """Verifies that LLaMA 3.3 and 3.1 models are recognized and routed to Groq."""
    assert ModelRegistry.is_valid_model("llama-3.3-70b-versatile")
    assert ModelRegistry.is_valid_model("llama-3.1-8b-instant")
    assert ModelRegistry.resolve_provider_for_model("llama-3.3-70b-versatile") == "groq"
    assert ModelRegistry.resolve_provider_for_model("llama-3.1-8b-instant") == "groq"
    assert ModelRegistry.resolve_provider_for_model("openai/gpt-oss-120b") == "groq"
