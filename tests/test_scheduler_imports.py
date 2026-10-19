import pytest
import scheduler.cron_scheduler as cs
import core.telegram_vault as tv
import time


def test_scheduler_modules_and_time_imports():
    # Verify time is imported in scheduler.cron_scheduler and core.telegram_vault
    assert hasattr(cs, "time"), "scheduler.cron_scheduler must have time imported"
    assert hasattr(tv, "time"), "core.telegram_vault must have time imported"
    assert callable(cs.time.time)
    assert callable(tv.time.sleep)


def test_no_undefined_standard_modules_across_repo():
    import os
    import ast
    import builtins

    builtin_names = set(dir(builtins))
    found_issues = []

    for root, _, files in os.walk('.'):
        if any(skip in root for skip in ('.venv', '__pycache__', '.git', 'node_modules')):
            continue
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(root, f)
                with open(p, 'r', encoding='utf-8') as src:
                    try:
                        tree = ast.parse(src.read(), filename=p)
                    except Exception:
                        continue

                defined = set(builtin_names)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            defined.add(a.asname or a.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        for a in node.names:
                            defined.add(a.asname or a.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defined.add(node.name)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        defined.add(node.id)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                        mod_name = node.value.id
                        if mod_name in ('time', 'sys', 'os', 'math', 'json', 're', 'logging', 'requests', 'pd', 'np') and mod_name not in defined:
                            found_issues.append(f"{p}:{node.lineno}: undefined module {mod_name}.{node.attr}")

    assert len(found_issues) == 0, f"Found undefined standard modules: {found_issues}"


def test_afternoon_scan_schedule_configuration():
    from core.config import SCHEDULE_AFTERNOON_HOUR, SCHEDULE_AFTERNOON_MINUTE, SCHEDULE_MORNING_HOUR, SCHEDULE_MORNING_MINUTE
    assert SCHEDULE_AFTERNOON_HOUR == 15
    assert SCHEDULE_AFTERNOON_MINUTE == 0
    assert SCHEDULE_MORNING_HOUR == 10
    assert SCHEDULE_MORNING_MINUTE == 20


def test_keepalive_target_url_rejects_testserver():
    from scheduler.cron_scheduler import register_external_request_origin, get_keepalive_target_url
    from execution.state_engine import state_engine

    # Ensure testserver is ignored and never stored
    register_external_request_origin("http", "testserver")
    saved = state_engine.get_kv("RENDER_PUBLIC_URL")
    assert saved is None or "testserver" not in str(saved)

    url = get_keepalive_target_url()
    assert "testserver" not in url


def test_next_scheduled_scan_300pm_resolution():
    from core.market_calendar import NSEMarketCalendar
    import datetime
    from core.data_quality import IST

    # At 12:00 PM on a trading day, next scan should be 15:00 IST
    mock_noon = IST.localize(datetime.datetime(2026, 10, 14, 12, 0, 0))
    scan_info = NSEMarketCalendar.get_next_scheduled_scan(mock_noon)
    assert scan_info["target_time"].endswith("15:00:00 IST")
    assert "3:00 PM" in scan_info["label"]
    assert scan_info["scan_type"] == "AFTERNOON_300PM"


def test_get_institutional_status_imported_in_scheduler():
    import scheduler.cron_scheduler as cs
    assert hasattr(cs, "get_institutional_status"), "get_institutional_status must be imported in scheduler.cron_scheduler"
    assert callable(cs.get_institutional_status)
    status = cs.get_institutional_status()
    assert isinstance(status, dict)


def test_no_undefined_symbols_recursive_across_repo():
    import symtable
    import os
    import builtins

    builtin_names = set(dir(builtins))

    def walk_scopes(t):
        yield t
        for c in t.get_children():
            yield from walk_scopes(c)

    errors = []
    for r, _, files in os.walk('.'):
        if any(x in r for x in ['.git', '__pycache__', '.venv', 'venv']):
            continue
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(r, f)
                with open(p, 'r', encoding='utf-8') as fh:
                    t = symtable.symtable(fh.read(), p, 'exec')
                top_ids = set(t.get_identifiers())
                for sc in walk_scopes(t):
                    for s in sc.get_symbols():
                        if s.is_global() and not s.is_declared_global() and s.get_name() not in top_ids and s.get_name() not in builtin_names:
                            errors.append(f"{p} -> {sc.get_name()}: undefined {s.get_name()}")

    assert len(errors) == 0, f"Found undefined symbols in codebase: {errors}"


def test_nifty500_reconcile_open_positions_date_handling():
    import pandas as pd
    from engines.nifty500_engine import reconcile_open_nifty500_positions
    from execution.state_engine import state_engine

    # Test with dummy DataFrame and ensure no NameError is thrown
    dummy_df = pd.DataFrame([{
        "symbol": "MOCK_STOCK",
        "date": pd.Timestamp("2026-10-15"),
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 10000
    }])
    # Even if there are no open positions, function returns []
    events = reconcile_open_nifty500_positions(dummy_df)
    assert isinstance(events, list)


def test_afternoon_scan_worker_mock_execution(monkeypatch):
    import pandas as pd
    from execution.state_engine import state_engine
    import scheduler.cron_scheduler as cs

    # Mock InstitutionalEngine.run to return a mock BUY_NOW signal
    mock_df = pd.DataFrame([{
        "symbol": "TCS",
        "bucket": "BUY_NOW",
        "close": 3500.0,
        "entry": 3500.0,
        "sl": 3400.0,
        "target": 3700.0
    }])
    monkeypatch.setattr(cs.InstitutionalEngine, "run", lambda self, auto_execute=True, mode_override="afternoon": mock_df)
    monkeypatch.setattr(cs, "run_nifty500_pipeline_sync", lambda mode="SCHEDULED_300PM", auto_execute=True: {"orders_executed": 1})

    # Intercept execution_queue.submit_task to capture worker function
    submitted_tasks = []
    monkeypatch.setattr(cs.execution_queue, "submit_task", lambda task_fn, task_name, task_type, deduplicate=True: submitted_tasks.append((task_fn, task_name)))

    # Force calendar to return True
    monkeypatch.setattr(cs.NSEMarketCalendar, "is_trading_day", lambda d: True)
    # Ensure telemetry is hermetic and doesn't skip
    t_dict = {}
    monkeypatch.setattr(state_engine, "get_scheduled_scans_telemetry", lambda: t_dict)
    monkeypatch.setattr(state_engine, "record_scheduled_scan_outcome", lambda run_key, status, **kw: t_dict.update({run_key: {"status": status, **kw}}))

    cs.scheduled_afternoon_scan_job()

    assert len(submitted_tasks) == 1
    worker_fn, task_name = submitted_tasks[0]
    assert "Afternoon" in task_name

    # Execute the worker function synchronously to verify zero NameError / exceptions
    worker_fn()

    # Verify telemetry recorded SUCCESS and correct buy_now_count
    telemetry = state_engine.get_scheduled_scans_telemetry()
    afternoon = telemetry.get("afternoon_305", {})
    assert afternoon.get("status") == "SUCCESS", f"Expected SUCCESS but got: {afternoon}"
    assert afternoon.get("buy_now_count") == 1
    assert "BUY NOW" in afternoon.get("note", "")



