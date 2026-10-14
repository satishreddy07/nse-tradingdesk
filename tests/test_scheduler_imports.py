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

