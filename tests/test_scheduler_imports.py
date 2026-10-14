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
