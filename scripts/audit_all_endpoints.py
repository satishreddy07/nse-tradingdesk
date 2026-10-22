#!/usr/bin/env python3
"""
Automated Production Health & Regression Audit Script.
Checks all 21 core API endpoints across all trading desk modules on the live VPS server.
"""

import urllib.request
import json
import sys

BASE_URL = "https://nseuniverse.duckdns.org"

ENDPOINTS = [
    ("/health", "Health Check"),
    ("/api/status", "System Status"),
    ("/api/nifty/live-spot", "Nifty Live Spot"),
    ("/api/stream/continuous", "Kotak Stream Continuous"),
    ("/api/nifty/intraday-expiry?interval=15m", "Intraday Expiry 15m"),
    ("/api/nifty/intraday-advanced", "Intraday Advanced"),
    ("/api/nifty/intraday-ml", "Intraday ML Slot Model"),
    ("/api/nifty/options-chain?expiry_type=weekly", "Options Chain Weekly"),
    ("/api/nifty/options-chain?expiry_type=monthly", "Options Chain Monthly"),
    ("/api/options/portfolio", "Options Portfolio"),
    ("/api/portfolio/PORTFOLIO_1/summary", "Portfolio 1 Summary"),
    ("/api/portfolio/PORTFOLIO_1/positions", "Portfolio 1 Positions"),
    ("/api/portfolio/PORTFOLIO_1/history?limit=5", "Portfolio 1 History"),
    ("/api/portfolio/PORTFOLIO_2/summary", "Portfolio 2 Summary"),
    ("/api/portfolio/PORTFOLIO_3/summary", "Portfolio 3 Summary"),
    ("/api/screener/compare/summary", "Screener Dual Audit Summary"),
    ("/api/screener/presets", "Screener Presets"),
    ("/api/scan/progress", "Scan Progress Queue"),
    ("/api/ml/overview", "ML Performance Overview"),
    ("/api/commodities/summary", "Commodities Summary"),
    ("/api/commodities/quotes", "Commodities Quotes")
]

def run_audit():
    print(f"Starting System Audit against {BASE_URL}...")
    passed = 0
    failed = 0

    for ep, name in ENDPOINTS:
        url = BASE_URL + ep
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NSETradingDesk-Auditor/1.0"})
            with urllib.request.urlopen(req, timeout=12) as res:
                code = res.getcode()
                raw = res.read().decode("utf-8", errors="ignore")
                try:
                    data = json.loads(raw)
                    status_field = data.get("status", "OK")
                    count = len(data) if isinstance(data, (dict, list)) else 1
                    msg = f"{count} items, status={status_field}"
                except Exception:
                    msg = f"{len(raw)} bytes"
                print(f"[PASS] [{code}] {name:<28} {ep:<45} -> {msg}")
                passed += 1
        except Exception as e:
            print(f"[FAIL] [ERR] {name:<28} {ep:<45} -> {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Audit Complete: {passed}/{len(ENDPOINTS)} endpoints PASSED, {failed} FAILED.")
    print("=" * 80)
    return failed == 0

if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
