#!/usr/bin/env python3
"""
Root entrypoint for Portfolio 3 Paper Trading & Backtesting Engine.
Executes the production-grade hybrid squeeze & split-exit algorithm.
"""
from engines.portfolio3_engine import (
    init_db,
    update_open_positions,
    evaluate_watchlist,
    scan_and_populate_watchlist,
    generate_report,
    get_chartink_universe,
    get_stock_data,
    DB_PATH,
    FRICTION_PCT,
    SCAN_CLAUSE
)

if __name__ == "__main__":
    init_db()
    # Step 1: Manage running trades
    update_open_positions()
    # Step 2: Trigger new breakouts
    evaluate_watchlist()
    # Step 3: Scan for next-day candidates
    scan_and_populate_watchlist()
    # Step 4: Display portfolio dashboard
    generate_report()
