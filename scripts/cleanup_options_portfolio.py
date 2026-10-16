import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "data/swing_trades.db"
apply_changes = "--apply" in sys.argv

print(f"Connecting to {db_path} (apply={apply_changes})...")

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    
    # 1. Count active strategies to preserve
    active_rows = conn.execute("SELECT id, strategy_name, status, created_at, realized_pnl FROM options_strategies WHERE status = 'ACTIVE'").fetchall()
    print(f"\n[PRESERVE] Found {len(active_rows)} ACTIVE strategies to KEEP:")
    for r in active_rows:
        print(f"  * ID {r['id']}: {r['strategy_name']} (Status: {r['status']}, Realized: {r['realized_pnl']})")

    # 2. Count closed strategies to delete
    closed_rows = conn.execute("SELECT id, strategy_name, status, realized_pnl FROM options_strategies WHERE status != 'ACTIVE'").fetchall()
    print(f"\n[DELETE] Found {len(closed_rows)} CLOSED strategies to remove.")

    # 3. Count closed intraday trade log records
    log_rows = conn.execute("SELECT id, strategy_id, status FROM intraday_options_trade_log WHERE status != 'ACTIVE'").fetchall()
    print(f"[DELETE] Found {len(log_rows)} intraday_options_trade_log rows to remove.")

    if apply_changes:
        c1 = conn.execute("DELETE FROM options_strategies WHERE status != 'ACTIVE'").rowcount
        c2 = conn.execute("DELETE FROM intraday_options_trade_log WHERE status != 'ACTIVE'").rowcount
        conn.commit()
        print(f"\n[COMMITTED] Successfully deleted {c1} non-active options_strategies and {c2} trade logs.")
        
        # Verify remaining
        remaining = conn.execute("SELECT id, strategy_name, status FROM options_strategies").fetchall()
        print(f"[VERIFY] Remaining options_strategies ({len(remaining)}):")
        for r in remaining:
            print(f"  * ID {r['id']}: {r['strategy_name']} (Status: {r['status']})")
    else:
        print("\n[DRY RUN] Run with --apply to execute changes.")
