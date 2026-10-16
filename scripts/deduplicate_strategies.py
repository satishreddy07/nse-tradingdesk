import sys
import sqlite3
import datetime
import os

def deduplicate_database(db_path: str):
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist.")
        return

    print(f"=== Inspecting and Deduplicating: {db_path} ===")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check if options_strategies exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='options_strategies'")
    if not c.fetchone():
        print("No options_strategies table found.")
        conn.close()
        return

    # Check columns
    cols = [r[1] for r in c.execute("PRAGMA table_info(options_strategies)").fetchall()]
    print(f"Columns: {cols}")

    # Fetch all active strategies
    c.execute("SELECT * FROM options_strategies WHERE status = 'ACTIVE' ORDER BY id DESC")
    active_rows = c.fetchall()
    print(f"\nTotal ACTIVE strategies found: {len(active_rows)}")

    # Group by (horizon/mode, expiry_type)
    # We want max 1 active INTRADAY, max 1 active WEEKLY, max 1 active MONTHLY
    seen_horizons = {}
    to_keep = []
    to_close = []

    for row in active_rows:
        mode = (row["auto_trade_mode"] or "MANUAL").upper()
        exp_type = (row["expiry_type"] or "WEEKLY").upper()
        
        # Determine category key
        if mode == "INTRADAY":
            key = "INTRADAY"
        elif "WEEKLY" in exp_type or mode == "WEEKLY":
            key = "WEEKLY"
        elif "MONTHLY" in exp_type or mode == "MONTHLY":
            key = "MONTHLY"
        else:
            key = f"{mode}_{exp_type}"

        if key not in seen_horizons:
            seen_horizons[key] = row["strategy_id"]
            to_keep.append((key, row))
        else:
            to_close.append((key, row))

    print(f"\nStrategies to KEEP ({len(to_keep)}):")
    for key, r in to_keep:
        print(f"  [KEEP - {key}] ID: {r['strategy_id']} | Name: {r['strategy_name']} | Expiry: {r['expiry_date']} | Created: {r['created_at']}")

    print(f"\nStrategies to CLOSE as DUPLICATES ({len(to_close)}):")
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key, r in to_close:
        print(f"  [CLOSE - {key}] ID: {r['strategy_id']} | Name: {r['strategy_name']} | Created: {r['created_at']}")
        c.execute("""
            UPDATE options_strategies
            SET status = 'CLOSED',
                closed_at = ?,
                exit_reason = 'DEDUPLICATED_BY_HARD_GATE',
                exit_note = 'Closed redundant duplicate strategy to enforce 1-active-strategy hard gate'
            WHERE strategy_id = ?
        """, (now_ts, r["strategy_id"]))

    conn.commit()

    # Re-verify active count
    c.execute("SELECT strategy_id, strategy_name, auto_trade_mode, expiry_type, expiry_date, status, created_at FROM options_strategies WHERE status='ACTIVE'")
    remaining_active = c.fetchall()
    print(f"\nVerification: Total remaining ACTIVE strategies: {len(remaining_active)}")
    for r in remaining_active:
        print(f"  -> ACTIVE: {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[6]}")

    conn.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/var/lib/nse-tradingdesk/swing_trades.db"
    deduplicate_database(path)
