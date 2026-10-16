import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "data/swing_trades.db"
print(f"Connecting to {db_path}...")

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    print("--- Before Cleanup ---")
    rows = conn.execute("SELECT status, count(id) as cnt FROM options_strategies GROUP BY status").fetchall()
    for r in rows:
        print(f"  options_strategies status={r['status']}: {r['cnt']}")

    log_rows = conn.execute("SELECT status, count(id) as cnt FROM intraday_options_trade_log GROUP BY status").fetchall()
    for r in log_rows:
        print(f"  intraday_options_trade_log status={r['status']}: {r['cnt']}")

    # Check active strategies details
    active_rows = conn.execute("SELECT id, strategy_name, status, created_at, realized_pnl FROM options_strategies WHERE status = 'ACTIVE'").fetchall()
    print(f"Active strategies to KEEP ({len(active_rows)}):")
    for r in active_rows:
        print(f"  ID {r['id']}: {r['strategy_name']} (created: {r['created_at']}, realized: {r['realized_pnl']})")
