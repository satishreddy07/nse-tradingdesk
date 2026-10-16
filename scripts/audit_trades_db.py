import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=== OPTIONS STRATEGIES ===")
c.execute("SELECT strategy_id, strategy_name, expiry_date, expiry_type, created_at, status, realized_pnl FROM options_strategies")
for r in c.fetchall():
    print(dict(r))

print("\n=== WEEKLY TRADE LOG ===")
c.execute("SELECT * FROM weekly_options_trade_log ORDER BY id DESC LIMIT 5")
for r in c.fetchall():
    print(dict(r))

print("\n=== MONTHLY TRADE LOG ===")
c.execute("SELECT * FROM monthly_options_trade_log ORDER BY id DESC LIMIT 5")
for r in c.fetchall():
    print(dict(r))

conn.close()
