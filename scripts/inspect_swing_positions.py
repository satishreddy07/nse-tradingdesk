import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=== SCAN LOGS FOR RPEL ===")
c.execute("SELECT * FROM scan_logs WHERE details_json LIKE '%RPEL%' OR message LIKE '%RPEL%'")
for r in c.fetchall():
    d = dict(r)
    print(f"[{d['timestamp']}] Type: {d['scan_type']} | Msg: {d['message']}")
    if d.get("details_json"):
        print(f"   Details: {d['details_json']}")

print("\n=== LATEST 5 SCAN LOGS ===")
c.execute("SELECT id, timestamp, scan_type, portfolio_type, status, message FROM scan_logs ORDER BY id DESC LIMIT 5")
for r in c.fetchall():
    d = dict(r)
    print(f"ID {d['id']} [{d['timestamp']}] {d['scan_type']} ({d['portfolio_type']}): {d['message']}")

conn.close()
