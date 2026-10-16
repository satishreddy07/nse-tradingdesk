import sqlite3
import json

conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== LATEST SIGNALS FOR INDHOTEL ===")
rows = cur.execute("SELECT * FROM latest_signals WHERE symbol = 'INDHOTEL'").fetchall()
print(f"Signals found: {len(rows)}")
for r in rows:
    print(dict(r))

print("\n=== SWING DECISION AUDIT FOR INDHOTEL ===")
rows = cur.execute("SELECT * FROM swing_decision_audit WHERE symbol = 'INDHOTEL'").fetchall()
print(f"Audit records found: {len(rows)}")
for r in rows:
    print(dict(r))

conn.close()
