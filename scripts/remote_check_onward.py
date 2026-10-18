import sqlite3

conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

r = c.execute("SELECT * FROM latest_signals WHERE symbol = 'ONWARDTEC'").fetchone()
if r:
    print("=== ONWARDTEC LATEST SIGNALS ROW ===")
    for k in r.keys():
        print(f"{k}: {r[k]}")

print("=== SWING DECISION AUDIT ===")
audit_rows = c.execute("SELECT * FROM swing_decision_audit WHERE symbol = 'ONWARDTEC'").fetchall()
for ar in audit_rows:
    for k in ar.keys():
        print(f"{k}: {ar[k]}")
