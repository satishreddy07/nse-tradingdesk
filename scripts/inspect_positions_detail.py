import sqlite3

conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== ADVANCE DETAILS ===")
for r in cur.execute("SELECT * FROM active_positions WHERE symbol = 'ADVANCE'"):
    for k in r.keys():
        print(f"  {k}: {r[k]}")

print("\n=== UTTAMSUGAR DETAILS ===")
for r in cur.execute("SELECT * FROM active_positions WHERE symbol = 'UTTAMSUGAR'"):
    for k in r.keys():
        print(f"  {k}: {r[k]}")
