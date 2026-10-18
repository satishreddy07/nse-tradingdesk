import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== quant_trade_dataset row 708 ===")
r = conn.execute("SELECT * FROM quant_trade_dataset WHERE id = 708").fetchone()
if r:
    for k in r.keys():
        v = r[k]
        if v is not None and str(v).strip():
            print(f"  {k}: {v}")

print("\n=== unified_trade_learning_dataset row 78 (COUNTERFACTUAL) ===")
r78 = conn.execute("SELECT * FROM unified_trade_learning_dataset WHERE id = 78").fetchone()
if r78:
    for k in r78.keys():
        v = r78[k]
        if v is not None and str(v).strip():
            print(f"  {k}: {v}")

print("\n=== unified_trade_learning_dataset row 79 (LIVE) ===")
r79 = conn.execute("SELECT * FROM unified_trade_learning_dataset WHERE id = 79").fetchone()
if r79:
    for k in r79.keys():
        v = r79[k]
        if v is not None and str(v).strip():
            print(f"  {k}: {v}")

conn.close()
