import sqlite3

conn = sqlite3.connect("/var/lib/nse-tradingdesk/swing_trades.db")
conn.row_factory = sqlite3.Row

print("=== quant_trade_dataset status breakdown ===")
rows = conn.execute("SELECT trade_type, status, count(*) as cnt FROM quant_trade_dataset GROUP BY trade_type, status").fetchall()
for r in rows:
    print(f"  trade_type={r['trade_type']}, status={r['status']} -> {r['cnt']} records")

print("\n=== unified_trade_learning_dataset status breakdown ===")
rows2 = conn.execute("SELECT record_type, status, count(*) as cnt FROM unified_trade_learning_dataset GROUP BY record_type, status").fetchall()
for r in rows2:
    print(f"  record_type={r['record_type']}, status={r['status']} -> {r['cnt']} records")

conn.close()
