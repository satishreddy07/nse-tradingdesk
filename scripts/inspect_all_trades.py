import sqlite3

conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== ALL ACTIVE POSITIONS ===")
for r in cur.execute("SELECT symbol, portfolio_type, entry_price, current_price, quantity, stop_loss, target, t1_hit, t1_pnl FROM active_positions"):
    print(dict(r))

print("\n=== ALL TRADE HISTORY ===")
for r in cur.execute("SELECT * FROM trade_history"):
    print(dict(r))
