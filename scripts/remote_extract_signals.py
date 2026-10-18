import sqlite3
import json
import sys

# Read symbols from stdin or file
symbols_file = sys.argv[1] if len(sys.argv) > 1 else None
if symbols_file:
    with open(symbols_file, 'r') as f:
        symbols = json.load(f)
else:
    symbols = []

conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
c = conn.cursor()

if symbols:
    placeholders = ",".join("?" for _ in symbols)
    query = f"SELECT symbol, bucket, score, dist_to_ema20, intraday_pct, ml_verdict, rejection_reason FROM latest_signals WHERE symbol IN ({placeholders})"
    rows = c.execute(query, symbols).fetchall()
else:
    rows = c.execute("SELECT symbol, bucket, score, dist_to_ema20, intraday_pct, ml_verdict, rejection_reason FROM latest_signals").fetchall()

res = {r[0]: {'bucket': r[1], 'score': r[2], 'dist_to_ema20': r[3], 'intraday_pct': r[4], 'ml_verdict': r[5], 'rejection_reason': r[6]} for r in rows}
print(json.dumps(res))
