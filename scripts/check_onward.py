import subprocess

cmd = [
    "ssh",
    "root@104.168.91.147",
    "python3",
    "-c",
    "import sqlite3; conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db'); print('latest_signals:', conn.execute('SELECT symbol, portfolio_type, bucket, score, dist_to_ema20, ml_verdict, rejection_reason FROM latest_signals WHERE symbol LIKE \"%ONWARD%\"').fetchall()); print('ohlcv_master_store:', conn.execute('SELECT symbol, last_close, change_pct, ema20 FROM ohlcv_master_store WHERE symbol LIKE \"%ONWARD%\"').fetchall())"
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
