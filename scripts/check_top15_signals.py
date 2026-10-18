import subprocess

top15 = [
    'ONWARDTEC', 'HBLENGINE', 'EUROPRATIK', 'TIPSMUSIC', 'TECHM',
    'SHANTIGOLD', 'MVGJL', 'SHRIPISTON', 'PNGSREVA', 'MBEL',
    'THANGAMAYL', 'SMSPHARMA', 'WIPRO', 'SKYGOLD', 'RPGLIFE'
]

syms_str = ",".join(f"'{s}'" for s in top15)

script = f"""
import sqlite3
conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db')
c = conn.cursor()
query = "SELECT symbol, bucket, score, dist_to_ema20, intraday_pct, ml_verdict, rejection_reason FROM latest_signals WHERE symbol IN ({syms_str})"
rows = c.execute(query).fetchall()
print(f'Found {{len(rows)}} of 15 in latest_signals:')
for r in rows:
    print(r)
"""

cmd = ["ssh", "root@104.168.91.147", "python3", "-c", script]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
