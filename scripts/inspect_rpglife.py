import subprocess

cmd = [
    "ssh",
    "root@104.168.91.147",
    "python3",
    "-c",
    "import sqlite3; conn = sqlite3.connect('/var/lib/nse-tradingdesk/swing_trades.db'); conn.row_factory = sqlite3.Row; r = dict(conn.execute(\"SELECT * FROM latest_signals WHERE symbol = 'RPGLIFE'\").fetchone()); print({k: r[k] for k in r if r[k] is not None})"
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
