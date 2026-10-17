import os
import subprocess

ssh_key = os.environ.get("VPS_SSH_KEY", "")
vps_host = os.environ.get("VPS_HOST", "")

if not ssh_key or not vps_host:
    print("Error: Please set VPS_SSH_KEY and VPS_HOST environment variables to execute remote diagnostics.")
    exit(1)

remote_cmd = """python3 -c "
import sqlite3
for path in ['/var/backups/nse-tradingdesk/swing_trades.pre-migration-20261015-180717.db', '/var/backups/nse-tradingdesk/swing_trades.backup-20261015-171058.db']:
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        print('=== ' + path + ' ===')
        print('Active:')
        for r in conn.execute('SELECT symbol, quantity, entry_price, t1_hit, t1_pnl, entry_date FROM active_positions'):
            print(' ', dict(r))
        print('History:')
        for r in conn.execute('SELECT * FROM trade_history'):
            print(' ', dict(r))
    except Exception as e:
        print('Error with', path, e)
"
"""

cmd = ["ssh", "-n", "-i", ssh_key, f"root@{vps_host}", remote_cmd]
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("STDOUT:")
print(res.stdout.encode("ascii", errors="backslashreplace").decode("ascii"))
