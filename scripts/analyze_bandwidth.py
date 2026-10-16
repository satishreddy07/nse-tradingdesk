import collections
import re
import subprocess
import os

print("=== VPS Bandwidth & Traffic Diagnostics ===")

# 1. Network interface byte counts
try:
    with open("/proc/net/dev") as f:
        for line in f:
            if "eth0:" in line:
                parts = line.split()
                rx_bytes = int(parts[1])
                tx_bytes = int(parts[9])
                print(f"Total eth0 Lifetime: RX={rx_bytes / 1024 / 1024:.2f} MB, TX={tx_bytes / 1024 / 1024:.2f} MB (Total={(rx_bytes + tx_bytes) / 1024 / 1024:.2f} MB)")
except Exception as e:
    print(f"Error reading /proc/net/dev: {e}")

# 2. System uptime (to know over how many hours/days this 4GB accumulated)
try:
    with open("/proc/uptime") as f:
        uptime_sec = float(f.read().split()[0])
        print(f"System Uptime: {uptime_sec / 3600:.1f} hours ({uptime_sec / 86400:.2f} days)")
except Exception as e:
    print(f"Error reading /proc/uptime: {e}")

# 3. Analyze journalctl for nse-tradingdesk HTTP requests
try:
    cmd = ["journalctl", "-u", "nse-tradingdesk.service", "--since", "2026-10-16 00:00:00", "--no-pager"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate()
    
    endpoint_counts = collections.Counter()
    total_reqs = 0
    pattern = re.compile(r'"(GET|POST)\s+([^\s\?]+)')
    
    for line in out.splitlines():
        m = pattern.search(line)
        if m:
            total_reqs += 1
            method, path = m.group(1), m.group(2)
            endpoint_counts[(method, path)] += 1
            
    print(f"\nTotal HTTP requests recorded today: {total_reqs}")
    print("Top 15 Requested Endpoints:")
    for (m, path), count in endpoint_counts.most_common(15):
        print(f"  {count:5d} requests: {m} {path}")
except Exception as e:
    print(f"Error analyzing journalctl: {e}")

# 4. Check Nginx log if exists
try:
    if os.path.exists("/var/log/nginx/access.log"):
        nginx_counts = collections.Counter()
        nginx_bytes = 0
        nginx_reqs = 0
        with open("/var/log/nginx/access.log") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 9:
                    nginx_reqs += 1
                    try:
                        b = int(parts[9])
                        nginx_bytes += b
                    except ValueError:
                        pass
        print(f"\nNginx Access Log: {nginx_reqs} requests, {nginx_bytes / 1024 / 1024:.2f} MB transferred since Nginx install.")
except Exception as e:
    print(f"Error reading nginx log: {e}")
