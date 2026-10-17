import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:8000/api/status")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())

print(f"Status: {data.get('status')}")
print(f"Current Time (IST): {data.get('current_time_ist')}")
print(f"Market Status: {data.get('market_status')}")
print(f"Scheduler Running: {data.get('scheduler_running')}")
print(f"Trading Gate Open: {data.get('trading_gate_open')} ({data.get('trading_gate_message')})")
print(f"Next Scan: {data.get('next_scan')}")

jobs = data.get("jobs", [])
print(f"\nActive Autonomous Background Jobs ({len(jobs)} total):")
for j in jobs:
    print(f"  * {j.get('name')} -> Next Run: {j.get('next_run_time')}")
