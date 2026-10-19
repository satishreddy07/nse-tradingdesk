import subprocess

def run_ssh(cmd_str):
    print(f"\n--- Executing: {cmd_str} ---")
    res = subprocess.run(["ssh", "root@104.168.91.147", cmd_str], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res

# 1. Check git status on VPS
run_ssh("cd /opt/nse-tradingdesk && git status")

# 2. Git pull
run_ssh("cd /opt/nse-tradingdesk && git pull origin main")

# 3. Find service name running /opt/nse-tradingdesk
run_ssh("systemctl list-units --type=service | grep -i nse || grep -rnw '/etc/systemd/system/' -e '/opt/nse-tradingdesk' 2>/dev/null")

# 4. Restart service or supervisor
run_ssh("systemctl restart nse-tradingdesk.service || systemctl restart nse-tradingdesk || pkill -f 'uvicorn.*web.api:app'")

# 5. Check if process is running
run_ssh("sleep 3; ps aux | grep uvicorn")

# 6. Verify API endpoints
run_ssh("curl -s http://127.0.0.1:8000/api/status | head -c 200")
print("\nCheck movers research API:")
run_ssh("curl -s http://127.0.0.1:8000/api/research/movers-accumulation | head -c 300")
